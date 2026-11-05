"""Testes do projeto. Rode com:  uv run tests/test_pipeline.py   (ou com pytest)

Os primeiros testes são rápidos e verificam uma peça cada. O último é de integração: roda a avaliação completa
(~2 min) e confere os limites de qualidade que o README promete.
"""
import json
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanecast import quantile_model, simulate  # noqa: E402
from sanecast.cleaning import clean_history  # noqa: E402
from sanecast.backtest import rolling_origin, summarize  # noqa: E402
from sanecast.config import DATA_DIR, DEFAULT  # noqa: E402
from sanecast.dashboard import build_dashboard  # noqa: E402
from sanecast.data import load_future, load_minutes, load_series, load_truth  # noqa: E402
from sanecast.day_features import FLAGS, build_day_features  # noqa: E402
from sanecast.evaluate import horizon_label, interval_score, pinball  # noqa: E402
from sanecast.forecasting import recovery_band, run_day  # noqa: E402
from sanecast.pipeline import Forecaster  # noqa: E402
from sanecast.profile import LevelModel, select_ridge  # noqa: E402
from sanecast.snapshots import day_ahead_band  # noqa: E402
from sanecast.workflow import run_evaluation  # noqa: E402

CHANNELS = ("app", "site", "whatsapp")


REFERENCE_DIR = DATA_DIR / "reference"  # dados fixos em REFERENCE_DAY: os limites dos testes valem para eles, não para o dia de hoje


def ensure_data():
    if not (REFERENCE_DIR / "transactions_1min.csv").exists():
        simulate.simulate(REFERENCE_DIR, today=simulate.REFERENCE_DAY)


def test_simulator_has_short_and_long_incidents_that_fit_in_one_day():
    durations = [event[2] for event in simulate.EVENTS]
    assert min(durations) <= 30 and max(durations) >= 180, durations
    assert all(int(e[1][:2]) * 60 + int(e[1][3:]) + e[2] <= 1440 for e in simulate.EVENTS)


def test_model_input_has_no_incident_column():
    ensure_data()
    columns = list(pd.read_csv(REFERENCE_DIR / "transactions_1min.csv", nrows=1).columns)
    assert columns == ["ts", "channel", "count"], columns


def test_state_machine_freezes_alpha_during_outage_and_ignores_retention():
    profile = np.full(288, 100.0)
    obs = profile.copy()
    obs[100:106] = 10.0  # queda de 30 min
    obs[106:112] = 180.0  # retenção logo depois
    run = run_day(obs, profile, DEFAULT)
    assert "OUTAGE" in set(run.state[100:112])
    assert run.state[103] == "OUTAGE"
    # com o alfa congelado, a previsão continua perto do perfil mesmo em cima do dado sujo
    assert abs(run.alpha[105] - 1.0) < 0.05
    assert abs(run.forecast[1][106] - 100.0) < 6.0
    # depois que o volume volta ao normal, o modelo volta a aprender
    assert run.state[150] == "NORMAL" and abs(run.alpha[150] - 1.0) < 0.05


def test_state_machine_keeps_the_outage_state_for_incidents_of_any_duration():
    profile = np.full(288, 100.0)
    for hours in (1, 4, 8):  # queda de 1, 4 e 8 horas, sempre com retenção depois
        length = hours * 12
        obs = profile.copy()
        obs[100 : 100 + length] = 10.0
        obs[100 + length : 100 + length + 12] = 180.0
        run = run_day(obs, profile, DEFAULT)
        assert set(run.state[103 : 100 + length]) == {"OUTAGE"}, hours  # confirmada em 15 min e mantida até o fim
        assert abs(run.alpha[100 + length - 1] - 1.0) < 0.05, hours  # o ritmo de antes da queda segue congelado
        assert abs(run.forecast[1][100 + length - 1] - 100.0) < 6.0, hours
        assert run.state[100 + length + 60] == "NORMAL" and abs(run.alpha[100 + length + 60] - 1.0) < 0.05, hours


def test_state_machine_keeps_a_long_anomalous_surge_out_of_the_alpha():
    profile = np.full(288, 100.0)
    obs = profile.copy()
    obs[100:136] = 300.0  # pico anormal de 3 horas
    run = run_day(obs, profile, DEFAULT)
    assert set(run.state[103:136]) == {"SURGE"}
    assert abs(run.alpha[135] - 1.0) < 0.05
    assert run.state[200] == "NORMAL"


def _history_with_outage(outage_bins, retention_factor, retention_bins):
    days = pd.date_range("2026-01-01", periods=14, freq="D")
    rng = np.random.default_rng(1)
    shape = 300 + 200 * np.sin(np.linspace(0, np.pi, 288))
    rate = np.tile(shape, (14, 1))
    rate[7, 120 : 120 + outage_bins] *= 0.1
    rate[7, 120 + outage_bins : 120 + outage_bins + retention_bins] *= retention_factor
    return clean_history(rng.poisson(rate).astype(float), days, DEFAULT)


def test_cleaning_marks_an_outage_and_its_retention_whatever_their_duration():
    for outage, retention, factor in ((6, 6, 1.7), (24, 24, 1.7), (48, 12, 1.7), (24, 30, 1.35)):  # 1.35 passa só pela regra da retenção
        mask = _history_with_outage(outage, factor, retention)
        end = 120 + outage + retention
        assert mask[7, 125 : 120 + outage].mean() > 0.95, (outage, retention)  # a queda inteira
        assert mask[7, 120 + outage + 2 : end - 2].mean() > 0.8, (outage, retention)  # a retenção, seja quanto durar
        assert mask[7, end + 6 : end + 40].mean() < 0.05, (outage, retention)  # depois dela, nada é marcado
        assert mask[np.arange(14) != 7].mean() < 0.01, (outage, retention)  # os outros dias ficam intactos


def test_cleaning_does_not_extend_the_mark_when_there_is_no_retention():
    mask = _history_with_outage(24, 1.0, 0)
    assert mask[7, 120 + 24 + 4 :].mean() < 0.02  # a queda termina e o dia volta a valer


def test_cleaning_ignores_a_slow_drift_below_the_reference_but_marks_an_abrupt_drop():
    """Dia com forma própria (ponte, Carnaval) afunda a razão sem degrau; só queda abrupta vale como evento."""
    days = pd.date_range("2026-01-01", periods=14, freq="D")
    shape = 300 + 200 * np.sin(np.linspace(0, np.pi, 288))
    rate = np.tile(shape, (14, 1))
    rate[7, 100:200] *= np.linspace(1.0, 0.4, 100)  # desce devagar, sem borda
    rate[9, 100:150] *= 0.1  # queda abrupta
    mask = clean_history(np.random.default_rng(1).poisson(rate).astype(float), days, DEFAULT)
    assert mask[7].mean() < 0.02
    assert mask[9, 105:145].mean() > 0.95


def test_state_machine_flags_anomalous_surge_but_not_a_seasonal_rise_already_in_the_profile():
    profile = np.full(288, 100.0)
    surge = profile.copy()
    surge[100:106] = 300.0  # pico que o perfil não explica
    assert "SURGE" in set(run_day(surge, profile, DEFAULT).state)
    seasonal = profile * 1.3  # dia inteiro 30% acima: o alfa absorve, não é evento
    assert set(run_day(seasonal, profile, DEFAULT).state) == {"NORMAL"}


def test_level_model_combines_effects_never_seen_together():
    """Feriado e início de mês nunca aparecem juntos no treino, mas a regressão combina os dois efeitos."""
    days = pd.date_range("2026-01-01", periods=300, freq="D")
    feats = build_day_features(days)
    rng = np.random.default_rng(0)
    truth = [(0.7 if f["holiday"] else 1.0) * (1.4 if f["month_start"] else 1.0) * (1.2 if f["dow"] >= 5 else 1.0)
             for f in feats]
    y = np.array([np.full(288, 100.0 * t * rng.lognormal(0, 0.01)) for t in truth])
    both = [i for i, f in enumerate(feats) if f["holiday"] and f["month_start"]]
    assert both, "o período precisa conter um feriado no início do mês"
    usable = [i for i in range(len(feats)) if i not in both]  # o treino nunca vê a combinação
    model = LevelModel(y, usable, feats)
    reference = model.predict(feats[usable[0]]) / truth[usable[0]]  # escala do modelo para o nível real
    for i in both:
        assert abs(model.predict(feats[i]) / (truth[i] * reference) - 1) < 0.1


def test_calendar_flags_cover_holidays_and_the_days_around_them_in_any_year():
    def flags(date):
        f = build_day_features(pd.DatetimeIndex([date]))[0]
        return {name for name in FLAGS if f[name]}
    assert "holiday" in flags("2027-02-09") and "carnival" in flags("2027-02-09")  # Carnaval, que depende da Páscoa
    assert "holiday" in flags("2027-03-26")  # Sexta-feira Santa
    assert "holiday" in flags("2026-06-04")  # Corpus Christi
    assert "holiday" in flags("2028-12-25") and "holiday" in flags("2030-11-20")  # outros anos, sem lista fixa
    assert flags("2026-04-20") == {"bridge", "mid_month"}  # segunda antes do feriado de terça
    assert "post" in flags("2026-04-22") and "eve" in flags("2026-06-03")
    assert "long_weekend" in flags("2026-11-21") and "long_weekend" in flags("2026-11-22")  # sexta 20/11 é feriado
    assert "black_friday" in flags("2026-11-27") and "cyber_monday" in flags("2026-11-30")
    assert "commercial" in flags("2026-05-10") and "new_year" in flags("2026-12-30") and "month_end" in flags("2026-02-28")


def test_level_model_learns_the_effect_of_each_seasonal_day_that_shows_up_in_the_history():
    ensure_data()
    days, series = load_series(REFERENCE_DIR)
    forecaster = Forecaster(days, series)
    prepared = forecaster.prepare(days[-1])["app"]
    effects = np.exp(prepared.level_model.beta[7 : 7 + len(FLAGS)])  # depois do intercepto e dos 6 dias da semana
    for name, estimated in zip(FLAGS, effects):
        seen = sum(forecaster.feats[d][name] for d in prepared.usable)
        if seen >= 5:  # um efeito visto menos vezes que isso fica sujeito ao acaso do próprio dia
            assert abs(estimated / simulate.SEASONAL_EFFECTS[name] - 1) < 0.08, (name, seen, estimated)


def test_reference_history_ends_on_the_reference_day():
    ensure_data()
    days, _ = load_series(REFERENCE_DIR)
    assert days[-1] == pd.Timestamp(simulate.REFERENCE_DAY) and len(days) == simulate.HISTORY_DAYS


def test_future_window_starts_the_day_after_the_history_and_shows_the_holiday_dip():
    ensure_data()
    days, _ = load_series(REFERENCE_DIR)
    future_days, future = load_future(REFERENCE_DIR)
    assert (future_days[0] - days[-1]).days == 1 and len(future_days) == simulate.FUTURE_DAYS
    totals = pd.Series(future["app"].sum(axis=1), index=future_days)
    around = totals[pd.to_datetime(["2026-11-03", "2026-11-04"])].mean()  # Finados é segunda; compara com dias úteis seguintes
    assert totals[pd.Timestamp("2026-11-02")] < 0.9 * around, (totals["2026-11-02"], around)


def test_day_ahead_band_covers_about_80_percent_without_seeing_the_day():
    ensure_data()
    days, series = load_series(REFERENCE_DIR)
    truth = load_truth(days, REFERENCE_DIR)
    forecaster = Forecaster(days, series)
    for channel in CHANNELS:
        inside = []
        for d in range(len(days) - 7, len(days)):  # cada dia só usa o histórico anterior a ele
            prepared = forecaster.prepare(days[d])
            lo, hi = day_ahead_band(forecaster, prepared, channel)
            profile = forecaster.profile_for(d, prepared[channel])
            clean = truth["affected"][channel][d] == 0
            inside.append(((series[channel][d] >= profile * lo) & (series[channel][d] <= profile * hi))[clean])
        coverage = np.concatenate(inside).mean()
        assert 0.70 < coverage < 0.90, (channel, coverage)


def test_dashboard_payload_has_history_with_snapshots_and_future_with_projection_only(tmp_path=None):
    ensure_data()
    days, series = load_series(REFERENCE_DIR)
    forecaster = Forecaster(days, series)
    models = quantile_model.train(forecaster, days[-3])
    future_days, future = load_future(REFERENCE_DIR)
    out = Path(tmp_path or DATA_DIR.parent / "out") / "dashboard_test.html"
    minute_days, minute_series = load_minutes(REFERENCE_DIR)
    minute_counts = {str(d.date()): {c: m[i] for c, m in minute_series.items()} for i, d in enumerate(minute_days)}
    build_dashboard(forecaster, models, 2, out, (future_days[:3], {c: m[:3] for c, m in future.items()}), minute_counts)
    html = out.read_text(encoding="utf-8")
    payload = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    out.unlink()
    assert [d["kind"] for d in payload["days"]] == ["past", "past", "today", "future", "future", "future"]
    assert payload["today"] == str(days[-1].date())
    today = payload["days"][2]
    assert today["nows"][0] == 1 and today["nows"][-1] == 287 and set(today["channels"]["app"]["snaps"]) == {str(n) for n in today["nows"]}
    app = today["channels"]["app"]
    assert len(app["real1"]) == 1440 and sum(app["real1"]) == sum(app["real"])  # contagem por minuto, para a granularidade de 1 minuto
    assert "snaps" not in payload["days"][3]["channels"]["app"] and len(payload["days"][3]["channels"]["app"]["lo"]) == 288


def test_conformal_shift_restores_the_coverage_of_a_band_that_is_too_narrow():
    rng = np.random.default_rng(0)
    y = rng.normal(1.0, 0.1, 4000)
    lo, hi = np.full(4000, 0.95), np.full(4000, 1.05)  # a faixa deveria ir de 0,87 a 1,13
    assert np.mean((y >= lo) & (y <= hi)) < 0.45
    shift = quantile_model.conformal_shift(lo[:2000], hi[:2000], y[:2000], 0.2)
    assert shift > 0
    inside = np.mean((y[2000:] >= lo[2000:] - shift) & (y[2000:] <= hi[2000:] + shift))
    assert 0.77 < inside < 0.83, inside


def test_probabilistic_metrics_on_tiny_examples():
    target = np.array([1.0, 2.0, 3.0])
    assert pinball(target, np.array([2.0, 2.0, 2.0]), 0.9) == (0.1 * 1 + 0 + 0.9 * 1) / 3
    # faixa [1, 2]: largura 1; o alvo 3 fica 1 acima, com custo 2/0,2 * 1 = 10
    assert abs(interval_score(target, np.full(3, 1.0), np.full(3, 2.0)) - (1 + 1 + 11) / 3) < 1e-12
    assert [horizon_label(h) for h in (1, 3, 12, 96, "day")] == ["5min", "15min", "1h", "8h", "day_ahead"]


def test_fast_path_confirms_an_extreme_drop_in_one_bin_only_where_the_noise_allows():
    obs_big = np.full(288, 10000.0)
    obs_big[100:106] = 1000.0  # queda de 90% em volume alto
    run = run_day(obs_big, np.full(288, 10000.0), DEFAULT)
    assert run.state[100] == "OUTAGE" and abs(run.alpha[100] - 1.0) < 0.05
    slow = run_day(obs_big, np.full(288, 10000.0), replace(DEFAULT, fast_low_ratio=0.0, fast_high_ratio=float("inf")))
    assert slow.state[100] == "NORMAL" and slow.state[102] == "OUTAGE"
    # em volume baixo um bin isolado a 30% é ruído de contagem: não dispara a via rápida
    profile_small = np.full(288, 25.0)
    obs_small = profile_small.copy()
    obs_small[100] = 7.0
    assert set(run_day(obs_small, profile_small, DEFAULT).state) == {"NORMAL"}


def test_slow_path_catches_a_ramped_degradation_that_the_alpha_absorbs_and_ignores_a_noisy_flat_day():
    profile = np.full(288, 5000.0)
    off = replace(DEFAULT, slow_window=0)
    for factor, expected in ((0.75, "OUTAGE"), (1.3, "SURGE")):
        obs = profile.copy()
        obs[100:108] *= np.linspace(1.0, factor, 8)  # rampa de 40 min, sem salto de bin a bin
        obs[108:220] *= factor
        assert set(run_day(obs, profile, off).state) == {"NORMAL"}  # sem a via lenta o alfa absorve o desvio
        run = run_day(obs, profile, DEFAULT)
        assert expected in set(run.state[108:200])
        assert abs(run.alpha[180] - 1.0) < 0.05  # o alfa ficou congelado no nível pré-evento
    noisy = np.random.default_rng(3).poisson(5000.0, 288) * np.random.default_rng(4).normal(1.0, 0.02, 288)
    assert set(run_day(noisy, profile, DEFAULT).state) == {"NORMAL"}


def test_recovery_band_is_narrow_for_large_counts_and_wide_for_small_ones():
    low, high = recovery_band(10000, DEFAULT)
    assert (low, high) == (1 - DEFAULT.recovery_margin, 1 + DEFAULT.recovery_margin)
    assert recovery_band(20, DEFAULT) == (DEFAULT.recovery_low, DEFAULT.recovery_high)  # ruído grande: os limites de sempre


def test_event_age_grows_through_the_event_and_the_retention_tail_then_resets():
    states = ["NORMAL"] * 3 + ["OUTAGE"] * 4 + ["NORMAL"] * 20
    age = quantile_model.state_age(states, DEFAULT)
    assert (age[:3] == 0).all()
    assert list(age[3:7]) == [1 + DEFAULT.min_run_bins + i for i in range(4)]
    assert age[7] == age[6] + 1 and (age[7 + DEFAULT.post_event_bins :] == 0).all()
    assert age[6 + DEFAULT.post_event_bins] > 0
    assert (quantile_model.state_age(states, replace(DEFAULT, widen_event_band=False)) == 0).all()


def _trending_level_model(**kwargs):
    days = pd.date_range("2026-01-01", periods=300, freq="D")
    feats = build_day_features(days)
    slope = np.where(np.arange(300) < 260, 0.0, 0.01 * (np.arange(300) - 260))  # plano e, nos últimos 40 dias, sobe 1% ao dia
    y = np.array([np.full(288, 100.0 * np.exp(s)) for s in slope])
    usable = list(range(300))
    future = build_day_features(pd.date_range(days[-1] + pd.Timedelta(days=1), periods=30, freq="D"))
    return LevelModel(y, usable, feats, **kwargs), future, np.exp(slope[-1])


def test_level_model_hinges_follow_a_recent_change_of_slope_and_the_trend_stops_after_the_cap():
    plain, future, last = _trending_level_model()
    hinged, _, _ = _trending_level_model(knots=(90, 30))
    expected = 100.0 * last * np.exp(0.01)  # um dia à frente, mantendo o ritmo recente
    ref = hinged.levels[0] / 1.0  # escala do modelo: nível observado no primeiro dia equivale a 100
    first = lambda m: m.predict(future[0]) / ref * 100.0  # noqa: E731
    assert abs(first(hinged) / expected - 1) < abs(first(plain) / expected - 1)
    capped = _trending_level_model(knots=(90, 30), trend_cap=7)[0]
    flat = [capped.predict(f) for f in future[7:]]  # depois do limite só o calendário muda o nível
    assert max(flat) / min(flat) < 1.5  # sem o limite, 23 dias a mais de inclinação somariam mais de 20%
    for f in future[7:]:
        twin = [g for g in future[7:] if g["dow"] == f["dow"] and g["month_start"] == f["month_start"] and g["vec"].tolist() == f["vec"].tolist()]
        values = [capped.predict(g) for g in twin]
        assert max(values) / min(values) < 1.0001


def test_hard_simulator_writes_full_days_up_to_the_reference_day(tmp_path=None):
    import tempfile
    from sanecast import simulate_hard

    out = Path(tmp_path or tempfile.mkdtemp())
    simulate_hard.simulate_hard(out, seed=3, today=simulate.REFERENCE_DAY)
    days, series = load_series(out)
    assert all(series[c].shape[1] == 288 for c in CHANNELS)
    assert days[-1] == pd.Timestamp(simulate.REFERENCE_DAY)
    assert not np.isnan(np.asarray(series["app"], float)).any()


def test_hard_simulator_campaigns_only_lift_the_history_and_the_day_shock_has_the_requested_spread():
    from sanecast import simulate_hard

    rng = np.random.default_rng(5)
    n = simulate.HISTORY_DAYS + simulate.FUTURE_DAYS
    campaigns = simulate_hard._campaigns(rng, n)
    for factor in campaigns.values():
        assert (factor >= 1).all() and (factor[simulate.HISTORY_DAYS :] == 1).all()
    assert any((f > 1).any() for f in campaigns.values())
    common = simulate_hard._ar1(n, np.random.default_rng(1), 0.02)
    assert 0.01 < common.std() < 0.03  # o desvio pedido, com a memória entre dias


def test_select_ridge_picks_a_grid_value_and_falls_back_when_there_is_no_room_to_validate():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    target = X @ np.array([0.5, -0.2, 0.1, 0.0]) + rng.normal(0, 0.05, 300)
    grid = (1e-4, 1e-2, 1.0)
    assert select_ridge(X, target, grid, default=1e-3) in grid
    assert select_ridge(X[:80], target[:80], grid, default=1e-3) == 1e-3


def test_rolling_origin_backtest_stacks_each_period_and_summarizes_the_spread():
    ensure_data()
    days, series = load_series(REFERENCE_DIR)
    metrics, coverage = rolling_origin(days, series, load_truth(days, REFERENCE_DIR), n_origins=2, window_days=7)
    assert set(metrics.origin) == {0, 1} and set(coverage.origin) == {0, 1}
    summary = summarize(metrics, ["wape_exp"])
    assert summary.runs.max() == 2 and summary.wape_exp_std.notna().any()  # um cenário raro pode faltar numa das janelas


def test_full_evaluation_meets_quality_limits():
    ensure_data()
    days, series = load_series(REFERENCE_DIR)
    result = run_evaluation(days, series, load_truth(days, REFERENCE_DIR), "2026-10-01", "2026-10-21")
    m = result.metrics.set_index(["channel", "scenario", "horizon", "method"])
    cov = result.coverage.set_index(["channel", "scenario", "horizon"]).coverage

    for channel in CHANNELS:
        found, correct, false_alarm = result.cleaning[channel]
        assert found > 0.80, f"{channel}: acha só {found:.0%} dos bins sujos"  # a cauda final de uma retenção longa já é quase normal
        assert correct > 0.50, f"{channel}: só {correct:.0%} das marcações estão corretas"  # canal pequeno gera trechos falsos isolados
        assert false_alarm < 0.005, f"{channel}: marcou sujeira em dia sazonal ({false_alarm:.2%})"
        for scenario in ("normal", "month_start", "special"):
            # o erro contra o observado fica a menos de 1,5 ponto percentual do piso de ruído; a oscilação lenta do simulador não é previsível
            floor = m.loc[(channel, scenario, "any", "noise_floor"), "wape_obs"]
            assert m.loc[(channel, scenario, "5min", "model"), "wape_obs"] < floor + 0.015
        # o feriado é um dia só, então o erro contra o observado depende do sorteio do ruído daquele dia. Compara-se com a taxa verdadeira
        holiday = m.loc[(channel, "holiday", "5min", "model"), "wape_exp"]
        normal = m.loc[(channel, "normal", "5min", "model"), "wape_exp"]
        assert holiday < 1.5 * normal + 0.01, (channel, holiday, normal)
        # em evento, de 20 minutos a 4 horas, o modelo prevê o que teria acontecido; sem tratamento a previsão acompanha a queda ou o pico
        model_error = m.loc[(channel, "event", "5min", "model"), "wape_exp"]
        raw_error = m.loc[(channel, "event", "5min", "no_cleaning"), "wape_exp"]
        assert model_error < raw_error / 3, (channel, model_error, raw_error)
        # feriado e início de mês: o perfil de dias parecidos vence os modelos estatísticos, que não conhecem o calendário
        for scenario in ("holiday", "month_start", "special"):
            mine = m.loc[(channel, scenario, "day_ahead", "profile_day_ahead"), "wape_exp"]
            theirs = m.loc[(channel, scenario, "day_ahead", "statsforecast_mstl"), "wape_exp"]
            assert mine < theirs / 2, (channel, scenario, mine, theirs)
        # faixa P10 a P90 calibrada em dias comuns
        for scenario in ("normal", "month_start", "holiday", "special"):
            assert 0.72 < cov[channel, scenario, "5min"] < 0.88, (channel, scenario, cov[channel, scenario, "5min"])
        # a calibração conformal mantém a cobertura nos horizontes mais longos
        for horizon in ("1h", "2h", "4h"):
            assert 0.70 < cov[channel, "normal", horizon] < 0.90, (channel, horizon, cov[channel, "normal", horizon])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, func in tests:
        func()
        print(f"ok  {name}")
    print(f"{len(tests)} testes passaram")
