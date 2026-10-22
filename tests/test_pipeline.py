"""Testes do projeto. Rode com:  uv run tests/test_pipeline.py   (ou com pytest)

Os primeiros testes são rápidos e verificam uma peça cada. O último é de integração: roda a avaliação completa
(~2 min) e confere os limites de qualidade que o README promete.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanecast import quantile_model, simulate  # noqa: E402
from sanecast.cleaning import clean_history  # noqa: E402
from sanecast.config import DATA_DIR, DEFAULT  # noqa: E402
from sanecast.dashboard import build_dashboard  # noqa: E402
from sanecast.data import load_future, load_minutes, load_series, load_truth  # noqa: E402
from sanecast.day_features import FLAGS, build_day_features  # noqa: E402
from sanecast.forecasting import run_day  # noqa: E402
from sanecast.pipeline import Forecaster  # noqa: E402
from sanecast.profile import LevelModel  # noqa: E402
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


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, func in tests:
        func()
        print(f"ok  {name}")
    print(f"{len(tests)} testes passaram")
