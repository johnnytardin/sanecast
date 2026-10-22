"""Painel HTML único e autocontido para simular o modelo no navegador.

O painel mostra os últimos dias, o dia de referência e os próximos dias simulados, por canal:
- modo "Dia": observado até o horário "agora", projeção do resto do dia, faixa P10 a P90 e o que aconteceu depois;
- modo "Vários dias": uma linha do tempo com todos os dias do período.

As projeções de todos os horários são calculadas aqui e gravadas como JSON dentro do HTML; o navegador só desenha.
O dia de referência é o último dia do histórico. Os dias seguintes só têm projeção (perfil do dia e faixa) e, na simulação,
o valor simulado para comparar.
"""
import json
from pathlib import Path

import pandas as pd

from .config import BIN_MINUTES, BINS_PER_DAY
from .day_features import build_day_features
from .snapshots import compute_snapshots, day_ahead_band

TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"
STATE_CODES = {"NORMAL": 0, "OUTAGE": 1, "SURGE": 2, "RECOVERY": 3}
# marcas do calendário que o painel mostra como etiqueta de cada dia, na ordem de exibição
TAG_LABELS = [("holiday", "feriado"), ("eve", "véspera de feriado"), ("post", "dia seguinte ao feriado"), ("bridge", "ponte"),
              ("long_weekend", "fim de semana prolongado"), ("carnival", "Carnaval"), ("black_friday", "Black Friday"),
              ("cyber_monday", "Cyber Monday"), ("commercial", "data comercial"), ("new_year", "virada do ano"),
              ("month_start", "início de mês"), ("mid_month", "meio de mês"), ("month_end", "fim de mês"),
              ("year_end", "fim de ano"), ("school_break", "férias escolares")]
WEEKDAYS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
PAST_NOW_STEP = 6  # dias anteriores: um horário "agora" a cada 30 min, mais o fim do dia
LAST_NOW_BIN = BINS_PER_DAY - 1  # o último "agora" é 23:55, quando não sobra nada para projetar
DEFAULT_NOW_BIN = (15 * 60 + 30) // BIN_MINUTES  # horário "agora" com que o painel abre (15:30)


def _round(values):
    return [round(float(v), 1) for v in values]


def _tags(feat):
    return [text for key, text in TAG_LABELS if feat[key]]


def _snaps(channel_data):
    return {str(now): {"p50": _round(p50), "lo": _round(lo), "hi": _round(hi)}
            for now, (p50, lo, hi) in channel_data["snaps"].items()}


def _history_day(forecaster, d, snapshot, bands, kind):
    """Dia com histórico: observado, perfil, faixa do dia e projeções online por horário."""
    channels = {}
    for channel, c in snapshot.items():
        lo, hi = bands[channel]
        channels[channel] = {
            "real": [int(v) for v in c["real"]], "profile": _round(c["profile"]),
            "lo": _round(c["profile"] * lo), "hi": _round(c["profile"] * hi),
            "state": [STATE_CODES[s] for s in c["state"]], "alpha": [round(float(a), 3) for a in c["alpha"]],
            "snaps": _snaps(c),
        }
    return _day_entry(forecaster.days[d], forecaster.feats[d], kind, channels, sorted(next(iter(snapshot.values()))["snaps"]))


def _day_entry(date, feat, kind, channels, nows):
    return {"date": str(date.date()), "weekday": WEEKDAYS[date.dayofweek], "kind": kind, "tags": _tags(feat),
            "nows": nows, "channels": channels}


def _future_day(forecaster, prepared, bands, date, feat, observed):
    """Dia depois do histórico: só perfil e faixa; `observed` é o valor simulado (None com dados reais)."""
    channels = {}
    for channel, p in prepared.items():
        profile = forecaster.profile_of(feat, p)
        lo, hi = bands[channel]
        channels[channel] = {"real": [int(v) for v in observed[channel]], "profile": _round(profile),
                             "lo": _round(profile * lo), "hi": _round(profile * hi)}
    return _day_entry(date, feat, "future", channels, [])


def build_dashboard(forecaster, models, n_past, out_path, future=(None, None), minute_counts=None):
    """Escreve o painel em `out_path`.

    `n_past`: quantos dias antes do dia de referência entram no painel. `future`: (dias, séries) da janela futura simulada.
    `minute_counts`: {data: {canal: 1440 contagens}} para a granularidade de 1 minuto (opcional).
    `models` deve ter sido treinado com dados anteriores ao primeiro dia mostrado.
    """
    today = len(forecaster.days) - 1
    past = list(range(today - n_past, today))
    days = []
    if past:
        past_snaps = compute_snapshots(forecaster, models, past, list(range(PAST_NOW_STEP, LAST_NOW_BIN, PAST_NOW_STEP)) + [LAST_NOW_BIN])
        prepared = forecaster.prepare(forecaster.days[past[0]])
        bands = {ch: day_ahead_band(forecaster, prepared, ch) for ch in forecaster.series}
        days += [_history_day(forecaster, d, past_snaps[d], bands, "past") for d in past]
    today_snaps = compute_snapshots(forecaster, models, [today], list(range(1, LAST_NOW_BIN + 1)))
    prepared = forecaster.prepare(forecaster.days[today])
    bands = {ch: day_ahead_band(forecaster, prepared, ch) for ch in forecaster.series}
    days.append(_history_day(forecaster, today, today_snaps[today], bands, "today"))
    future_days, future_series = future
    if future_days is not None:
        feats = build_day_features(future_days)
        for i, (date, feat) in enumerate(zip(future_days, feats)):
            days.append(_future_day(forecaster, prepared, bands, date, feat, {ch: m[i] for ch, m in future_series.items()}))

    for day in days:
        for channel, c in day["channels"].items():
            row = (minute_counts or {}).get(day["date"], {}).get(channel)
            if row is not None:
                c["real1"] = [int(v) for v in row]

    payload = json.dumps({
        "today": str(forecaster.days[today].date()),
        "defaultNow": DEFAULT_NOW_BIN,
        "days": days}, separators=(",", ":"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload), encoding="utf-8")
    return out_path
