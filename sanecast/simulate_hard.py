"""Simulador difícil: o mesmo formato de arquivos de `simulate.py`, mas sem as premissas que favorecem o modelo.

O simulador simples usa efeitos multiplicativos, a mesma forma de curva em todo dia e eventos iguais em todo histórico, que é
justamente a estrutura do modelo. Aqui, em cada canal:
- os efeitos sazonais se somam em vez de se multiplicar, e cada canal reage com uma intensidade (o whatsapp quase não sente feriado);
- fim de semana e feriado têm outra forma de curva, mais plana e com pico em outro horário;
- o nível tem ciclo anual, um degrau permanente de +12% e uma mudança de tendência;
- o nível de cada dia tem memória (AR(1) entre dias) e há dias anormais que nenhum calendário explica (±8% a 20%);
- um choque de nível comum a todos os canais (clima, feriado local, instabilidade geral) soma-se ao de cada canal;
- há um efeito de quinzena (dias 19 a 21) que o calendário do modelo não conhece e campanhas de 2 a 5 dias, sem aviso, que elevam um canal ou todos;
- a contagem é superdispersa (gama por minuto, mais ruído que Poisson);
- os eventos são sorteados por semente: quedas fortes com retenção variável, degradações parciais (20% a 50%) com rampa, picos,
  microquedas de 5 a 10 minutos, degradações lentas (15% a 40%, rampa de 30 a 60 minutos) e incidentes de 2 a 4 horas, em um canal ou em todos.
Não há dado real aqui: serve para medir quanto do resultado vem do formato do simulador.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .day_features import build_day_features
from .simulate import CHANNELS, FUTURE_DAYS, HISTORY_DAYS, SEASONAL_EFFECTS, _minutes, intraday_drift

STYLE = {  # intensidade do calendário, curva do dia (hora, largura, peso), piso, achatamento no fim de semana e efeito de sábado e domingo
    "app": dict(scale=1.0, peaks=((12, 2.2, 1.0), (19, 1.8, 1.3)), floor=0.35, flat=0.4, weekend=(0.9, 0.8)),
    "site": dict(scale=1.4, peaks=((11, 2.5, 1.0), (15, 2.0, 1.1)), floor=0.25, flat=0.6, weekend=(0.55, 0.4)),
    "whatsapp": dict(scale=0.6, peaks=((13, 2.5, 1.0), (21, 2.0, 1.4)), floor=0.3, flat=0.2, weekend=(1.1, 1.15)),
}
WEEKDAY = [1, 1, 1, 1.02, 1.1]  # segunda a sexta
DAY_OFF_FLAGS = ("holiday", "carnival", "long_weekend", "bridge", "new_year")


def _shape(hour, style, flat_weight):
    base = style["floor"] + sum(w * np.exp(-((hour - h) / s) ** 2) for h, s, w in style["peaks"])
    flat = 0.5 + np.exp(-((hour - 14) / 4.5) ** 2)
    shape = (1 - flat_weight) * base / base.mean() + flat_weight * flat / flat.mean()
    return shape / shape.mean()


def _calendar_factor(feat, style):
    dow = feat["dow"]
    weekday = style["weekend"][dow - 5] if dow >= 5 else WEEKDAY[dow]
    seasonal = sum(effect - 1 for name, effect in SEASONAL_EFFECTS.items() if feat[name])  # soma, não produto
    day = feat["day"].day
    start = 0.4 if 5 <= day <= 7 else 0.2 if 8 <= day <= 10 else 0.0
    payday = 0.08 if 19 <= day <= 21 else 0.0  # quinzena: o calendário do modelo só conhece o início do mês
    return max(weekday + style["scale"] * seasonal, 0.2) * (1 + (start + payday) * style["scale"])


def _day_flat_weight(feat, style):
    if any(feat[name] for name in DAY_OFF_FLAGS):
        return 0.8
    return style["flat"] if feat["dow"] >= 5 else 0.0


def _slow_level(days, rng):
    """Ciclo anual, degrau de +12% e mudança de tendência, mais memória entre dias e dias anormais sem causa conhecida."""
    n = len(days)
    index = np.arange(n) - (HISTORY_DAYS - 1)  # 0 é hoje
    annual = 1 + 0.06 * np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 30) / 365)
    trend = 1.2 + 0.001 * np.minimum(index + 120, 0) + 0.0003 * np.maximum(index + 120, 0)
    step = np.where(index >= -60, 1.12, 1.0)
    level = _ar1(n, rng, 0.04)
    unknown = np.ones(n)
    chosen = rng.choice(np.arange(10, HISTORY_DAYS + FUTURE_DAYS), size=12, replace=False)
    unknown[chosen] = 1 + rng.choice([-1, 1], size=len(chosen)) * rng.uniform(0.08, 0.2, size=len(chosen))
    return annual * trend * step * np.exp(level) * unknown


def _ar1(n, rng, sd, phi=0.5):
    noise, out = rng.standard_normal(n) * sd * np.sqrt(1 - phi**2), np.empty(n)
    out[0] = rng.standard_normal() * sd
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def _campaigns(rng, n_days):
    """Campanhas sem aviso: 4 a 6 sorteadas no histórico, de 2 a 5 dias, +10% a +35% com queda gradual, em um canal ou em todos.
    Retorna o fator por canal (dias,). O futuro não tem campanha."""
    factor = {c: np.ones(n_days) for c in CHANNELS}
    for _ in range(int(rng.integers(4, 7))):
        start, length = int(rng.integers(20, HISTORY_DAYS - 5)), int(rng.integers(2, 6))
        lift = rng.uniform(0.10, 0.35) * np.linspace(1, 0.5, length)
        target = list(CHANNELS) if rng.random() < 0.25 else [str(rng.choice(list(CHANNELS)))]
        for c in target:
            factor[c][start : start + length] *= 1 + lift
    return factor


def expected_rate(channel, level, days, rng, common=None, campaign=None):
    style, hour = STYLE[channel], np.arange(1440) / 60
    feats = build_day_features(days)
    shapes = {w: _shape(hour, style, w) for w in {_day_flat_weight(f, style) for f in feats}}
    calendar = np.array([_calendar_factor(f, style) for f in feats]) * _slow_level(days, rng)
    if common is not None:
        calendar = calendar * common
    if campaign is not None:
        calendar = calendar * campaign
    shape = np.array([shapes[_day_flat_weight(f, style)] for f in feats])
    return calendar[:, None] * level * shape * intraday_drift(len(days), rng, sd=0.05)


def _events(rng, n_days):
    """Eventos sorteados nos dias do histórico: dicionários com dia, início, duração, canal (None é todos), razão e retenção."""
    events = []
    for day in range(1, n_days):
        if rng.random() > 0.2:
            continue
        kind = rng.choice(["drop", "long", "partial", "spike", "micro", "slow"], p=[0.3, 0.1, 0.15, 0.2, 0.1, 0.15])
        channel = None if rng.random() < 0.3 else str(rng.choice(list(CHANNELS)))
        duration = {"drop": rng.integers(15, 91), "long": rng.integers(120, 241), "partial": rng.integers(30, 181),
                    "spike": rng.integers(15, 181), "micro": rng.integers(5, 11), "slow": rng.integers(120, 301)}[kind]
        ratio = {"drop": rng.uniform(0.05, 0.35), "long": rng.uniform(0.05, 0.3), "partial": rng.uniform(0.5, 0.8),
                 "spike": rng.uniform(1.4, 2.5), "micro": rng.uniform(0.1, 0.3), "slow": rng.uniform(0.6, 0.85)}[kind]
        retention = kind in ("drop", "long") and rng.random() < 0.7
        events.append(dict(day=day, start=int(rng.integers(7 * 60, 22 * 60 - duration)), duration=int(duration), channel=channel,
                           kind=str(kind), ratio=float(ratio), ramp=int(rng.integers(5, 16)) if kind in ("partial", "spike") else int(rng.integers(30, 61)) if kind == "slow" else 0,
                           retention=float(rng.uniform(0.5, 1.0) * 0.8 * (1 - ratio)) if retention else 0.0,
                           tau_scale=float(rng.uniform(0.7, 1.5))))
    return events


def event_factor(channel, n_days, events):
    factor = np.ones((n_days, 1440))
    for e in events:
        if e["channel"] not in (None, channel):
            continue
        d, s, dur, ramp = e["day"], e["start"], e["duration"], e["ramp"]
        weight = np.minimum(np.arange(1, dur + 1) / ramp, 1) if ramp else np.ones(dur)
        factor[d, s : s + dur] *= 1 + (e["ratio"] - 1) * weight
        if ramp:  # a rampa de saída continua depois do fim
            down = np.linspace(1, 0, ramp, endpoint=False)[1:]
            end = s + dur
            factor[d, end : end + len(down)] *= 1 + (e["ratio"] - 1) * down
        if e["retention"] > 0:
            tau = max(10, dur / 3) * e["tau_scale"]
            k = np.arange(min(int(3 * tau), 240, 1440 - s - dur))
            factor[d, s + dur : s + dur + len(k)] *= 1 + e["retention"] * np.exp(-k / tau)
    return factor


def _counts(rate, rng, cv=0.12):
    """Poisson com taxa que varia por minuto (gama de média 1): contagens superdispersas."""
    return rng.poisson(rate * rng.gamma(1 / cv**2, cv**2, size=rate.shape))


def simulate_hard(out_dir: Path = DATA_DIR, seed: int = 42, today=None):
    """Grava os três CSVs de `simulate.simulate`, com o simulador difícil. O futuro não tem eventos."""
    rng = np.random.default_rng(seed)
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    all_days = pd.date_range(end=today + pd.Timedelta(days=FUTURE_DAYS), periods=HISTORY_DAYS + FUTURE_DAYS, freq="D")
    history, future = all_days[:HISTORY_DAYS], all_days[HISTORY_DAYS:]
    events = _events(rng, HISTORY_DAYS)
    common = np.exp(_ar1(len(all_days), rng, 0.02))
    campaigns = _campaigns(rng, len(all_days))
    inputs, truth, ahead = [], [], []
    for channel, level in CHANNELS.items():
        rate = expected_rate(channel, level, all_days, rng, common, campaigns[channel])
        past = rate[:HISTORY_DAYS]
        factor = event_factor(channel, HISTORY_DAYS, events)
        inputs.append(pd.DataFrame({"ts": _minutes(history), "channel": channel, "count": _counts(past * factor, rng).ravel()}))
        truth.append(pd.DataFrame({"ts": _minutes(history), "channel": channel, "expected": past.ravel().round(3),
                                   "affected": (factor != 1).ravel().astype(int)}))
        ahead.append(pd.DataFrame({"ts": _minutes(future), "channel": channel, "count": _counts(rate[HISTORY_DAYS:], rng).ravel()}))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(inputs).to_csv(out_dir / "transactions_1min.csv", index=False)
    pd.concat(truth).to_csv(out_dir / "truth_1min.csv", index=False)
    pd.concat(ahead).to_csv(out_dir / "future_1min.csv", index=False)
    return out_dir / "transactions_1min.csv"
