"""Simulador de transações por minuto, com sazonalidade e eventos.

Gera três canais com volumes diferentes e três arquivos em data/:
- transactions_1min.csv (ts, channel, count): a única entrada do modelo, dos últimos HISTORY_DAYS dias até "hoje"
  (a data atual do computador, ou a data passada em `today`);
- truth_1min.csv (ts, channel, expected, affected): taxa verdadeira e bins afetados, usado só para medir o acerto;
- future_1min.csv (ts, channel, count): os FUTURE_DAYS dias depois de hoje, sorteados à parte. Só o painel lê este arquivo,
  para comparar a projeção com o que "aconteceria". Com dados reais ele não existe.

Sazonalidade: fim de semana, feriado e os dias à volta dele (véspera, dia seguinte, ponte, fim de semana prolongado), início, meio e fim de mês,
Carnaval, Black Friday, datas comerciais, fim de ano, virada do ano e férias escolares (SEASONAL_EFFECTS), crescimento de 0,1% ao dia
e deriva lenta do nível ao longo do dia.
Eventos (EVENTS): quedas seguidas de retenção e picos anômalos, de 20 minutos a 4 horas. A retenção dura tanto quanto a queda
(no mínimo 30 minutos, no máximo 4 horas).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .day_features import build_day_features

HISTORY_DAYS = 400  # dias de histórico simulado, contando hoje: mais de um ano, para cada dia sazonal aparecer ao menos uma vez
FUTURE_DAYS = 30  # dias simulados depois de hoje, usados só pelo painel
REFERENCE_DAY = "2026-10-21"  # último dia do histórico de referência, fixo para a documentação e os testes
CHANNELS = {"app": 200, "site": 40, "whatsapp": 5}  # transações por minuto, em média

# (dias antes de hoje, hora, duração em min, canal ou None para todos, efeito)
# efeito > 0: queda (0.9 = perde 90% do volume), seguida de retenção
# efeito < 0: pico anômalo sem retenção (-1.2 = volume sobe 120%)
EVENTS = [
    (189, "11:00", 30, None, 0.9),
    (154, "15:00", 25, "app", 0.85),
    (133, "19:00", 30, "site", 0.8),
    (118, "14:00", 30, "site", -1.2),
    (91, "09:00", 30, None, 0.9),
    (77, "20:00", 30, "app", -0.9),
    (64, "13:00", 30, "whatsapp", 0.8),
    (42, "16:00", 20, "app", 0.9),
    (33, "10:00", 30, "site", 0.8),
    (27, "17:30", 30, "whatsapp", -1.3),
    (19, "14:00", 30, None, 0.85),
    (15, "19:00", 30, "app", -0.8),
    (13, "10:30", 30, "app", 0.9),
    (9, "18:00", 25, "site", 0.7),
    (7, "13:00", 30, "whatsapp", -1.5),
    (6, "20:00", 30, "whatsapp", 0.8),
    (0, "11:00", 30, "app", 0.9),
    (0, "13:30", 30, "whatsapp", -1.2),
    # incidentes mais longos: de 1 a 4 horas
    (150, "09:30", 120, "app", 0.9),
    (112, "15:00", 60, "site", 0.85),
    (98, "10:00", 180, None, 0.9),
    (70, "18:00", 240, "whatsapp", 0.8),
    (50, "11:30", 120, "app", -0.9),
    (38, "13:00", 180, "site", -1.0),
    (22, "08:00", 90, "app", 0.9),
    (11, "16:00", 150, None, 0.85),
]
assert all(0 < e[2] and int(e[1][:2]) * 60 + int(e[1][3:]) + e[2] <= 1440 for e in EVENTS), "todo evento começa e termina no mesmo dia"


# efeito de cada dia sazonal sobre o volume (multiplicam entre si). O início de mês é tratado à parte, por dia do mês
SEASONAL_EFFECTS = {
    "holiday": 0.7, "eve": 1.15, "post": 0.92, "bridge": 0.82, "long_weekend": 0.88, "carnival": 0.75, "black_friday": 1.7,
    "cyber_monday": 1.25, "commercial": 1.1, "new_year": 0.85, "year_end": 1.1, "school_break": 0.93, "month_end": 1.08, "mid_month": 1.05,
}
WEEKDAY_EFFECT = [1, 1, 1, 1.02, 1.1, 0.9, 0.8]  # segunda a domingo


def calendar_factor(feat):
    factor = WEEKDAY_EFFECT[feat["dow"]]
    for name, effect in SEASONAL_EFFECTS.items():
        factor *= effect if feat[name] else 1
    return factor * (1.4 if 5 <= feat["day"].day <= 7 else 1.2 if 8 <= feat["day"].day <= 10 else 1)


def expected_rate(level, days, rng, first_index=0):
    """Taxa esperada (transações/min) por dia x minuto, sem evento. `first_index` é a posição do 1º dia desde o início do histórico."""
    hour = np.arange(1440) / 60
    shape = 0.35 + np.exp(-((hour - 12) / 2.2) ** 2) + 1.3 * np.exp(-((hour - 19) / 1.8) ** 2)  # picos às 12h e 19h
    shape = shape / shape.mean()
    day_factor = [
        calendar_factor(feat)
        * (1.2 + 0.001 * (first_index + i - HISTORY_DAYS + 1))  # crescimento de 0,1% ao dia; hoje vale 1,2
        * rng.lognormal(0, 0.03)  # variação própria de cada dia
        for i, feat in enumerate(build_day_features(days))
    ]
    return np.array(day_factor)[:, None] * level * shape[None, :] * intraday_drift(len(days), rng)


def intraday_drift(n_days, rng, sd=0.04, memory_minutes=180):
    """Fator multiplicativo que varia devagar ao longo do dia (processo AR(1) em escala log, desvio `sd`)."""
    phi = np.exp(-1 / memory_minutes)
    noise = rng.standard_normal((n_days, 1440)) * sd * np.sqrt(1 - phi**2)
    log_drift = np.empty((n_days, 1440))
    log_drift[:, 0] = rng.standard_normal(n_days) * sd
    for minute in range(1, 1440):
        log_drift[:, minute] = phi * log_drift[:, minute - 1] + noise[:, minute]
    return np.exp(log_drift)


def event_factor(channel, days):
    """Fator por dia x minuto: queda ou pico durante o evento e retenção depois da queda."""
    factor = np.ones((len(days), 1440))
    for days_ago, hhmm, duration, event_channel, effect in EVENTS:
        if event_channel not in (None, channel):
            continue
        i = len(days) - 1 - days_ago
        start = int(hhmm[:2]) * 60 + int(hhmm[3:])
        factor[i, start : start + duration] = 1 - effect
        if effect > 0:  # retenção: decai depois da queda e dura 3 constantes de tempo (30 min para queda de 30 min, no máximo 4 h)
            tau = max(10, duration / 3)
            k = np.arange(min(int(3 * tau), 240, 1440 - start - duration))
            factor[i, start + duration : start + duration + len(k)] = 1 + 0.8 * effect * np.exp(-k / tau)
    return factor


def _minutes(days):
    return (days.values[:, None] + np.arange(1440).astype("timedelta64[m]")).ravel()


def _simulate_future(seed, today):
    """Contagens dos FUTURE_DAYS dias depois de hoje, sem eventos (não se sabe o que vai acontecer)."""
    rng = np.random.default_rng(seed + 1)  # sorteio próprio: o histórico não muda quando a janela futura muda
    days = pd.date_range(today + pd.Timedelta(days=1), periods=FUTURE_DAYS, freq="D")
    first_index = HISTORY_DAYS
    frames = [pd.DataFrame({"ts": _minutes(days), "channel": channel,
                            "count": rng.poisson(expected_rate(level, days, rng, first_index)).ravel()})
              for channel, level in CHANNELS.items()]
    return pd.concat(frames)


def simulate(out_dir: Path = DATA_DIR, seed: int = 42, today=None):
    """Grava os três CSVs em `out_dir`. `today` (padrão: data do computador) é o último dia do histórico.

    A mesma semente e a mesma data geram os mesmos dados. Em outra data o calendário muda (feriados, início de mês)
    e os eventos acompanham: cada um acontece sempre a tantos dias antes de hoje.
    """
    rng = np.random.default_rng(seed)
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    days = pd.date_range(end=today, periods=HISTORY_DAYS, freq="D")
    ts = _minutes(days)
    inputs, truth = [], []
    for channel, level in CHANNELS.items():
        rate = expected_rate(level, days, rng)
        factor = event_factor(channel, days)
        count = rng.poisson(rate * factor).ravel()
        inputs.append(pd.DataFrame({"ts": ts, "channel": channel, "count": count}))
        truth.append(pd.DataFrame({"ts": ts, "channel": channel, "expected": rate.ravel().round(3),
                                   "affected": (factor != 1).ravel().astype(int)}))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(inputs).to_csv(out_dir / "transactions_1min.csv", index=False)
    pd.concat(truth).to_csv(out_dir / "truth_1min.csv", index=False)
    _simulate_future(seed, today).to_csv(out_dir / "future_1min.csv", index=False)
    return out_dir / "transactions_1min.csv"
