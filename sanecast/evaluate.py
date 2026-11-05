"""Métricas de erro.

Há dois erros, porque transações são contagens e o observado varia em torno da taxa verdadeira (desvio padrão = raiz da taxa):
- wape_obs: erro contra o observado. Uma previsão perfeita ainda erra cerca de 0,8 / raiz(taxa) por bin (`noise_floor`).
- wape_exp: erro contra a taxa verdadeira, que só existe na simulação.

WAPE = soma(|alvo - previsão|) / soma(alvo).
Cenários: normal, month_start (início de mês), holiday (feriado) e event (bins de queda, retenção ou pico).
"""
import numpy as np
import pandas as pd

from .config import BIN_MINUTES
from .day_features import FLAGS

BAND_MISCOVERAGE = 0.2  # a faixa P10 a P90 deve errar 20% das vezes


def horizon_label(h):
    """Nome do horizonte: 1 bin vira "5min", 12 vira "1h", "day" vira "day_ahead"."""
    if h == "day":
        return "day_ahead"
    minutes = int(h) * BIN_MINUTES
    return f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes}min"
# dias com efeito forte, além de feriado e início de mês: o cenário "special"
SPECIAL_FLAGS = ("eve", "post", "bridge", "long_weekend", "carnival", "black_friday", "cyber_monday", "commercial", "new_year")


def scenario_masks(feats, eval_idx, affected):
    """Máscaras (dias, bins) de cada cenário. `affected` é a matriz do gabarito para os dias avaliados."""
    ones = np.ones((1, affected.shape[1]), bool)
    holiday = np.array([feats[d]["holiday"] for d in eval_idx])[:, None] * ones
    month_start = np.array([feats[d]["month_start"] and not feats[d]["holiday"] for d in eval_idx])[:, None] * ones
    special = np.array([any(feats[d][name] for name in SPECIAL_FLAGS) and not feats[d]["holiday"] and not feats[d]["month_start"]
                        for d in eval_idx])[:, None] * ones
    event = affected == 1
    return {"normal": ~event & ~holiday & ~month_start & ~special, "month_start": ~event & month_start,
            "holiday": ~event & holiday, "special": ~event & special, "event": event}


def wape(target, forecast, mask):
    ok = mask & ~np.isnan(forecast)
    return float(np.abs(target[ok] - forecast[ok]).sum() / target[ok].sum()) if ok.any() else np.nan


def pinball(target, forecast, q):
    """Perda pinball média do quantil `q` (0 a 1). Usa o mesmo alvo e a mesma previsão em qualquer escala."""
    diff = target - forecast
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_score(target, lo, hi, miscoverage=BAND_MISCOVERAGE):
    """Interval score de Gneiting e Raftery (2007): largura da faixa mais 2/alfa vezes o quanto o alvo ficou fora dela.
    Menor é melhor, e premia faixa estreita só quando ela cobre."""
    below, above = np.maximum(lo - target, 0), np.maximum(target - hi, 0)
    return float(np.mean((hi - lo) + (2 / miscoverage) * (below + above)))


def mase_scale(y, first_eval, weeks=4):
    """Erro absoluto médio da previsão "mesmo bin, 7 dias antes" nas `weeks` semanas anteriores ao período avaliado."""
    first = max(first_eval - 7 * weeks, 7)
    return float(np.abs(y[first:first_eval] - y[first - 7 : first_eval - 7]).mean())


def noise_floor(expected, mask):
    """WAPE contra o observado de uma previsão perfeita: E|Poisson(l) - l| ~ raiz(2 l / pi)."""
    return float(np.sqrt(2 * expected[mask] / np.pi).sum() / expected[mask].sum())


def evaluate(feats, eval_idx, series, truth, methods):
    """Tabela com uma linha por canal, cenário, horizonte e método.

    methods: {nome: {canal: {horizonte: matriz (dias avaliados, bins)}}}, com o horizonte em bins (1, 12, ...) ou "day".
    """
    rows = []
    for channel in series:
        observed = series[channel][eval_idx]
        expected = truth["expected"][channel][eval_idx]
        scale = mase_scale(series[channel], eval_idx[0])
        masks = scenario_masks(feats, eval_idx, truth["affected"][channel][eval_idx])
        for scenario, mask in masks.items():
            if mask.any() and scenario != "event":
                rows.append((channel, scenario, "any", "noise_floor", noise_floor(expected, mask), np.nan, np.nan))
            for method, by_channel in methods.items():
                for h, forecast in by_channel[channel].items():
                    ok = mask & ~np.isnan(forecast)
                    mase = np.nan if scenario == "event" or not ok.any() or scale == 0 else np.abs(observed - forecast)[ok].mean() / scale
                    rows.append((channel, scenario, horizon_label(h), method,
                                 np.nan if scenario == "event" else wape(observed, forecast, mask),
                                 wape(expected, forecast, mask), mase))
    return pd.DataFrame(rows, columns=["channel", "scenario", "horizon", "method", "wape_obs", "wape_exp", "mase"])


def coverage(series, truth, feats, eval_idx, bands):
    """Cobertura de [P10, P90], largura relativa, interval score relativo e pinball relativo dos três quantis.

    `bands`: {canal: {h: {quantil: matriz}}}. Em evento o alvo é a taxa verdadeira. Largura, interval score e pinball são divididos
    pela média do alvo, para valerem em qualquer volume. Menor é melhor, e a cobertura deve ficar perto de 80%."""
    rows = []
    for channel in series:
        observed = series[channel][eval_idx]
        expected = truth["expected"][channel][eval_idx]
        masks = scenario_masks(feats, eval_idx, truth["affected"][channel][eval_idx])
        for scenario, mask in masks.items():
            target = expected if scenario == "event" else observed
            for h, q in bands[channel].items():
                lo, hi = q[min(q)], q[max(q)]
                ok = mask & ~np.isnan(lo)
                if not ok.any():
                    continue
                mean = target[ok].mean()
                pin = np.mean([pinball(target[ok], q[level][ok], level) for level in q])
                rows.append((channel, scenario, horizon_label(h), float(((target >= lo) & (target <= hi))[ok].mean()),
                             float((hi - lo)[ok].mean() / mean), interval_score(target[ok], lo[ok], hi[ok]) / mean, pin / mean))
    return pd.DataFrame(rows, columns=["channel", "scenario", "horizon", "coverage", "width", "interval_score", "pinball"])


def cleaning_accuracy(prepared, truth_affected, feats):
    """Compara a máscara da limpeza com o gabarito.

    Retorna {canal: (bins sujos achados, marcações corretas, alarme falso em dia sazonal)}, todos em fração.
    """
    out = {}
    for channel, p in prepared.items():
        real = truth_affected[channel][p.pool] == 1
        seasonal = np.array([feats[d]["month_start"] or any(feats[d][name] for name in FLAGS) for d in p.pool])
        seasonal = seasonal[:, None] * np.ones((1, real.shape[1]), bool)
        found = (p.mask & real).sum() / real.sum()
        correct = (p.mask & real).sum() / p.mask.sum()
        false_alarm = (p.mask & ~real & seasonal).sum() / seasonal.sum()
        out[channel] = (float(found), float(correct), float(false_alarm))
    return out
