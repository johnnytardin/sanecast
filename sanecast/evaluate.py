"""Métricas de erro.

Há dois erros, porque transações são contagens e o observado varia em torno da taxa verdadeira (desvio padrão = raiz da taxa):
- wape_obs: erro contra o observado. Uma previsão perfeita ainda erra cerca de 0,8 / raiz(taxa) por bin (`noise_floor`).
- wape_exp: erro contra a taxa verdadeira, que só existe na simulação.

WAPE = soma(|alvo - previsão|) / soma(alvo).
Cenários: normal, month_start (início de mês), holiday (feriado) e event (bins de queda, retenção ou pico).
"""
import numpy as np
import pandas as pd

from .day_features import FLAGS

HORIZON_LABELS = {1: "5min", 12: "1h", "day": "day_ahead"}
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


def noise_floor(expected, mask):
    """WAPE contra o observado de uma previsão perfeita: E|Poisson(l) - l| ~ raiz(2 l / pi)."""
    return float(np.sqrt(2 * expected[mask] / np.pi).sum() / expected[mask].sum())


def evaluate(feats, eval_idx, series, truth, methods):
    """Tabela com uma linha por canal, cenário, horizonte e método.

    methods: {nome: {canal: {horizonte: matriz (dias avaliados, bins)}}}, com horizonte 1, 12 ou "day".
    """
    rows = []
    for channel in series:
        observed = series[channel][eval_idx]
        expected = truth["expected"][channel][eval_idx]
        masks = scenario_masks(feats, eval_idx, truth["affected"][channel][eval_idx])
        for scenario, mask in masks.items():
            if mask.any() and scenario != "event":
                rows.append((channel, scenario, "any", "noise_floor", noise_floor(expected, mask), np.nan))
            for method, by_channel in methods.items():
                for h, forecast in by_channel[channel].items():
                    rows.append((channel, scenario, HORIZON_LABELS[h], method,
                                 np.nan if scenario == "event" else wape(observed, forecast, mask),
                                 wape(expected, forecast, mask)))
    return pd.DataFrame(rows, columns=["channel", "scenario", "horizon", "method", "wape_obs", "wape_exp"])


def coverage(series, truth, feats, eval_idx, bands):
    """Fração do alvo dentro de [P10, P90]. `bands`: {canal: {h: {quantil: matriz}}}. Em evento o alvo é a taxa verdadeira."""
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
                rows.append((channel, scenario, HORIZON_LABELS[h], float(((target >= lo) & (target <= hi))[ok].mean())))
    return pd.DataFrame(rows, columns=["channel", "scenario", "horizon", "coverage"])


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
