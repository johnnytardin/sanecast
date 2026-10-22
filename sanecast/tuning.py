"""Grade de hiperparâmetros com janela de validação e janela de teste.

Cada variante muda alguns campos de `Config`. O erro é medido contra a taxa verdadeira (wape_exp) nos bins sem evento,
em duas janelas: validação (dias -50 a -21 antes do último dia dos dados) e teste (dias -20 a 0). Os dias de cada janela
são previstos com dias anteriores a ela.
A limpeza é reaproveitada entre variantes que não mudam os parâmetros dela.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import DEFAULT
from .evaluate import scenario_masks, wape

VALIDATION_DAYS_AGO = (50, 21)
TEST_DAYS_AGO = (20, 0)

# nome da variante -> campos de Config que mudam em relação ao padrão
GRID = {
    "default": {},
    "profile_k 10": dict(profile_k=10),
    "profile_k 30": dict(profile_k=30),
    "similarity_margin 0.15": dict(similarity_margin=0.15),
    "similarity_margin 0.6": dict(similarity_margin=0.6),
    "similarity_margin 1.0": dict(similarity_margin=1.0),
    "shape_min_days 3": dict(shape_min_days=3),
    "shape_min_days 8": dict(shape_min_days=8),
    "smooth_window 3": dict(smooth_window=3),
    "smooth_window 7": dict(smooth_window=7),
    "smooth_target_count 0 (janela fixa)": dict(smooth_target_count=0.0),
    "smooth_target_count 8000": dict(smooth_target_count=8000.0),
    "alpha_target_count 1000": dict(alpha_target_count=1000.0),
    "alpha_target_count 6000": dict(alpha_target_count=6000.0),
    "alpha_max_window 36": dict(alpha_max_window=36),
    "alpha_prior_bins 3": dict(alpha_prior_bins=3),
    "alpha_prior_bins 12": dict(alpha_prior_bins=12),
    "alpha_tau 12": dict(alpha_tau=12),
    "alpha_tau 48": dict(alpha_tau=48),
}


def score(forecaster, window, truth):
    """Erro (%) contra a taxa verdadeira nos bins sem evento, por canal e horizonte (5 min, 1 h, perfil do dia)."""
    start, end = window
    run = forecaster.run(start, end)
    row = {}
    for channel, channel_run in run.channels.items():
        expected = truth["expected"][channel][run.eval_idx]
        masks = scenario_masks(forecaster.feats, run.eval_idx, truth["affected"][channel][run.eval_idx])
        clean = ~masks["event"]
        for label, forecast in (("5min", channel_run.forecast[1]), ("1h", channel_run.forecast[12]),
                                ("day", channel_run.profile)):
            row[f"{channel}_{label}"] = 100 * wape(expected, forecast, clean)
    row["mean"] = float(np.mean(list(row.values())))
    return row


def run_grid(forecaster, truth, grid=None, windows=None):
    """Roda cada variante nas duas janelas. Retorna uma tabela ordenada pelo erro médio da validação."""
    grid = GRID if grid is None else grid
    last = forecaster.days[-1]
    windows = windows or {name: tuple(last - pd.Timedelta(days=n) for n in ago)
                          for name, ago in (("validation", VALIDATION_DAYS_AGO), ("test", TEST_DAYS_AGO))}
    rows = []
    for name, changes in grid.items():
        variant = forecaster.with_config(replace(DEFAULT, **changes))
        row = {"variant": name}
        for window_name, window in windows.items():
            row.update({f"{window_name}:{k}": v for k, v in score(variant, window, truth).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("validation:mean").reset_index(drop=True)
