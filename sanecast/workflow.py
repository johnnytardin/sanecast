"""Avaliação de um período: roda o modelo, os modelos de comparação e calcula as métricas.

    days, series = load_series()
    result = run_evaluation(days, series, load_truth(days), "AAAA-MM-DD", "AAAA-MM-DD")
    result.metrics    # erro por canal, cenário, horizonte e método
    result.coverage   # cobertura da faixa P10 a P90
    result.cleaning   # acerto da limpeza no histórico
"""
from dataclasses import dataclass

import pandas as pd

from . import baselines, quantile_model
from .config import DEFAULT
from .evaluate import cleaning_accuracy, coverage, evaluate
from .pipeline import Forecaster


@dataclass
class Evaluation:
    metrics: pd.DataFrame
    coverage: pd.DataFrame
    cleaning: dict  # {canal: (bins sujos achados, marcações corretas, alarme falso em dia sazonal)}


def run_evaluation(days, series, truth, start, end, cfg=DEFAULT, with_statsforecast=True):
    forecaster = Forecaster(days, series, cfg)
    run = forecaster.run(start, end)
    raw = forecaster.run(start, end, treat=False)  # o mesmo modelo, sem limpeza nem estados
    models = quantile_model.train(forecaster, start)
    bands = quantile_model.predict(models, forecaster, run)

    channels = list(series)
    mid = sorted(cfg.quantiles)[len(cfg.quantiles) // 2]
    methods = {
        "model": {c: run.channels[c].forecast for c in channels},
        "model_lgbm_p50": {c: {h: bands[c][h][mid] for h in cfg.horizons} for c in channels},
        "no_cleaning": {c: raw.channels[c].forecast for c in channels},
        "last_week": baselines.last_week(series, run.eval_idx, cfg.horizons),
        "profile_day_ahead": {c: {"day": run.channels[c].profile} for c in channels},
    }
    if with_statsforecast:
        methods.update(baselines.statsforecast_day_ahead(forecaster, start, run.eval_idx))

    return Evaluation(
        metrics=evaluate(forecaster.feats, run.eval_idx, series, truth, methods),
        coverage=coverage(series, truth, forecaster.feats, run.eval_idx, bands),
        cleaning=cleaning_accuracy(forecaster.prepare(start), truth["affected"], forecaster.feats),
    )
