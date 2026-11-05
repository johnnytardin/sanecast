"""Validação com origem móvel e várias seeds do simulador.

Uma avaliação isolada (21 dias, uma seed) deixa cada feriado como um único dia sorteado. Aqui o modelo é avaliado em vários
períodos consecutivos, cada um previsto só com o que veio antes dele (origem móvel), e em vários históricos simulados
(seeds). O resultado traz a média e o desvio entre as execuções, e diz se uma diferença entre dois métodos é maior que a
variação do sorteio.

    table = rolling_origin(days, series, truth, n_origins=3, window_days=21)
    summarize(table)                         # média e desvio de cada métrica
    seeds_backtest([1, 2, 3], n_origins=3)   # a mesma coisa em vários históricos simulados
"""
import tempfile
from pathlib import Path

import pandas as pd

from . import simulate
from .config import DEFAULT
from .data import load_series, load_truth
from .workflow import run_evaluation

KEYS = ["channel", "scenario", "horizon", "method"]


def rolling_origin(days, series, truth, cfg=DEFAULT, n_origins=3, window_days=21, run_id=0):
    """Avalia `n_origins` períodos seguidos de `window_days` dias, do mais antigo ao mais recente, terminando no último dia dos dados.

    Retorna (métricas, cobertura): tabelas de `run_evaluation` empilhadas, com as colunas `origin` (0 é o período mais recente) e `run`.
    """
    last = days[-1]
    metrics, coverage = [], []
    for origin in range(n_origins):
        end = last - pd.Timedelta(days=origin * window_days)
        start = end - pd.Timedelta(days=window_days - 1)
        result = run_evaluation(days, series, truth, start, end, cfg, with_statsforecast=False)
        metrics.append(result.metrics.assign(origin=origin, run=run_id))
        coverage.append(result.coverage.assign(origin=origin, run=run_id))
    return pd.concat(metrics, ignore_index=True), pd.concat(coverage, ignore_index=True)


def seeds_backtest(seeds, cfg=DEFAULT, n_origins=3, window_days=21, generator=simulate.simulate):
    """Simula um histórico por seed (até `simulate.REFERENCE_DAY`) e roda `rolling_origin` em cada um. `generator` é `simulate.simulate` ou `simulate_hard.simulate_hard`."""
    metrics, coverage = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            folder = Path(tmp) / str(seed)
            generator(folder, seed=seed, today=simulate.REFERENCE_DAY)
            days, series = load_series(folder)
            m, c = rolling_origin(days, series, load_truth(days, folder), cfg, n_origins, window_days, run_id=seed)
            metrics.append(m)
            coverage.append(c)
    return pd.concat(metrics, ignore_index=True), pd.concat(coverage, ignore_index=True)


def summarize(table, value_columns):
    """Média, desvio e número de execuções de cada coluna, por canal, cenário, horizonte e método (ou os que existirem).

    Cada execução é um par (seed, período). O desvio mede quanto o número muda de um sorteio para outro."""
    keys = [k for k in KEYS if k in table]
    grouped = table.groupby(keys, sort=False)[list(value_columns)]
    out = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std")).join(grouped.size().rename("runs"))
    return out.reset_index()


def compare(table, value, baseline, challenger):
    """Diferença `challenger - baseline` de `value` em cada execução, por canal, cenário e horizonte.

    Retorna a média da diferença, o desvio e quantas execuções o desafiante venceu. Se a média é menor que o desvio, a
    diferença pode ser só sorteio."""
    keys = [k for k in KEYS if k != "method" and k in table] + ["run", "origin"]
    wide = table[table.method.isin([baseline, challenger])].pivot_table(index=keys, columns="method", values=value)
    diff = (wide[challenger] - wide[baseline]).rename("diff").reset_index()
    group = diff.groupby([k for k in keys if k not in ("run", "origin")], sort=False)["diff"]
    return group.agg(mean="mean", std="std", wins=lambda x: int((x < 0).sum()), runs="size").reset_index()

