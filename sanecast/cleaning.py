"""Limpeza do histórico: marca os bins que os dias parecidos não explicam.

Para cada dia `d`:
1. referência = mediana dos `cleaning_k` dias mais parecidos de `d` (sem o próprio `d`), multiplicada pelo nível do dia
   (mediana de y / referência no trecho diurno);
2. razão = média móvel de 3 bins de y / referência;
3. queda: `min_run_bins` bins seguidos com razão abaixo de `low_ratio`, de qualquer duração. Depois dela vem a retenção:
   os bins marcados continuam enquanto a razão fica acima de `retention_exit_ratio` e terminam quando `min_run_bins` bins seguidos
   voltam a ficar abaixo dela. Nenhuma duração é presumida;
4. pico anômalo: `min_run_bins` bins seguidos com razão acima de `high_ratio`.
A 2ª passada refaz a referência sem os bins marcados na 1ª e mede o nível do dia só nos bins que sobraram, para uma queda
longa não puxar o nível para baixo. Bins com referência abaixo de MIN_REFERENCE_COUNT não são julgados.
"""
import numpy as np

from .config import DAY_END_BIN, DAY_START_BIN
from .day_features import build_day_features
from .profile import similar_days_median

MIN_REFERENCE_COUNT = 15
MIN_LEVEL_BINS = 24  # bins diurnos livres de marca necessários para medir o nível do dia
OUTAGE, RETENTION, SURGE = 1, 2, 3


def find_runs(condition, min_len):
    """Trechos (início, fim) com `min_len` ou mais posições seguidas verdadeiras."""
    runs, t = [], 0
    while t < len(condition):
        j = t
        while j < len(condition) and condition[j]:
            j += 1
        if j - t >= min_len:
            runs.append((t, j))
        t = max(j, t + 1)
    return runs


def retention_end(ratio, reliable, start, cfg):
    """Bin em que termina a retenção que começa em `start`: o primeiro de `min_run_bins` bins seguidos abaixo de
    `retention_exit_ratio`, mais `retention_margin_bins` porque a média móvel borra a borda. Bins sem referência confiável
    contam como abaixo, para o ruído da madrugada não estender a retenção."""
    below = (ratio <= cfg.retention_exit_ratio) | ~reliable
    t = start
    while t + cfg.min_run_bins <= len(ratio) and not below[t : t + cfg.min_run_bins].all():
        t += 1
    return min(t + cfg.retention_margin_bins, len(ratio))


def clean_history(y, days, cfg):
    """Máscara booleana (dias, bins) com True nos bins sujos de `y`."""
    feats = build_day_features(days)
    n_days = len(days)
    marks = np.zeros(y.shape, np.int8)
    for _ in range(cfg.cleaning_passes):
        y_clean = np.where(marks > 0, np.nan, y)
        found = np.zeros(y.shape, np.int8)
        for d in range(n_days):
            others = [e for e in range(n_days) if e != d and np.isnan(y_clean[e]).mean() <= cfg.max_dirty_fraction]
            reference = similar_days_median(feats[d], others, y_clean, feats, cfg.cleaning_k)
            level = (y_clean[d] / np.maximum(reference, 1.0))[DAY_START_BIN:DAY_END_BIN]
            if np.isfinite(level).sum() < MIN_LEVEL_BINS:  # quase todo o trecho diurno está marcado
                level = (y[d] / np.maximum(reference, 1.0))[DAY_START_BIN:DAY_END_BIN]
            reference = reference * np.nanmedian(level)
            ratio = np.convolve(y[d], np.ones(3) / 3, "same") / np.maximum(reference, 1e-9)
            reliable = reference >= MIN_REFERENCE_COUNT
            for a, b in find_runs((ratio < cfg.low_ratio) & reliable, cfg.min_run_bins):
                found[d, max(a - 1, 0) : b] = OUTAGE
                end = retention_end(ratio, reliable, b, cfg)
                after = found[d, b:end]
                after[after == 0] = RETENTION
            for a, b in find_runs((ratio > cfg.high_ratio) & reliable, cfg.min_run_bins):
                segment = found[d, max(a - 1, 0) : b + 1]
                segment[segment == 0] = SURGE
        marks = found
    return marks > 0
