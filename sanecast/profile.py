"""Perfil do dia: o volume esperado de cada bin antes de ver o dia.

perfil = forma média dos dias parecidos x nível do dia previsto por regressão.
"""
import warnings

import numpy as np
import pandas as pd

from .config import DAY_END_BIN, DAY_START_BIN
from .day_features import MONTH_START_DAYS, day_distance


def _nanmedian(matrix):
    with warnings.catch_warnings():  # colunas só com NaN ficam NaN e são interpoladas depois
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(matrix, axis=0)


def similar_days_median(target, pool, y_clean, feats, k):
    """Mediana, bin a bin, dos `k` dias de `pool` mais parecidos com `target`, suavizada em 3 bins."""
    distances = np.array([day_distance(target, feats[e]) for e in pool])
    picked = np.asarray(pool)[np.argsort(distances, kind="stable")[:k]]
    profile = pd.Series(_nanmedian(y_clean[picked])).interpolate(limit_direction="both")
    return profile.rolling(3, center=True, min_periods=1).mean().to_numpy()


def smooth_profile(profile, cfg, n_days):
    """Média móvel centrada. A janela tem `smooth_window` bins e cresce onde o volume é baixo,
    até juntar `smooth_target_count` transações (somando os `n_days` dias), com teto em `smooth_max_window`."""
    cumulative = np.concatenate([[0.0], np.cumsum(profile)])
    base_half = (cfg.smooth_window - 1) // 2
    max_half = (cfg.smooth_max_window - 1) // 2
    out = np.empty(len(profile))
    for i, value in enumerate(profile):
        windows_needed = cfg.smooth_target_count / (n_days * max(value, 0.1))
        half = int(min(max_half, max(base_half, np.ceil((windows_needed - 1) / 2))))
        lo, hi = max(0, i - half), min(len(profile), i + half + 1)
        out[i] = (cumulative[hi] - cumulative[lo]) / (hi - lo)
    return out


class LevelModel:
    """Nível do dia por regressão log-linear (mínimos quadrados com ridge):

        log(nível) = intercepto + dia da semana + início de mês (por dia do mês) + cada marca de FLAGS + tendência

    Os efeitos se somam em escala log, então se multiplicam no nível. O nível de cada dia é a mediana de
    y / referência no trecho diurno, onde a referência é a mediana de todos os dias usáveis.
    """

    def __init__(self, y_clean, usable, feats, ridge=1e-3):
        self.first_day = feats[0]["day"]
        reference = np.maximum(_nanmedian(y_clean[usable]), 1.0)
        window = slice(DAY_START_BIN, DAY_END_BIN)
        self.levels = {d: float(np.nanmedian(y_clean[d, window] / reference[window])) for d in usable}
        X = np.array([self._design(feats[d]) for d in usable])
        target = np.log([self.levels[d] for d in usable])
        penalty = ridge * np.eye(X.shape[1])
        penalty[0, 0] = 0.0  # o intercepto não é penalizado
        self.beta = np.linalg.solve(X.T @ X + penalty, X.T @ target)

    def _design(self, f):
        dow = [1.0 if f["dow"] == k else 0.0 for k in range(1, 7)]  # segunda-feira é a referência
        month = [1.0 if (f["month_start"] and f["day"].day == k) else 0.0 for k in MONTH_START_DAYS]
        trend = (f["day"] - self.first_day).days / 100.0
        return np.array([1.0, *dow, *f["vec"].astype(float), *month, trend])

    def predict(self, feat):
        """Nível esperado do dia, na mesma escala de `levels`."""
        return float(np.exp(self._design(feat) @ self.beta))


def daily_profile(target, pool, y_clean, feats, cfg, model):
    """Perfil de `target`: média das formas dos dias parecidos (y / nível de cada um) x nível previsto.

    Entram até `profile_k` dias dentro de `similarity_margin` do mais parecido. Se houver menos que
    `shape_min_days`, a margem aumenta de 0,1 em 0,1 até alcançar.
    """
    distances = np.array([day_distance(target, feats[e]) for e in pool])
    order = np.argsort(distances, kind="stable")
    margin = cfg.similarity_margin
    while True:
        chosen = order[distances[order] <= distances[order[0]] + margin]
        if len(chosen) >= min(cfg.shape_min_days, len(order)) or margin > 10:
            break
        margin += 0.1
    picked = np.asarray(pool)[chosen[: cfg.profile_k]]
    shapes = y_clean[picked] / np.array([model.levels[e] for e in picked])[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shape = np.nanmean(shapes, axis=0)
    shape = pd.Series(shape).interpolate(limit_direction="both").to_numpy()
    return smooth_profile(shape * model.predict(target), cfg, n_days=len(picked))
