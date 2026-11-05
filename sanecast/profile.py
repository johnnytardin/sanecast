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


def _ridge_fit(X, target, ridge, scale=None):
    """`scale` multiplica a penalidade de cada coluna (padrão: 1)."""
    penalty = ridge * np.diag(np.ones(X.shape[1]) if scale is None else scale)
    penalty[0, 0] = 0.0  # o intercepto não é penalizado
    return np.linalg.solve(X.T @ X + penalty, X.T @ target)


def select_ridge(X, target, grid, default, folds=4, fold_days=14, scale=None):
    """Penalidade ridge com menor erro de previsão fora da amostra, por validação com origem móvel.

    Cada dobra ajusta com os dias anteriores e prevê os `fold_days` seguintes (as `folds` últimas janelas, em ordem de data).
    O erro é o quadrático em escala log, a mesma em que a regressão trabalha. `X` e `target` seguem a ordem do tempo.
    Com menos de `folds` * `fold_days` + 60 dias, devolve `default`: não há como validar."""
    n = len(target)
    if n < folds * fold_days + 60:
        return default
    errors = np.zeros(len(grid))
    for k in range(folds):
        cut = n - (folds - k) * fold_days
        for i, ridge in enumerate(grid):
            beta = _ridge_fit(X[:cut], target[:cut], ridge, scale)
            errors[i] += np.mean((X[cut : cut + fold_days] @ beta - target[cut : cut + fold_days]) ** 2)
    return float(grid[int(np.argmin(errors))])


class LevelModel:
    """Nível do dia por regressão log-linear (mínimos quadrados com ridge):

        log(nível) = intercepto + dia da semana + início de mês (por dia do mês) + cada marca de FLAGS + tendência

    Os efeitos se somam em escala log, então se multiplicam no nível. O nível de cada dia é a mediana de
    y / referência no trecho diurno, onde a referência é a mediana de todos os dias usáveis.

    A tendência é uma reta mais uma dobra para cada valor de `knots` (dias antes do fim do histórico), com penalidade `knot_scale` vezes
    maior que a dos demais termos. A reta sozinha extrapola mal quando o nível muda de ritmo (degrau, ciclo anual), e as dobras deixam o fim
    do histórico mandar na tendência sem que uma inclinação ruidosa domine. Depois de `trend_cap` dias do fim do histórico a tendência para de crescer.

    `ridge` é a penalidade. Com `ridge_grid`, a penalidade é escolhida entre os candidatos por validação temporal
    (`select_ridge`) e `ridge` vira o valor usado quando há poucos dias para validar.
    """

    def __init__(self, y_clean, usable, feats, ridge=1e-3, ridge_grid=None, knots=(), knot_scale=1000.0, trend_cap=7):
        self.first_day = feats[0]["day"]
        self.end_day = feats[max(usable)]["day"] + pd.Timedelta(days=1)  # primeiro dia depois do histórico usado
        self.knots, self.trend_cap = tuple(knots), trend_cap
        reference = np.maximum(_nanmedian(y_clean[usable]), 1.0)
        window = slice(DAY_START_BIN, DAY_END_BIN)
        self.levels = {d: float(np.nanmedian(y_clean[d, window] / reference[window])) for d in usable}
        X = np.array([self._design(feats[d]) for d in usable])
        target = np.log([self.levels[d] for d in usable])
        scale = np.ones(X.shape[1])
        scale[X.shape[1] - len(self.knots) :] = knot_scale  # as dobras são as últimas colunas
        self.ridge = select_ridge(X, target, ridge_grid, ridge, scale=scale) if ridge_grid else ridge
        self.beta = _ridge_fit(X, target, self.ridge, scale)

    def _design(self, f):
        dow = [1.0 if f["dow"] == k else 0.0 for k in range(1, 7)]  # segunda-feira é a referência
        month = [1.0 if (f["month_start"] and f["day"].day == k) else 0.0 for k in MONTH_START_DAYS]
        day = min(f["day"], self.end_day + pd.Timedelta(days=self.trend_cap))
        trend = (day - self.first_day).days / 100.0
        hinges = [max(0.0, (day - (self.end_day - pd.Timedelta(days=k))).days) / 100.0 for k in self.knots]
        return np.array([1.0, *dow, *f["vec"].astype(float), *month, trend, *hinges])

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
