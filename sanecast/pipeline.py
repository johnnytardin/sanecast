"""Fluxo principal: limpa o histórico, monta o perfil de cada dia e roda a previsão online.

    days, series = load_series()
    forecaster = Forecaster(days, series)
    run = forecaster.run("AAAA-MM-DD", "AAAA-MM-DD")   # prevê cada dia do período usando só dias anteriores ao início
    run.channels["app"].forecast[1]                     # previsão de 5 min à frente: matriz (dias, 288)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cleaning import clean_history
from .config import BINS_PER_DAY, DEFAULT
from .day_features import build_day_features
from .forecasting import run_day
from .profile import LevelModel, daily_profile


@dataclass
class Prepared:
    """Histórico de um canal depois da limpeza."""
    clean: np.ndarray  # (dias, bins) contagens, com NaN nos bins sujos dos dias de treino
    usable: list  # índices dos dias que servem de dia parecido
    mask: np.ndarray  # (dias de treino, bins) True onde a limpeza marcou sujeira
    pool: list  # índices dos dias de treino
    level_model: LevelModel  # regressão do nível do dia


@dataclass
class ChannelRun:
    forecast: dict  # {h: (dias avaliados, bins)} previsão alinhada ao bin previsto
    state: np.ndarray  # (dias avaliados, bins) estado online
    profile: np.ndarray  # (dias avaliados, bins) perfil do dia
    features: np.ndarray  # (dias avaliados, bins, atributos) usados pelo LightGBM


@dataclass
class PeriodRun:
    eval_idx: list  # índices dos dias avaliados dentro de `days`
    channels: dict  # {canal: ChannelRun}


class Forecaster:
    def __init__(self, days, series, cfg=DEFAULT, _cleaning_cache=None):
        self.days, self.series, self.cfg = days, series, cfg
        self.feats = build_day_features(days)
        self._prepared = {}
        self._cleaning_cache = {} if _cleaning_cache is None else _cleaning_cache  # compartilhado entre configs

    def with_config(self, cfg):
        """Mesmos dados com outra configuração, reaproveitando a limpeza quando os parâmetros dela não mudam."""
        return Forecaster(self.days, self.series, cfg, self._cleaning_cache)

    def _cleaning_key(self, channel, train_end):
        c = self.cfg
        return (channel, train_end, c.cleaning_k, c.cleaning_passes, c.low_ratio, c.high_ratio, c.min_run_bins,
                c.retention_margin_bins, c.retention_exit_ratio, c.max_dirty_fraction)

    def prepare(self, train_end, treat=True):
        """Limpa o histórico anterior a `train_end`. Retorna {canal: Prepared}."""
        train_end = pd.Timestamp(train_end)
        key = (train_end, treat, self.cfg.max_dirty_fraction)
        if key in self._prepared:
            return self._prepared[key]
        pool = [i for i, d in enumerate(self.days) if d < train_end]
        prepared = {}
        for channel, y in self.series.items():
            if treat:
                ckey = self._cleaning_key(channel, train_end)
                if ckey not in self._cleaning_cache:
                    self._cleaning_cache[ckey] = clean_history(y[pool], self.days[pool], self.cfg)
                mask = self._cleaning_cache[ckey]
            else:
                mask = np.zeros((len(pool), BINS_PER_DAY), bool)
            clean = y.copy()
            clean[pool] = np.where(mask, np.nan, y[pool])
            usable = [e for e in pool if np.isnan(clean[e]).mean() <= self.cfg.max_dirty_fraction]
            prepared[channel] = Prepared(clean, usable, mask, pool, LevelModel(clean, usable, self.feats))
        self._prepared[key] = prepared
        return prepared

    def profile_for(self, day_index, prepared, exclude=None):
        """Perfil de um dia do histórico. `exclude` tira um dia da base (usado no treino do LightGBM)."""
        return self.profile_of(self.feats[day_index], prepared, exclude)

    def profile_of(self, day_feat, prepared, exclude=None):
        """Perfil de qualquer dia a partir do calendário dele (`build_day_features`), inclusive dias futuros."""
        usable = [e for e in prepared.usable if e != exclude]
        return daily_profile(day_feat, usable, prepared.clean, self.feats, self.cfg, prepared.level_model)

    def run(self, start, end, treat=True):
        """Previsão online de cada dia entre `start` e `end` (inclusive), usando só dias anteriores a `start`."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        eval_idx = [i for i, d in enumerate(self.days) if start <= d <= end]
        prepared = self.prepare(start, treat)
        channels = {}
        for channel, y in self.series.items():
            n_feat = 1 + len(self.cfg.ratio_lags)
            forecast = {h: np.full((len(eval_idx), BINS_PER_DAY), np.nan) for h in self.cfg.horizons}
            state = np.empty((len(eval_idx), BINS_PER_DAY), object)
            profile = np.zeros((len(eval_idx), BINS_PER_DAY))
            features = np.zeros((len(eval_idx), BINS_PER_DAY, n_feat))
            for k, d in enumerate(eval_idx):
                profile[k] = self.profile_for(d, prepared[channel])
                day_run = run_day(y[d], profile[k], self.cfg, treat)
                for h in self.cfg.horizons:
                    forecast[h][k] = day_run.forecast[h]
                state[k], features[k] = day_run.state, day_run.features
            channels[channel] = ChannelRun(forecast, state, profile, features)
        return PeriodRun(eval_idx, channels)
