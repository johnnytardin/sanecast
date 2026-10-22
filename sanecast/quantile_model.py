"""LightGBM quantílico (P10, P50, P90) sobre o perfil do dia.

O alvo é a razão y(t+h) / perfil(t+h). Um modelo por quantil, o mesmo para todos os canais.
Atributos: alfa, razões observado/perfil recentes, horizonte, log do perfil, hora do bin previsto (seno e cosseno),
dia da semana, feriado, véspera, dia seguinte, ponte, fim de semana prolongado, início e fim de mês e canal.

Treino: só linhas com estado NORMAL e sem bins marcados como sujos no atributo nem no alvo.
Previsão: em OUTAGE, SURGE ou RECOVERY o centro da faixa vem da previsão do modelo base (alfa congelado)
e o LightGBM define só a largura relativa da faixa.
"""
import lightgbm as lgb
import numpy as np

from .config import BINS_PER_DAY
from .forecasting import run_day

PARAMS = dict(objective="quantile", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, verbose=-1)


def build_attributes(features, profile, day_feat, channel_id, cfg):
    """Uma matriz de atributos por horizonte, com uma linha para cada bin t em que t + h cabe no dia."""
    blocks = []
    for h in cfg.horizons:
        t = np.arange(BINS_PER_DAY - h)
        target = t + h
        n = len(t)
        blocks.append(np.column_stack([
            features[t], np.full(n, h), np.log1p(profile[target]),
            np.sin(2 * np.pi * target / BINS_PER_DAY), np.cos(2 * np.pi * target / BINS_PER_DAY),
            np.full(n, day_feat["dow"]), np.full(n, day_feat["holiday"]), np.full(n, day_feat["eve"]), np.full(n, day_feat["post"]),
            np.full(n, day_feat["bridge"]), np.full(n, day_feat["long_weekend"]), np.full(n, day_feat["month_start"]),
            np.full(n, day_feat["month_end"]), np.full(n, channel_id)]))
    return blocks


def train(forecaster, train_end):
    """Treina um modelo por quantil com os dias anteriores a `train_end`. Retorna {quantil: modelo}."""
    cfg = forecaster.cfg
    prepared = forecaster.prepare(train_end)
    rows, targets = [], []
    for channel_id, (channel, y) in enumerate(forecaster.series.items()):
        p = prepared[channel]
        for i, d in enumerate(p.pool):
            profile = forecaster.profile_for(d, p, exclude=d)  # perfil sem o próprio dia
            day_run = run_day(y[d], profile, cfg)
            for h, block in zip(cfg.horizons, build_attributes(day_run.features, profile, forecaster.feats[d],
                                                              channel_id, cfg)):
                t = np.arange(BINS_PER_DAY - h)
                clean = (day_run.state[t] == "NORMAL") & ~p.mask[i, t] & ~p.mask[i, t + h]
                rows.append(block[clean])
                targets.append((y[d][t + h] / np.maximum(profile[t + h], 1.0))[clean])
    X, Y = np.vstack(rows), np.concatenate(targets)
    categorical = [X.shape[1] - 1]  # a última coluna é o canal
    return {q: lgb.train({**PARAMS, "alpha": q}, lgb.Dataset(X, Y, categorical_feature=categorical), cfg.lgbm_rounds)
            for q in cfg.quantiles}


def predict(models, forecaster, run):
    """Faixas P10/P50/P90 de um PeriodRun. Retorna {canal: {h: {quantil: matriz (dias, bins)}}}."""
    cfg = forecaster.cfg
    quantiles = list(models)
    mid = len(quantiles) // 2
    bands = {}
    for channel_id, channel in enumerate(forecaster.series):
        cr = run.channels[channel]
        bands[channel] = {h: {q: np.full(cr.profile.shape, np.nan) for q in quantiles} for h in cfg.horizons}
        for k, d in enumerate(run.eval_idx):
            blocks = build_attributes(cr.features[k], cr.profile[k], forecaster.feats[d], channel_id, cfg)
            for h, block in zip(cfg.horizons, blocks):
                t = np.arange(BINS_PER_DAY - h)
                pred = np.sort(np.column_stack([models[q].predict(block) for q in quantiles]), axis=1)
                normal = cr.state[k][t] == "NORMAL"
                center = cr.forecast[h][k][t + h]
                for j, q in enumerate(quantiles):
                    from_model = pred[:, j] * np.maximum(cr.profile[k][t + h], 1.0)
                    around_base = center * pred[:, j] / np.maximum(pred[:, mid], 1e-9)
                    bands[channel][h][q][k, t + h] = np.where(normal, from_model, around_base)
    return bands
