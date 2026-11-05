"""LightGBM quantílico (P10, P50, P90) sobre o perfil do dia.

O alvo é a razão y(t+h) / perfil(t+h). Um modelo por quantil, o mesmo para todos os canais.
Atributos: alfa, razões observado/perfil recentes, horizonte, log do perfil, hora do bin previsto (seno e cosseno),
dia da semana, feriado, véspera, dia seguinte, ponte, fim de semana prolongado, início e fim de mês e canal.

Treino: só linhas com estado NORMAL e sem bins marcados como sujos no atributo nem no alvo.
Calibração (conformal quantile regression, Romano et al. 2019): os últimos `calibration_days` dias do treino ficam fora do ajuste
e medem o quanto a faixa P10 a P90 erra. O erro de cobertura, por canal e horizonte, vira uma correção que alarga (ou estreita)
a faixa para ela cobrir 80% nos dias de calibração, sem supor distribuição nenhuma.
Previsão: em OUTAGE, SURGE ou RECOVERY o centro da faixa vem da previsão do modelo base (alfa congelado)
e o LightGBM define só a largura relativa da faixa. Como o alfa congelado envelhece, a largura é a do horizonte (idade + h),
não a de h: a incerteza de quem prevê com um alfa de `idade` intervalos atrás é a de um horizonte maior.
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


class BandModels(dict):
    """{quantil: modelo LightGBM}, mais a correção conformal `shift[(canal, h)]`, na escala da razão y / perfil."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift = {}


def conformal_shift(lo, hi, y, miscoverage):
    """Correção de CQR: o quantil (1 - alfa) dos erros max(lo - y, y - hi), com a correção de amostra finita.
    Soma-se à faixa nos dois lados (negativa, estreita a faixa)."""
    score = np.maximum(lo - y, y - hi)
    level = min(1.0, np.ceil((len(score) + 1) * (1 - miscoverage)) / len(score))
    return float(np.quantile(score, level, method="higher"))


def adjust(models, pred, channel_id, h):
    """Aplica a correção conformal. `pred`: (n, quantis) já ordenado, na escala da razão. A mediana não muda."""
    shift = models.shift.get((channel_id, h), 0.0)
    if shift == 0.0:
        return pred
    out = pred.copy()
    mid = pred.shape[1] // 2
    out[:, 0] = np.minimum(np.maximum(pred[:, 0] - shift, 0.0), pred[:, mid])
    out[:, -1] = np.maximum(pred[:, -1] + shift, pred[:, mid])
    return out


def state_age(state, cfg):
    """Intervalos desde que o alfa foi confiável pela última vez, para cada bin. 0 em NORMAL sem evento por perto.

    Fora de NORMAL: os intervalos seguidos fora de NORMAL mais `min_run_bins` (o alfa congelado é o de antes da confirmação).
    Depois que o estado volta a NORMAL a idade continua crescendo por `post_event_bins`, porque o alfa volta a aprender enquanto
    a cauda da retenção ainda contamina o dado. Com `widen_event_band` desligado, tudo é 0."""
    age = np.zeros(len(state), int)
    if not cfg.widen_event_band:
        return age
    run, last, tail = 0, 0, 0
    for i, st in enumerate(state):
        if st != "NORMAL":
            run += 1
            last, tail = run + cfg.min_run_bins, cfg.post_event_bins
        elif tail > 0:
            run, last, tail = 0, last + 1, tail - 1
        else:
            run, last = 0, 0
        age[i] = last
    return age


def relative_widths(pred):
    """Largura relativa inferior e superior da faixa, por linha: (1 - P10/P50, P90/P50 - 1). `pred`: (n, quantis) ordenado."""
    mid = pred.shape[1] // 2
    centre = np.maximum(pred[:, mid], 1e-9)
    return 1 - pred[:, 0] / centre, pred[:, -1] / centre - 1


def train(forecaster, train_end):
    """Treina um modelo por quantil com os dias anteriores a `train_end` e calibra a faixa nos últimos `calibration_days`.
    Retorna BandModels ({quantil: modelo} com a correção conformal)."""
    cfg = forecaster.cfg
    prepared = forecaster.prepare(train_end)
    n_cal = cfg.calibration_days if len(prepared[next(iter(prepared))].pool) >= 3 * cfg.calibration_days else 0
    fit_rows, fit_targets, cal_rows, cal_targets = [], [], [], []
    for channel_id, (channel, y) in enumerate(forecaster.series.items()):
        p = prepared[channel]
        for i, d in enumerate(p.pool):
            in_calibration = i >= len(p.pool) - n_cal
            profile = forecaster.profile_for(d, p, exclude=d)  # perfil sem o próprio dia
            day_run = run_day(y[d], profile, cfg)
            for h, block in zip(cfg.horizons, build_attributes(day_run.features, profile, forecaster.feats[d],
                                                              channel_id, cfg)):
                t = np.arange(BINS_PER_DAY - h)
                clean = (day_run.state[t] == "NORMAL") & ~p.mask[i, t] & ~p.mask[i, t + h]
                (cal_rows if in_calibration else fit_rows).append(block[clean])
                (cal_targets if in_calibration else fit_targets).append((y[d][t + h] / np.maximum(profile[t + h], 1.0))[clean])
    X, Y = np.vstack(fit_rows), np.concatenate(fit_targets)
    categorical = [X.shape[1] - 1]  # a última coluna é o canal
    models = BandModels({q: lgb.train({**PARAMS, "alpha": q}, lgb.Dataset(X, Y, categorical_feature=categorical), cfg.lgbm_rounds)
                         for q in cfg.quantiles})
    if n_cal:
        Xc, Yc = np.vstack(cal_rows), np.concatenate(cal_targets)
        pred = np.sort(np.column_stack([models[q].predict(Xc) for q in cfg.quantiles]), axis=1)
        h_col, channel_col = 1 + len(cfg.ratio_lags), Xc.shape[1] - 1
        miscoverage = 1 - (max(cfg.quantiles) - min(cfg.quantiles))
        for channel_id in range(len(forecaster.series)):
            for h in cfg.horizons:
                pick = (Xc[:, channel_col] == channel_id) & (Xc[:, h_col] == h)
                if pick.sum() >= 50:  # com menos que isso a correção seria ruído
                    models.shift[(channel_id, h)] = conformal_shift(pred[pick, 0], pred[pick, -1], Yc[pick], miscoverage)
    return models


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
            preds = {h: adjust(models, np.sort(np.column_stack([models[q].predict(block) for q in quantiles]), axis=1), channel_id, h)
                     for h, block in zip(cfg.horizons, blocks)}
            age = state_age(cr.state[k], cfg)
            widths = {g: relative_widths(pred) for g, pred in preds.items()}
            for h in cfg.horizons:
                t = np.arange(BINS_PER_DAY - h)
                pred = preds[h]
                normal = (cr.state[k][t] == "NORMAL") & (age[t] == 0)  # NORMAL e sem evento recente
                center = cr.forecast[h][k][t + h]
                low_w, high_w = widths[h][0].copy(), widths[h][1].copy()
                for i in np.flatnonzero(age[t] > 0):  # largura do horizonte (idade + h), entre os horizontes que existem nesse bin
                    avail = [g for g in cfg.horizons if i < BINS_PER_DAY - g]
                    low_w[i] = np.interp(age[i] + h, avail, [widths[g][0][i] for g in avail])
                    high_w[i] = np.interp(age[i] + h, avail, [widths[g][1][i] for g in avail])
                for j, q in enumerate(quantiles):
                    from_model = pred[:, j] * np.maximum(cr.profile[k][t + h], 1.0)
                    if j < mid:
                        around_base = center * (1 - low_w)
                    elif j > mid:
                        around_base = center * (1 + high_w)
                    else:
                        around_base = center
                    bands[channel][h][q][k, t + h] = np.where(normal, from_model, around_base)
    return bands
