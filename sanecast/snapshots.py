"""Projeções "como se fosse agora".

O modelo só olha bins anteriores, então uma passada pelo dia contém o que seria visto em qualquer horário.
Para cada horário `now`:
- p50: projeção do resto do dia. O centro é a mediana do LightGBM nos horizontes de `cfg.horizons`, interpolada entre eles:
  o LightGBM corrige a projeção base (alfa de `now` x perfil, com o efeito do alfa decaindo), e a correção volta a 1 depois do
  maior horizonte. Em OUTAGE, SURGE ou RECOVERY o centro é a projeção base, com o alfa congelado;
- lo e hi: faixa P10 a P90, já calibrada (conformal). A largura relativa vem do LightGBM, é interpolada entre os horizontes
  e, depois do maior, cresce com (h / h_max) ** band_growth_exponent.
- perto do fim do dia só cabem os horizontes menores, e a faixa usa os que couberem. Às 23:55 não sobra intervalo para projetar.
"""
import numpy as np

from .config import BINS_PER_DAY
from .forecasting import project_rest_of_day, run_day
from .quantile_model import adjust, build_attributes, state_age


def band_widths(ratios_by_horizon, horizons, n_future, exponent, age=0):
    """Largura relativa inferior e superior da faixa para cada bin futuro (1 a n_future bins à frente).

    `age`: intervalos desde que o alfa foi confiável pela última vez (veja `quantile_model.state_age`); a faixa usa o horizonte `age + h`."""
    h = np.arange(1, n_future + 1) + age
    lower = np.array([1 - r[0] / r[1] for r in ratios_by_horizon])
    upper = np.array([r[2] / r[1] - 1 for r in ratios_by_horizon])
    grow = np.maximum(h / max(horizons), 1.0) ** exponent
    return np.interp(h, horizons, lower) * grow, np.interp(h, horizons, upper) * grow


def lgbm_center(base, alpha, horizons_ratios, cfg):
    """Projeção base corrigida pela mediana do LightGBM.

    `horizons_ratios`: [(h, razão mediana prevista)]. Em cada horizonte, a correção é a razão do LightGBM dividida pela razão da projeção
    base; entre horizontes ela é interpolada, e depois do último volta a 1 com a mesma constante do alfa."""
    hs = np.array([h for h, _ in horizons_ratios], float)
    base_ratio = 1 + (alpha - 1) * np.exp(-hs / cfg.alpha_tau)
    factor = np.array([r for _, r in horizons_ratios]) / base_ratio
    h = np.arange(1, len(base) + 1)
    beyond = np.maximum(h - hs[-1], 0)
    return base * (1 + (np.interp(h, hs, factor) - 1) * np.exp(-beyond / cfg.alpha_tau))


def compute_snapshots(forecaster, models, day_indices, now_bins):
    """Retorna {dia: {canal: dict(real, profile, state, alpha, snaps={now: (p50, lo, hi)})}}.

    O histórico é o anterior ao primeiro dia pedido; `models` deve ter sido treinado com dados anteriores a esse dia.
    """
    cfg = forecaster.cfg
    prepared = forecaster.prepare(forecaster.days[day_indices[0]])
    quantiles = list(models)
    all_nows = [n for n in now_bins if n <= BINS_PER_DAY - 1]
    out = {}
    for d in day_indices:
        out[d] = {}
        for channel_id, (channel, y) in enumerate(forecaster.series.items()):
            profile = forecaster.profile_for(d, prepared[channel])
            run = run_day(y[d], profile, cfg)
            age = state_age(run.state, cfg)
            blocks = build_attributes(run.features, profile, forecaster.feats[d], channel_id, cfg)
            predicted = []  # por horizonte: {now: [P10, P50, P90]}; só existe quando o bin previsto cabe no dia
            for h, block in zip(cfg.horizons, blocks):
                rows = [n for n in all_nows if n < len(block)]
                pred = (adjust(models, np.sort(np.column_stack([models[q].predict(block[rows]) for q in quantiles]), axis=1),
                               channel_id, h) if rows else [])
                predicted.append(dict(zip(rows, pred)))
            snaps = {}
            for now in all_nows:
                n_future = BINS_PER_DAY - 1 - now
                if n_future == 0:  # último bin do dia: não sobra nada para projetar
                    snaps[now] = (np.empty(0), np.empty(0), np.empty(0))
                    continue
                available = [(h, predicted[j][now]) for j, h in enumerate(cfg.horizons) if now in predicted[j]]
                w_lo, w_hi = band_widths([r for _, r in available], [h for h, _ in available], n_future, cfg.band_growth_exponent, age[now])
                p50 = project_rest_of_day(profile, run.alpha[now], now, cfg)
                if run.state[now] == "NORMAL" and age[now] == 0:
                    p50 = lgbm_center(p50, run.alpha[now], [(h, r[len(r) // 2]) for h, r in available], cfg)
                snaps[now] = (p50, p50 * (1 - w_lo), p50 * (1 + w_hi))
            out[d][channel] = dict(real=y[d], profile=profile, state=run.state, alpha=run.alpha, snaps=snaps)
    return out


def day_ahead_band(forecaster, prepared, channel, n_days=28, half_window=6, quantiles=(0.1, 0.9)):
    """Faixa relativa P10 e P90 da projeção feita antes do dia começar (dia seguinte ou mais à frente).

    Mede y / perfil nos últimos `n_days` dias limpos de `prepared`, cada um com o perfil calculado sem ele mesmo.
    Cada bin usa os bins vizinhos (± `half_window`), porque o ruído de contagem muda ao longo do dia.
    Retorna (razão inferior, razão superior), cada uma com 288 valores.
    """
    p = prepared[channel]
    recent = set(p.pool[-n_days:])
    days = [d for d in p.usable if d in recent]
    ratios = np.array([p.clean[d] / np.maximum(forecaster.profile_for(d, p, exclude=d), 1.0) for d in days])
    pad = np.pad(ratios, ((0, 0), (half_window, half_window)), mode="edge")
    around = np.stack([pad[:, b : b + 2 * half_window + 1].ravel() for b in range(BINS_PER_DAY)])
    return tuple(np.nanquantile(around, q, axis=1) for q in quantiles)
