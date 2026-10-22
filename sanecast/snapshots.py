"""Projeções "como se fosse agora".

O modelo só olha bins anteriores, então uma passada pelo dia contém o que seria visto em qualquer horário.
Para cada horário `now`:
- p50: projeção do resto do dia (alfa de `now` x perfil, com o efeito do alfa decaindo);
- lo e hi: faixa P10 a P90. A largura relativa vem do LightGBM em 5 min e 1 h, é interpolada entre os dois
  e, depois de 1 h, cresce com (h / h_max) ** band_growth_exponent.
- depois das 22:55 o horizonte de 1 h não cabe mais no dia: a faixa usa só o de 5 min e cresce com h ** band_growth_exponent.
  Às 23:55 não sobra intervalo para projetar.
"""
import numpy as np

from .config import BINS_PER_DAY
from .forecasting import project_rest_of_day, run_day
from .quantile_model import build_attributes


def band_widths(ratios_by_horizon, horizons, n_future, exponent):
    """Largura relativa inferior e superior da faixa para cada bin futuro (1 a n_future bins à frente)."""
    h = np.arange(1, n_future + 1)
    lower = np.array([1 - r[0] / r[1] for r in ratios_by_horizon])
    upper = np.array([r[2] / r[1] - 1 for r in ratios_by_horizon])
    grow = np.maximum(h / max(horizons), 1.0) ** exponent
    return np.interp(h, horizons, lower) * grow, np.interp(h, horizons, upper) * grow


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
            blocks = build_attributes(run.features, profile, forecaster.feats[d], channel_id, cfg)
            predicted = []  # por horizonte: {now: [P10, P50, P90]}; só existe quando o bin previsto cabe no dia
            for block in blocks:
                rows = [n for n in all_nows if n < len(block)]
                pred = np.sort(np.column_stack([models[q].predict(block[rows]) for q in quantiles]), axis=1) if rows else []
                predicted.append(dict(zip(rows, pred)))
            snaps = {}
            for now in all_nows:
                n_future = BINS_PER_DAY - 1 - now
                if n_future == 0:  # último bin do dia: não sobra nada para projetar
                    snaps[now] = (np.empty(0), np.empty(0), np.empty(0))
                    continue
                available = [(h, predicted[j][now]) for j, h in enumerate(cfg.horizons) if now in predicted[j]]
                w_lo, w_hi = band_widths([r for _, r in available], [h for h, _ in available], n_future, cfg.band_growth_exponent)
                p50 = project_rest_of_day(profile, run.alpha[now], now, cfg)
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
