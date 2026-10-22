"""Previsão dentro do dia: fator alfa e máquina de estados.

A cada bin t, depois de observar o bin:
1. alfa mede o ritmo do dia contra o perfil numa janela que junta cerca de `alpha_target_count` transações;
2. a máquina de estados decide se há um evento em andamento;
3. a previsão para t+h é (1 + (alfa - 1) * exp(-h / alpha_tau)) * perfil(t+h).

Estados:
  NORMAL    atualiza o alfa
  OUTAGE    3 bins seguidos abaixo de low_ratio. Congela o alfa de antes da queda. Sai com 2 bins acima de outage_exit_ratio
  SURGE     3 bins seguidos acima de high_ratio. Congela o alfa
  RECOVERY  depois da queda: ignora o pico de retenção
SURGE e RECOVERY voltam a NORMAL com 3 bins seguidos entre recovery_low e recovery_high.
Bins de evento não entram no cálculo do alfa.
"""
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT


@dataclass
class DayRun:
    forecast: dict  # {h: array} previsão do bin t+h feita no bin t, alinhada ao bin previsto (NaN no começo)
    state: np.ndarray  # estado de cada bin
    features: np.ndarray  # [alfa, razões observado/perfil em ratio_lags]; as razões valem 1 fora de NORMAL
    alpha: np.ndarray  # alfa de cada bin


def run_day(obs, profile, cfg=DEFAULT, treat=True):
    """Percorre o dia bin a bin, vendo só o passado de cada bin. Com `treat=False` não há detecção de eventos."""
    n = len(profile)
    forecast = {h: np.full(n, np.nan) for h in cfg.horizons}
    states = ["NORMAL"] * n
    features = np.ones((n, 1 + len(cfg.ratio_lags)))
    valid = np.ones(n, bool)  # bins que entram no cálculo do alfa
    alphas, frozen, state = [], 1.0, "NORMAL"
    n_low = n_high = n_ok = n_in_state = 0
    prior = cfg.alpha_prior_bins * profile.mean()
    for t in range(n):
        reference_alpha = (alphas[-1] if alphas else 1.0) if state == "NORMAL" else frozen
        ratio = obs[t] / max(reference_alpha * profile[t], 1.0)
        if treat:
            if state == "NORMAL":
                n_low = n_low + 1 if ratio < cfg.low_ratio else 0
                n_high = n_high + 1 if ratio > cfg.high_ratio else 0
                if n_low >= cfg.min_run_bins or n_high >= cfg.min_run_bins:
                    state = "OUTAGE" if n_low >= cfg.min_run_bins else "SURGE"
                    frozen = alphas[t - cfg.min_run_bins] if t >= cfg.min_run_bins else 1.0
                    n_ok = n_in_state = 0
                    valid[max(t - cfg.min_run_bins + 1, 0) : t + 1] = False
            elif state == "OUTAGE":
                valid[t] = False
                n_in_state += 1
                n_ok = n_ok + 1 if ratio > cfg.outage_exit_ratio else 0
                if n_ok >= 2:
                    state, n_ok, n_in_state = "RECOVERY", 0, 0
                elif n_in_state > cfg.max_event_bins:
                    state, n_low = "NORMAL", 0
                    valid[max(t - 5, 0) : t + 1] = True
            else:  # RECOVERY e SURGE
                valid[t] = False
                n_in_state += 1
                n_ok = n_ok + 1 if cfg.recovery_low <= ratio <= cfg.recovery_high else 0
                if n_ok >= 3 or n_in_state > cfg.max_event_bins:
                    state, n_low, n_high = "NORMAL", 0, 0
                    valid[max(t - 2, 0) : t + 1] = True
        if state == "NORMAL":
            n_bins = int(np.clip(np.ceil(cfg.alpha_target_count / max(profile[t], 1.0)),
                                 cfg.alpha_min_window, cfg.alpha_max_window))
            window = slice(max(t - n_bins + 1, 0), t + 1)
            ok = valid[window]
            expected = profile[window][ok].sum()
            raw = obs[window][ok].sum() / expected if expected > 0 else 1.0
            # alfa = (soma observada + prior * alfa anterior) / (soma esperada + prior)
            alpha = (expected * raw + prior * (alphas[-1] if alphas else 1.0)) / (expected + prior)
            for j, lag in enumerate(cfg.ratio_lags):
                lo = max(t - lag + 1, 0)
                features[t, 1 + j] = obs[lo : t + 1].sum() / max(profile[lo : t + 1].sum(), 1.0)
        else:
            alpha = frozen
        features[t, 0] = alpha
        alphas.append(alpha)
        states[t] = state
        for h in cfg.horizons:
            if t + h < n:
                forecast[h][t + h] = (1 + (alpha - 1) * np.exp(-h / cfg.alpha_tau)) * profile[t + h]
    return DayRun(forecast, np.array(states), features, np.array(alphas))


def project_rest_of_day(profile, alpha, now_bin, cfg=DEFAULT):
    """Projeção central (P50) dos bins depois de `now_bin`, a partir do alfa daquele bin."""
    horizon = np.arange(1, len(profile) - now_bin)
    return (1 + (alpha - 1) * np.exp(-horizon / cfg.alpha_tau)) * profile[now_bin + 1 :]
