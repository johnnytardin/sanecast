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
SURGE e RECOVERY voltam a NORMAL com 3 bins seguidos em torno de 1, dentro de `recovery_band`.
Via rápida: um bin só abaixo de `fast_low_ratio` (ou acima de `fast_high_ratio`) já confirma o evento, desde que também esteja a `fast_z`
desvios de contagem do esperado. Em volume baixo o ruído sozinho passa dos limiares fixos, e a condição de ruído impede o falso alarme.
Via lenta: uma degradação ou alta de 12% ou mais com rampa longa é absorvida pelo alfa e nunca cruza a razão contra o alfa atual. A via lenta compara a soma dos
últimos `slow_window` bins com o alfa de `slow_lag` bins antes e confirma depois de `slow_run_bins` bins seguidos fora de `slow_low_ratio`/`slow_high_ratio` e do ruído de contagem
(`slow_z`). O evento lento entra como OUTAGE ou SURGE com o alfa de antes do desvio congelado e sai pela faixa de retorno, como o pico.
Bins de evento não entram no cálculo do alfa.
"""
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT


def recovery_band(expected, cfg):
    """Faixa de razão que vale como "voltou ao normal". Em volume alto o ruído de contagem é pequeno e uma retenção de 10% já é fora do normal,
    então a faixa estreita até `recovery_margin`. Em volume baixo ela alarga com o ruído (`recovery_z` desvios) até os limites
    `recovery_low` e `recovery_high`."""
    noise = cfg.recovery_z / np.sqrt(max(expected, 1.0))
    return 1 - np.clip(noise, cfg.recovery_margin, 1 - cfg.recovery_low), 1 + np.clip(noise, cfg.recovery_margin, cfg.recovery_high - 1)


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
    n_low = n_high = n_fast_low = n_fast_high = n_slow_low = n_slow_high = n_ok = n_in_state = 0
    slow = False  # o evento em andamento veio da via lenta e sai pela faixa de retorno
    prior = cfg.alpha_prior_bins * profile.mean()
    for t in range(n):
        reference_alpha = (alphas[-1] if alphas else 1.0) if state == "NORMAL" else frozen
        ratio = obs[t] / max(reference_alpha * profile[t], 1.0)
        if treat:
            if state == "NORMAL":
                n_low = n_low + 1 if ratio < cfg.low_ratio else 0
                n_high = n_high + 1 if ratio > cfg.high_ratio else 0
                noise = cfg.fast_z / np.sqrt(max(reference_alpha * profile[t], 1.0))  # o limiar rápido nunca fica dentro do ruído de contagem
                n_fast_low = n_fast_low + 1 if ratio < min(cfg.fast_low_ratio, 1 - noise) else 0
                n_fast_high = n_fast_high + 1 if ratio > max(cfg.fast_high_ratio, 1 + noise) else 0
                is_low = n_low >= cfg.min_run_bins or n_fast_low >= cfg.fast_run_bins
                is_high = n_high >= cfg.min_run_bins or n_fast_high >= cfg.fast_run_bins
                is_slow = False
                if cfg.slow_window and not (is_low or is_high) and t >= cfg.slow_window + cfg.slow_lag:
                    window = slice(t - cfg.slow_window + 1, t + 1)
                    expected = alphas[t - cfg.slow_lag] * profile[window].sum()
                    ratio_slow = obs[window].sum() / max(expected, 1.0)
                    noise = cfg.slow_z / np.sqrt(max(expected, 1.0))
                    clean = valid[t - cfg.slow_window - cfg.slow_lag + 1 : t + 1].all()
                    n_slow_low = n_slow_low + 1 if clean and ratio_slow < min(cfg.slow_low_ratio, 1 - noise) else 0
                    n_slow_high = n_slow_high + 1 if clean and ratio_slow > max(cfg.slow_high_ratio, 1 + noise) else 0
                    is_slow = max(n_slow_low, n_slow_high) >= cfg.slow_run_bins
                if is_slow:
                    state, slow = ("OUTAGE" if n_slow_low >= cfg.slow_run_bins else "SURGE"), True
                    onset = max(t - cfg.slow_window - cfg.slow_run_bins + 2, 0)  # o desvio já aparecia na janela que deu o primeiro aviso
                    frozen = alphas[max(onset - cfg.slow_lag // 2, 0)]
                    n_ok = n_in_state = 0
                    valid[onset : t + 1] = False
                elif is_low or is_high:
                    run_bins = max(n_low if is_low else n_high, 1)  # os bins do começo do evento, que já estão fora do normal
                    state, slow = ("OUTAGE" if is_low else "SURGE"), False
                    frozen = alphas[t - run_bins] if t >= run_bins else 1.0
                    n_ok = n_in_state = 0
                    valid[max(t - run_bins + 1, 0) : t + 1] = False
            elif state == "OUTAGE" and not slow:
                valid[t] = False
                n_in_state += 1
                n_ok = n_ok + 1 if ratio > cfg.outage_exit_ratio else 0
                if n_ok >= 2:
                    state, n_ok, n_in_state = "RECOVERY", 0, 0
                elif n_in_state > cfg.max_event_bins:
                    state, n_low, n_fast_low, n_slow_low, n_slow_high = "NORMAL", 0, 0, 0, 0
                    valid[max(t - 5, 0) : t + 1] = True
            else:  # RECOVERY e SURGE
                valid[t] = False
                n_in_state += 1
                low, high = recovery_band(reference_alpha * profile[t], cfg)
                n_ok = n_ok + 1 if low <= ratio <= high else 0
                if n_ok >= 3 or n_in_state > cfg.max_event_bins:
                    state, n_low, n_high, n_fast_low, n_fast_high, n_slow_low, n_slow_high = "NORMAL", 0, 0, 0, 0, 0, 0
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
