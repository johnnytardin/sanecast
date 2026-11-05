"""Caminhos, constantes e hiperparâmetros. Todo número ajustável do modelo está em `Config`."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

BIN_MINUTES = 5
BINS_PER_DAY = 288  # 24 * 60 / BIN_MINUTES
MINUTES_PER_DAY = 1440
DAY_START_BIN, DAY_END_BIN = 60, 276  # trecho 05:00 a 23:00, usado para medir o nível do dia


@dataclass(frozen=True)
class Config:
    # ---- perfil do dia ----
    profile_k: int = 20  # máximo de dias parecidos usados para a forma do dia
    similarity_margin: float = 0.3  # só entram dias até esta distância do mais parecido
    shape_min_days: int = 5  # mínimo de dias para estimar a forma; a margem cresce até alcançar
    smooth_window: int = 5  # janela fixa de suavização no tempo, em bins
    smooth_target_count: float = 4000.0  # a janela cresce onde há poucas transações, até juntar esta quantidade
    smooth_max_window: int = 9  # janela máxima da suavização adaptativa

    # ---- limpeza do histórico ----
    cleaning_k: int = 7  # dias parecidos usados como referência
    cleaning_passes: int = 2  # a 2ª passada refaz a referência sem os trechos achados na 1ª
    low_ratio: float = 0.55  # razão observado/esperado abaixo disso é queda
    high_ratio: float = 1.45  # razão acima disso é pico anômalo
    edge_step: float = 1.4  # a limpeza só marca um trecho que começa com mudança de pelo menos esta razão (1 desliga)
    min_run_bins: int = 3  # bins seguidos para confirmar um evento (3 bins = 15 min)
    fast_low_ratio: float = 0.35  # via rápida: `fast_run_bins` bins seguidos abaixo disso confirmam a queda; 0 desliga
    fast_high_ratio: float = 1.8  # via rápida do pico: bins seguidos acima disso; infinito desliga
    fast_run_bins: int = 1
    fast_z: float = 6.0  # a via rápida também exige estar a `fast_z` desvios de contagem do esperado
    slow_window: int = 12  # via lenta: bins somados para medir o desvio (0 desliga)
    slow_lag: int = 12  # o desvio é medido contra o alfa de tantos bins antes, que um evento lento ainda não contaminou
    slow_low_ratio: float = 0.88  # via lenta: razão da janela abaixo disso (e fora do ruído) é degradação
    slow_high_ratio: float = 1.14  # via lenta: razão da janela acima disso (e fora do ruído) é pico lento
    slow_run_bins: int = 6  # bins seguidos fora dos limiares para confirmar pela via lenta
    slow_z: float = 6.0  # a via lenta também exige estar a `slow_z` desvios de contagem do esperado
    retention_exit_ratio: float = 1.10  # a retenção acaba quando a razão fica abaixo disso por min_run_bins bins seguidos
    retention_margin_bins: int = 2  # bins marcados logo depois do fim da queda, porque a média móvel de 3 bins borra a borda
    max_dirty_fraction: float = 0.3  # dia com mais que isso marcado como sujo não serve de dia parecido

    # ---- fator alfa (ritmo do dia) ----
    alpha_target_count: float = 3000.0  # a janela do alfa junta aproximadamente esta quantidade de transações esperadas
    alpha_min_window: int = 6  # janela mínima do alfa, em bins
    alpha_max_window: int = 72  # janela máxima do alfa, em bins
    alpha_prior_bins: float = 6  # peso do valor anterior, em bins de volume esperado
    alpha_tau: float = 24  # o efeito do alfa decai com exp(-h / tau), h em bins

    # ---- estados online ----
    recovery_low: float = 0.8  # limites da faixa de razão considerada "voltou ao normal" (a faixa real é mais estreita em canal de volume alto)
    recovery_high: float = 1.25
    recovery_margin: float = 0.05  # menor meia-largura dessa faixa em torno de 1
    recovery_z: float = 2.5  # a meia-largura é no mínimo recovery_z desvios do ruído de contagem do intervalo (1/raiz da contagem esperada)
    outage_exit_ratio: float = 0.8  # razão acima disso encerra a queda
    max_event_bins: int = BINS_PER_DAY  # um evento pode durar até o fim do dia; no dia seguinte a máquina de estados recomeça em NORMAL

    # ---- saídas ----
    horizons: tuple = (1, 3, 6, 12, 24, 48, 96)  # horizontes previstos pelo LightGBM, em bins: 5 min, 15 min, 30 min, 1 h, 2 h, 4 h e 8 h
    ratio_lags: tuple = (1, 3, 6, 12)  # janelas (bins) das razões observado/perfil usadas como atributos
    quantiles: tuple = (0.1, 0.5, 0.9)
    lgbm_rounds: int = 150  # árvores por quantil
    band_growth_exponent: float = 0.2  # depois do maior horizonte, a faixa cresce com (h / h_max) ** expoente
    calibration_days: int = 28  # últimos dias do treino reservados para calibrar a faixa (conformal); 0 desliga a calibração
    widen_event_band: bool = True  # em evento a faixa usa o horizonte (idade do alfa congelado + h), não só h
    post_event_bins: int = 12  # intervalos depois do evento em que o alfa ainda aprende com a cauda da retenção e a faixa segue alargada

    # ---- nível do dia ----
    # candidatos da penalidade ridge para a validação temporal; vazio usa 1e-3. Desligado: as dobras quase não têm feriado e a escolha
    # cai no valor mais forte (0,1), o que encolhe os efeitos raros e piora o feriado (veja docs/RESULTS.md)
    ridge_grid: tuple = ()
    level_knots: tuple = (90, 30)  # dobras da tendência, em dias antes do fim do histórico; vazio deixa só a reta
    level_knot_scale: float = 1000.0  # penalidade das dobras, em múltiplos da penalidade ridge dos demais termos
    trend_cap_days: int = 7  # a tendência para de crescer este número de dias depois do fim do histórico


DEFAULT = Config()
