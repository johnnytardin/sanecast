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
    min_run_bins: int = 3  # bins seguidos para confirmar um evento (3 bins = 15 min)
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
    recovery_low: float = 0.8  # faixa de razão considerada "voltou ao normal"
    recovery_high: float = 1.25
    outage_exit_ratio: float = 0.8  # razão acima disso encerra a queda
    max_event_bins: int = BINS_PER_DAY  # um evento pode durar até o fim do dia; no dia seguinte a máquina de estados recomeça em NORMAL

    # ---- saídas ----
    horizons: tuple = (1, 12)  # horizontes previstos pelo LightGBM, em bins: 5 min e 1 h
    ratio_lags: tuple = (1, 3, 6, 12)  # janelas (bins) das razões observado/perfil usadas como atributos
    quantiles: tuple = (0.1, 0.5, 0.9)
    lgbm_rounds: int = 150  # árvores por quantil
    band_growth_exponent: float = 0.2  # depois do maior horizonte, a faixa cresce com (h / h_max) ** expoente


DEFAULT = Config()
