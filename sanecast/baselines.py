"""Modelos de comparação.

- last_week: o mesmo bin 7 dias antes, com o dado bruto;
- statsforecast_day_ahead: SeasonalNaive (sazonalidade semanal) e MSTL (sazonalidade diária e semanal) do StatsForecast,
  ajustados no histórico já limpo. Não usam calendário de feriados.
"""
import numpy as np
import pandas as pd

from .config import BINS_PER_DAY

WEEKS_OF_HISTORY = 8


def last_week(series, eval_idx, horizons):
    """Mesmo bin, 7 dias antes, para cada horizonte."""
    shifted = np.array(eval_idx) - 7
    return {channel: {h: y[shifted] for h in horizons} for channel, y in series.items()}


def statsforecast_day_ahead(forecaster, start, eval_idx):
    """Previsão do período inteiro com os dados até `start`. Retorna {método: {canal: {"day": matriz}}}."""
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, SeasonalNaive

    prepared = forecaster.prepare(start)
    frames = []
    for channel, p in prepared.items():
        recent = p.pool[-7 * WEEKS_OF_HISTORY :]
        y = pd.Series(p.clean[recent].ravel()).interpolate(limit_direction="both").to_numpy()
        frames.append(pd.DataFrame({"unique_id": channel, "y": y,
                                    "ds": pd.date_range(forecaster.days[recent[0]], periods=len(y), freq="5min")}))
    sf = StatsForecast(models=[SeasonalNaive(season_length=7 * BINS_PER_DAY),
                               MSTL(season_length=[BINS_PER_DAY, 7 * BINS_PER_DAY])], freq="5min", n_jobs=1)
    result = sf.forecast(df=pd.concat(frames), h=len(eval_idx) * BINS_PER_DAY)
    names = {"SeasonalNaive": "statsforecast_seasonal_naive", "MSTL": "statsforecast_mstl"}
    return {names[col]: {channel: {"day": np.clip(g.sort_values("ds")[col].to_numpy().reshape(len(eval_idx), -1), 0, None)}
                         for channel, g in result.groupby("unique_id")} for col in names}
