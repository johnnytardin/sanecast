"""Leitura dos CSVs e agrupamento em bins de 5 minutos.

Entrada (uma linha por canal e por minuto):

    ts,channel,count
    AAAA-MM-DD 14:05:00,app,1043

Cada canal vira uma matriz (dias, 288). Todos os canais precisam ter dias completos.
"""
from pathlib import Path

import pandas as pd

from .config import BINS_PER_DAY, DATA_DIR, MINUTES_PER_DAY


def _to_bins(df, value_columns):
    df = df.copy()
    df["bin"] = pd.to_datetime(df["ts"]).dt.floor("5min")
    return df.groupby(["channel", "bin"])[value_columns].agg(
        {c: ("max" if c == "affected" else "sum") for c in value_columns}).reset_index()


def _reshape(grouped, column, days):
    """Transforma a série longa de cada canal em uma matriz (dias, 288)."""
    out = {}
    for channel, g in grouped.groupby("channel"):
        if len(g) != len(days) * BINS_PER_DAY:
            raise ValueError(f"Canal '{channel}' com dados incompletos: esperado {len(days) * BINS_PER_DAY} bins, "
                             f"recebido {len(g)}. Confira se não há dias parciais ou minutos faltando.")
        out[channel] = g.sort_values("bin")[column].to_numpy().reshape(len(days), BINS_PER_DAY).astype(float)
    return out


def load_series(data_dir: Path = DATA_DIR):
    """Retorna (days, series): `days` é o índice de datas e `series[channel]` a matriz (dias, 288) de contagens."""
    df = pd.read_csv(data_dir / "transactions_1min.csv", parse_dates=["ts"])
    missing = {"ts", "channel", "count"} - set(df.columns)
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {sorted(missing)}")
    days = pd.DatetimeIndex(sorted(df["ts"].dt.normalize().unique()))
    return days, _reshape(_to_bins(df, ["count"]), "count", days)


def load_truth(days, data_dir: Path = DATA_DIR):
    """Gabarito do simulador: {"expected": {canal: matriz}, "affected": {canal: matriz}}."""
    df = pd.read_csv(data_dir / "truth_1min.csv", parse_dates=["ts"])
    grouped = _to_bins(df, ["expected", "affected"])
    return {"expected": _reshape(grouped, "expected", days), "affected": _reshape(grouped, "affected", days)}


def load_future(data_dir: Path = DATA_DIR):
    """Dias simulados depois do histórico, só para o painel comparar. Retorna (days, series) ou (None, None) se não houver."""
    path = data_dir / "future_1min.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=["ts"])
    days = pd.DatetimeIndex(sorted(df["ts"].dt.normalize().unique()))
    return days, _reshape(_to_bins(df, ["count"]), "count", days)


def load_minutes(data_dir: Path = DATA_DIR, name: str = "transactions_1min.csv"):
    """Contagens por minuto, sem agrupar: retorna (dias, {canal: matriz (dias, 1440)}) ou (None, None) se o arquivo não existir.

    O painel usa essas contagens para mostrar o observado com granularidade de 1 minuto.
    """
    path = data_dir / name
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=["ts"]).sort_values(["channel", "ts"])
    days = pd.DatetimeIndex(sorted(df["ts"].dt.normalize().unique()))
    out = {}
    for channel, g in df.groupby("channel"):
        if len(g) != len(days) * MINUTES_PER_DAY:
            raise ValueError(f"Canal '{channel}' com dados incompletos: esperado {len(days) * MINUTES_PER_DAY} minutos, recebido {len(g)}.")
        out[channel] = g["count"].to_numpy().reshape(len(days), MINUTES_PER_DAY)
    return days, out
