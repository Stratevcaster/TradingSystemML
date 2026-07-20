"""Labels and the assembled dataset."""
from __future__ import annotations

import pandas as pd

from ..config import Config
from .features import build_features
from .loaders import load_ohlcv


def make_label(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Binary label: 1 if future return over `horizon` exceeds threshold."""
    future_ret = df["close"].shift(-cfg.horizon) / df["close"] - 1
    return (future_ret > cfg.up_threshold).astype("float")


def forward_return(df: pd.DataFrame, cfg: Config) -> pd.Series:
    return df["close"].shift(-cfg.horizon) / df["close"] - 1


def build_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, ohlcv) aligned and cleaned (warm-up + tail dropped)."""
    ohlcv = load_ohlcv(cfg.ticker, cfg.period, cfg.interval)
    X = build_features(ohlcv, cfg)
    y = make_label(ohlcv, cfg)

    data = X.copy()
    data["_y"] = y
    data = data.dropna()
    y = data.pop("_y")
    return data, y, ohlcv.loc[data.index]
