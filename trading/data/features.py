"""Causal feature engineering (technical + cross-asset + funding).

Every feature uses only past/current information, so there is no look-ahead
leakage into the label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from . import loaders


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    for k in (1, 2, 3, 5, 10, 20):
        out[f"ret_{k}"] = close.pct_change(k)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd"] = macd / close
    out["macd_signal"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    out["rsi_14"] = _rsi(close, 14)

    for w in (10, 20, 50):
        out[f"sma_dist_{w}"] = close / close.rolling(w).mean() - 1

    out["vol_10"] = close.pct_change().rolling(10).std()
    out["vol_20"] = close.pct_change().rolling(20).std()
    out["atr_norm"] = _atr(df, 14) / close
    out["hl_range"] = (df["high"] - df["low"]) / close
    out["close_pos"] = (close - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    out["vol_chg"] = df["volume"].pct_change()
    out["vol_z"] = ((df["volume"] - df["volume"].rolling(20).mean())
                    / df["volume"].rolling(20).std())
    return out


def cross_asset_features(index: pd.DatetimeIndex, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for sym, close in loaders.load_cross_assets(cfg).items():
        c = close.reindex(index, method="ffill")
        key = sym.replace("-", "").replace("^", "").replace("=", "").replace(".", "")
        out[f"{key}_ret_1"] = c.pct_change(1)
        out[f"{key}_ret_4"] = c.pct_change(4)
        out[f"{key}_sma_dist_10"] = c / c.rolling(10).mean() - 1
    return out


def funding_features(index: pd.DatetimeIndex, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    funding = loaders.load_funding(cfg, index)
    if funding is not None:
        out["funding"] = funding
        out["funding_ma4"] = funding.rolling(4).mean()
        out["funding_chg"] = funding.diff()
    return out


def onchain_features(index: pd.DatetimeIndex, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    if not cfg.onchain:
        return out
    for name, series in loaders.load_onchain(cfg, index).items():
        out[f"oc_{name}_chg1"] = series.pct_change(1)
        out[f"oc_{name}_chg4"] = series.pct_change(4)
        out[f"oc_{name}_dist"] = series / series.rolling(10).mean() - 1
    return out


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    feats = [technical_features(df),
             cross_asset_features(df.index, cfg),
             funding_features(df.index, cfg),
             onchain_features(df.index, cfg)]
    out = pd.concat(feats, axis=1)
    out["dow"] = df.index.dayofweek
    return out.replace([np.inf, -np.inf], np.nan)
