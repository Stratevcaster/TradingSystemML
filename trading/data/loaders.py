"""Data loading: primary OHLCV, cross-asset series, and funding rates."""
from __future__ import annotations

import os

import pandas as pd
import requests
import yfinance as yf

from ..config import Config


def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download OHLCV as a clean DataFrame with lowercase columns."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker!r}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.dropna()


def load_cross_assets(cfg: Config) -> dict[str, pd.Series]:
    """Return a {ticker: close-series} map, skipping any that fail to load."""
    out: dict[str, pd.Series] = {}
    for sym in cfg.cross_assets:
        try:
            close = load_ohlcv(sym, cfg.period, cfg.interval)["close"]
            out[sym] = close
        except Exception as exc:  # noqa: BLE001 - network/listing issues are non-fatal
            print(f"[warn] skipping cross-asset {sym}: {exc}")
    return out


def load_funding(cfg: Config, index: pd.DatetimeIndex) -> pd.Series | None:
    """Fetch Binance perpetual funding rates and resample onto `index`.

    Uses the public endpoint (no API key). Returns None on any failure so the
    pipeline degrades gracefully when the endpoint is unreachable/geo-blocked.
    """
    if not cfg.funding_symbol:
        return None
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    rows: list[dict] = []
    end = int(pd.Timestamp.utcnow().timestamp() * 1000)
    try:
        for _ in range(40):  # page backwards, 1000 rows each (~8h cadence)
            resp = requests.get(url, params={"symbol": cfg.funding_symbol,
                                             "endTime": end, "limit": 1000},
                                timeout=10)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            rows = batch + rows
            end = batch[0]["fundingTime"] - 1
            if len(batch) < 1000:
                break
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] funding rates unavailable: {exc}")
        return None

    if not rows:
        return None
    s = pd.Series(
        [float(r["fundingRate"]) for r in rows],
        index=pd.to_datetime([r["fundingTime"] for r in rows], unit="ms"),
    ).sort_index()
    # Aggregate to the bar frequency (sum of funding paid over each bar).
    rule = "1W" if cfg.interval == "1wk" else "1D"
    agg = s.resample(rule).sum()
    agg.index = agg.index.tz_localize(None)
    return agg.reindex(index, method="ffill")


def _blockchain_chart(chart: str) -> pd.Series | None:
    """Fetch one free blockchain.com chart as a date-indexed Series."""
    url = f"https://api.blockchain.info/charts/{chart}"
    try:
        resp = requests.get(url, params={"timespan": "all", "format": "json",
                                         "sampled": "true"}, timeout=15)
        resp.raise_for_status()
        vals = resp.json().get("values", [])
    except Exception as exc:  # noqa: BLE001 - non-fatal
        print(f"[warn] on-chain chart {chart!r} unavailable: {exc}")
        return None
    if not vals:
        return None
    s = pd.Series([v["y"] for v in vals],
                  index=pd.to_datetime([v["x"] for v in vals], unit="s"))
    return s.sort_index()


def _glassnode_metric(cfg: Config) -> pd.Series | None:
    """Optional Glassnode metric, enabled only when GLASSNODE_API_KEY is set."""
    key = os.environ.get("GLASSNODE_API_KEY")
    if not key:
        return None
    url = f"https://api.glassnode.com/v1/metrics/{cfg.glassnode_metric}"
    try:
        resp = requests.get(url, params={"a": "BTC", "i": "24h", "api_key": key},
                            timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Glassnode metric unavailable: {exc}")
        return None
    if not data:
        return None
    s = pd.Series([d.get("v") for d in data],
                  index=pd.to_datetime([d["t"] for d in data], unit="s"))
    return s.sort_index().astype(float)


def load_onchain(cfg: Config, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Return a {name: series} map of on-chain metrics aligned to `index`.

    Uses the free blockchain.com charts API (no key). If GLASSNODE_API_KEY is
    present, an extra Glassnode metric is added. All sources degrade
    gracefully when unreachable.
    """
    out: dict[str, pd.Series] = {}
    for chart in cfg.onchain_charts:
        s = _blockchain_chart(chart)
        if s is not None:
            name = chart.replace("-", "_")
            out[name] = s.reindex(index, method="ffill")

    gn = _glassnode_metric(cfg)
    if gn is not None:
        out["glassnode"] = gn.reindex(index, method="ffill")
    return out
