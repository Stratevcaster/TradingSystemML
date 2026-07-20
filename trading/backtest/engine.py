"""Walk-forward backtest driven by purged out-of-fold predictions.

Position sizing is edge-based on the calibrated primary probability, so the
backtest is leakage-free (predictions come from purged walk-forward CV).
Trades are non-overlapping (re-enter every `horizon` bars) to keep returns
independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data import forward_return
from ..models import train


def run(cfg: Config, verbose: bool = True) -> dict:
    res = train(cfg, verbose=False)
    proba = res["proba_cal"]              # purged OOF, calibrated
    ohlcv = res["ohlcv"]

    idx = proba.index
    fwd = forward_return(ohlcv, cfg).reindex(idx)

    # Edge-based fractional sizing from the calibrated probability.
    edge = ((proba - cfg.long_threshold) / max(1e-9, 1 - cfg.long_threshold))
    size = edge.clip(lower=0, upper=cfg.max_position)

    # Non-overlapping entries only.
    take = np.zeros(len(idx), dtype=bool)
    take[::cfg.horizon] = True
    active = take & (size.values > 0)

    sizes = size.values[active]
    rets = fwd.values[active]
    valid = ~np.isnan(rets)
    sizes, rets = sizes[valid], rets[valid]

    trade_ret = sizes * rets - cfg.fee * (sizes > 0)
    equity = np.cumprod(1 + trade_ret)

    n = len(trade_ret)
    total = float(equity[-1] - 1) if n else 0.0
    win_rate = float((trade_ret > 0).mean()) if n else 0.0
    bh = float(ohlcv["close"].reindex(idx).iloc[-1]
               / ohlcv["close"].reindex(idx).iloc[0] - 1)
    days = (idx[-1] - idx[0]).days or 1
    cagr = (1 + total) ** (365 / days) - 1 if total > -1 else -1.0
    per_year = 252 / cfg.horizon if cfg.interval == "1d" else 52 / cfg.horizon
    sharpe = (float(np.mean(trade_ret) / np.std(trade_ret) * np.sqrt(per_year))
              if n and np.std(trade_ret) > 0 else 0.0)

    summary = {
        "trades": n, "win_rate": win_rate, "total_return": total,
        "cagr": cagr, "sharpe": sharpe, "buy_hold_return": bh,
        "oof_start": idx[0].date().isoformat(),
        "oof_end": idx[-1].date().isoformat(),
    }
    if verbose:
        print(f"\n=== Backtest {cfg.ticker} ({cfg.interval}, h={cfg.horizon}) "
              f"OOF {summary['oof_start']} -> {summary['oof_end']} ===")
        print(f"trades={n}  win_rate={win_rate:.1%}")
        print(f"strategy total={total:.1%}  CAGR={cagr:.1%}  Sharpe={sharpe:.2f}")
        print(f"buy&hold total={bh:.1%}")
    return summary
