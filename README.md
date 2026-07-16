# TradingSystemML

A lean, modern ML pipeline that predicts the **direction** (up/down) of an
asset over a future horizon and backtests a simple long/flat strategy.
Example asset: **BTC-USD**.

This is a full rewrite of an old 2020 LSTM project. See
[Why it was rewritten](#why-the-rewrite) below.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train + evaluate (purged-CV OOF metrics + feature importance)
python -m trading train    --ticker BTC-USD --interval 1wk --horizon 4

# Predict the latest bar (calibrated P(up), meta confidence, position size)
python -m trading predict  --ticker BTC-USD --interval 1wk --horizon 4

# Leakage-free backtest on out-of-fold predictions vs buy & hold
python -m trading backtest --ticker BTC-USD --interval 1wk --horizon 4
```

`--interval` accepts `1d` (daily) or `1wk` (weekly bars — more signal, less
noise). `--horizon` is the number of bars ahead to predict.

## Design

```
trading/
├── config.py          # single dataclass of all settings
├── cli.py / __main__  # command-line interface
├── data/
│   ├── loaders.py     # yfinance OHLCV, cross-asset series, Binance funding
│   ├── features.py    # causal technical + cross-asset + funding features
│   └── labels.py      # labels + assembled dataset
├── models/
│   ├── cv.py          # purged walk-forward CV with embargo
│   └── classifier.py  # LightGBM primary + meta model, calibration, sizing
└── backtest/
    └── engine.py      # edge-based, non-overlapping backtest vs buy & hold
```

**Techniques applied**

- **Weekly bars / longer horizons** to raise signal-to-noise.
- **Cross-asset & macro features**: ETH, DXY, gold, S&P 500, VIX, plus
  Binance perpetual **funding rates** (public API, no key; skipped gracefully
  if the network blocks it).
- **On-chain features**: BTC network metrics (tx count, active addresses,
  hash rate, miner revenue) from the free blockchain.com charts API — no key
  required. Set `GLASSNODE_API_KEY` to add an extra Glassnode metric
  (`glassnode_metric`, default SOPR). Any unreachable source is skipped.
- **Purged walk-forward CV with embargo** — training folds are strictly in the
  past and the `horizon + embargo` overlap is purged, eliminating look-ahead.
- **Isotonic probability calibration** fit on out-of-fold predictions.
- **Meta-labeling** — a second model estimates whether a primary long signal
  is likely correct.
- **Edge-based position sizing** scaled by calibrated probability and meta
  confidence, capped by `max_position`.

## Why the rewrite

The original code had fundamental issues that made results unreliable and
the code unrunnable on modern stacks:

- **Data leakage**: scaler fit on the full dataset and `train_test_split`
  with `shuffle=True` on time series. Fixed with purged walk-forward CV and
  train-only fitting (LightGBM needs no scaling).
- **Predicted absolute price** with MSE → the model learned near-persistence.
  Now predicts **direction**, the quantity that matters for trading.
- **Transposed input tensor** and a broken accuracy metric (`int(acc)` → 0).
- **17 separate LSTMs** (one per horizon), 400 epochs each. Replaced by one
  fast LightGBM model (trains in seconds).
- **Broken/deprecated deps** (`yahoo_fin`, `numba.cuda`, private pandas
  imports) and a **hardcoded API token**. Removed.
- Deleted an unrelated empty `keylogger.py`.
- **Predicted absolute price** with MSE → the model learned near-persistence.
  Now predicts **direction**, the quantity that matters for trading.
- **Transposed input tensor** and a broken accuracy metric (`int(acc)` → 0).
- **17 separate LSTMs** (one per horizon), 400 epochs each. Replaced by one
  fast LightGBM model (trains in seconds).
- **Broken/deprecated deps** (`yahoo_fin`, `numba.cuda`, private pandas
  imports) and a **hardcoded API token**. Removed.
- Deleted an unrelated empty `keylogger.py`.
