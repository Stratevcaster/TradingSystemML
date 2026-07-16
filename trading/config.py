"""Central configuration for the trading-signal pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


@dataclass
class Config:
    # --- Data ---
    ticker: str = "BTC-USD"
    period: str = "10y"          # how much history to download
    interval: str = "1wk"        # "1d" for daily, "1wk" for weekly bars

    # Cross-asset / macro tickers pulled from Yahoo and turned into features.
    cross_assets: tuple[str, ...] = ("ETH-USD", "DX-Y.NYB", "GC=F", "^GSPC", "^VIX")
    # Binance perpetual symbol for funding-rate features (public API, no key).
    funding_symbol: str | None = "BTCUSDT"

    # On-chain features from the free blockchain.com charts API (no key).
    onchain: bool = True
    onchain_charts: tuple[str, ...] = (
        "n-transactions", "n-unique-addresses", "hash-rate", "miners-revenue",
    )
    # Optional Glassnode metric, used only when GLASSNODE_API_KEY is set.
    glassnode_metric: str = "indicators/sopr"

    # --- Target ---
    horizon: int = 4            # predict direction N bars ahead (weeks if 1wk)
    # A move counts as "up" only if the future return exceeds this threshold.
    up_threshold: float = 0.0

    # --- Purged walk-forward CV ---
    n_splits: int = 6
    embargo: int = 2            # bars embargoed on each side of every test fold

    # --- Model (LightGBM) ---
    params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "n_estimators": 3000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })
    early_stopping_rounds: int = 150
    calibrate: bool = True      # isotonic probability calibration

    # --- Meta-labeling + sizing ---
    meta_labeling: bool = True
    long_threshold: float = 0.55   # primary P(up) needed to consider a trade
    meta_threshold: float = 0.50   # meta P(correct) needed to actually trade
    max_position: float = 1.0      # cap on fractional position size
    fee: float = 0.0004            # per-trade cost

    @property
    def model_path(self) -> Path:
        tag = f"{self.ticker.replace('-', '_')}_{self.interval}_h{self.horizon}"
        return ARTIFACTS / f"{tag}.joblib"


DEFAULT = Config()
