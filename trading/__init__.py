"""Modern, lean trading-signal ML package.

Predicts the direction (up/down) of an asset over a future horizon using
engineered technical + cross-asset + funding features, a LightGBM primary
model with purged walk-forward CV, isotonic calibration, meta-labeling and
edge-based position sizing.
"""

__all__ = ["config", "data", "models", "backtest"]
