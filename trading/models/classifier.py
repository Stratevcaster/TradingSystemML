"""LightGBM primary + meta model with purged-CV OOF, calibration, sizing."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..config import Config
from ..data import build_dataset
from .cv import purged_walk_forward


@dataclass
class Bundle:
    primary: LGBMClassifier
    meta: LGBMClassifier | None
    calibrator: IsotonicRegression | None
    features: list[str]
    cfg: Config


def _fit_lgbm(cfg: Config, X: pd.DataFrame, y: pd.Series) -> LGBMClassifier:
    """Fit one LightGBM using a chronological tail of the data for early stop."""
    params = dict(cfg.params)
    n_estimators = params.pop("n_estimators")
    model = LGBMClassifier(n_estimators=n_estimators, **params)

    cut = int(len(X) * 0.85)
    if cut < len(X) and y.iloc[cut:].nunique() > 1:
        model.fit(X.iloc[:cut], y.iloc[:cut],
                  eval_set=[(X.iloc[cut:], y.iloc[cut:])],
                  callbacks=[early_stopping(cfg.early_stopping_rounds, verbose=False)])
    else:
        model.fit(X, y)
    return model


def _oof_predictions(cfg: Config, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Out-of-fold primary probabilities from purged walk-forward CV."""
    oof = pd.Series(np.nan, index=X.index)
    for tr, te in purged_walk_forward(len(X), cfg.n_splits, cfg.horizon, cfg.embargo):
        if y.iloc[tr].nunique() < 2:
            continue
        model = _fit_lgbm(cfg, X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def train(cfg: Config, verbose: bool = True) -> dict:
    X, y, ohlcv = build_dataset(cfg)

    # 1) Honest out-of-fold predictions for evaluation, calibration, meta-labels.
    oof = _oof_predictions(cfg, X, y)
    mask = oof.notna()
    Xo, yo, oofo = X[mask], y[mask], oof[mask]

    # 2) Probability calibration (isotonic) fit on OOF predictions.
    calibrator = None
    proba_cal = oofo.copy()
    if cfg.calibrate and yo.nunique() > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oofo.values, yo.values)
        proba_cal = pd.Series(calibrator.predict(oofo.values), index=oofo.index)

    # 3) Meta-labeling: given a primary long signal, predict if it is correct.
    meta = None
    if cfg.meta_labeling:
        signal = proba_cal >= cfg.long_threshold
        if signal.sum() > 30 and yo[signal].nunique() > 1:
            Xm = Xo[signal].copy()
            Xm["primary_proba"] = proba_cal[signal].values
            meta = _fit_lgbm(cfg, Xm, yo[signal])

    # 4) Final production models trained on all available data.
    primary = _fit_lgbm(cfg, X, y)

    metrics = _evaluate(cfg, yo, oofo, proba_cal)
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = Bundle(primary, meta, calibrator, list(X.columns), cfg)
    joblib.dump(bundle, cfg.model_path)

    if verbose:
        _print_report(cfg, primary, X, metrics)

    return {"bundle": bundle, "metrics": metrics, "X": X, "y": y,
            "ohlcv": ohlcv, "oof": oof, "proba_cal": proba_cal}


def _evaluate(cfg: Config, y: pd.Series, raw: pd.Series, cal: pd.Series) -> dict:
    pred = (cal >= 0.5).astype(int)
    out = {
        "oof_accuracy": accuracy_score(y, pred),
        "oof_f1": f1_score(y, pred, zero_division=0),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }
    if y.nunique() > 1:
        out["oof_auc_raw"] = roc_auc_score(y, raw)
        out["oof_auc_cal"] = roc_auc_score(y, cal)
    return out


def _print_report(cfg: Config, model, X, metrics: dict) -> None:
    print(f"\n=== {cfg.ticker}  interval={cfg.interval}  horizon={cfg.horizon} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    imp = pd.Series(model.feature_importances_, index=X.columns)
    print("\nTop features:")
    print(imp.sort_values(ascending=False).head(12).to_string())


def load(cfg: Config) -> Bundle:
    return joblib.load(cfg.model_path)


def _calibrated(bundle: Bundle, proba: np.ndarray) -> np.ndarray:
    if bundle.calibrator is not None:
        return bundle.calibrator.predict(proba)
    return proba


def position_size(bundle: Bundle, X_row: pd.DataFrame) -> dict:
    """Return primary proba, meta confidence, and a fractional position size."""
    cfg = bundle.cfg
    raw = bundle.primary.predict_proba(X_row[bundle.features])[:, 1]
    proba = _calibrated(bundle, raw)[0]

    meta_conf = 1.0
    if bundle.meta is not None:
        Xm = X_row[bundle.features].copy()
        Xm["primary_proba"] = proba
        meta_conf = float(bundle.meta.predict_proba(Xm)[:, 1][0])

    size = 0.0
    if proba >= cfg.long_threshold and meta_conf >= cfg.meta_threshold:
        edge = (proba - cfg.long_threshold) / max(1e-9, 1 - cfg.long_threshold)
        size = min(cfg.max_position, edge * meta_conf)
    return {"proba_up": float(proba), "meta_conf": meta_conf, "size": float(size)}
