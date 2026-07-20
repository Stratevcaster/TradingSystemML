"""Command-line interface."""
from __future__ import annotations

import argparse
import dataclasses

from .config import Config
from . import backtest, models
from .data import build_features
from .data.loaders import load_ohlcv


def _cfg_from_args(args) -> Config:
    fields = {f.name for f in dataclasses.fields(Config)}
    overrides = {k: v for k, v in vars(args).items()
                 if k in fields and v is not None}
    return dataclasses.replace(Config(), **overrides)


def _predict(cfg: Config) -> None:
    bundle = models.load(cfg)
    ohlcv = load_ohlcv(cfg.ticker, cfg.period, cfg.interval)
    X = build_features(ohlcv, cfg).dropna()
    row = X.iloc[[-1]]
    out = models.position_size(bundle, row)
    print(f"{X.index[-1].date()}  {cfg.ticker} ({cfg.interval}, h={cfg.horizon})")
    print(f"  P(up)={out['proba_up']:.3f}  meta={out['meta_conf']:.3f}  "
          f"position={out['size']:.2f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="trading")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "predict", "backtest"):
        sp = sub.add_parser(name)
        sp.add_argument("--ticker", type=str)
        sp.add_argument("--horizon", type=int)
        sp.add_argument("--interval", type=str, choices=["1d", "1wk"])
        sp.add_argument("--period", type=str)

    args = p.parse_args(argv)
    cfg = _cfg_from_args(args)

    if args.cmd == "train":
        models.train(cfg)
    elif args.cmd == "predict":
        _predict(cfg)
    elif args.cmd == "backtest":
        backtest.run(cfg)


if __name__ == "__main__":
    main()
