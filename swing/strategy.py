"""
strategy.py  -  turns a strategy config into signals and runs the engine for one period.

A strategy config is a plain dict:
  {"setup": "breakout", "setup_params": {...}, "filters": {...},
   "exits": {...}, "sizing": {...}, "rank": "random" | "score"}

Periods: "dev", "val", "final" (see config.SPLITS). "final" is refused unless allow_final=True,
and run.py only passes that from the `final` command. Every run is appended to
results/variant_log.csv so the number of variants tried is on record.
"""
import copy
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as C
import data
import engine
import features
import filters
import metrics
import setups

VARIANT_LOG = C.RESULTS_DIR / "variant_log.csv"


def naive_config(setup: str) -> dict:
    return {"setup": setup, "setup_params": copy.deepcopy(C.SETUP_PARAMS[setup]), "filters": {},
            "exits": copy.deepcopy(C.NAIVE_EXITS[setup]), "sizing": copy.deepcopy(C.NAIVE_SIZING),
            "rank": "random"}


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def period_bounds(asset: str, period: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    sp = C.SPLITS[asset]
    ts = lambda s: pd.Timestamp(s, tz="UTC")
    if period == "dev":
        return ts(sp["trade_start"]), ts(sp["dev_end"]) + pd.Timedelta(hours=23)
    if period == "val":
        return ts(sp["dev_end"]) + pd.Timedelta(days=1), ts(sp["val_end"]) + pd.Timedelta(hours=23)
    if period == "final":
        return ts(sp["val_end"]) + pd.Timedelta(days=1), ts("2100-01-01")
    raise ValueError(period)


def tradable(asset: str, f: pd.DataFrame) -> pd.Series:
    ok = f.age >= C.WARMUP_BARS
    if asset == "crypto":
        ok &= f.dollar_vol30 >= C.CRYPTO_MIN_DOLLAR_VOL
    return ok


class Prepared:
    """Everything the engine needs for one (asset, config, period)."""
    def __init__(self, syms, events, dates):
        self.syms, self.events, self.dates = syms, events, dates


def prepare(asset: str, cfg: dict, period: str, allow_final: bool = False) -> Prepared:
    if period == "final" and not allow_final:
        raise RuntimeError("final period is locked. Only `run.py final` may touch it.")
    start, end = period_bounds(asset, period)
    features.build_feature_cache(asset)
    uni = data.universe(asset)
    data_end = uni.end.max()
    setup_fn = setups.SETUPS[cfg["setup"]]
    long_only = C.LONG_ONLY[asset]
    frames = []
    for _, u in uni.iterrows():
        if u.end < start or u.start > end:
            continue
        f = features.load(asset, u.symbol)
        sig = setup_fn(f, cfg["setup_params"])
        sig = filters.apply(asset, u.symbol, f, sig, cfg)
        tm = cfg["exits"].get("trail_ma")
        f = f.assign(matrail=f.close.rolling(int(tm)).mean() if tm else np.nan)
        ok = tradable(asset, f)
        win = (f.index >= start) & (f.index <= end)
        if not win.any():
            continue
        ends_early = u.end < data_end and u.end <= end     # history stops inside this window
        frames.append((u.symbol, f.loc[win], sig.loc[win], ok.loc[win], ends_early))
    dates = pd.DatetimeIndex(sorted(set().union(*[set(fr[1].index) for fr in frames])))
    syms = []  # type: List[engine.SymArrays]
    events = {}  # type: Dict[int, list]
    raw_scores = {}  # type: Dict[int, list]
    for sid, (name, f, sig, ok, ends_early) in enumerate(frames):
        u = dates.searchsorted(f.index)
        syms.append(engine.SymArrays(name, u, f, sig, bool(ends_early), ok.values))
        sides = [(1, "long", "long_stop")] if long_only else [(1, "long", "long_stop"), (-1, "short", "short_stop")]
        for side, col, stop_col in sides:
            rows = np.flatnonzero(sig[col].values & ok.values)
            stops = sig[stop_col].values
            score = sig.score.values
            for row in rows:
                events.setdefault(int(u[row]), []).append((sid, int(row), side, float(stops[row]), float(score[row])))
            for row in np.flatnonzero(sig[col + "_raw"].values & ok.values):
                raw_scores.setdefault(int(u[row]), []).append(float(score[row]))
    if cfg.get("filters", {}).get("score"):
        sp = filters.layer_params(cfg)["score"]
        events = filters.score_gate(events, dates, sp["top_pct"], sp["min_history"], raw_scores)
    return Prepared(syms, events, dates)


def log_variant(asset: str, period: str, cfg: dict, label: str, seed: int, stats: dict) -> None:
    VARIANT_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "asset": asset, "period": period,
           "label": label, "hash": cfg_hash(cfg), "seed": seed,
           **{k: stats.get(k) for k in ("sharpe", "cagr", "maxdd", "trades", "exp_r")},
           "config": json.dumps(cfg, sort_keys=True, default=str)}
    pd.DataFrame([row]).to_csv(VARIANT_LOG, mode="a", header=not VARIANT_LOG.exists(), index=False)


def run(asset: str, cfg: dict, period: str = "dev", seed: int = 0, label: str = "",
        allow_final: bool = False, prepared: Optional[Prepared] = None, log: bool = True,
        cost_mult: float = 1.0):
    prep = prepared or prepare(asset, cfg, period, allow_final)
    res = engine.simulate(asset, prep.syms, prep.events, prep.dates, cfg, seed, cost_mult)
    stats = metrics.summarize(res)
    if log:
        log_variant(asset, period, cfg, label, seed, stats)
    return res, stats
