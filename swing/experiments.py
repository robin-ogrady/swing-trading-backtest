"""
experiments.py  -  named variants built on top of the naive config, plus a multi-seed runner.
"""
import copy
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as C
import filters
import strategy

FILTER_LAYERS = ["regime", "vol", "chop", "rs", "volume", "event", "score rank only", "score top 20%"]
METRIC_KEYS = ["sharpe", "cagr", "maxdd", "sortino", "trades_per_yr", "exp_r", "profit_factor", "win_rate",
               "exposure", "trades"]


def layer_config(asset: str, setup: str, name: str) -> Optional[dict]:
    """Naive config plus one layer, or None when the layer does not apply."""
    cfg = strategy.naive_config(setup)
    if name == "naive":
        return cfg
    if name in FILTER_LAYERS:
        key = "score" if name.startswith("score") else name
        if not filters.APPLICABLE[key](asset, setup):
            return None
        if name == "score rank only":
            cfg["rank"] = "score"
        elif name == "score top 20%":
            cfg["rank"] = "score"
            cfg["filters"] = {"score": True}
        else:
            cfg["filters"] = {name: True}
        return cfg
    if name in C.SIZING_VARIANTS:
        cfg["sizing"].update(copy.deepcopy(C.SIZING_VARIANTS[name]))
        return cfg
    if name in C.EXIT_VARIANTS:
        if setup == "rsi2":
            return None
        cfg["exits"] = copy.deepcopy(C.EXIT_VARIANTS[name])
        return cfg
    raise ValueError(name)


def all_layer_names() -> List[str]:
    return ["naive"] + FILTER_LAYERS + list(C.SIZING_VARIANTS) + list(C.EXIT_VARIANTS)


class PrepCache:
    """Signal preparation depends only on asset, setup, setup params, filters and period."""
    def __init__(self):
        self.d = {}

    def get(self, asset: str, cfg: dict, period: str, allow_final: bool = False):
        key = (asset, period, cfg["setup"], json.dumps(cfg["setup_params"], sort_keys=True),
               json.dumps(cfg.get("filters", {}), sort_keys=True),
               json.dumps(cfg.get("layer_params", {}), sort_keys=True), cfg["exits"].get("trail_ma"))
        if key not in self.d:
            self.d[key] = strategy.prepare(asset, cfg, period, allow_final)
        return self.d[key]


def event_signature(prep) -> Tuple[int, int]:
    n = sum(len(v) for v in prep.events.values())
    h = hash(tuple(sorted((k, e[0], e[2]) for k, v in prep.events.items() for e in v)))
    return n, h


def multi_seed(asset: str, cfg: dict, period: str, prep, n_seeds: int, label: str,
               log: bool = True) -> Tuple[dict, List[dict], object]:
    """Runs n_seeds random-pick seeds. Returns (median stats, per-seed stats, seed-0 result)."""
    per = []
    res0 = None
    for s in range(n_seeds):
        res, st = strategy.run(asset, cfg, period, seed=s, label=label, prepared=prep, log=log and s == 0)
        per.append(st)
        if s == 0:
            res0 = res
    med = {k: float(np.nanmedian([p.get(k, np.nan) for p in per])) for k in METRIC_KEYS}
    med["sharpe_min"] = float(np.nanmin([p.get("sharpe", np.nan) for p in per]))
    med["sharpe_max"] = float(np.nanmax([p.get("sharpe", np.nan) for p in per]))
    return med, per, res0
