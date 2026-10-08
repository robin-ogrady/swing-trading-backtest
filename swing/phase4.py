"""
phase4.py  -  ablation, picked configs, sensitivity, random-entry Monte Carlo, bootstrap,
regime / yearly breakdown and the multiple-testing haircut.

Order matters for honesty:
  1. make_picks() reads the phase 3 DEV results and writes results/phase4_picks.json.
     If the file exists it is reused, never rebuilt, so picks cannot drift after validation.
  2. only then are validation-period runs made.
"""
import copy
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

import config as C
import data
import experiments
import features
import filters
import strategy

PICKS_PATH = C.RESULTS_DIR / "phase4_picks.json"
ALL_EXIT = "partial 2R + BE + 20MA"
ALL_SIZING = {"heat_cap": 0.06, "dd_reduce": {"dd": 0.10, "factor": 0.5}}
FILTER_KEYS = ["regime", "vol", "chop", "rs", "volume", "event", "score"]
PICK_ASSETS = ["stocks", "crypto"]          # forex dropped from picks (nothing worked on dev)


# ------------------------------------------------------------------ configs
def all_layers(asset: str, setup: str, minus: Optional[str] = None) -> dict:
    """Every applicable layer on (textbook values). minus = one group to leave out."""
    cfg = strategy.naive_config(setup)
    fl = {k: True for k in FILTER_KEYS if filters.APPLICABLE[k](asset, setup) and k != minus}
    cfg["filters"] = fl
    if fl.get("score"):
        cfg["rank"] = "score"
    if setup != "rsi2" and minus != "exit":
        cfg["exits"] = copy.deepcopy(C.EXIT_VARIANTS[ALL_EXIT])
    if minus != "sizing":
        cfg["sizing"].update(copy.deepcopy(ALL_SIZING))
    return cfg


def minus_groups(asset: str, setup: str) -> List[str]:
    g = [k for k in FILTER_KEYS if filters.APPLICABLE[k](asset, setup)]
    if setup != "rsi2":
        g.append("exit")
    return g + ["sizing"]


def make_picks() -> dict:
    if PICKS_PATH.exists():
        return json.loads(PICKS_PATH.read_text())
    p3 = pd.read_csv(C.RESULTS_DIR / "phase3_layers.csv")
    picks = {"made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "rule": "filters scored + on dev; best + exit on dev; sizing left at naive (it moved DD, not edge)",
             "configs": {}}
    for asset in PICK_ASSETS:
        for setup in ("breakout", "pullback", "rsi2"):
            d = p3[(p3.asset == asset) & (p3.setup == setup)].set_index("layer")
            plus = set(d.index[d.verdict == "+"])
            cfg = strategy.naive_config(setup)
            names = []
            fl = {k: True for k in ("regime", "vol", "chop", "rs", "volume", "event") if k in plus}
            names += list(fl)
            if "score top 20%" in plus:
                fl["score"] = True
                cfg["rank"] = "score"
                names.append("score top 20%")
            elif "score rank only" in plus:
                cfg["rank"] = "score"
                names.append("score rank only")
            cfg["filters"] = fl
            ex = d.loc[[e for e in C.EXIT_VARIANTS if e in plus]] if setup != "rsi2" else d.iloc[:0]
            if len(ex):
                best = ex.d_sharpe.idxmax()
                cfg["exits"] = copy.deepcopy(C.EXIT_VARIANTS[best])
                names.append(best)
            picks["configs"][f"{asset}/{setup}"] = {"layers": names, "cfg": cfg}
    PICKS_PATH.write_text(json.dumps(picks, indent=1))
    return picks


# ------------------------------------------------------------------ sensitivity
def _perturb(value, m, kind="plain"):
    if kind == "top":            # percentile threshold: move the size of the kept tail
        return 100 - (100 - value) * m
    if kind == "top_frac":
        return min(0.99, 1 - (1 - value) * m)
    v = value * m
    return int(round(v)) if isinstance(value, int) else round(v, 4)


def sensitivity_params(asset: str, cfg: dict) -> List[Tuple[str, callable]]:
    """(label, function(cfg, m) -> new cfg) for every number that shapes this config."""
    out = []

    def setter(path, kind="plain"):
        def f(c, m):
            c = copy.deepcopy(c)
            node = c
            for k in path[:-1]:
                node = node.setdefault(k, {})
            base = node.get(path[-1])
            if base is None:
                base = C.LAYERS[path[1]][path[2]] if path[0] == "layer_params" else None
            node[path[-1]] = _perturb(base, m, kind)
            return c
        return f

    for k, v in cfg["setup_params"].items():
        if isinstance(v, (int, float)):
            out.append((f"setup.{k}", setter(["setup_params", k])))
    fl = cfg.get("filters", {})
    if fl.get("regime") and asset == "stocks":
        out.append(("regime.breadth_min", setter(["layer_params", "regime", "breadth_min"])))
    if fl.get("vol"):
        out.append(("vol.max_pct", setter(["layer_params", "vol", "max_pct"], "top_frac")))
    if fl.get("chop"):
        out.append(("chop.adx_min", setter(["layer_params", "chop", "adx_min"])))
    if fl.get("rs"):
        out.append(("rs.min_pct", setter(["layer_params", "rs", "min_pct"], "top")))
    if fl.get("volume"):
        key = "breakout_min" if cfg["setup"] == "breakout" else "pullback_max"
        out.append((f"volume.{key}", setter(["layer_params", "volume", key])))
    if fl.get("event"):
        out.append(("event.window_before", setter(["layer_params", "event", "window_before"])))
    if fl.get("score"):
        out.append(("score.top_pct", setter(["layer_params", "score", "top_pct"])))
    ex = cfg["exits"]
    for k in ("target_r", "max_hold", "trail_ma", "trail_atr"):
        if ex.get(k):
            out.append((f"exit.{k}", setter(["exits", k])))
    for k in ("frac", "at_r", "after_bars"):
        if ex.get("partial") and ex["partial"].get(k):
            out.append((f"exit.partial.{k}", setter(["exits", "partial", k])))
    if ex.get("time_stop"):
        out.append(("exit.time_stop.bars", setter(["exits", "time_stop", "bars"])))
        out.append(("exit.time_stop.min_r", setter(["exits", "time_stop", "min_r"])))
    out.append(("sizing.risk_pct", setter(["sizing", "risk_pct"])))
    return out


# ------------------------------------------------------------------ Monte Carlo
def random_entry_runs(asset: str, cfg: dict, prep, n: int, mode: str = "same_day",
                      seed0: int = 1000) -> pd.DataFrame:
    """
    Replace the entry signals with random ones, keep exits, sizing and limits.
      same_day: each real signal becomes a random tradable symbol on the same day and side,
                with the same stop distance in ATR units. Tests symbol selection.
      any_day:  same number of signals and sides, random days and symbols. Tests timing too.
    """
    syms = prep.syms
    nd = len(prep.dates)
    cand_s, cand_r, cand_k = [], [], []
    for sid, sa in enumerate(syms):
        good = np.flatnonzero(sa.ok & (sa.atr > 0))
        cand_s.append(np.full(len(good), sid))
        cand_r.append(good)
        cand_k.append(sa.u[good])
    cs, cr, ck = np.concatenate(cand_s), np.concatenate(cand_r), np.concatenate(cand_k)
    order = np.argsort(ck, kind="stable")
    cs, cr, ck = cs[order], cr[order], ck[order]
    start = np.searchsorted(ck, np.arange(nd))
    count = np.searchsorted(ck, np.arange(nd), side="right") - start
    ev = [(k, e[2], abs(syms[e[0]].c[e[1]] - e[3]) / syms[e[0]].atr[e[1]])
          for k, lst in prep.events.items() for e in lst]
    ek = np.array([e[0] for e in ev])
    eside = np.array([e[1] for e in ev])
    edist = np.array([e[2] for e in ev])
    days_ok = np.flatnonzero(count > 0)
    rows = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        k = ek if mode == "same_day" else np.sort(rng.choice(days_ok, size=len(ek)))
        side = eside if mode == "same_day" else rng.permutation(eside)
        j = start[k] + (rng.random(len(k)) * count[k]).astype(int)
        s_, r_ = cs[j], cr[j]
        events = {}  # type: Dict[int, list]
        for kk, sid, row, sd, dist in zip(k.tolist(), s_.tolist(), r_.tolist(), side.tolist(), edist.tolist()):
            sa = syms[sid]
            events.setdefault(kk, []).append((sid, row, sd, sa.c[row] - sd * dist * sa.atr[row], np.nan))
        p = strategy.Prepared(syms, events, prep.dates)
        _, st = strategy.run(asset, cfg, "mc", seed=i, prepared=p, log=False)
        rows.append({k2: st.get(k2, np.nan) for k2 in ("sharpe", "cagr", "exp_r", "maxdd", "trades")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ bootstrap
def block_bootstrap_sharpe(ret: pd.Series, ppy: int, n: int = 2000, block: int = 20,
                           seed: int = 7) -> Tuple[float, float]:
    r = ret.dropna().values
    T = len(r)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, T - block, size=nb)
        x = np.concatenate([r[s:s + block] for s in st])[:T]
        sd = x.std()
        out[i] = x.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


def bootstrap_mean(x: np.ndarray, n: int = 2000, seed: int = 7) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x))).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


# ------------------------------------------------------------------ regimes
def market_regime(asset: str) -> pd.Series:
    """bull: proxy above a rising 200 SMA. bear: below a falling one. chop: anything else."""
    m = features.compute(data.load_spy(), "stocks") if asset == "stocks" else features.load("crypto", "BTCUSDT")
    up = (m.close > m.sma200) & (m.sma200_slope21 > 0)
    dn = (m.close < m.sma200) & (m.sma200_slope21 < 0)
    reg = pd.Series(np.where(up, "bull", np.where(dn, "bear", "chop")), index=m.index)
    return reg[m.sma200_slope21.notna()]


# ------------------------------------------------------------------ haircut
def haircut(sharpe: float, years: float, all_sr: np.ndarray, all_years: np.ndarray) -> dict:
    """Harvey & Liu (2015) haircut: Bonferroni and BHY adjusted p-values over every variant tried."""
    N = len(all_sr)
    t = sharpe * np.sqrt(years)
    p = 2 * (1 - sps.norm.cdf(abs(t)))
    p_all = 2 * (1 - sps.norm.cdf(np.abs(all_sr * np.sqrt(all_years))))
    p_bon = min(1.0, p * N)
    # BHY step-up adjusted p for this p-value
    ps = np.sort(np.append(p_all, p))
    M = len(ps)
    cm = np.sum(1.0 / np.arange(1, M + 1))
    adj = ps * M * cm / np.arange(1, M + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    p_bhy = float(min(1.0, adj[np.searchsorted(ps, p)]))

    def sr_from_p(pp):
        if pp >= 1:
            return 0.0
        return sps.norm.ppf(1 - pp / 2) / np.sqrt(years) * np.sign(sharpe)

    sb, sh = sr_from_p(p_bon), sr_from_p(p_bhy)
    return {"t": t, "p": p, "N": N, "p_bonferroni": p_bon, "sr_bonferroni": sb, "p_bhy": p_bhy, "sr_bhy": sh,
            "haircut_bhy": 1 - sh / sharpe if sharpe > 0 else np.nan}
