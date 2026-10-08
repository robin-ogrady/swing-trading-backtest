"""
filters.py  -  the "human judgment" layers. Each one is an independent toggle in cfg["filters"].

  regime   stand down unless the market trend agrees.
           stocks: SPY above its 200 SMA and breadth (share above 50 SMA) >= 50%. Long only.
           crypto: longs only while BTC is above its 200 SMA, shorts only while below.
           forex:  per pair, longs above its own 200 SMA, shorts below.
  vol      stand down when market 20-day vol is above the 90th percentile of its past year
           (stocks SPY, crypto BTC, forex the pair itself).
  chop     trend setups skip symbols with ADX(14) < 20.
  rs       longs need a cross-sectional relative strength percentile >= 80, shorts <= 20.
  volume   breakout day volume >= 1.4x its 50-day average; pullback bars' volume below average.
  event    stocks: skip when an earnings-like gap is likely in the next ~2 weeks
           (bars since the last gap-on-volume-spike day is near a multiple of 63).
  score    rank signals by a 0-100 quality score and take only the top X% (threshold from
           the scores of all PAST signals, recomputed monthly, so no lookahead).

The score is always computed so ranking can use it. It is the mean of component
percentiles (vs all tradable symbols that day), flipped for shorts where direction matters.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C
import context

# which layers make sense where (anything else is skipped, not counted as a variant)
APPLICABLE = {
    "regime": lambda a, s: True,
    "vol": lambda a, s: True,
    "chop": lambda a, s: s == "breakout",          # pullback already needs ADX > 25; rsi2 is mean reversion
    "rs": lambda a, s: True,
    "volume": lambda a, s: a != "forex" and s in ("breakout", "pullback"),
    "event": lambda a, s: a == "stocks",
    "score": lambda a, s: True,
}


def layer_params(cfg: dict) -> dict:
    """config.LAYERS with any per-config overrides from cfg["layer_params"] (used by sensitivity tests)."""
    L = {k: dict(v) for k, v in C.LAYERS.items()}
    for k, v in cfg.get("layer_params", {}).items():
        L[k].update(v)
    return L


def _col(ctx: dict, key: str, sym: str, idx) -> pd.Series:
    p = ctx.get(key)
    if p is None or sym not in p:
        return pd.Series(np.nan, index=idx)
    return p[sym].reindex(idx)


def score(asset: str, setup: str, sym: str, f: pd.DataFrame, ctx: dict) -> (pd.Series, pd.Series):
    """Returns (long score, short score), 0-100."""
    idx = f.index
    P = lambda k: _col(ctx, k, sym, idx)
    comps_l, comps_s = [], []

    def directional(p):          # high percentile is good for longs, bad for shorts
        comps_l.append(p)
        comps_s.append(100 - p)

    def both(p):                 # same meaning for both sides
        comps_l.append(p)
        comps_s.append(p)

    if setup == "breakout":
        directional(P("rs_raw"))
        both(100 - P("tightness"))
        if asset != "forex":
            both(100 - P("vol10_ratio"))
        directional(P("sma20_slope5"))
        comps_l.append(pd.Series(np.where(f.higher_low > 0, 100.0, 0.0), index=idx))
        comps_s.append(pd.Series(np.where(f.higher_low < 0, 100.0, 0.0), index=idx))
        comps_l.append(100 * (1 - ((f.close - f.hh20) / f.atr14) / 2).clip(0, 1))
        comps_s.append(100 * (1 - ((f.ll20 - f.close) / f.atr14) / 2).clip(0, 1))
    elif setup == "pullback":
        directional(P("rs_raw"))
        both(P("adx"))
        directional(100 - P("body_ratio10"))
        if asset != "forex":
            both(100 - P("vol3_ratio"))
        directional(P("ema50_slope10"))
    elif setup == "rsi2":
        comps_l.append(100 * (1 - f.rsi2 / 10).clip(0, 1))
        comps_s.append(100 * (1 - (100 - f.rsi2) / 10).clip(0, 1))
        directional(P("trend200"))
        directional(P("rs_raw"))
    sl = pd.concat(comps_l, axis=1).mean(axis=1)
    ss = pd.concat(comps_s, axis=1).mean(axis=1)
    return sl, ss


def apply(asset: str, sym: str, f: pd.DataFrame, sig: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Signal hook for strategy.prepare: masks entries and fills the score."""
    fl = cfg.get("filters", {})
    ctx = context.load(asset)
    L = layer_params(cfg)
    idx = f.index
    mkt = ctx["market"].reindex(idx)
    sig = sig.copy()
    ok_l = pd.Series(True, index=idx)
    ok_s = pd.Series(True, index=idx)
    if fl.get("regime"):
        if asset == "stocks":
            up = mkt.mkt_up.astype(float).fillna(0) > 0
            ok_l &= up & (mkt.breadth50 >= L["regime"]["breadth_min"])
        elif asset == "crypto":
            up = mkt.mkt_up.astype(float)
            ok_l &= up == 1
            ok_s &= up == 0
        else:
            ok_l &= f.close > f.sma200
            ok_s &= f.close < f.sma200
    if fl.get("vol"):
        vp = f.vol20_pct252 if asset == "forex" else mkt.mkt_vol_pct
        calm = vp <= L["vol"]["max_pct"]
        ok_l &= calm
        ok_s &= calm
    if fl.get("chop"):
        trending = f.adx >= L["chop"]["adx_min"]
        ok_l &= trending
        ok_s &= trending
    if fl.get("rs"):
        rs = _col(ctx, "rs_raw", sym, idx)
        ok_l &= rs >= L["rs"]["min_pct"]
        ok_s &= rs <= 100 - L["rs"]["min_pct"]
    if fl.get("volume"):
        if cfg["setup"] == "breakout":
            v = f.vol_ratio >= L["volume"]["breakout_min"]
        else:
            v = f.vol3_ratio <= L["volume"]["pullback_max"]
        ok_l &= v
        ok_s &= v
    if fl.get("event"):
        e = L["event"]
        b = f.bars_since_event
        ph = b % e["period"]
        near = (b >= 40) & (b <= e["max_bars"]) & ((ph >= e["period"] - e["window_before"]) | (ph <= e["window_after"]))
        ok_l &= ~near
        ok_s &= ~near
    sig["long_raw"] = sig["long"]           # before any layer: the score gate ranks against these
    sig["short_raw"] = sig["short"]
    sig["long"] = sig["long"] & ok_l.fillna(False)
    sig["short"] = sig["short"] & ok_s.fillna(False)
    sl, ss = score(asset, cfg["setup"], sym, f, ctx)
    sig["score"] = np.where(sig["short_raw"], ss, sl)
    return sig


def score_gate(events: Dict[int, list], dates: pd.DatetimeIndex, top_pct: float,
               min_history: int, raw_scores: Dict[int, list]) -> Dict[int, list]:
    """
    Keep only signals whose score is in the top X% of the scores of all EARLIER raw setup
    signals (every signal the setup produced on a tradable bar, before the other layers).
    Threshold recomputed at each month start. Ranking against the raw setup signals keeps
    "top 20%" meaning the same whether or not other layers are stacked on top.
    """
    out = {}
    hist = []  # type: List[float]
    thr = np.inf
    month = None
    for k in sorted(set(events) | set(raw_scores)):
        m = (dates[k].year, dates[k].month)
        if m != month:
            month = m
            h = np.array(hist)
            h = h[~np.isnan(h)]
            thr = np.quantile(h, 1 - top_pct / 100) if len(h) >= min_history else np.inf
        ev = events.get(k)
        if ev:
            keep = [e for e in ev if e[4] == e[4] and e[4] >= thr]
            if keep:
                out[k] = keep
        hist.extend(raw_scores.get(k, []))
    return out
