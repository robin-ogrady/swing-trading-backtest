"""
setups.py  -  entry rules. Each setup takes one symbol's feature frame and returns a frame of
signals decided on the close of bar t (the engine fills them at the next open):

  long, short         bool   entry signal
  long_stop, short_stop      stop price for the trade
  long_exit, short_exit bool setup-specific exit signal (only used when exits.native_exit is on)
  score               float  0-100 setup quality (filled in by filters.py, NaN here)
"""
from typing import Callable, Dict

import numpy as np
import pandas as pd


def _clip_stop(close: pd.Series, raw_stop: pd.Series, atr: pd.Series, lo: float, hi: float,
               side: int) -> pd.Series:
    dist = (side * (close - raw_stop)).clip(lower=lo * atr, upper=hi * atr)
    return close - side * dist


def _out(f: pd.DataFrame, long, short, long_stop, short_stop, long_exit=None, short_exit=None):
    ok = f.atr14.notna() & (f.atr14 > 0)
    false = pd.Series(False, index=f.index)
    return pd.DataFrame({
        "long": (long & ok).fillna(False).astype(bool),
        "short": (short & ok).fillna(False).astype(bool),
        "long_stop": long_stop, "short_stop": short_stop,
        "long_exit": (false if long_exit is None else long_exit).fillna(False).astype(bool),
        "short_exit": (false if short_exit is None else short_exit).fillna(False).astype(bool),
        "score": np.nan,
    }, index=f.index)


def breakout(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """First close above the prior N-bar high (short: first close below the prior N-bar low)."""
    n = int(p["base_len"])
    hh = f.high.rolling(n).max().shift(1) if n != 20 else f.hh20
    ll = f.low.rolling(n).min().shift(1) if n != 20 else f.ll20
    c = f.close
    long = (c > hh) & (c.shift(1) <= hh.shift(1))
    short = (c < ll) & (c.shift(1) >= ll.shift(1))
    ls = _clip_stop(c, f.low, f.atr14, p["stop_atr_min"], p["stop_atr_max"], 1)
    ss = _clip_stop(c, f.high, f.atr14, p["stop_atr_min"], p["stop_atr_max"], -1)
    return _out(f, long, short, ls, ss)


def pullback(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Raschke Holy Grail / bone zone: trend stack + ADX, dip to the 20 EMA, close back over the 9 EMA."""
    k = int(p["lookback"])
    c = f.close
    trend_up = (f.ema9 > f.ema20) & (f.ema20 > f.ema50) & (f.adx > p["adx_min"])
    trend_dn = (f.ema9 < f.ema20) & (f.ema20 < f.ema50) & (f.adx > p["adx_min"])
    touched_dn = (f.low <= f.ema20).astype(float).rolling(k).max() > 0
    touched_up = (f.high >= f.ema20).astype(float).rolling(k).max() > 0
    long = trend_up & touched_dn & (c > f.ema9) & (c.shift(1) <= f.ema9.shift(1))
    short = trend_dn & touched_up & (c < f.ema9) & (c.shift(1) >= f.ema9.shift(1))
    ls = _clip_stop(c, f.low.rolling(k).min(), f.atr14, p["stop_atr_min"], p["stop_atr_max"], 1)
    ss = _clip_stop(c, f.high.rolling(k).max(), f.atr14, p["stop_atr_min"], p["stop_atr_max"], -1)
    return _out(f, long, short, ls, ss)


def rsi2(f: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Connors RSI(2) mean reversion in the direction of the 200-day trend."""
    ma = f.sma200 if int(p["ma"]) == 200 else f.close.rolling(int(p["ma"])).mean()
    c = f.close
    long = (c > ma) & (f.rsi2 < p["entry"])
    short = (c < ma) & (f.rsi2 > 100 - p["entry"])
    ls = c - p["stop_atr"] * f.atr14
    ss = c + p["stop_atr"] * f.atr14
    return _out(f, long, short, ls, ss, long_exit=f.rsi2 > p["exit"], short_exit=f.rsi2 < 100 - p["exit"])


SETUPS = {"breakout": breakout, "pullback": pullback, "rsi2": rsi2}  # type: Dict[str, Callable]
