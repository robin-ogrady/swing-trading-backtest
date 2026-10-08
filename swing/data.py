"""
data.py  -  load the raw daily parquet files, clean them, split them into clean segments
and cache one parquet per segment in cache/clean/<asset>/.

Cleaning rules (all counted and printed by run.py data):
  all      drop duplicate / NaN / non-positive rows, force high/low to contain open and close,
           split the series at long date gaps (each piece becomes its own symbol).
  stocks   start after the last cluster of flat (H == L) or zero-volume bars (old Yahoo history).
           No bad-tick filter: tested, it mostly deleted real crash days (Mar 2020, Oct 1987).
  forex    G10 pairs only by default, drop weekend stubs, remove bad ticks (Yahoo 2008 has
           misdated rows), volume set to NaN (always 0 in the source).
           Yahoo's forex "close" is a stale snapshot taken near the open since about 2009,
           so the real end-of-day price is the next bar's open. close[t] is replaced with
           open[t+1] everywhere (before 2009 the two differ by 1-2 bps). Yahoo's high/low window
           also misses part of the day, so on weekdays followed by a trading day the range is
           widened to contain the close. Before a weekend the close can sit outside the range
           and the engine treats it as a gap.
  crypto   drop stablecoins / fiat / wrapped duplicates, split at redenominations
           (price jumps 20x+ in a day while coin volume jumps the other way).

The bad-tick rule looks one bar ahead. That is data cleaning, not a trading signal:
a bar that is removed never existed in a usable way.
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C

CLEAN_DIR = C.CACHE_DIR / "clean"


def raw_symbols(asset: str) -> List[str]:
    return sorted(p.stem for p in (C.DATA_DIR / asset / "1d").glob("*.parquet"))


def load_raw(asset: str, sym: str) -> pd.DataFrame:
    return pd.read_parquet(C.DATA_DIR / asset / "1d" / f"{sym}.parquet")


# ------------------------------------------------------------------ helpers
def _basic(df: pd.DataFrame, st: Dict[str, int]) -> pd.DataFrame:
    n0 = len(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=C.OHLC)
    df = df[(df[C.OHLC] > 0).all(axis=1)].copy()
    st["rows_dropped_basic"] += n0 - len(df)
    hi = df[C.OHLC].max(axis=1)
    lo = df[C.OHLC].min(axis=1)
    st["ohlc_fixed"] += int(((hi != df.high) | (lo != df.low)).sum())
    df["high"], df["low"] = hi, lo
    neg = df.volume < 0
    st["negative_volume_nulled"] += int(neg.sum())
    df.loc[neg, "volume"] = np.nan
    return df


def _spike_index(close: pd.Series, floor: float) -> pd.Index:
    r = np.log(close).diff()
    sig = r.abs().rolling(60, min_periods=20).median().shift(1).bfill() * 1.4826
    thr = np.maximum(C.SPIKE_Z * sig, floor)
    nxt = r.shift(-1)
    big = (r.abs() > thr) & (nxt.abs() > thr)
    rev = (np.sign(r) != np.sign(nxt)) & ((r + nxt).abs() < 0.5 * np.minimum(r.abs(), nxt.abs()))
    return close.index[big & rev]


def _remove_spikes(df: pd.DataFrame, floor: float, st: Dict[str, int]) -> pd.DataFrame:
    for _ in range(3):
        bad = _spike_index(df.close, floor)
        if len(bad) == 0:
            break
        st["bad_ticks_removed"] += len(bad)
        df = df.drop(bad)
    return df


def _stock_clean_start(df: pd.DataFrame, st: Dict[str, int]) -> pd.DataFrame:
    bad = ((df.high == df.low) | (df.volume <= 0)).astype(int)
    cluster = bad.rolling(20, min_periods=1).sum() >= C.STOCK_BAD_BARS_IN_20
    if cluster.any():
        last = cluster[cluster].index[-1]
        n0 = len(df)
        df = df.loc[df.index > last]
        st["rows_dropped_bad_history"] += n0 - len(df)
    return df


def _redenom_breaks(df: pd.DataFrame) -> List[pd.Timestamp]:
    r = np.log(df.close).diff()
    v = df.volume.replace(0, np.nan)
    before = v.rolling(5, min_periods=1).median().shift(1)
    after = v[::-1].rolling(5, min_periods=1).median()[::-1]
    lvr = np.log(after / before)
    lim = np.log(C.REDENOM_FACTOR)
    hit = (r.abs() > lim) & (np.sign(lvr) == -np.sign(r)) & ((lvr + r).abs() < np.log(10))
    return list(df.index[hit.fillna(False)])


def _split(df: pd.DataFrame, max_gap: int, breaks: List[pd.Timestamp]) -> List[pd.DataFrame]:
    cut = pd.Series(df.index.to_series().diff().dt.days > max_gap, index=df.index)
    for b in breaks:
        cut.loc[b] = True
    seg = cut.cumsum()
    return [g for _, g in df.groupby(seg)]


def forex_pairs(universe: str) -> List[str]:
    syms = raw_symbols("forex")
    if universe == "all":
        return syms
    g = set(C.G10_CCY)
    return [s for s in syms if s[:3] in g and s[3:] in g]


def eligible_raw(asset: str) -> List[str]:
    if asset == "forex":
        return forex_pairs(C.FOREX_UNIVERSE)
    if asset == "crypto":
        ex = set(C.CRYPTO_EXCLUDE)
        return [s for s in raw_symbols("crypto") if s[:-4] not in ex]
    return raw_symbols(asset)


# ------------------------------------------------------------------ main clean
def clean_symbol(asset: str, sym: str, st: Dict[str, int]) -> List[pd.DataFrame]:
    df = _basic(load_raw(asset, sym), st)
    breaks = []  # type: List[pd.Timestamp]
    if asset == "stocks":
        df = _stock_clean_start(df, st)
    elif asset == "forex":
        n0 = len(df)
        df = df[df.index.dayofweek < 5]
        st["weekend_rows_dropped"] += n0 - len(df)
        df = _remove_spikes(df, C.SPIKE_FLOOR["forex"], st)
        nxt = df.open.shift(-1)
        st["fx_close_outside_range"] += int(((nxt > df.high) | (nxt < df.low)).sum())
        df = df.assign(close=nxt, volume=np.nan).iloc[:-1]
        # price moves continuously from open to the true close on a normal weekday, so the bar's
        # range must contain the close. Across weekends / holidays the gap stays a real gap.
        next_day = pd.Series(df.index, index=df.index).shift(-1).fillna(df.index[-1])
        cont = (next_day - df.index.to_series()).dt.days == 1
        df.loc[cont, "high"] = np.maximum(df.high[cont], df.close[cont])
        df.loc[cont, "low"] = np.minimum(df.low[cont], df.close[cont])
    elif asset == "crypto":
        breaks = _redenom_breaks(df)
        st["redenominations"] += len(breaks)
    segs = _split(df, C.MAX_GAP_DAYS[asset], breaks)
    st["segments_total"] += len(segs)
    keep = [s for s in segs if len(s) >= C.MIN_ROWS]
    st["segments_too_short"] += len(segs) - len(keep)
    return keep


def build_clean_cache(asset: str, force: bool = False) -> pd.DataFrame:
    """Cleans every eligible file of one asset class. Returns the universe table."""
    out = CLEAN_DIR / asset
    uni_path = out / "_universe.csv"
    if uni_path.exists() and not force:
        return pd.read_csv(uni_path, parse_dates=["start", "end"])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.parquet"):
        old.unlink()
    st = {k: 0 for k in ["rows_dropped_basic", "ohlc_fixed", "negative_volume_nulled", "rows_dropped_bad_history",
                         "bad_ticks_removed", "weekend_rows_dropped", "fx_close_outside_range",
                         "redenominations", "segments_total", "segments_too_short"]}
    rows = []
    elig = eligible_raw(asset)
    for sym in elig:
        segs = clean_symbol(asset, sym, st)
        for i, seg in enumerate(segs):
            name = sym if len(segs) == 1 and i == 0 else f"{sym}@{seg.index[0]:%Y%m%d}"
            seg.to_parquet(out / f"{name}.parquet")
            rows.append({"symbol": name, "source": sym, "rows": len(seg),
                         "start": seg.index[0], "end": seg.index[-1]})
    uni = pd.DataFrame(rows)
    uni.to_csv(uni_path, index=False)
    stats = dict(st, raw_files=len(raw_symbols(asset)), eligible_files=len(elig), kept_segments=len(uni))
    pd.Series(stats).to_csv(out / "_clean_stats.csv", header=["value"])
    return uni


def clean_stats(asset: str) -> pd.Series:
    return pd.read_csv(CLEAN_DIR / asset / "_clean_stats.csv", index_col=0)["value"]


def universe(asset: str) -> pd.DataFrame:
    return build_clean_cache(asset)


def load(asset: str, sym: str) -> pd.DataFrame:
    return pd.read_parquet(CLEAN_DIR / asset / f"{sym}.parquet")


def close_panel(asset: str, start: Optional[str] = None) -> pd.DataFrame:
    """Dates x symbols panel of closes. Only the close column, so it stays small."""
    cols = {}
    for sym in universe(asset).symbol:
        c = load(asset, sym).close
        cols[sym] = c.loc[pd.Timestamp(start, tz="UTC"):] if start else c
    return pd.DataFrame(cols).sort_index()


# ------------------------------------------------------------------ SPY (market proxy + benchmark)
EXTRA_DIR = C.ROOT / "extra_data"


def download_spy() -> pd.DataFrame:
    """SPY daily from Yahoo, dividend adjusted (so buy-and-hold is total return)."""
    import yfinance as yf
    raw = yf.download("SPY", period="max", interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1) if "SPY" in raw.columns.get_level_values(1) else raw.droplevel(0, axis=1)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    df.columns.name = None
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EXTRA_DIR / "SPY.parquet")
    return df


def load_spy() -> pd.DataFrame:
    p = EXTRA_DIR / "SPY.parquet"
    return pd.read_parquet(p) if p.exists() else download_spy()
