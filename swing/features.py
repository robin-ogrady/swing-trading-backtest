"""
features.py  -  per-symbol indicators, cached as one parquet per symbol.

Every column at row t uses only bars up to and including t (no lookahead). Columns that
describe "the prior N bars" (hh20, vol_avg50) exclude bar t itself.
Bump FEATURE_VERSION whenever a formula changes so the cache rebuilds.
Stored as float32 and only from 2 years before the asset's trade_start (older bars are warm-up only).
"""
import numpy as np
import pandas as pd

import config as C
import data

FEATURE_VERSION = 3
FEAT_DIR = C.CACHE_DIR / f"features_v{FEATURE_VERSION}"


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df.close.shift(1)
    return pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df.high.diff()
    dn = -df.low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = wilder(true_range(df), n)
    pdi = 100 * wilder(pdm, n) / atr
    mdi = 100 * wilder(mdm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, n), "pdi": pdi, "mdi": mdi})


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def compute(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    c, h, l = df.close, df.high, df.low
    f = df[["open", "high", "low", "close", "volume"]].copy()
    for n in (9, 20, 50):
        f[f"ema{n}"] = ema(c, n)
    for n in (10, 20, 50, 150, 200):
        f[f"sma{n}"] = c.rolling(n).mean()
    f["sma200_slope21"] = f.sma200 / f.sma200.shift(21) - 1
    f["atr14"] = wilder(true_range(df), 14)
    f["atr_pct"] = f.atr14 / c
    f = f.join(adx(df, 14))
    f["rsi2"] = rsi(c, 2)
    f["hh20"] = h.rolling(20).max().shift(1)
    f["ll20"] = l.rolling(20).min().shift(1)
    f["hi252"] = h.rolling(252, min_periods=126).max()
    f["lo252"] = l.rolling(252, min_periods=126).min()
    for n in (21, 63, 126):
        f[f"ret{n}"] = c / c.shift(n) - 1
    if asset != "forex":
        v = df.volume
        f["vol_avg50"] = v.rolling(50, min_periods=20).mean().shift(1)
        f["vol_ratio"] = v / f.vol_avg50
        f["dollar_vol30"] = (c * v).rolling(30, min_periods=20).median()
        f["vol10_ratio"] = v.rolling(10).mean().shift(1) / f.vol_avg50      # base volume dry-up
        f["vol3_ratio"] = v.rolling(3).mean().shift(1) / f.vol_avg50        # pullback volume
        # earnings-like event: big gap on a volume spike
        gap = (df.open / c.shift(1) - 1).abs()
        event = (gap > np.maximum(0.03, 2 * f.atr_pct.shift(1))) & (f.vol_ratio > 2.0)
        last = pd.Series(np.where(event, np.arange(len(f)), np.nan), index=f.index).ffill()
        f["bars_since_event"] = np.arange(len(f)) - last
    # quality components for the selectivity score (prior bars only, today excluded)
    f["range10"] = (h.rolling(10).max() - l.rolling(10).min()).shift(1)
    f["range50"] = (h.rolling(50).max() - l.rolling(50).min()).shift(1)
    f["tightness"] = f.range10 / f.range50                                     # lower = tighter base
    f["higher_low"] = l.rolling(10).min().shift(1) / l.rolling(10).min().shift(11) - 1
    f["sma20_slope5"] = f.sma20 / f.sma20.shift(5) - 1
    f["ema50_slope10"] = f.ema50 / f.ema50.shift(10) - 1
    body = (c - df.open)
    red = (-body).clip(lower=0).where(body < 0)
    green = body.clip(lower=0).where(body > 0)
    f["body_ratio10"] = red.rolling(10, min_periods=1).mean() / green.rolling(10, min_periods=1).mean()
    f["trend200"] = c / f.sma200 - 1
    f["rs_raw"] = (f.ret21 + f.ret63 + f.ret126) / 3
    lr = np.log(c).diff()
    f["vol20"] = lr.rolling(20).std() * np.sqrt(C.PERIODS_PER_YEAR[asset])
    f["vol20_pct252"] = f.vol20.rolling(252, min_periods=126).rank(pct=True)
    f["age"] = np.arange(1, len(f) + 1)
    return f


def build_feature_cache(asset: str, force: bool = False) -> None:
    out = FEAT_DIR / asset
    done = out / "_done"
    if done.exists() and not force:
        return
    out.mkdir(parents=True, exist_ok=True)
    keep_from = pd.Timestamp(C.SPLITS[asset]["trade_start"], tz="UTC") - pd.DateOffset(years=2)
    for sym in data.universe(asset).symbol:
        f = compute(data.load(asset, sym), asset)
        f = f.loc[f.index >= keep_from]
        if len(f):
            f.astype("float32").to_parquet(out / f"{sym}.parquet", compression="zstd")
    done.write_text("ok")


def load(asset: str, sym: str) -> pd.DataFrame:
    return pd.read_parquet(FEAT_DIR / asset / f"{sym}.parquet")
