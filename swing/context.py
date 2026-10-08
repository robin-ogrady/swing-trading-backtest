"""
context.py  -  cross-sectional panels and market regime series, cached per asset class.

Panels (dates x symbols, float32): each quality component turned into a percentile (0-100)
among the symbols that are tradable that day. Breadth: share of tradable symbols above their
50-day SMA. Market proxy: SPY for stocks, BTC for crypto (forex has none: per-pair only).
Everything at date t uses data up to t only.
"""
from typing import Dict

import numpy as np
import pandas as pd

import config as C
import data
import features
import strategy

PANEL_COLS = ["rs_raw", "tightness", "vol10_ratio", "vol3_ratio", "sma20_slope5", "adx",
              "body_ratio10", "ema50_slope10", "trend200"]
PANEL_DIR = C.CACHE_DIR / f"panels_v{features.FEATURE_VERSION}"
_MEM = {}  # type: Dict[str, dict]


def _market(asset: str) -> pd.DataFrame:
    if asset == "stocks":
        spy = data.load_spy()
        m = features.compute(spy, "stocks")
    elif asset == "crypto":
        m = features.load("crypto", "BTCUSDT")
    else:
        return pd.DataFrame()
    return pd.DataFrame({"mkt_up": m.close > m.sma200, "mkt_vol_pct": m.vol20_pct252})


def build(asset: str, force: bool = False) -> None:
    out = PANEL_DIR / asset
    if (out / "_done").exists() and not force:
        return
    out.mkdir(parents=True, exist_ok=True)
    cols = {k: {} for k in PANEL_COLS}
    above50 = {}
    for sym in data.universe(asset).symbol:
        f = features.load(asset, sym)
        ok = strategy.tradable(asset, f)
        for k in PANEL_COLS:
            if k in f:
                cols[k][sym] = f[k].where(ok)
        above50[sym] = (f.close > f.sma50).astype(float).where(ok & f.sma50.notna())
    for k, d in cols.items():
        if d:
            pct = pd.DataFrame(d).rank(axis=1, pct=True) * 100
            pct.astype("float32").to_parquet(out / f"{k}.parquet", compression="zstd")
    a50 = pd.DataFrame(above50)
    breadth = a50.sum(axis=1) / a50.notna().sum(axis=1) * 100
    ctx = pd.DataFrame({"breadth50": breadth, "n_tradable": a50.notna().sum(axis=1)})
    mk = _market(asset)
    if len(mk):
        ctx = ctx.join(mk, how="left")
    ctx.to_parquet(out / "_market.parquet")
    (out / "_done").write_text("ok")


def load(asset: str) -> dict:
    """{'market': DataFrame by date, '<component>': percentile panel}. Kept in memory once loaded."""
    if asset not in _MEM:
        build(asset)
        out = PANEL_DIR / asset
        d = {"market": pd.read_parquet(out / "_market.parquet")}
        for k in PANEL_COLS:
            p = out / f"{k}.parquet"
            if p.exists():
                d[k] = pd.read_parquet(p)
        _MEM[asset] = d
    return _MEM[asset]
