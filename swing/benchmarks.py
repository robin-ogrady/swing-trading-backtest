"""
benchmarks.py  -  buy-and-hold references and the market proxies used by regime filters.

  equal-weight index: each day, the average close-to-close return of every universe member
  that has at least WARMUP_BARS of its own history that day (crypto: and passes the liquidity
  filter). Rebalanced daily. For stocks this is built from today's S&P 500 members, so it
  carries the same survivorship bias as the strategy results.
"""
import numpy as np
import pandas as pd

import config as C
import data

BENCH_DIR = C.CACHE_DIR / "bench"


def _eligible_mask(asset: str, closes: pd.DataFrame) -> pd.DataFrame:
    age = closes.notna().cumsum()
    ok = age >= C.WARMUP_BARS
    if asset == "crypto":
        dv = {}
        for sym in closes.columns:
            df = data.load(asset, sym)
            dv[sym] = (df.close * df.volume).rolling(30, min_periods=20).median()
        dv = pd.DataFrame(dv).reindex(closes.index)
        ok &= dv >= C.CRYPTO_MIN_DOLLAR_VOL
    return ok


def equal_weight_index(asset: str, force: bool = False) -> pd.Series:
    """Daily returns of the equal-weight basket. Eligibility is decided on the prior close."""
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCH_DIR / f"{asset}_ew.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)["ret"]
    closes = data.close_panel(asset)
    rets = closes.pct_change(fill_method=None)
    ok = _eligible_mask(asset, closes).shift(1, fill_value=False)
    r = rets.where(ok).mean(axis=1, skipna=True)
    n = rets.where(ok).notna().sum(axis=1)
    r = r[n > 0]
    pd.DataFrame({"ret": r, "n": n[n > 0]}).to_parquet(path)
    return r


def single_asset(asset: str, sym: str) -> pd.Series:
    return data.load(asset, sym).close.pct_change().dropna()


def perf(ret: pd.Series, ppy: int) -> dict:
    ret = ret.dropna()
    if len(ret) < 20:
        return {}
    eq = (1 + ret).cumprod()
    yrs = (ret.index[-1] - ret.index[0]).days / 365.25
    sd = ret.std()
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan,
            "sharpe": ret.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan,
            "maxdd": (eq / eq.cummax() - 1).min()}
