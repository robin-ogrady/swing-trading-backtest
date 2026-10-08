"""
metrics.py  -  portfolio and trade statistics.
"""
import numpy as np
import pandas as pd

import config as C


def longest_losing_streak(r: pd.Series) -> int:
    best = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


def equity_stats(eq: pd.Series, ppy: int) -> dict:
    ret = eq.pct_change().dropna()
    if len(ret) < 20 or eq.iloc[0] <= 0:
        return {}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    end = eq.iloc[-1] / eq.iloc[0]
    sd = ret.std()
    down = np.sqrt((np.minimum(ret, 0) ** 2).mean())
    dd = eq / eq.cummax() - 1
    return {
        "cagr": end ** (1 / yrs) - 1 if end > 0 and yrs > 0 else -1.0,
        "vol": sd * np.sqrt(ppy),
        "sharpe": ret.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan,
        "sortino": ret.mean() / down * np.sqrt(ppy) if down > 0 else np.nan,
        "maxdd": dd.min(),
    }


def trade_stats(tr: pd.DataFrame, years: float) -> dict:
    if tr.empty:
        return {"trades": 0}
    r = tr.r
    win = tr.pnl > 0
    gp = tr.pnl[win].sum()
    gl = -tr.pnl[~win].sum()
    ret_pct = tr.side * (tr.exit / tr.entry - 1)
    return {
        "trades": len(tr),
        "trades_per_yr": len(tr) / years if years > 0 else np.nan,
        "win_rate": win.mean(),
        "avg_win_r": r[win].mean() if win.any() else np.nan,
        "avg_loss_r": r[~win].mean() if (~win).any() else np.nan,
        "avg_win_pct": ret_pct[win].mean() if win.any() else np.nan,
        "avg_loss_pct": ret_pct[~win].mean() if (~win).any() else np.nan,
        "exp_r": r.mean(),
        "profit_factor": gp / gl if gl > 0 else np.nan,
        "avg_bars": tr.bars.mean(),
        "lose_streak": longest_losing_streak(tr.sort_values("exit_date").pnl),
        "cost_share": tr.costs.sum() / max(gp, 1e-9),
    }


def summarize(res) -> dict:
    ppy = C.PERIODS_PER_YEAR[res.asset]
    eq = res.equity
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    out = equity_stats(eq, ppy)
    out.update(trade_stats(res.trades, yrs))
    out["exposure"] = res.gross.mean()
    out["time_in_mkt"] = (res.npos > 0).mean()
    out["avg_positions"] = res.npos.mean()
    return out
