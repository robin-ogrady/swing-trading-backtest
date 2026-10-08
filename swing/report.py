"""
report.py  -  tables and charts.
"""
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402

# validated reference palette, first three categorical slots (safe all-pairs) + neutral ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BENCH = ["#52514e", "#8a8984"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def pct(x, d=1):
    return "   n/a" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}"


def stats_table(rows: List[dict]) -> str:
    """rows: dicts with 'name' plus metric keys. Returns a fixed-width text table."""
    cols = [("CAGR", "cagr", pct), ("Sharpe", "sharpe", num), ("Sortino", "sortino", num),
            ("MaxDD", "maxdd", pct), ("Trades", "trades", lambda v: "" if pd.isna(v) else f"{int(v)}"),
            ("/yr", "trades_per_yr", lambda v: num(v, 0)), ("Win", "win_rate", lambda v: pct(v, 0)),
            ("AvgW R", "avg_win_r", num), ("AvgL R", "avg_loss_r", num), ("Exp R", "exp_r", lambda v: num(v, 3)),
            ("PF", "profit_factor", num), ("Expo", "exposure", num), ("InMkt", "time_in_mkt", lambda v: pct(v, 0)),
            ("LoseRun", "lose_streak", lambda v: "" if pd.isna(v) else f"{int(v)}"),
            ("Cost/GP", "cost_share", lambda v: pct(v, 0))]
    w0 = max(len(r["name"]) for r in rows) + 2
    head = "".ljust(w0) + "".join(h.rjust(9) for h, _, _ in cols)
    lines = [head]
    for r in rows:
        lines.append(r["name"].ljust(w0) + "".join(f(r.get(k, np.nan)).rjust(9) for _, k, f in cols))
    return "\n".join(lines)


def side_stats(trades: pd.DataFrame) -> str:
    out = []
    for side, lab in ((1, "long"), (-1, "short")):
        t = trades[trades.side == side]
        if len(t):
            out.append(f"{lab} {len(t)} trades, win {pct((t.pnl > 0).mean(), 0)}, exp R {t.r.mean():.3f}, "
                       f"total P&L ${t.pnl.sum():,.0f}")
    return " | ".join(out)


def yearly_returns(eq: pd.Series) -> pd.Series:
    y = eq.groupby(eq.index.year).last()
    first = eq.iloc[0]
    return y / y.shift(1).fillna(first) - 1


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def equity_chart(path, title: str, curves: Dict[str, pd.Series], benches: Dict[str, pd.Series]):
    """Top: growth of $1 (log). Bottom: drawdown. Same x, separate y (no dual axis)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]}, facecolor=SURFACE)
    for ax in (a1, a2):
        _style(ax)
    for i, (name, eq) in enumerate(curves.items()):
        g = eq / eq.iloc[0]
        a1.plot(g.index, g.values, color=SERIES[i % 3], linewidth=1.6, label=name)
        a1.annotate(name, (g.index[-1], g.values[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK, va="center")
        dd = eq / eq.cummax() - 1
        a2.plot(dd.index, dd.values * 100, color=SERIES[i % 3], linewidth=1.2)
    for j, (name, r) in enumerate(benches.items()):
        g = (1 + r).cumprod()
        a1.plot(g.index, g.values, color=BENCH[j % 2], linewidth=1.3, linestyle="--", label=name)
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.set_ylabel("growth of $1 (log)", color=INK2, fontsize=9)
    a2.set_ylabel("drawdown %", color=INK2, fontsize=9)
    a1.set_title(title, color=INK, fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def layer_dots(path, df: pd.DataFrame, title: str):
    """Small multiples, one per asset: change in median Sharpe vs naive for each layer, one dot per setup."""
    assets = [a for a in C.ASSETS if a in set(df.asset)]
    layers = [l for l in dict.fromkeys(df.layer) if l != "naive"]
    setups_ = ["breakout", "pullback", "rsi2"]
    fig, axes = plt.subplots(1, len(assets), figsize=(4.2 * len(assets), 0.32 * len(layers) + 1.6),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    y = {l: i for i, l in enumerate(layers)}
    lim = max(0.5, float(np.nanmax(np.abs(df.d_sharpe.values))) * 1.1)
    for ax, asset in zip(axes, assets):
        _style(ax)
        ax.axvspan(-0.1, 0.1, color=GRID, alpha=0.6, linewidth=0)
        ax.axvline(0, color=INK2, linewidth=0.8)
        for i, s in enumerate(setups_):
            d = df[(df.asset == asset) & (df.setup == s) & (df.layer != "naive")]
            ax.scatter(d.d_sharpe, [y[l] + (i - 1) * 0.22 for l in d.layer], s=26, color=SERIES[i],
                       edgecolors=SURFACE, linewidths=0.8, label=s, zorder=3)
        ax.set_title(asset, color=INK, fontsize=10, loc="left")
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("change in median Sharpe vs naive", color=INK2, fontsize=8)
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(layers, fontsize=8, color=INK)
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def mc_hist(path, runs: dict):
    """Small multiples: Sharpe of random-entry runs, with the real config's Sharpe marked."""
    keys = list(runs)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.4 * nrow), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, keys):
        _style(ax)
        sims, real = runs[k]
        sims = sims[~np.isnan(sims)]
        ax.hist(sims, bins=40, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=0.5)
        ax.axvline(real, color=SERIES[1], linewidth=2)
        beat = (sims < real).mean() * 100
        ax.set_title(f"{k}\nreal {real:.2f}, beats {beat:.0f}% of random", color=INK, fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("Random entries, same days, same exits and sizing: Sharpe distribution (orange = real)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
