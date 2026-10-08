"""
engine.py  -  portfolio simulation with one shared equity pot.

Day k (union of every symbol's dates), for each pending entry / open position whose symbol
has a bar today:
  1. pending entry fills at today's open; cancelled if the open is already through the stop.
     Size = risk cash / |open - stop|, capped by MAX_POSITION_PCT, GROSS_CAP and heat cap.
  2. exits decided on yesterday's close (time stop, native exit, MA trail) fill at the open.
  3. stop: open through the stop -> filled at the open (gap), else low/high touches -> at the stop.
  4. partial / target: filled at the level, or at the open if it gapped past it.
     A bar that touches both stop and target counts as a stop (worst case).
  5. on the close: trail updates, time stop, native exit, "partial after N bars" are decided
     for the next open. A symbol whose history ends (delisting, relabel) is closed at its last close.
Then equity is marked at the close, and new signals from today's close are queued for the next
open. Signals only exist on bars where the symbol is tradable and inside [start, end].
Nothing after `end` is ever read; runs past the validation end need allow_final=True.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C


class SymArrays:
    """Numpy arrays for one symbol, trimmed to the simulation window."""
    __slots__ = ("name", "u", "o", "h", "l", "c", "atr", "matrail", "lx", "sx",
                 "n", "ends_early", "ok")

    def __init__(self, name, u, f, sig, ends_early, ok=None):
        self.name = name
        self.u = u
        self.o = f.open.values.astype(float)
        self.h = f.high.values.astype(float)
        self.l = f.low.values.astype(float)
        self.c = f.close.values.astype(float)
        self.atr = f.atr14.values.astype(float)
        self.matrail = f.matrail.values.astype(float) if "matrail" in f else np.full(len(u), np.nan)
        self.lx = sig.long_exit.values
        self.sx = sig.short_exit.values
        self.n = len(u)
        self.ends_early = ends_early
        self.ok = np.ones(len(u), dtype=bool) if ok is None else np.asarray(ok, dtype=bool)


class Pos:
    __slots__ = ("s", "side", "state", "row", "stop", "init_stop", "risk_cash", "qty", "qty0",
                 "entry", "entry_date", "signal_date", "bars", "target", "partial_px",
                 "partial_done", "exit_pending", "partial_pending", "pnl", "costs", "best_close",
                 "atr0", "last_close", "init_risk")

    def __init__(self, s, side, row, stop, risk_cash, signal_date, atr0):
        self.s, self.side, self.row, self.stop, self.init_stop = s, side, row, stop, stop
        self.state = "pending"
        self.risk_cash, self.signal_date, self.atr0 = risk_cash, signal_date, atr0
        self.qty = self.qty0 = 0.0
        self.entry = self.target = self.partial_px = np.nan
        self.entry_date = None
        self.bars = 0
        self.partial_done = False
        self.exit_pending = None
        self.partial_pending = False
        self.pnl = self.costs = 0.0
        self.best_close = np.nan
        self.last_close = np.nan
        self.init_risk = 0.0


class Result:
    def __init__(self, equity, gross, npos, trades, asset, start, end):
        self.equity, self.gross, self.npos, self.trades = equity, gross, npos, trades
        self.asset, self.start, self.end = asset, start, end


def simulate(asset: str, syms: List[SymArrays], events: Dict[int, list], dates: pd.DatetimeIndex,
             cfg: dict, seed: int = 0, cost_mult: float = 1.0) -> Result:
    ex = cfg["exits"]
    sz = cfg["sizing"]
    rank = cfg.get("rank", "random")
    cost_rate = (C.COSTS[asset]["fee_bps"] + C.COSTS[asset]["slip_bps"]) / 1e4 * cost_mult
    borrow = C.SHORT_COST_BPS_PER_DAY[asset] / 1e4 * cost_mult
    max_pos_pct = C.MAX_POSITION_PCT[asset]
    gross_cap = C.GROSS_CAP[asset]
    heat_cap = sz.get("heat_cap")
    risk_pct = sz["risk_pct"] / 100.0
    max_n = int(sz["max_positions"])
    atr_mode = sz.get("mode", "risk") == "atr"
    atr_mult = float(sz.get("atr_mult", 2.0))
    dd_cfg = sz.get("dd_reduce")
    target_r = ex.get("target_r")
    max_hold = ex.get("max_hold")
    native = ex.get("native_exit", False)
    part = ex.get("partial")            # {"frac": .33, "at_r": 2.0} or {"frac": .5, "after_bars": 3}
    be_after_partial = ex.get("breakeven", True)
    trail_ma = ex.get("trail_ma")       # 10 or 20: exit on first close through that SMA
    trail_atr = ex.get("trail_atr")     # chandelier: best close -/+ k * ATR
    time_stop = ex.get("time_stop")     # {"bars": 10, "min_r": 0.5}: dead trade exit
    trail_wait = bool(part and part.get("trail_after_partial"))   # MA trail only after the partial

    spp = cfg["setup_params"]
    stop_lo = spp.get("stop_atr_min", spp.get("stop_atr", 0.0))
    stop_hi = spp.get("stop_atr_max", spp.get("stop_atr", np.inf))
    rng = np.random.default_rng(seed)
    cash = C.START_EQUITY
    peak = cash
    positions = []  # type: List[Pos]
    trades = []
    nd = len(dates)
    eq_out = np.empty(nd)
    gross_out = np.empty(nd)
    npos_out = np.empty(nd, dtype=int)
    equity = cash

    def close_out(p: Pos, qty: float, px: float, reason: Optional[str], date):
        nonlocal cash
        pnl = p.side * qty * (px - p.entry)
        cost = abs(qty * px) * cost_rate
        cash += pnl - cost
        p.pnl += pnl
        p.costs += cost
        p.qty -= qty
        if reason is not None:
            sa = syms[p.s]
            trades.append((sa.name, p.side, p.signal_date, p.entry_date, p.entry, p.init_stop,
                           p.qty0, date, px, reason, p.pnl - p.costs, p.costs,
                           (p.pnl - p.costs) / p.init_risk if p.init_risk > 0 else np.nan, p.bars))

    for k in range(nd):
        date = dates[k]
        gross_prev = sum(abs(p.qty * p.last_close) for p in positions if p.state == "open")
        heat_prev = sum(max(0.0, p.side * (p.entry - p.stop)) * p.qty
                        for p in positions if p.state == "open")
        still = []
        for p in positions:
            sa = syms[p.s]
            r = p.row + 1
            if r >= sa.n or sa.u[r] != k:
                # an order that can no longer fill (history ended, or 5 days without a bar) is dropped
                if not (p.state == "pending" and (r >= sa.n or k - sa.u[r - 1] > 5)):
                    still.append(p)
                continue
            p.row = r
            o, h, l, c = sa.o[r], sa.h[r], sa.l[r], sa.c[r]
            side = p.side
            just_filled = False
            if p.state == "pending":
                if side * (o - p.stop) <= 0:
                    continue                                   # gapped through the stop: cancel
                # keep the stop distance inside the setup's ATR band, measured from the real entry
                dd = min(max(abs(o - p.stop), stop_lo * p.atr0), stop_hi * p.atr0)
                p.stop = p.init_stop = o - side * dd
                dist = atr_mult * p.atr0 if atr_mode else abs(o - p.stop)
                qty = p.risk_cash / dist
                qty = min(qty, max_pos_pct * equity / o, max(0.0, gross_cap * equity - gross_prev) / o)
                if heat_cap is not None:
                    room = heat_cap * equity - heat_prev
                    qty = min(qty, max(0.0, room) / abs(o - p.stop))
                if qty * o < 0.001 * equity:
                    continue
                p.qty = p.qty0 = qty
                p.entry = o
                p.entry_date = date
                p.state = "open"
                p.init_risk = qty * abs(o - p.stop)
                d = abs(o - p.stop)
                if target_r:
                    p.target = o + side * target_r * d
                if part and part.get("at_r"):
                    p.partial_px = o + side * part["at_r"] * d
                cost = qty * o * cost_rate
                cash -= cost
                p.costs += cost
                p.best_close = o
                gross_prev += qty * o
                heat_prev += p.init_risk
                just_filled = True
            else:
                if p.exit_pending:
                    close_out(p, p.qty, o, p.exit_pending, date)
                    continue
                if p.partial_pending:
                    close_out(p, p.qty * part["frac"], o, None, date)
                    p.partial_pending = False
                    p.partial_done = True
                    if be_after_partial:
                        p.stop = p.entry if side * (p.entry - p.stop) > 0 else p.stop
                if side * (o - p.stop) <= 0:
                    close_out(p, p.qty, o, "stop_gap", date)
                    continue
            # intrabar stop, then partial, then target
            lo_touch = l if side > 0 else h
            if side * (lo_touch - p.stop) <= 0:
                close_out(p, p.qty, p.stop, "stop", date)
                continue
            hi_touch = h if side > 0 else l
            if part and not p.partial_done and not np.isnan(p.partial_px) and side * (hi_touch - p.partial_px) >= 0:
                px = o if (not just_filled and side * (o - p.partial_px) > 0) else p.partial_px
                close_out(p, p.qty * part["frac"], px, None, date)
                p.partial_done = True
                if be_after_partial:
                    p.stop = p.entry
            if target_r and side * (hi_touch - p.target) >= 0:
                px = o if (not just_filled and side * (o - p.target) > 0) else p.target
                close_out(p, p.qty, px, "target", date)
                continue
            # close-of-bar decisions for the next open
            p.bars += 1
            p.last_close = c
            if side * (c - p.best_close) > 0:
                p.best_close = c
            if side < 0 and borrow:
                b = abs(p.qty * c) * borrow
                cash -= b
                p.costs += b
            if sa.ends_early and r == sa.n - 1:
                close_out(p, p.qty, c, "segment_end", date)
                continue
            if trail_atr and not np.isnan(sa.atr[r]):
                ts = p.best_close - side * trail_atr * sa.atr[r]
                if side * (ts - p.stop) > 0:
                    p.stop = ts
            if trail_ma:
                ma = sa.matrail[r]
                if not np.isnan(ma) and side * (c - ma) < 0 and (p.partial_done or not trail_wait):
                    p.exit_pending = "trail_ma"
            if native and (sa.lx[r] if side > 0 else sa.sx[r]):
                p.exit_pending = "native"
            if time_stop and p.bars >= time_stop["bars"] and not p.partial_done:
                if side * (c - p.entry) < time_stop["min_r"] * abs(p.entry - p.init_stop):
                    p.exit_pending = "time_stop"
            if max_hold and p.bars >= max_hold:
                p.exit_pending = p.exit_pending or "max_hold"
            if part and part.get("after_bars") and not p.partial_done and p.bars >= part["after_bars"] \
                    and side * (c - p.entry) > 0:
                p.partial_pending = True
            still.append(p)
        positions = still

        # mark to market
        unreal = 0.0
        gross = 0.0
        for p in positions:
            if p.state == "open":
                unreal += p.side * p.qty * (p.last_close - p.entry)
                gross += abs(p.qty * p.last_close)
        equity = cash + unreal
        peak = max(peak, equity)
        eq_out[k] = equity
        gross_out[k] = gross / equity if equity > 0 else np.nan
        npos_out[k] = sum(1 for p in positions if p.state == "open")
        if equity <= 0:
            eq_out[k:] = equity
            gross_out[k:] = np.nan
            npos_out[k:] = 0
            break

        # new entries from today's close
        ev = events.get(k)
        if ev and k < nd - 1:
            held = {p.s for p in positions}
            cand = [e for e in ev if e[0] not in held]
            if cand:
                if rank == "score":
                    sc = np.nan_to_num(np.array([e[4] for e in cand], dtype=float), nan=-1e9)
                    order = np.lexsort((rng.random(len(cand)), -sc))
                else:
                    order = rng.permutation(len(cand))
                rp = risk_pct
                if dd_cfg and equity < peak * (1 - dd_cfg["dd"]):
                    rp *= dd_cfg["factor"]
                n_now = len(positions)
                pend_gross = 0.0
                pend_risk = 0.0
                for i in order:
                    if n_now >= max_n:
                        break
                    s, row, side, stop, score = cand[i]
                    sa = syms[s]
                    c = sa.c[row]
                    d = abs(c - stop)
                    if not d > 0:
                        continue
                    risk_cash = equity * rp
                    est = min(risk_cash / (atr_mult * sa.atr[row] if atr_mode else d) * c, max_pos_pct * equity)
                    if gross + pend_gross + est > gross_cap * equity * 1.0001:
                        continue
                    if heat_cap is not None and heat_prev + pend_risk + est / c * d > heat_cap * equity:
                        continue
                    positions.append(Pos(s, side, row, stop, risk_cash, date, sa.atr[row]))
                    n_now += 1
                    pend_gross += est
                    pend_risk += est / c * d

    # close anything still open on the last day
    last = dates[min(k, nd - 1)]
    for p in positions:
        if p.state == "open":
            close_out(p, p.qty, p.last_close, "end", last)
    eq = pd.Series(eq_out[:k + 1], index=dates[:k + 1])
    if len(trades):
        eq.iloc[-1] = cash
    cols = ["symbol", "side", "signal_date", "entry_date", "entry", "stop", "qty", "exit_date", "exit",
            "reason", "pnl", "costs", "r", "bars"]
    return Result(eq, pd.Series(gross_out[:k + 1], index=dates[:k + 1]),
                  pd.Series(npos_out[:k + 1], index=dates[:k + 1]),
                  pd.DataFrame(trades, columns=cols), asset, dates[0], dates[-1])
