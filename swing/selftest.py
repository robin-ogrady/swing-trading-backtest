"""
selftest.py  -  checks the engine on a hand-made price series. Run: python3 selftest.py
"""
import numpy as np
import pandas as pd

import config as C
import engine


def make(o, h, l, c, long_sig, stop):
    idx = pd.date_range("2020-01-01", periods=len(o), freq="D", tz="UTC")
    f = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "atr14": 1.0}, index=idx)
    sig = pd.DataFrame({"long_exit": False, "short_exit": False}, index=idx)
    sa = engine.SymArrays("TEST", np.arange(len(o)), f, sig, False)
    events = {i: [(0, i, 1, stop, np.nan)] for i, s in enumerate(long_sig) if s}
    return [sa], events, idx


def cfg(target_r=2.0, max_hold=20):
    return {"setup_params": {"stop_atr_min": 0.0, "stop_atr_max": 100.0},
            "exits": {"target_r": target_r, "max_hold": max_hold},
            "sizing": {"risk_pct": 1.0, "max_positions": 1, "heat_cap": None}, "rank": "random"}


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    assert cond


C.COSTS["stocks"] = {"fee_bps": 0.0, "slip_bps": 0.0}
C.MAX_POSITION_PCT["stocks"] = 100.0
C.GROSS_CAP["stocks"] = 100.0

# signal on bar 0 close, fill bar 1 open = 100, stop 95 -> risk 5, target 110 hit on bar 3
syms, ev, idx = make(o=[99, 100, 101, 104, 108], h=[100, 102, 103, 111, 109], l=[98, 99, 100, 103, 107],
                     c=[99, 101, 102, 108, 108], long_sig=[1, 0, 0, 0, 0], stop=95.0)
r = engine.simulate("stocks", syms, ev, idx, cfg())
t = r.trades.iloc[0]
check("fills at next open", t.entry == 100 and t.entry_date == idx[1])
check("target hit at 2R = 110", t.exit == 110 and t.reason == "target")
check("pnl = 1% risk * 2R", abs(t.pnl - 2000) < 1e-6 and abs(t.r - 2.0) < 1e-9)
check("equity matches trades", abs(r.equity.iloc[-1] - C.START_EQUITY - r.trades.pnl.sum()) < 1e-6)

# gap through the stop: open 90 below stop 95 -> filled at 90, not 95
syms, ev, idx = make(o=[99, 100, 100, 90, 90], h=[100, 101, 101, 91, 91], l=[98, 99, 99, 88, 89],
                     c=[99, 100, 100, 90, 90], long_sig=[1, 0, 0, 0, 0], stop=95.0)
t = engine.simulate("stocks", syms, ev, idx, cfg()).trades.iloc[0]
check("gap through stop fills at the open", t.exit == 90 and t.reason == "stop_gap" and abs(t.r + 2.0) < 1e-9)

# stop and target in the same bar -> stop (worst case)
syms, ev, idx = make(o=[99, 100, 100, 100], h=[100, 101, 112, 101], l=[98, 99, 94, 99],
                     c=[99, 100, 100, 100], long_sig=[1, 0, 0, 0], stop=95.0)
t = engine.simulate("stocks", syms, ev, idx, cfg()).trades.iloc[0]
check("stop and target same bar -> stop", t.exit == 95 and t.reason == "stop")

# entry bar opens below the stop -> order cancelled
syms, ev, idx = make(o=[99, 94, 100], h=[100, 95, 101], l=[98, 93, 99], c=[99, 94, 100],
                     long_sig=[1, 0, 0], stop=95.0)
check("gap below stop on entry cancels", engine.simulate("stocks", syms, ev, idx, cfg()).trades.empty)

# max hold: exit decided on close of bar N, filled next open
syms, ev, idx = make(o=[99, 100, 100, 100, 100, 103], h=[100, 101, 101, 101, 101, 104],
                     l=[98, 99, 99, 99, 99, 102], c=[99, 100, 100, 100, 100, 103],
                     long_sig=[1, 0, 0, 0, 0, 0], stop=95.0)
t = engine.simulate("stocks", syms, ev, idx, cfg(target_r=None, max_hold=3)).trades.iloc[0]
check("max hold exits at next open", t.reason == "max_hold" and t.exit_date == idx[4] and t.bars == 3)

# costs: 10 bps per side
C.COSTS["stocks"] = {"fee_bps": 5.0, "slip_bps": 5.0}
syms, ev, idx = make(o=[99, 100, 101, 104, 108], h=[100, 102, 103, 111, 109], l=[98, 99, 100, 103, 107],
                     c=[99, 101, 102, 108, 108], long_sig=[1, 0, 0, 0, 0], stop=95.0)
t = engine.simulate("stocks", syms, ev, idx, cfg()).trades.iloc[0]
check("costs charged both sides", abs(t.costs - 200 * (100 + 110) * 0.001) < 1e-6)
print("all engine checks passed")
