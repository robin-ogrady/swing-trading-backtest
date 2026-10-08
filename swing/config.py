"""
config.py  -  every tunable number for the swing project lives here.

Nothing in here should be changed after looking at validation or final-period results
without logging it as a new variant (see VARIANT_LOG in results/).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"            # ~/CODE/data (read only)
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

ASSETS = ["stocks", "forex", "crypto"]
OHLC = ["open", "high", "low", "close"]

# ------------------------------------------------------------------ universe
MIN_ROWS = 500              # per cleaned segment, not per raw file
WARMUP_BARS = 252           # a symbol can only trade once it has this many bars of its own history
PERIODS_PER_YEAR = {"stocks": 252, "forex": 260, "crypto": 365}

# a gap longer than this (calendar days) splits a series into separate segments.
# Positions are force-closed at the end of a segment (delisting, suspension, relabel).
MAX_GAP_DAYS = {"stocks": 14, "forex": 30, "crypto": 7}

# stocks: old Yahoo history has stretches of flat (H == L) or zero-volume bars.
# A symbol starts after the last 20-bar window holding at least this many such bars.
STOCK_BAD_BARS_IN_20 = 3

# bad-tick filter (forex only): a bar is removed when its return AND the next
# return are both huge (z-score vs trailing robust vol AND above an absolute floor)
# and they cancel out. Real moves that stick are kept.
SPIKE_Z = 8.0
SPIKE_FLOOR = {"forex": 0.02}

# crypto: a one-day price change bigger than this factor with an inverse jump in coin
# volume is treated as a redenomination / token swap and splits the series.
REDENOM_FACTOR = 20.0

# forex: which pairs to trade. "g10" = the 28 pairs of USD EUR JPY GBP CHF AUD CAD NZD.
# "all" adds the exotic crosses (worse data, much wider real spreads).
FOREX_UNIVERSE = "g10"
G10_CCY = ["USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"]

# crypto: not tradable risk assets (stablecoins, fiat, wrapped / staked duplicates, gold token)
CRYPTO_EXCLUDE = ["AEUR", "AUD", "BUSD", "EUR", "EURI", "FDUSD", "GBP", "PAX", "PAXG", "TUSD",
                  "USDC", "USDP", "USDS", "XUSD", "WBTC", "WBETH", "BNSOL"]
# point-in-time liquidity filter: 30-day median of close * volume (USDT)
CRYPTO_MIN_DOLLAR_VOL = 1_000_000

# ------------------------------------------------------------------ periods
# trade_start: first date a trade can open (earlier data is only indicator warm-up).
# dev: tune here.  val: check here.  final: after val_end, run ONCE (run.py final).
SPLITS = {
    "stocks": {"trade_start": "1995-01-01", "dev_end": "2019-12-31", "val_end": "2022-12-31"},
    "forex":  {"trade_start": "2005-01-01", "dev_end": "2019-12-31", "val_end": "2022-12-31"},
    "crypto": {"trade_start": "2018-06-01", "dev_end": "2020-12-31", "val_end": "2022-12-31"},
}

# ------------------------------------------------------------------ costs
# per side, in basis points of traded notional
COSTS = {
    "stocks": {"fee_bps": 5.0, "slip_bps": 5.0},
    "forex":  {"fee_bps": 0.0, "slip_bps": 2.0},    # half spread + slippage on G10
    "crypto": {"fee_bps": 10.0, "slip_bps": 10.0},
}
# daily cost of holding a short, bps of notional (borrow / funding)
SHORT_COST_BPS_PER_DAY = {"stocks": 1.0, "forex": 0.0, "crypto": 1.0}

# ------------------------------------------------------------------ strategy
START_EQUITY = 100_000.0
LONG_ONLY = {"stocks": True, "forex": False, "crypto": False}   # crypto/forex get mirrored shorts
# hard portfolio limits (not tuned): biggest single position and total gross exposure, x equity
MAX_POSITION_PCT = {"stocks": 0.20, "forex": 1.50, "crypto": 0.20}
GROSS_CAP = {"stocks": 1.0, "forex": 6.0, "crypto": 1.0}

# setup rules. Numbers are the textbook values, not tuned.
SETUP_PARAMS = {
    # close above the prior 20-bar high. stop = breakout-day low, distance clipped to [min, max] ATR
    "breakout": {"base_len": 20, "stop_atr_min": 0.5, "stop_atr_max": 1.5},
    # 9 > 20 > 50 EMA, ADX > 25, low touched the 20 EMA in the last 5 bars, close back above the 9 EMA.
    # stop = lowest low of the last 5 bars, clipped to [min, max] ATR
    "pullback": {"adx_min": 25.0, "lookback": 5, "stop_atr_min": 0.5, "stop_atr_max": 2.0},
    # Connors: close above 200 SMA and RSI(2) < 10. exit RSI(2) > 70 or 10 bars.
    # Connors uses no stop; a 3 ATR disaster stop is used so risk-based sizing works.
    "rsi2": {"ma": 200, "entry": 10.0, "exit": 70.0, "stop_atr": 3.0},
}

# naive baseline = the setup rule + dumb exits + fixed sizing, no filter layers.
# signals beyond the free slots are picked at random (no ranking).
NAIVE_EXITS = {
    "breakout": {"target_r": 2.0, "max_hold": 20, "native_exit": False},
    "pullback": {"target_r": 2.0, "max_hold": 20, "native_exit": False},
    "rsi2":     {"target_r": None, "max_hold": 10, "native_exit": True},
}
NAIVE_SIZING = {"risk_pct": 1.0, "max_positions": 10, "heat_cap": None}

# ------------------------------------------------------------------ filter layers (phase 3)
# one textbook value each. Phase 4 moves them +/-30% to test sensitivity.
LAYERS = {
    "regime": {"breadth_min": 50.0},
    "vol": {"max_pct": 0.90},
    "chop": {"adx_min": 20.0},
    "rs": {"min_pct": 80.0},
    "volume": {"breakout_min": 1.4, "pullback_max": 1.0},
    "event": {"period": 63, "window_before": 11, "window_after": 3, "max_bars": 260},
    "score": {"top_pct": 20.0, "min_history": 100},
}

# sizing variants tested alone (values from the brief)
SIZING_VARIANTS = {
    "risk 0.25%": {"risk_pct": 0.25},
    "risk 0.5%": {"risk_pct": 0.5},
    "risk 2%": {"risk_pct": 2.0},
    "ATR sizing": {"mode": "atr", "atr_mult": 2.0},     # size to 2 ATR instead of the stop distance
    "heat cap 4%": {"heat_cap": 0.04},
    "heat cap 6%": {"heat_cap": 0.06},
    "heat cap 8%": {"heat_cap": 0.08},
    "max 5 positions": {"max_positions": 5},
    "halve risk in 10% DD": {"dd_reduce": {"dd": 0.10, "factor": 0.5}},
}

# exit variants tested alone (trend setups only; rsi2 keeps its Connors exits)
EXIT_VARIANTS = {
    # Qullamaggie: sell 1/3 after 3 days if in profit, stop to breakeven, trail the rest under the 10 SMA
    "partial 3d + BE + 10MA": {"target_r": None, "max_hold": 100, "native_exit": False, "breakeven": True,
                               "partial": {"frac": 0.33, "after_bars": 3, "trail_after_partial": True},
                               "trail_ma": 10},
    # sell half at 2R, stop to breakeven, trail the rest under the 20 SMA
    "partial 2R + BE + 20MA": {"target_r": None, "max_hold": 100, "native_exit": False, "breakeven": True,
                               "partial": {"frac": 0.5, "at_r": 2.0, "trail_after_partial": True},
                               "trail_ma": 20},
    "ATR trail 2.5": {"target_r": None, "max_hold": 100, "native_exit": False, "trail_atr": 2.5},
    # naive exits plus: out after 5 bars unless the close is at least +0.5R
    "time stop 5d": {"target_r": 2.0, "max_hold": 20, "native_exit": False,
                     "time_stop": {"bars": 5, "min_r": 0.5}},
}
