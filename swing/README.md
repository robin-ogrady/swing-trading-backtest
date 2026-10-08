# swing

**Findings: see `results/SUMMARY.md`.** Short version: no rule set beat buy-and-hold out of sample.

Can rule-based price action swing trading, plus "human" filter layers, beat buy-and-hold
out of sample after costs? Stocks, forex, crypto. Daily bars from `data/` next to the `swing/` folder (built by `download_market_data.py`).

## Run

```
python3 swing/run.py data            # phase 1: clean + cache + universe summary
python3 swing/run.py data --force    # rebuild cache from raw files
python3 swing/run.py baseline        # phase 2: naive baselines, dev period only
python3 swing/run.py layers          # phase 3: each layer alone vs naive, dev only (~4 min)
python3 swing/run.py ablation        # phase 4a: ablation + picked configs, dev and validation (~15 min)
python3 swing/run.py robust          # phase 4b: sensitivity, Monte Carlo, bootstrap, regimes, haircut (~20 min)
python3 swing/run.py final           # phase 5: final period, ONCE (writes results/FINAL_RUN.lock)
python3 swing/selftest.py            # engine checks on a hand-made price series
```

SPY (for the stock benchmark and regime filter) is pulled once with yfinance into `extra_data/SPY.parquet`.

Needs: pandas, pyarrow, numpy, scipy, matplotlib (`python3 -m pip install <pkg>`). Python 3.9.

## Layout

| file | what |
|---|---|
| `config.py` | every parameter: universe rules, periods, costs |
| `data.py` | load raw parquet, clean, split into segments, cache per symbol |
| `benchmarks.py` | equal-weight buy-and-hold index per asset class, BTC |
| `features.py` | per-symbol indicators (no lookahead), cached |
| `setups.py` | entry rules: breakout, pullback, rsi2 |
| `context.py` | cross-sectional percentile panels (RS, quality), breadth, SPY/BTC regime, cached |
| `filters.py` | the "human" layers: regime, vol, chop, rs, volume, event, score |
| `experiments.py` | named layer variants, multi-seed runner |
| `phase4.py` | all-layers / picked configs, sensitivity, random-entry Monte Carlo, bootstrap, regimes, haircut |
| `engine.py` | portfolio simulation: shared equity, next-open fills, gap-through-stop fills, limits |
| `strategy.py` | config -> signals -> engine; period windows; final-period lock; variant log |
| `metrics.py`, `report.py` | stats, tables, charts |
| `run.py` | entry point |
| `cache/` | cleaned bars (one parquet per symbol), benchmark series. Safe to delete, rebuilt on demand |
| `results/` | text summaries, tables, charts |

## Periods (walk-forward)

| asset | trade from | develop | validate | final (run once) |
|---|---|---|---|---|
| stocks | 1995 | to 2019 | 2020-2022 | 2023 on |
| forex | 2005 | to 2019 | 2020-2022 | 2023 on |
| crypto | mid 2018 | to 2020 | 2021-2022 | 2023 on |

## Data cleaning (see `data.py` docstring)

- Stocks: old Yahoo history has flat or zero-volume bars; each stock starts after the last such cluster.
- Forex: G10 pairs only (28). Since about 2009 Yahoo's forex "close" is a stale snapshot near the
  open, so close is replaced by the next bar's open. Yahoo's high/low window also does not cover the
  full 24h (next open lands outside the prior range ~25% of the time), so on weekdays the range is
  widened to include the close; weekend gaps stay real gaps. Bad ticks (misdated 2008 rows etc.)
  removed. Volume is always 0: no volume logic.
- Crypto: stablecoins, fiat, wrapped duplicates removed. Series split at redenominations and at
  gaps over 7 days (each piece is its own symbol; open positions are closed at the piece's end).

## Backtest rules

- Signals on the close, fills at the next open. Open through the stop: filled at the open.
  Stop and target in the same bar: counted as a stop. Entry that opens through its stop: cancelled.
- Stop distance is kept inside the setup's ATR band measured from the real fill.
- Costs per side: stocks 5 + 5 bps, forex 2 bps, crypto 10 + 10 bps. Shorts pay 1 bp/day (stocks, crypto).
- Limits: max position 20% of equity (forex 150%), gross exposure 1x (forex 6x), max 10 positions.
- The final period is locked in code (`strategy.prepare` refuses it without `allow_final`).
- Every run is logged to `results/variant_log.csv`; distinct configs = variants tried.

## Known biases

- Stocks are today's S&P 500: survivorship bias inflates every long stock result.
- `trim_small_files.py` already deleted crypto pairs under 500 rows, so short-lived dead coins are gone.
- The 500-row minimum is itself mild lookahead (it knows a symbol lived 500 days).
