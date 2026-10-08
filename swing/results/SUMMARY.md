# Swing rules vs buy-and-hold: summary

Daily bars, stocks (today's S&P 500), G10 forex, Binance crypto. All results after costs.
Periods: develop to 2019 (crypto to 2020), validate 2020-22 (crypto 2021-22), final 2023 to Sep 2026, run once.

## Bottom line

1. **Did the rules beat buy-and-hold risk-adjusted out of sample? No.** In the final period, the best stock config
   had Sharpe 0.64 against SPY's 1.43. The best crypto config had 0.16 against BTC's 1.16. Forex never got meaningfully
   above zero.
2. **Which human layers mattered?** Stacking all of them (sit out bad markets, be selective, control risk) beat the
   naive rule in 13 of 18 out-of-sample comparisons. It did that mostly by trading less and holding less. It makes
   losing systems lose less. It did not create an edge over the market. No single layer had a reliable effect.
3. **Which asset class showed an edge? None.** The closest was RSI(2) dip-buying in stocks. It beat random entries in
   development and validation, but faded in the final period and failed its pre-registered test.
4. **Is anything overfit, survivorship-inflated or luck?** The crypto development results were overfit to one alt
   season. Every long stock result is flattered by survivorship. The one entry signal with evidence could not be told
   apart from luck in the final period.

The pre-registered primary hypothesis was: stocks RSI(2) plus relative strength plus quality score beats SPY on
Sharpe and beats 95% of random-entry runs, 2023 on. **It failed both parts:** Sharpe 0.56 vs 1.43, and it beat
86.6% of random runs (p = 0.13). Per the pre-registration, no rule set in this project is considered to have shown
an edge.

## What was tested

- **Three setups:**
  - breakout from a 20-day base (Minervini/Qullamaggie style)
  - trend pullback (Raschke Holy Grail: 9 > 20 > 50 EMA, ADX > 25, dip to the 20 EMA)
  - Connors RSI(2) mean reversion
- **Shorts:** crypto and forex got mirrored shorts. Stocks were long only.
- **Naive version:** the setup rule, a fixed stop, a 2R target, a 20-day max hold and 1% risk per trade. When more
  signals come in than there are free slots, trades are picked at random.
- **Human layers, each as a toggle:**
  - market regime, with a stand-down state
  - volatility stand-down
  - chop filter (ADX)
  - relative strength percentile
  - 0-100 quality score with a top-20% gate
  - breakout volume confirmation
  - earnings-gap proxy
  - sizing: 0.25 to 2% risk, ATR sizing, heat caps 4/6/8%, max positions, halving risk in a drawdown
  - exits: Qullamaggie partials, partial at 2R plus a 20MA trail, ATR trail, time stop
- **Simulation:** a portfolio with shared equity. Signals are taken on the close and filled at the next open. A bar
  that gaps through the stop fills at the open. A bar that hits both stop and target counts as a stop.
- **Costs per side:** stocks 10 bps, forex 2 bps, crypto 20 bps. Shorts pay 1 bp/day.

## Results by period

Sharpe is the median of 10 random-pick seeds. PICKED means the layers that helped on development, frozen before
validation was run. ALL means every textbook layer stacked, never selected on results.

| | Dev | Val | Final | Final CAGR | Final max DD |
|---|---|---|---|---|---|
| **SPY buy & hold** | 0.61 | 0.41 | **1.43** | 22.4% | -18.8% |
| S&P 500-today equal weight (survivor biased) | 1.05 | 0.60 | 1.33 | 19.8% | -17.9% |
| stocks breakout, naive | -0.22 | 0.02 | -0.04 | -1.8% | -26.9% |
| stocks breakout, ALL | 0.12 | 0.29 | 0.50 | 5.0% | -13.7% |
| stocks breakout, PICKED | 0.32 | -0.56 | 0.36 | 4.7% | -19.4% |
| stocks pullback, naive | 0.48 | -0.15 | 0.18 | 1.5% | -19.7% |
| stocks pullback, ALL | 0.58 | 0.58 | 0.43 | 4.2% | -13.9% |
| stocks pullback, PICKED | 0.61 | 0.21 | 0.26 | 3.0% | -19.7% |
| stocks RSI(2), naive | 0.63 | 0.55 | 0.25 | 2.6% | -20.3% |
| stocks RSI(2), ALL | 0.63 | 0.09 | 0.64 | 4.3% | -15.5% |
| stocks RSI(2), PICKED | 0.96 | 0.53 | 0.56 | 7.1% | -14.9% |
| **BTC buy & hold** | 1.10 | -0.03 | **1.16** | 54.4% | -53.0% |
| crypto equal weight (incl. dead coins) | 0.42 | 0.94 | 0.15 | -13.7% | -90.5% |
| crypto breakout, naive | 0.30 | 0.84 | -0.46 | -28.4% | -87.1% |
| crypto breakout, ALL | 0.26 | 0.94 | -0.05 | -1.6% | -22.6% |
| crypto breakout, PICKED | 1.24 | 0.68 | 0.11 | -4.5% | -69.5% |
| crypto pullback, naive | 0.12 | 1.53 | 0.16 | -0.5% | -51.2% |
| crypto pullback, ALL | 0.93 | 0.87 | -0.61 | -8.8% | -44.7% |
| crypto pullback, PICKED | 1.66 | 0.21 | -0.61 | -17.5% | -53.8% |
| crypto RSI(2), naive | -1.13 | 0.01 | -0.81 | -17.5% | -54.2% |
| crypto RSI(2), ALL | 0.09 | -0.96 | -0.44 | -3.3% | -15.7% |
| crypto RSI(2), PICKED | -0.35 | 0.76 | -0.60 | -12.2% | -47.1% |
| forex, best of 6 (RSI(2), ALL) | 0.03 | 0.31 | 0.22 | 0.9% | -8.2% |

Forex's other five configs were negative in the final period. Forex buy-and-hold is roughly 0.

**Validation nuance:** in 2020-22, three stock configs had a Sharpe above SPY's 0.41. They were RSI(2) naive 0.55,
RSI(2) picked 0.53 and pullback ALL 0.58. They got there by holding less through the 2020 crash and the 2022 bear.
In the 2023-26 bull market they made 3 to 7% a year while SPY made 22%. They had lower drawdowns than SPY, but not
by enough to make up for that.

## Which human layers mattered

| Layer | Verdict |
|---|---|
| **Full stack (ALL)** | Beat naive in 13 of 18 out-of-sample comparisons: stocks 5/6, forex 5/6, crypto 3/6. It cut trades 4-5x and exposure a lot, so fewer costs and smaller drawdowns. It did not beat buy-and-hold. |
| **Exits that let winners run** | Strongest single layer on dev: partial at 2R, stop to breakeven, trail the 20MA. Out of sample it was mixed, and the crypto gains were one alt season. The fixed 2R target of the naive rule clearly hurt. |
| Qullamaggie partials (1/3 after 3 days, trail 10MA) | Too tight for stocks and forex. Helped crypto on dev only. |
| **Market regime** | Halved breakout drawdowns. Hurt pullbacks. Not a reliable Sharpe improver. |
| Relative strength, quality score | Helped mean reversion and crypto pullback on dev. Hurt every breakout. Did not hold in validation. |
| Volume confirmation, earnings proxy, chop, vol stand-down | Noise. |
| Sizing, heat caps, drawdown de-risking | Change drawdown, not edge. None improved Sharpe consistently. |
| **Picking layers from dev results** | No better than stacking every textbook layer. A layer's effect on dev had a rank correlation of **-0.07** with its effect in validation. Only 48% of the layers that helped on dev still helped in validation. |

Entry signals, tested against 1,000 random entries with the same exits and sizing:
- **Breakouts:** no better than random stocks on the same days. In validation they were worse (1st percentile).
- **Pullbacks:** no better than random.
- **Stocks RSI(2):** the only entry with evidence. It beat 99.9% of random runs in dev and 99.8% in validation, but
  only 86.6% in the final period.

Market regime: almost every config made its money in bull markets and lost it in chop. Stocks RSI(2) was the
exception, and it was positive in the 2022 bear.

## What discretionary skill could not be coded

- **Catalysts and news.** There was no earnings calendar, news, guidance, token unlock or listing data. The
  Zarattini "stocks in play" result depends on relative volume plus a catalyst, and we only had a gap-on-volume
  proxy.
- **Sector and theme leadership.** There was no sector data, and a lot of real leadership is sector-driven.
- **Reading a chart.** The 0-100 base-quality score is a crude numeric stand-in for a trader's eye (shape, overhead
  supply, context). As coded, it made breakouts worse.
- **Intraday execution.** Daily bars force buying at the next open, after the breakout has already happened. Real
  breakout traders buy the pivot intraday with a stop at the low of the day. This penalizes breakouts specifically,
  and is the biggest fairness gap in the test.
- **Conviction sizing and pyramiding.** Adding to winners and sizing up on the best setups were not modeled.
- **Knowing when to stop trading.** Only crudely modeled, as halving risk in a 10% drawdown.
- **Survivor stories.** Online success stories come from the traders who survived. This test has no equivalent of
  "only the winners post", which is exactly the point.

## How likely is any edge to be overfit, survivorship-inflated or luck

- **Multiple testing.** 350 variants were tried on development: 141 stocks, 135 crypto, 74 forex. After the
  Harvey-Liu BHY haircut, the only picked config whose dev Sharpe survives meaningfully is stocks RSI(2), at 0.96
  down to 0.63. Crypto pullback keeps 0.73 under BHY but drops to 0 under Bonferroni, and its best results came from
  2020 alt season.
- **Crypto overfitting (flagged in advance).** The crypto dev universe had only 7 to 36 tradable coins. The two best
  dev results were crypto pullback at 1.66 on 46 trades and crypto breakout at 1.24. Both were built on Q4 2020 alt
  season. For crypto breakout, the top 10 trades made 70 to 83% of the profit. They fell to 0.21 and 0.68 in validation, and -0.61
  and 0.11 in the final period. Crypto pullback also collapses when the ADX threshold moves 30%.
- **Survivorship.** The stock universe is today's S&P 500. Its equal-weight buy-and-hold made 19.6% a year in dev
  vs SPY's 10.1%. That tailwind flatters every long stock result, and dip-buying (RSI(2)) most of all, because the
  stocks that dipped and then died are missing. Even with that tailwind, no stock config beat SPY in the final
  period. `trim_small_files.py` had also deleted crypto pairs under 500 rows, so fast-dying coins are missing too.
- **Luck.** Stocks RSI(2)'s final-period Sharpe 95% interval is -0.37 to 1.55, which includes zero. Its edge over
  random entries went from p = 0.001 in dev to p = 0.002 in validation to p = 0.13 in the final. It is consistent
  with a real but small short-term reversal effect that survivorship inflated, and that a strong bull market then
  swamped.
- **Process issue (disclosed).** The quality-score gate was buggy when stacked with other layers: it ranked against
  signals the other layers had already filtered. I found this after the validation numbers had been seen. It was
  fixed to match its stand-alone definition, and the picks were not changed. The pre-fix output is in
  `phase4_ablation_BEFORE_gate_fix.txt`. The final period was run once, and `FINAL_RUN.lock` records when.

## Not modeled / caveats

- **Costs and taxes:**
  - no taxes
  - no forex swap/carry
  - crypto short funding is a flat 1 bp/day
  - costs are flat per trade, with no size-based market impact
- **Crypto liquidity:** a coin only trades once its 30-day median volume is over $1M.
- **Forex data:** Yahoo forex data needed repairs. Since about 2009 the "close" is a stale snapshot near the open,
  so close was replaced with the next open, and weekday ranges were widened to include that close.
- **Charts:** equity charts show a single random-pick seed. The tables use the median of 10 seeds, and seed-to-seed
  spread is large (stocks breakout naive dev Sharpe ranges from -0.42 to 0.00).

## Files

Everything is in `results/`:
- `phase1_data_summary.txt`
- `phase2_baselines.txt` and the `phase2_*_equity.png` charts
- `phase3_layers.txt` and `phase3_layers.png`
- `phase4_ablation.txt`, `phase4_robust.txt`, `phase4_montecarlo.png` and `phase4_*_val_equity.png`
- `phase5_final.txt` and the `phase5_*_final_equity.png` charts
- `variant_log.csv`, which lists every run
- `phase4_picks.json` and `phase5_preregistration.json`, the frozen picks and the pre-registered hypothesis

How to rerun is in `../README.md`. The final period is locked and will not run again.
