"""
run.py  -  entry point. Run from anywhere:

  python3 ~/CODE/swing/run.py data            phase 1: clean, cache, universe summary
  python3 ~/CODE/swing/run.py data --force    rebuild the clean cache from the raw files
  python3 ~/CODE/swing/run.py baseline        phase 2: naive baseline of each setup, dev period only
  python3 ~/CODE/swing/run.py layers          phase 3: each layer alone on top of naive, dev period only
  python3 ~/CODE/swing/run.py ablation        phase 4a: ablation table + picked configs, dev and validation
  python3 ~/CODE/swing/run.py robust          phase 4b: sensitivity, Monte Carlo, bootstrap, regimes, haircut
  python3 ~/CODE/swing/run.py final           phase 5: the untouched final period, ONCE (refuses a second run)
"""
import argparse
import json
import sys
from io import StringIO

import numpy as np
import pandas as pd

import benchmarks
import config as C
import data
import experiments
import metrics
import phase4
import report
import strategy


class Tee:
    """Prints to the terminal and keeps a copy for results/."""
    def __init__(self):
        self.buf = StringIO()

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.buf.write(s + "\n")

    def save(self, name):
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (C.RESULTS_DIR / name).write_text(self.buf.getvalue())


def pct(x):
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


def cmd_data(force: bool):
    out = Tee()
    for asset in C.ASSETS:
        uni = data.build_clean_cache(asset, force=force)
        st = data.clean_stats(asset)
        sp = C.SPLITS[asset]
        out(f"\n===== {asset.upper()} =====")
        out(f"raw files {st.raw_files}, eligible {st.eligible_files}, kept segments {st.kept_segments} "
            f"(segments too short: {st.segments_too_short})")
        fixes = ["ohlc_fixed", "negative_volume_nulled", "rows_dropped_bad_history", "bad_ticks_removed", "weekend_rows_dropped",
                 "fx_close_outside_range", "redenominations"]
        out("cleaning: " + ", ".join(f"{k}={int(st[k])}" for k in fixes if st[k]))
        split_srcs = uni.source.value_counts()
        split_srcs = split_srcs[split_srcs > 1]
        if len(split_srcs):
            out(f"split into several segments: {', '.join(split_srcs.index)}")
        out(f"rows: total {uni.rows.sum():,}, median per symbol {int(uni.rows.median())}")
        out(f"history: {uni.start.min():%Y-%m-%d} to {uni.end.max():%Y-%m-%d}")
        ended = uni[uni.end < uni.end.max() - pd.Timedelta(days=30)]
        out(f"segments that end early (delisted / suspended / relabelled): {len(ended)}")

        # tradable symbols (>= WARMUP_BARS of own history) on Jan 1 of each year
        counts = {}
        for y in range(pd.Timestamp(sp["trade_start"]).year, uni.end.max().year + 1):
            t = pd.Timestamp(f"{y}-01-01", tz="UTC")
            n = 0
            for _, r in uni.iterrows():
                if r.start <= t <= r.end:
                    ok_from = r.start + pd.Timedelta(days=int(C.WARMUP_BARS * 365 / C.PERIODS_PER_YEAR[asset]))
                    n += ok_from <= t
            counts[y] = n
        out("tradable symbols on Jan 1: " + " ".join(f"{y}:{n}" for y, n in counts.items()))
        out(f"periods: trade from {sp['trade_start']}, dev to {sp['dev_end']}, "
            f"validate to {sp['val_end']}, final after (locked)")

        # benchmark sanity check, dev and validation only (final period stays unseen)
        ppy = C.PERIODS_PER_YEAR[asset]
        refs = {"equal-weight": benchmarks.equal_weight_index(asset, force=force)}
        if asset == "crypto":
            refs["BTC"] = benchmarks.single_asset("crypto", "BTCUSDT")
        for name, r in refs.items():
            for lab, a, b in (("dev", sp["trade_start"], sp["dev_end"]),
                              ("val", pd.Timestamp(sp["dev_end"]) + pd.Timedelta(days=1), sp["val_end"])):
                p = benchmarks.perf(r.loc[pd.Timestamp(a, tz="UTC"):pd.Timestamp(b, tz="UTC")], ppy)
                if p:
                    out(f"  buy&hold {name:12s} {lab}: CAGR {pct(p['cagr'])}  Sharpe {p['sharpe']:.2f}  "
                        f"maxDD {pct(p['maxdd'])}")
    out("\nWARNING: stocks are today's S&P 500 members. Every stock result (and the stock "
        "equal-weight index) is inflated by survivorship bias, more so the further back it goes.")
    out("NOTE: trim_small_files.py already deleted crypto pairs under 500 rows, which removed many "
        "coins that listed and died fast, so crypto keeps some survivorship bias too.")
    out.save("phase1_data_summary.txt")


def bench_returns(asset: str, period: str) -> dict:
    """Buy-and-hold references for one period (daily returns)."""
    a, b = strategy.period_bounds(asset, period)
    out = {}
    if asset == "stocks":
        out["SPY buy&hold"] = data.load_spy().close.pct_change().dropna()
        out["S&P500-today EW (survivor biased)"] = benchmarks.equal_weight_index("stocks")
    elif asset == "crypto":
        out["BTC buy&hold"] = benchmarks.single_asset("crypto", "BTCUSDT")
        out["crypto EW buy&hold"] = benchmarks.equal_weight_index("crypto")
    return {k: v.loc[a:b] for k, v in out.items()}


def cmd_baseline(n_seeds: int = 20):
    out = Tee()
    period = "dev"
    rows, yearly = [], {}
    tdir = C.RESULTS_DIR / "trades"
    tdir.mkdir(parents=True, exist_ok=True)
    out("PHASE 2: naive baselines, DEV period only (validation and final stay unseen)")
    out("naive = setup rule + fixed stop + 2R target + 20-bar max hold (rsi2: Connors exits), "
        "1% risk per trade, max 10 positions, random pick when signals exceed free slots")
    for asset in C.ASSETS:
        a, b = strategy.period_bounds(asset, period)
        ppy = C.PERIODS_PER_YEAR[asset]
        out(f"\n===== {asset.upper()}  dev {a:%Y-%m-%d} to {b:%Y-%m-%d}  "
            f"({'long only' if C.LONG_ONLY[asset] else 'long + mirrored short'}) =====")
        table, curves, notes = [], {}, []
        benches = bench_returns(asset, period)
        for name, r in benches.items():
            st = metrics.equity_stats((1 + r).cumprod(), ppy)
            table.append(dict(name=name, **st))
        if asset == "forex":
            table.append(dict(name="cash (forex buy&hold ~ 0)", cagr=0.0))
        for setup in ("breakout", "pullback", "rsi2"):
            cfg = strategy.naive_config(setup)
            prep = strategy.prepare(asset, cfg, period)
            res, st = strategy.run(asset, cfg, period, seed=0, label=f"naive {setup}", prepared=prep)
            _, gross = strategy.run(asset, cfg, period, prepared=prep, log=False, cost_mult=0.0)
            sh = [strategy.run(asset, cfg, period, seed=s, prepared=prep, log=False)[1].get("sharpe", np.nan)
                  for s in range(1, n_seeds)]
            sh = np.array(sh + [st["sharpe"]])
            table.append(dict(name=f"naive {setup}", **st))
            curves[setup] = res.equity
            yearly[f"{asset}_{setup}"] = report.yearly_returns(res.equity)
            res.trades.to_csv(tdir / f"phase2_{asset}_{setup}.csv", index=False)
            rows.append(dict(asset=asset, setup=setup, **st, gross_exp_r=gross.get("exp_r"),
                             gross_cagr=gross.get("cagr"), seed_sharpe_min=sh.min(),
                             seed_sharpe_med=np.median(sh), seed_sharpe_max=sh.max()))
            n = f"  {setup:8s} before costs: CAGR {pct(gross.get('cagr'))}, exp R {gross.get('exp_r', np.nan):.3f}"
            n += f" | Sharpe over {n_seeds} random-pick seeds: {sh.min():.2f} to {sh.max():.2f} (median {np.median(sh):.2f})"
            if not C.LONG_ONLY[asset]:
                n += "\n           " + report.side_stats(res.trades)
            notes.append(n)
        out(report.stats_table(table))
        for n in notes:
            out(n)
        report.equity_chart(C.RESULTS_DIR / f"phase2_{asset}_equity.png",
                            f"{asset}: naive baselines, dev period, after costs", curves,
                            {k: v for k, v in benches.items()})
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "phase2_baselines.csv", index=False)
    pd.DataFrame(yearly).to_csv(C.RESULTS_DIR / "phase2_yearly_returns.csv")
    vl = pd.read_csv(strategy.VARIANT_LOG)
    out(f"\nvariants logged so far (distinct configs x asset, dev): "
        f"{vl[vl.period == 'dev'].groupby(['asset', 'hash']).ngroups}")
    out("Expo = average gross exposure / equity. InMkt = share of days with a position. "
        "Cost/GP = costs as a share of gross profits.")
    out.save("phase2_baselines.txt")


def cmd_layers(n_seeds: int = 10):
    out = Tee()
    period = "dev"
    out("PHASE 3: each layer ALONE on top of the naive baseline, DEV period only")
    out(f"every row = median over {n_seeds} random-pick seeds. dSh = change in median Sharpe vs naive.")
    out("verdict: + if dSh >= +0.10, - if <= -0.10, ~ otherwise (rough; formal tests come in phase 4)")
    rows = []
    pc = experiments.PrepCache()
    weekend = []
    for asset in C.ASSETS:
        for setup in ("breakout", "pullback", "rsi2"):
            base_cfg = experiments.layer_config(asset, setup, "naive")
            base_prep = pc.get(asset, base_cfg, period)
            base_sig = experiments.event_signature(base_prep)
            base, _, base_res = experiments.multi_seed(asset, base_cfg, period, base_prep, n_seeds, f"naive {setup}")
            if asset == "crypto":
                t = base_res.trades
                wk = t.entry_date.dt.dayofweek >= 5
                weekend.append(f"  {setup:8s} weekend entries {wk.sum():4d}: exp R {t.r[wk].mean():+.3f} | "
                               f"weekday entries {(~wk).sum():4d}: exp R {t.r[~wk].mean():+.3f}")
            out(f"\n===== {asset.upper()} / {setup}   naive Sharpe over seeds {base['sharpe_min']:.2f} to "
                f"{base['sharpe_max']:.2f} =====")
            out(f"{'layer':26s}{'Sharpe':>8s}{'dSh':>7s}{'CAGR':>8s}{'MaxDD':>8s}{'tr/yr':>7s}{'ExpR':>8s}"
                f"{'PF':>6s}{'Win':>6s}{'Expo':>6s}  v")
            for name in experiments.all_layer_names():
                cfg = experiments.layer_config(asset, setup, name)
                if cfg is None:
                    continue
                if name == "naive":
                    med = base
                else:
                    prep = pc.get(asset, cfg, period)
                    if cfg.get("filters") and experiments.event_signature(prep) == base_sig:
                        out(f"{name:26s}  same signals as naive (layer is redundant here), skipped")
                        continue
                    med, _, _ = experiments.multi_seed(asset, cfg, period, prep, n_seeds, f"{setup} + {name}")
                d = med["sharpe"] - base["sharpe"]
                v = "" if name == "naive" else ("+" if d >= 0.10 else "-" if d <= -0.10 else "~")
                out(f"{name:26s}{med['sharpe']:8.2f}{d:7.2f}{pct(med['cagr']):>8s}{pct(med['maxdd']):>8s}"
                    f"{med['trades_per_yr']:7.0f}{med['exp_r']:8.3f}{med['profit_factor']:6.2f}"
                    f"{med['win_rate'] * 100:5.0f}%{med['exposure']:6.2f}  {v}")
                rows.append(dict(asset=asset, setup=setup, layer=name, d_sharpe=d, verdict=v, **med))
    out("\nCRYPTO weekend check (naive, seed 0, entries filled Sat/Sun vs Mon-Fri):")
    for w in weekend:
        out(w)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "phase3_layers.csv", index=False)
    report.layer_dots(C.RESULTS_DIR / "phase3_layers.png", df,
                      "Phase 3: each layer alone vs naive (dev, median of 10 seeds). Grey band = +/-0.10")
    # which layers helped consistently
    out("\nLAYER SCORECARD (count of setup x asset combos where the layer is +, ~, -):")
    for name in [n for n in experiments.all_layer_names() if n != "naive"]:
        d = df[df.layer == name]
        if len(d):
            out(f"  {name:26s} +{(d.verdict == '+').sum()}  ~{(d.verdict == '~').sum()}  -{(d.verdict == '-').sum()}"
                f"   mean dSh {d.d_sharpe.mean():+.2f}")
    vl = pd.read_csv(strategy.VARIANT_LOG)
    out(f"\nvariants logged so far (distinct configs x asset, dev): "
        f"{vl[vl.period == 'dev'].groupby(['asset', 'hash']).ngroups}")
    out.save("phase3_layers.txt")


def _row(out, name, dv, vl, base_dv, base_vl):
    out(f"{name:28s}{dv['sharpe']:7.2f}{dv['sharpe'] - base_dv['sharpe']:+7.2f}   |"
        f"{vl['sharpe']:7.2f}{vl['sharpe'] - base_vl['sharpe']:+7.2f}{pct(vl['cagr']):>8s}{pct(vl['maxdd']):>8s}"
        f"{vl['trades_per_yr']:7.0f}{vl['exp_r']:8.3f}")


def cmd_ablation(n_seeds: int = 10):
    out = Tee()
    picks = phase4.make_picks()          # frozen from phase 3 dev results BEFORE any validation run
    out(f"PHASE 4a: ablation + picked configs. Picks frozen {picks['made']} from dev only: {picks['rule']}")
    out("NOTE: first run had a score-gate bug when stacked with other layers (gate ranked against already-"
        "filtered signals). Fixed to rank against all raw setup signals, as in phase 3. Picks unchanged. "
        "Pre-fix output kept in phase4_ablation_BEFORE_gate_fix.txt")
    out(f"median of {n_seeds} random-pick seeds. dSh = change vs naive in the same period.")
    pc = experiments.PrepCache()
    rows = []
    head = f"{'':28s}{'DEV Sh':>7s}{'dSh':>7s}   |{'VAL Sh':>7s}{'dSh':>7s}{'CAGR':>8s}{'MaxDD':>8s}{'tr/yr':>7s}{'ExpR':>8s}"

    def both(asset, cfg, label):
        res = {}
        for per in ("dev", "val"):
            prep = pc.get(asset, cfg, per)
            res[per], _, _ = experiments.multi_seed(asset, cfg, per, prep, n_seeds, label)
        return res

    val_curves = {a: {} for a in phase4.PICK_ASSETS}
    for asset in C.ASSETS:
        for setup in ("breakout", "pullback", "rsi2"):
            out(f"\n===== {asset.upper()} / {setup} =====")
            out(head)
            base = both(asset, strategy.naive_config(setup), f"naive {setup}")
            base_sig = experiments.event_signature(pc.get(asset, strategy.naive_config(setup), "dev"))
            variants = [("naive", strategy.naive_config(setup))]
            for name in experiments.all_layer_names()[1:]:
                cfg = experiments.layer_config(asset, setup, name)
                if cfg is None:
                    continue
                if cfg.get("filters") and experiments.event_signature(pc.get(asset, cfg, "dev")) == base_sig:
                    continue
                variants.append((f"+ {name}", cfg))
            variants.append(("ALL LAYERS", phase4.all_layers(asset, setup)))
            for g in phase4.minus_groups(asset, setup):
                variants.append((f"all minus {g}", phase4.all_layers(asset, setup, minus=g)))
            key = f"{asset}/{setup}"
            if key in picks["configs"]:
                variants.append(("PICKED", picks["configs"][key]["cfg"]))
            for name, cfg in variants:
                r = base if name == "naive" else both(asset, cfg, f"{setup} {name}")
                _row(out, name, r["dev"], r["val"], base["dev"], base["val"])
                rows.append(dict(asset=asset, setup=setup, variant=name,
                                 **{f"dev_{k}": v for k, v in r["dev"].items()},
                                 **{f"val_{k}": v for k, v in r["val"].items()}))
                if name == "PICKED":
                    out(f"  picked layers: {', '.join(picks['configs'][key]['layers']) or 'none (naive)'}")
                    res, _ = strategy.run(asset, cfg, "val", prepared=pc.get(asset, cfg, "val"), log=False)
                    val_curves[asset][setup] = res.equity
        if asset in phase4.PICK_ASSETS:
            benches = bench_returns(asset, "val")
            out(f"  buy&hold in validation: " + ", ".join(
                f"{k} Sharpe {metrics.equity_stats((1 + r).cumprod(), C.PERIODS_PER_YEAR[asset]).get('sharpe', np.nan):.2f} "
                f"CAGR {pct(metrics.equity_stats((1 + r).cumprod(), C.PERIODS_PER_YEAR[asset]).get('cagr'))}"
                for k, r in benches.items()))
            report.equity_chart(C.RESULTS_DIR / f"phase4_{asset}_val_equity.png",
                                f"{asset}: picked configs, VALIDATION period, after costs",
                                val_curves[asset], benches)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "phase4_ablation.csv", index=False)
    # does a layer's effect on dev predict its effect in validation?
    out("\nDO DEV LAYER EFFECTS HOLD IN VALIDATION? (single-layer rows only)")
    one = df[df.variant.str.startswith("+ ")].copy()
    base = df[df.variant == "naive"].set_index(["asset", "setup"])
    one["d_dev"] = one.dev_sharpe - one.set_index(["asset", "setup"]).index.map(base.dev_sharpe)
    one["d_val"] = one.val_sharpe - one.set_index(["asset", "setup"]).index.map(base.val_sharpe)
    for a, g in list(one.groupby("asset")) + [("all", one)]:
        rho = g[["d_dev", "d_val"]].corr(method="spearman").iloc[0, 1]
        plus = g[g.d_dev >= 0.10]
        out(f"  {a:7s} rank corr(dev effect, val effect) {rho:+.2f} | layers that helped on dev: {len(plus):3d}, "
            f"still helped in val: {(plus.d_val > 0).sum():3d} ({(plus.d_val > 0).mean() * 100:.0f}%)")
    vl = pd.read_csv(strategy.VARIANT_LOG)
    out(f"\nvariants tried on dev (distinct configs x asset): {vl[vl.period == 'dev'].groupby(['asset', 'hash']).ngroups}")
    out.save("phase4_ablation.txt")


def cmd_robust(n_mc: int = 1000, n_mc_any: int = 200):
    out = Tee()
    picks = phase4.make_picks()
    pc = experiments.PrepCache()
    out("PHASE 4b: robustness of the picked configs (stocks, crypto)")
    vl = pd.read_csv(strategy.VARIANT_LOG)
    dev_v = vl[vl.period == "dev"].drop_duplicates(["asset", "hash"])
    yrs_of = {a: (strategy.period_bounds(a, "dev")[1] - strategy.period_bounds(a, "dev")[0]).days / 365.25
              for a in C.ASSETS}
    all_sr = dev_v.sharpe.fillna(0).values
    all_yrs = dev_v.asset.map(yrs_of).values
    rows, mc_all = [], {}
    for key, pk in picks["configs"].items():
        asset, setup = key.split("/")
        cfg = pk["cfg"]
        ppy = C.PERIODS_PER_YEAR[asset]
        out(f"\n===== {key}  layers: {', '.join(pk['layers']) or 'none (naive)'} =====")
        rec = {"config": key}
        # --- sensitivity on dev
        base, _, _ = experiments.multi_seed(asset, cfg, "dev", pc.get(asset, cfg, "dev"), 5, f"{key} picked", log=False)
        out(f"sensitivity (dev, 5 seeds; base Sharpe {base['sharpe']:.2f}): each number moved -30% / +30%")
        collapses = []
        for label, fn in phase4.sensitivity_params(asset, cfg):
            shs = []
            for m in (0.7, 1.3):
                c2 = fn(cfg, m)
                med, _, _ = experiments.multi_seed(asset, c2, "dev", pc.get(asset, c2, "dev"), 5, f"{key} sens {label} x{m}")
                shs.append(med["sharpe"])
            flag = any((s < 0.5 * base["sharpe"]) or (np.sign(s) != np.sign(base["sharpe"])) for s in shs) \
                if base["sharpe"] > 0 else False
            if flag:
                collapses.append(label)
            out(f"  {label:26s} -30%: {shs[0]:5.2f}   +30%: {shs[1]:5.2f}   {'COLLAPSE' if flag else ''}")
        rec["sens_params"] = len(phase4.sensitivity_params(asset, cfg))
        rec["sens_collapses"] = len(collapses)
        # --- per period: actual, Monte Carlo, bootstrap
        for per in ("dev", "val"):
            prep = pc.get(asset, cfg, per)
            med, _, res = experiments.multi_seed(asset, cfg, per, prep, 10, f"{key} picked", log=False)
            mc = phase4.random_entry_runs(asset, cfg, prep, n_mc, "same_day")
            mca = phase4.random_entry_runs(asset, cfg, prep, n_mc_any, "any_day", seed0=50_000)
            mc_all[f"{key} {per}"] = (mc.sharpe.values, med["sharpe"])
            p_mc = (1 + (mc.sharpe >= med["sharpe"]).sum()) / (len(mc) + 1)
            p_mca = (1 + (mca.sharpe >= med["sharpe"]).sum()) / (len(mca) + 1)
            lo, hi = phase4.block_bootstrap_sharpe(res.equity.pct_change().dropna(), ppy)
            elo, ehi = phase4.bootstrap_mean(res.trades.r.values)
            out(f"{per.upper()}: Sharpe {med['sharpe']:.2f} (seed0 95% CI {lo:.2f} to {hi:.2f}), exp R "
                f"{med['exp_r']:.3f} (95% CI {elo:.3f} to {ehi:.3f}), CAGR {pct(med['cagr'])}, maxDD {pct(med['maxdd'])}")
            out(f"     random entries, same days ({n_mc}): median Sharpe {mc.sharpe.median():.2f}, "
                f"95th pct {mc.sharpe.quantile(.95):.2f} -> real beats {100 * (1 - p_mc):.1f}% (p={p_mc:.3f})")
            out(f"     random entries, any day  ({n_mc_any}): median Sharpe {mca.sharpe.median():.2f}, "
                f"95th pct {mca.sharpe.quantile(.95):.2f} -> real beats {100 * (1 - p_mca):.1f}% (p={p_mca:.3f})")
            rec.update({f"{per}_sharpe": med["sharpe"], f"{per}_sharpe_lo": lo, f"{per}_sharpe_hi": hi,
                        f"{per}_expr": med["exp_r"], f"{per}_expr_lo": elo, f"{per}_expr_hi": ehi,
                        f"{per}_mc_p": p_mc, f"{per}_mc_any_p": p_mca, f"{per}_mc_median": mc.sharpe.median()})
            if per == "dev":
                hc = phase4.haircut(med["sharpe"], yrs_of[asset], all_sr, all_yrs)
                out(f"     haircut over {hc['N']} dev variants: t={hc['t']:.2f}, p={hc['p']:.4f}; Bonferroni p "
                    f"{hc['p_bonferroni']:.3f} -> Sharpe {hc['sr_bonferroni']:.2f}; BHY p {hc['p_bhy']:.3f} -> "
                    f"Sharpe {hc['sr_bhy']:.2f} (haircut {report.pct(hc['haircut_bhy'], 0)})")
                rec.update({"hc_N": hc["N"], "dev_sr_bhy": hc["sr_bhy"], "dev_sr_bonf": hc["sr_bonferroni"]})
        # --- regimes and years, dev + val combined view (separate runs, seed 0)
        reg = phase4.market_regime(asset)
        lines = []
        for per in ("dev", "val"):
            res, _ = strategy.run(asset, cfg, per, prepared=pc.get(asset, cfg, per), log=False)
            r = res.equity.pct_change().dropna()
            g = reg.reindex(r.index).fillna("chop")
            t = res.trades
            tr = reg.reindex(t.entry_date).fillna("chop").values
            for name in ("bull", "chop", "bear"):
                x = r[g == name]
                if len(x) > 20:
                    sh = x.mean() / x.std() * np.sqrt(ppy) if x.std() > 0 else np.nan
                    lines.append(f"  {per} {name:5s} {len(x) / len(r) * 100:4.0f}% of days  ann.return "
                                 f"{x.mean() * ppy * 100:6.1f}%  Sharpe {sh:5.2f}  trades {int((tr == name).sum()):5d}  "
                                 f"exp R {t.r[tr == name].mean():+.3f}")
            yr = report.yearly_returns(res.equity)
            rec[f"{per}_years"] = " ".join(f"{y}:{v * 100:+.0f}%" for y, v in yr.items())
        out("by market regime (proxy above rising 200 SMA = bull, below falling = bear, else chop):")
        for l in lines:
            out(l)
        out(f"by year: {rec['dev_years']} | VAL {rec['val_years']}")
        rows.append(rec)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "phase4_robust.csv", index=False)
    report.mc_hist(C.RESULTS_DIR / "phase4_montecarlo.png", mc_all)
    vl = pd.read_csv(strategy.VARIANT_LOG)
    out(f"\nvariants tried on dev incl. sensitivity (distinct configs x asset): "
        f"{vl[vl.period == 'dev'].groupby(['asset', 'hash']).ngroups}")
    out.save("phase4_robust.txt")


def cmd_final(n_seeds: int = 10, n_mc: int = 1000):
    lock = C.RESULTS_DIR / "FINAL_RUN.lock"
    if lock.exists():
        print(f"The final period was already run ({lock.read_text().strip()}). Results: results/phase5_final.txt")
        return
    import hashlib
    picks = phase4.make_picks()
    digest = hashlib.sha1(phase4.PICKS_PATH.read_bytes()).hexdigest()[:12]
    lock.write_text(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} picks sha1 {digest}")
    out = Tee()
    out(f"PHASE 5: FINAL untouched period, run once. picks file sha1 {digest}, frozen {picks['made']}")
    prereg = json.loads((C.RESULTS_DIR / "phase5_preregistration.json").read_text())
    out(f"pre-registered primary hypothesis ({prereg['made']}): {prereg['primary']}")
    out("every candidate below is reported; nothing is dropped after seeing these numbers.")
    pc = experiments.PrepCache()
    rows = []
    for asset in C.ASSETS:
        a, b = strategy.period_bounds(asset, "final")
        ppy = C.PERIODS_PER_YEAR[asset]
        out(f"\n===== {asset.upper()} final period from {a:%Y-%m-%d} =====")
        table, curves = [], {}
        benches = bench_returns(asset, "final") if asset != "forex" else {}
        for name, r in benches.items():
            table.append(dict(name=name, **metrics.equity_stats((1 + r).cumprod(), ppy)))
        for setup in ("breakout", "pullback", "rsi2"):
            key = f"{asset}/{setup}"
            cands = [(f"naive {setup}", strategy.naive_config(setup)),
                     (f"ALL LAYERS {setup}", phase4.all_layers(asset, setup))]
            if key in picks["configs"]:
                cands.append((f"PICKED {setup}", picks["configs"][key]["cfg"]))
            for name, cfg in cands:
                prep = pc.get(asset, cfg, "final", allow_final=True)
                med, _, res = experiments.multi_seed(asset, cfg, "final", prep, n_seeds, name)
                row = dict(name=name, **med)
                if name.startswith("PICKED"):
                    mc = phase4.random_entry_runs(asset, cfg, prep, n_mc, "same_day")
                    row["mc_p"] = (1 + (mc.sharpe >= med["sharpe"]).sum()) / (len(mc) + 1)
                    lo, hi = phase4.block_bootstrap_sharpe(res.equity.pct_change().dropna(), ppy)
                    row["sh_lo"], row["sh_hi"] = lo, hi
                    curves[setup] = res.equity
                table.append(row)
                rows.append(dict(asset=asset, **row))
        out(report.stats_table(table))
        for r in table:
            if "mc_p" in r:
                out(f"  {r['name']}: Sharpe 95% CI {r['sh_lo']:.2f} to {r['sh_hi']:.2f}; beats "
                    f"{100 * (1 - r['mc_p']):.1f}% of {n_mc} random-entry runs (p={r['mc_p']:.3f})")
        if curves:
            report.equity_chart(C.RESULTS_DIR / f"phase5_{asset}_final_equity.png",
                                f"{asset}: picked configs, FINAL period, after costs", curves, benches)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "phase5_final.csv", index=False)
    out.save("phase5_final.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "baseline", "layers", "ablation", "robust", "final"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "data":
        cmd_data(a.force)
    elif a.cmd == "baseline":
        cmd_baseline()
    elif a.cmd == "layers":
        cmd_layers()
    elif a.cmd == "ablation":
        cmd_ablation()
    elif a.cmd == "robust":
        cmd_robust()
    elif a.cmd == "final":
        cmd_final()


if __name__ == "__main__":
    sys.exit(main())
