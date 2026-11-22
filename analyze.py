"""
Test whether order book imbalance predicts the next price move, then plot a liquidity heatmap.

Usage:
    python analyze.py                      # uses every file in data/
    python analyze.py --fee-bps 40 --heatmap-minutes 5

Outputs (in results/):
    report.md            signal stats per horizon + after-cost check
    heatmap.png          liquidity heatmap with mid price and imbalance
    deciles.png          future return by signal decile
    ic_by_horizon.png    predictive power vs. horizon
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

SIGNALS = {
    "imb_l1": "L1 imbalance (best bid vs ask size)",
    "imb_l5": "Depth imbalance (top 5 levels)",
    "ofi_1s": "Order flow imbalance (1s, normalized)",
}


# ---------- data ----------

def load(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv.gz")))
    if not files:
        raise SystemExit(f"No data in {data_dir}/ - run recorder.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64").dropna()
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    depth = sum(c.startswith("bid_px_") for c in df.columns)
    print(f"loaded {len(df):,} samples from {len(files)} file(s), depth {depth}, "
          f"{(df.ts.iloc[-1] - df.ts.iloc[0]) / 3600:.2f} h span")
    return df, depth


def features(df, gap_s):
    b1, a1 = df.bid_px_1, df.ask_px_1
    bq1, aq1 = df.bid_qty_1, df.ask_qty_1
    df["mid"] = (b1 + a1) / 2
    df["spread_bps"] = (a1 - b1) / df.mid * 1e4

    df["imb_l1"] = (bq1 - aq1) / (bq1 + aq1)
    bq5 = df[[f"bid_qty_{i}" for i in range(1, 6)]].sum(axis=1)
    aq5 = df[[f"ask_qty_{i}" for i in range(1, 6)]].sum(axis=1)
    df["imb_l5"] = (bq5 - aq5) / (bq5 + aq5)

    # Order flow imbalance (Cont, Kukanov & Stoikov 2014): net change in best-level
    # liquidity, counting price moves. Positive = buying pressure.
    B, A, QB, QA = (v.to_numpy(float) for v in (b1, a1, bq1, aq1))
    pb, pa, qb, qa = (np.r_[np.nan, v[:-1]] for v in (B, A, QB, QA))
    e = ((B >= pb) * QB - (B <= pb) * qb - (A <= pa) * QA + (A >= pa) * qa).astype(float)
    gap = (df.ts.diff() > gap_s).to_numpy()   # reconnects / missing data
    e[gap | np.isnan(e)] = 0.0
    gap = pd.Series(gap, index=df.index)
    df["seg"] = gap.cumsum()             # contiguous recording segments

    t = pd.to_datetime(df.ts, unit="s")
    ofi = pd.Series(e, index=t).rolling("1s").sum().values
    depth = pd.Series(((bq1 + aq1) / 2).values, index=t).rolling("60s").mean().values
    df["ofi_1s"] = ofi / depth
    return df


def add_targets(df, horizons, tol):
    ts, mid, seg = df.ts.values, df.mid.values, df.seg.values
    for h in horizons:
        j = np.searchsorted(ts, ts + h)
        ok = j < len(ts)
        jj = np.where(ok, j, 0)
        ok &= (ts[jj] - (ts + h) <= tol) & (seg[jj] == seg)   # no jumping across gaps
        df[f"ret_{h}s"] = np.where(ok, (mid[jj] / mid - 1) * 1e4, np.nan)   # bps
    return df


# ---------- stats ----------

def spearman(a, b):
    a = pd.Series(np.asarray(a, dtype="float64")).rank().to_numpy()
    b = pd.Series(np.asarray(b, dtype="float64")).rank().to_numpy()
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(df, horizons, interval, split):
    rows = []
    cut = int(len(df) * split)
    for s in SIGNALS:
        for h in horizons:
            d = df[[s, f"ret_{h}s"]].dropna()
            x, y = d[s], d[f"ret_{h}s"]
            ic = spearman(x, y)
            # Overlapping windows overstate significance; also report on non-overlapping samples
            step = max(1, int(round(h / interval)))
            dn = d.iloc[::step]
            ic_no = spearman(dn[s], dn[f"ret_{h}s"])
            t_no = ic_no * np.sqrt(max(len(dn) - 2, 1)) / np.sqrt(max(1 - ic_no ** 2, 1e-12))
            nz = (y != 0) & (x != 0)
            hit = (np.sign(x[nz]) == np.sign(y[nz])).mean()
            # Out-of-sample: fit slope on first part, score on the rest
            tr, te = d[d.index < cut], d[d.index >= cut]
            if len(tr) > 100 and len(te) > 100:
                beta = np.polyfit(tr[s].to_numpy(float), tr[f"ret_{h}s"].to_numpy(float), 1)
                pred = np.polyval(beta, te[s].to_numpy(float))
                yt = te[f"ret_{h}s"].to_numpy(float)
                r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
            else:
                r2 = np.nan
            rows.append(dict(signal=s, horizon_s=h, n=len(d), IC=ic, IC_nonoverlap=ic_no,
                             t_stat=t_no, hit_rate=hit, oos_R2=r2))
    return pd.DataFrame(rows)


def cost_test(df, signal, h, fee_bps, split, q=0.9):
    """Trade only the strongest 10% of signals (threshold set on training data),
    cross the spread in and out, pay taker fees both ways, no overlapping trades."""
    cut = int(len(df) * split)
    hi = df[signal].iloc[:cut].abs().quantile(q)
    te = df.iloc[cut:]
    ts, sig, ret, spr = (te[c].to_numpy(float) for c in ("ts", signal, f"ret_{h}s", "spread_bps"))
    gross, net, next_free = [], [], -np.inf
    for i in range(len(te)):
        if ts[i] < next_free or np.isnan(ret[i]) or abs(sig[i]) < hi:
            continue
        g = np.sign(sig[i]) * ret[i]
        gross.append(g)
        net.append(g - spr[i] - 2 * fee_bps)   # ~full spread round trip + fees
        next_free = ts[i] + h
    if not gross:
        return None
    return dict(signal=signal, horizon_s=h, trades=len(gross), gross_bps=np.mean(gross),
                spread_bps=np.mean(spr), fees_bps=2 * fee_bps, net_bps=np.mean(net),
                win_rate_net=np.mean(np.array(net) > 0))


# ---------- plots ----------

def plot_heatmap(df, depth, minutes, out):
    d = df[df.ts >= df.ts.iloc[-1] - minutes * 60]
    d = d[d.seg == d.seg.iloc[-1]]
    t0 = d.ts.iloc[0]
    tmin = (d.ts - t0).values / 60

    px = np.concatenate([d[f"{s}_px_{i}"].values for s in ("bid", "ask") for i in range(1, depth + 1)])
    qty = np.concatenate([d[f"{s}_qty_{i}"].values for s in ("bid", "ask") for i in range(1, depth + 1)])
    tt = np.tile(tmin, 2 * depth)

    tbins = min(600, max(50, len(d) // 5))
    lo, hi = np.percentile(px, [0.5, 99.5])
    H, xe, ye = np.histogram2d(tt, px, bins=[tbins, 250], range=[[0, tmin[-1]], [lo, hi]], weights=qty)
    cnt, _ = np.histogram(tmin, bins=xe)
    H = H / np.maximum(cnt, 1)[:, None]          # average resting size per sample
    H[H <= 0] = np.nan

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    m = ax.pcolormesh(xe, ye, H.T, cmap="magma", norm=LogNorm(), shading="auto")
    ax.plot(tmin, d.mid, color="#4fc3f7", lw=0.9, label="mid price")
    ax.set_ylabel("price")
    ax.set_title(f"Liquidity heatmap - last {minutes} min (brighter = more resting size)")
    ax.legend(loc="upper left")
    fig.colorbar(m, ax=[ax, ax2], pad=0.01, label="avg resting qty")

    sm = pd.Series(d.imb_l5.values).rolling(max(1, len(d) // tbins)).mean()
    ax2.fill_between(tmin, sm, 0, where=sm >= 0, color="#2e7d32", alpha=0.7, lw=0)
    ax2.fill_between(tmin, sm, 0, where=sm < 0, color="#c62828", alpha=0.7, lw=0)
    ax2.axhline(0, color="gray", lw=0.6)
    ax2.set_ylabel("top-5 imbalance\n(+ = bid heavy)")
    ax2.set_xlabel("minutes")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_deciles(df, h, out):
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(14, 4), sharey=True)
    for ax, (s, name) in zip(axes, SIGNALS.items()):
        d = df[[s, f"ret_{h}s"]].dropna()
        dec = pd.qcut(d[s].rank(method="first"), 10, labels=range(1, 11))
        m = d.groupby(dec, observed=True)[f"ret_{h}s"].mean()
        ax.bar(m.index.astype(int), m.values, color=np.where(m.values >= 0, "#2e7d32", "#c62828"))
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("signal decile (1 = most sell-heavy)")
    axes[0].set_ylabel(f"avg next {h}s move (bps)")
    fig.suptitle(f"Does imbalance predict the next {h}s? (a clean staircase = yes)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_ic(stats, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, name in SIGNALS.items():
        r = stats[stats.signal == s]
        ax.plot(r.horizon_s, r.IC_nonoverlap, marker="o", label=name)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("horizon (seconds, log scale)")
    ax.set_ylabel("rank correlation with future move")
    ax.set_title("Signal strength decays with horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--horizons", default="1,2,5,10,30,60", help="seconds")
    ap.add_argument("--interval", type=float, default=0.1, help="sampling interval used when recording")
    ap.add_argument("--fee-bps", type=float, default=40,
                    help="taker fee per side in bps (40 = 0.40%%; check your actual tier)")
    ap.add_argument("--split", type=float, default=0.7, help="train fraction (by time)")
    ap.add_argument("--heatmap-minutes", type=float, default=5)
    args = ap.parse_args()

    horizons = [float(h) if "." in h else int(h) for h in args.horizons.split(",")]
    os.makedirs(args.out, exist_ok=True)

    df, depth = load(args.data)
    df = features(df, gap_s=max(2.0, args.interval * 20))
    df = add_targets(df, horizons, tol=max(0.5, args.interval * 5))

    stats = evaluate(df, horizons, args.interval, args.split)
    best_h = horizons[min(1, len(horizons) - 1)]
    costs = [c for s in SIGNALS for h in horizons if (c := cost_test(df, s, h, args.fee_bps, args.split))]
    costs = pd.DataFrame(costs)

    plot_heatmap(df, depth, args.heatmap_minutes, os.path.join(args.out, "heatmap.png"))
    plot_deciles(df, best_h, os.path.join(args.out, "deciles.png"))
    plot_ic(stats, os.path.join(args.out, "ic_by_horizon.png"))

    fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("# Order book imbalance - results\n\n")
        f.write(f"- Samples: {len(df):,} | span {(df.ts.iloc[-1]-df.ts.iloc[0])/3600:.2f} h | "
                f"recording segments: {df.seg.nunique()}\n")
        f.write(f"- Median spread: {df.spread_bps.median():.2f} bps | fee assumption: {args.fee_bps} bps/side\n\n")
        f.write("## Predictive power\n\nIC = rank correlation between signal and the future mid move. "
                "`IC_nonoverlap` and `t_stat` use non-overlapping samples (the honest numbers). "
                "`oos_R2` is fit on the first "
                f"{int(args.split*100)}% of data, scored on the rest.\n\n")
        f.write(stats.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n\n")
        f.write("## Does it make money? (top-10% signals, cross spread, pay fees, test period only)\n\n")
        f.write((costs.to_string(index=False, float_format=lambda x: f"{x:.2f}") if len(costs) else "no trades") + "\n")
    print(stats.to_string(index=False, float_format=fmt))
    print()
    if len(costs):
        print(costs.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nsaved report + charts to {args.out}/")


if __name__ == "__main__":
    main()
