"""
Live order book imbalance dashboard (paper only - no real orders).

Usage:
    python3 live.py                       # BTC/USD, predict 2s ahead
    python3 live.py --horizon 5 --threshold 0.6 --depth 100

Shows:
  - rolling liquidity heatmap with mid price
  - live top-5 imbalance
  - scoreboard: every time imbalance is strong, it predicts the direction of the
    next move, checks itself after --horizon seconds, and keeps score
    (hit rate, avg move captured, paper P&L before and after fees)
Close the window or press Ctrl+C in Terminal to stop.
"""
import argparse
import asyncio
import csv
import os
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

from recorder import Book, stream


def wilson(k, n, z=1.96):
    """95% confidence interval for a hit rate."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


class State:
    def __init__(self, depth, window_s, interval, horizon, threshold, fee_bps):
        self.lock = threading.Lock()
        self.depth, self.interval = depth, interval
        self.horizon, self.threshold, self.fee_bps = horizon, threshold, fee_bps
        self.snaps = deque(maxlen=int(window_s / interval))
        self.pending = deque()
        self.results = deque(maxlen=5000)   # (t0, mid0, side, gross, net, spread, hit or None)
        self.last_pred_ts = 0.0
        self.status = "connecting..."
        os.makedirs("logs", exist_ok=True)
        self.log_path = time.strftime("logs/predictions_%Y%m%d_%H%M%S.csv")
        self.log_fh = open(self.log_path, "w", newline="")
        self.log = csv.writer(self.log_fh)
        self.log.writerow(["time", "horizon_s", "side", "l1_imbalance", "mid_start", "mid_end",
                           "move_bps", "gross_bps", "spread_bps", "net_bps", "hit"])

    def sample(self, book):
        r = book.row()
        now = time.time()
        if r is None:
            return
        N = self.depth
        bpx, bq, apx, aq = (np.array(r[i * N:(i + 1) * N]) for i in range(4))
        mid = (bpx[0] + apx[0]) / 2
        spread_bps = (apx[0] - bpx[0]) / mid * 1e4
        imb1 = (bq[0] - aq[0]) / (bq[0] + aq[0])
        b5, a5 = bq[:5].sum(), aq[:5].sum()
        imb5 = (b5 - a5) / (b5 + a5)
        with self.lock:
            self.status = "live"
            self.snaps.append((now, mid, imb5, bpx, bq, apx, aq))
            # resolve predictions whose horizon has passed
            while self.pending and now - self.pending[0][0] >= self.horizon:
                t0, mid0, side, spr0, sig0 = self.pending.popleft()
                move = (mid / mid0 - 1) * 1e4
                gross = side * move
                net = gross - spr0 - 2 * self.fee_bps
                hit = None if move == 0 else bool(np.sign(move) == side)
                self.results.append((t0, mid0, side, gross, net, spr0, hit))
                self.log.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), self.horizon,
                                   "UP" if side > 0 else "DOWN", f"{sig0:.3f}", mid0, mid, f"{move:.3f}",
                                   f"{gross:.3f}", f"{spr0:.3f}", f"{net:.3f}", "" if hit is None else int(hit)])
                self.log_fh.flush()
            # new prediction: strong L1 imbalance, one at a time per horizon
            if abs(imb1) >= self.threshold and now - self.last_pred_ts >= self.horizon:
                self.pending.append((now, mid, 1 if imb1 > 0 else -1, spread_bps, imb1))
                self.last_pred_ts = now


def run_feed(state, book, symbol, depth):
    async def sampler():
        while True:
            state.sample(book)
            await asyncio.sleep(state.interval)

    async def main():
        await asyncio.gather(stream(book, symbol, depth, stop_at=float("inf")), sampler())

    asyncio.run(main())


def build_dashboard(state, symbol):
    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#111")
    gs = fig.add_gridspec(3, 2, width_ratios=[4, 1.7], height_ratios=[3, 1, 0.05], hspace=0.25, wspace=0.08)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_i = fig.add_subplot(gs[1, 0], sharex=ax_h)
    ax_t = fig.add_subplot(gs[0:2, 1])
    for ax in (ax_h, ax_i, ax_t):
        ax.set_facecolor("#111")
        ax.tick_params(colors="#bbb")
        for s in ax.spines.values():
            s.set_color("#444")
    ax_t.axis("off")
    fig.suptitle(f"{symbol} live order book  |  paper predictions {state.horizon}s ahead", color="white")

    def update(_):
        with state.lock:
            snaps = list(state.snaps)
            results = list(state.results)
            pending = list(state.pending)
            n_pending, status = len(pending), state.status
        ax_h.clear(); ax_i.clear(); ax_t.clear(); ax_t.axis("off")
        if len(snaps) < 5:
            ax_t.text(0, 0.9, status, color="white", fontsize=14, transform=ax_t.transAxes)
            return
        now = snaps[-1][0]
        t = np.array([s[0] - now for s in snaps])
        mid = np.array([s[1] for s in snaps])
        imb = np.array([s[2] for s in snaps])
        px = np.concatenate([np.concatenate([s[3], s[5]]) for s in snaps])
        qty = np.concatenate([np.concatenate([s[4], s[6]]) for s in snaps])
        tt = np.repeat(t, 2 * state.depth)

        lo, hi = np.percentile(px, [1, 99])
        lo, hi = min(lo, mid.min()), max(hi, mid.max())
        tb = min(240, len(snaps))
        H, xe, ye = np.histogram2d(tt, px, bins=[tb, 120], range=[[t[0], 0], [lo, hi]], weights=qty)
        cnt, _ = np.histogram(t, bins=xe)
        H = H / np.maximum(cnt, 1)[:, None]
        H[H <= 0] = np.nan
        if np.isfinite(H).any():
            ax_h.pcolormesh(xe, ye, H.T, cmap="magma", norm=LogNorm(), shading="auto")
        ax_h.plot(t, mid, color="#4fc3f7", lw=1.2)
        # mark each call: triangle up/down, green = right, red = wrong, grey = flat or still pending
        calls = [(r[0], r[1], r[2], r[6]) for r in results] + [(p[0], p[1], p[2], "pending") for p in pending]
        for t0, m0, sd, hit in calls:
            if t0 - now < t[0]:
                continue
            c = "#9e9e9e" if hit is None or hit == "pending" else ("#00e676" if hit else "#ff1744")
            ax_h.scatter(t0 - now, m0, marker="^" if sd > 0 else "v", s=70, color=c,
                         edgecolors="white" if hit == "pending" else "black", linewidths=0.8, zorder=5)
        ax_h.set_ylim(lo, hi)
        ax_h.set_ylabel("price", color="#bbb")
        ax_h.set_title(f"mid {mid[-1]:,.2f}   (brighter = more resting size)", color="#ddd", fontsize=10)

        ax_i.fill_between(t, imb, 0, where=imb >= 0, color="#2e7d32", lw=0)
        ax_i.fill_between(t, imb, 0, where=imb < 0, color="#c62828", lw=0)
        ax_i.axhline(0, color="#666", lw=0.6)
        ax_i.set_ylim(-1, 1)
        ax_i.set_ylabel("top-5 imbalance", color="#bbb")
        ax_i.set_xlabel("seconds ago", color="#bbb")
        for ax in (ax_h, ax_i):
            ax.set_facecolor("#111"); ax.tick_params(colors="#bbb")

        cur = imb[-1]
        side = "BUY pressure" if cur > 0.15 else "SELL pressure" if cur < -0.15 else "balanced"
        col = "#66bb6a" if cur > 0.15 else "#ef5350" if cur < -0.15 else "#bbb"
        lines = [("NOW", "#888", 11), (f"{side}", col, 20), (f"imbalance {cur:+.2f}", col, 12), ("", "#111", 8)]
        if results:
            g = np.array([r[3] for r in results]); nt = np.array([r[4] for r in results])
            spr = np.array([r[5] for r in results])
            hits = [r[6] for r in results if r[6] is not None]
            hr = np.mean(hits) * 100 if hits else float("nan")
            lo_ci, hi_ci = wilson(sum(hits), len(hits))
            real = lo_ci > 0.5
            breakeven = (g.mean() - spr.mean()) / 2   # max fee per side that still breaks even
            lines += [
                ("SCOREBOARD", "#888", 11),
                (f"predictions  {len(results)}  (+{n_pending} pending)", "white", 11),
                (f"hit rate     {hr:.1f}%", "white", 14),
                (f"95% range    {lo_ci*100:.0f}% - {hi_ci*100:.0f}%  (n={len(hits)})", "white", 10),
                ("  -> better than a coin flip" if real else "  -> not proven yet (range includes 50%)",
                 "#66bb6a" if real else "#ffb74d", 10),
                ("", "#111", 6),
                (f"avg captured {g.mean():+.2f} bps/trade", "white", 11),
                (f"paper P&L    {g.sum():+.1f} bps gross", "#66bb6a" if g.sum() > 0 else "#ef5350", 11),
                ("", "#111", 6),
                ("COST CHECK", "#888", 11),
                (f"break-even fee  {breakeven:+.2f} bps/side", "white", 11),
                (f"your fee        {state.fee_bps:.2f} bps/side", "white", 11),
                (f"after fees      {nt.mean():+.1f} bps/trade", "#ef5350" if nt.mean() < 0 else "#66bb6a", 11),
                ("", "#111", 6),
                ("^ v = calls  green right  red wrong", "#888", 9),
                (f"log: {state.log_path}", "#666", 8),
            ]
        else:
            lines += [("SCOREBOARD", "#888", 11), (f"waiting for strong signal (|L1| >= {state.threshold})", "#bbb", 10),
                      (f"{n_pending} pending", "#bbb", 10)]
        y = 0.97
        for txt, c, fs in lines:
            ax_t.text(0.02, y, txt, color=c, fontsize=fs, family="monospace", transform=ax_t.transAxes, va="top")
            y -= 0.012 + fs * 0.0036

    anim = FuncAnimation(fig, update, interval=500, cache_frame_data=False)
    return fig, anim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--depth", type=int, default=25, choices=[10, 25, 100])
    ap.add_argument("--window", type=float, default=120, help="seconds of history on screen")
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--horizon", type=float, default=2, help="seconds ahead to predict")
    ap.add_argument("--threshold", type=float, default=0.5, help="min |L1 imbalance| to make a prediction")
    ap.add_argument("--fee-bps", type=float, default=40)
    args = ap.parse_args()

    state = State(args.depth, args.window, args.interval, args.horizon, args.threshold, args.fee_bps)
    book = Book(args.depth)
    threading.Thread(target=run_feed, args=(state, book, args.symbol, args.depth), daemon=True).start()
    fig, anim = build_dashboard(state, args.symbol)
    plt.show()


if __name__ == "__main__":
    main()
