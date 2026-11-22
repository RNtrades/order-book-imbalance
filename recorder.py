"""
Record a live order book from Kraken's public websocket (no account needed).

Usage:
    python recorder.py --symbol BTC/USD --depth 25 --interval 0.1 --minutes 120

Writes a gzipped CSV to data/ with one row per sample:
    ts, bid_px_1..N, bid_qty_1..N, ask_px_1..N, ask_qty_1..N
Leave it running for at least 1-2 hours before analyzing. More data = better.
"""
import argparse
import asyncio
import csv
import gzip
import json
import os
import time
from datetime import datetime, timezone

import websockets

URL = "wss://ws.kraken.com/v2"
STALE_S = 5      # don't save the book if it hasn't updated for this long
RECONNECT_S = 10  # reconnect if the book has been silent this long


class Book:
    def __init__(self, depth):
        self.depth = depth
        self.bids = {}
        self.asks = {}
        self.ready = False
        self.last_update = time.time()

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self.ready = False

    @staticmethod
    def _apply(levels, side):
        for lvl in levels:
            p, q = float(lvl["price"]), float(lvl["qty"])
            if q == 0:
                side.pop(p, None)
            else:
                side[p] = q

    def _trim(self):
        # Kraken requires trimming the book back to the subscribed depth after each update
        for side, rev in ((self.bids, True), (self.asks, False)):
            if len(side) > self.depth:
                for p in sorted(side, reverse=rev)[self.depth:]:
                    del side[p]

    def on_message(self, msg):
        self.last_update = time.time()
        for d in msg.get("data", []):
            if msg["type"] == "snapshot":
                self.reset()
            self._apply(d.get("bids", []), self.bids)
            self._apply(d.get("asks", []), self.asks)
            self._trim()
        if msg["type"] == "snapshot":
            self.ready = True

    def row(self):
        if time.time() - self.last_update > STALE_S:  # watchdog: frozen feed, don't record it
            return None
        if not self.ready or len(self.bids) < self.depth or len(self.asks) < self.depth:
            return None
        b = sorted(self.bids.items(), reverse=True)
        a = sorted(self.asks.items())
        if b[0][0] >= a[0][0]:  # crossed book = bad state, skip
            return None
        return ([p for p, _ in b] + [q for _, q in b] +
                [p for p, _ in a] + [q for _, q in a])


async def stream(book, symbol, depth, stop_at):
    while time.time() < stop_at:
        try:
            async with websockets.connect(URL, ping_interval=20, max_size=2**22) as ws:
                await ws.send(json.dumps({"method": "subscribe",
                                          "params": {"channel": "book", "symbol": [symbol], "depth": depth}}))
                print(f"connected, subscribed to {symbol} depth {depth}")
                book.last_update = time.time()
                while True:
                    # watchdog 1: connection dead (not even heartbeats) -> TimeoutError -> reconnect
                    raw = await asyncio.wait_for(ws.recv(), timeout=RECONNECT_S)
                    msg = json.loads(raw)
                    if msg.get("channel") == "book":
                        book.on_message(msg)
                    # watchdog 2: connection alive but the book stopped updating -> reconnect
                    if time.time() - book.last_update > RECONNECT_S:
                        raise RuntimeError(f"no book updates for {RECONNECT_S}s (feed frozen)")
                    if time.time() >= stop_at:
                        return
        except Exception as e:
            print(f"disconnected ({e!r}); reconnecting in 3s")
            book.reset()  # never sample a stale book across a reconnect
            await asyncio.sleep(3)


async def sampler(book, writer, fh, interval, stop_at):
    n, last_flush = 0, time.time()
    while time.time() < stop_at:
        r = book.row()
        if r is not None:
            writer.writerow([f"{time.time():.3f}"] + r)
            n += 1
        if time.time() - last_flush > 10:
            fh.flush()
            last_flush = time.time()
            print(f"{n} samples recorded")
        await asyncio.sleep(interval)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--depth", type=int, default=25, choices=[10, 25, 100])
    ap.add_argument("--interval", type=float, default=0.1, help="seconds between samples")
    ap.add_argument("--minutes", type=float, default=120)
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"data/{args.symbol.replace('/', '')}_{stamp}.csv.gz"
    N = args.depth
    header = (["ts"] + [f"bid_px_{i}" for i in range(1, N + 1)] + [f"bid_qty_{i}" for i in range(1, N + 1)] +
              [f"ask_px_{i}" for i in range(1, N + 1)] + [f"ask_qty_{i}" for i in range(1, N + 1)])

    stop_at = time.time() + args.minutes * 60
    book = Book(N)
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        print(f"writing to {path} (Ctrl+C to stop early; data so far is kept)")
        try:
            await asyncio.gather(stream(book, args.symbol, N, stop_at),
                                 sampler(book, w, fh, args.interval, stop_at))
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
    print("done:", path)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped")
