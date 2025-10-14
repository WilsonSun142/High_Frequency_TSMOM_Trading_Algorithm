# trade_algorithm_tsmom_multistock.py
# EDUCATIONAL PURPOSES ONLY — USE IBKR PAPER TRADING

from __future__ import annotations
from ib_insync import *
import numpy as np
import time, signal
from collections import deque, defaultdict
from datetime import datetime
import pytz

# ========= CONNECTION =========
HOST, PORT, CLIENT_ID = '127.0.0.1', 7497, 909

# ========= SLEEVE / RISK =========
HFT_SLEEVE_USD          = 150_000
PER_TRADE_RISK_USD      = 60
DAILY_LOSS_LIMIT        = -0.01 * HFT_SLEEVE_USD
OPEN_NOTIONAL_MAX       = 150_000

# ========= COST / TURNOVER GUARDS =========
DRY_RUN                 = True
MIN_SECONDS_BETWEEN_TR  = 8.0          # per symbol
MAX_TRADES_PER_MIN_SYM  = 3            # per symbol
SPREAD_SKIP_TICKS       = 2.0          # skip if spread > 2 ticks ($0.02)
TICK                    = 0.01
PER_TRADE_NOTIONAL_MAX  = 10_000
MAX_SHARES_PER_TRADE    = 3000

# ========= TSMOM SETTINGS =========
LOOKBACK_SAMPLES  = 120                # loop sleeps ~0.05s
EWMA_ALPHA        = 0.2
Z_WINDOW          = 300
Z_ENTRY           = 1.5
Z_EXIT            = 0.5
ALLOW_SHORT       = True
PRINT_EVERY       = 2.0

# ========= UNIVERSE =========
SYMBOLS      = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'SPY']
EXCHANGE, CCY = 'SMART', 'USD'

# ========= TIMEZONE =========
syd = pytz.timezone('Australia/Sydney')
def now_syd(): return datetime.now(tz=syd)

# ========= RUN TAG (scopes orders to this execution) =========
RUN_TAG = f"TSMOM-{int(time.time())}"  # e.g., "TSMOM-173..."; used in orderRef

# ========= HELPERS =========
def size_for_risk_stock(risk_usd, sl_dollars):
    if sl_dollars <= 0: return 0
    return max(1, int(risk_usd / sl_dollars))

def opposite(side): return 'SELL' if side.upper()=='BUY' else 'BUY'

def _is_working(status: str) -> bool:
    # "Inactive" can appear briefly for children; include Pending states.
    return status in {'Submitted','PreSubmitted','PendingSubmit','ApiPending','PendingCancel','Inactive'}

def our_trades_for(ib: IB, contract: Contract, order_ref_prefix: str):
    """
    Return working trades for this contract created by *this run* (orderRef startswith RUN_TAG).
    """
    out = []
    for tr in ib.trades():
        try:
            if tr.contract.conId != contract.conId:
                continue
            ref = getattr(tr.order, 'orderRef', '') or ''
            if ref.startswith(order_ref_prefix):
                st = tr.orderStatus.status if tr.orderStatus else ''
                if _is_working(st):
                    out.append(tr)
        except Exception:
            pass
    return out

def cancel_our_side(ib: IB, contract: Contract, order_ref_prefix: str, side: str):
    side = side.upper()
    for tr in our_trades_for(ib, contract, order_ref_prefix):
        try:
            if tr.order.action.upper() == side:
                ib.cancelOrder(tr.order)
        except Exception:
            pass

def flatten_ours(ib: IB, contract: Contract, qty: int, side: str, order_ref_prefix: str, dry_run: bool):
    """
    Close *our* exposure only:
      1) cancel our working orders on the closing side,
      2) send a market order with our orderRef to offset our position.
    """
    side = side.upper()
    cancel_our_side(ib, contract, order_ref_prefix, side)
    ib.sleep(0.25)
    if dry_run:
        print(f"[DRY] Flatten ours via {side} {qty}")
        return
    mkt = MarketOrder(side, qty)
    mkt.orderRef = order_ref_prefix + ":FLAT"
    ib.placeOrder(contract, mkt)

# ========= SIGNAL: TSMOM =========
class ReturnMomentum:
    """EWMA of lookback log return; Z-scored over rolling window."""
    def __init__(self, lookback=LOOKBACK_SAMPLES, alpha=EWMA_ALPHA, z_window=Z_WINDOW):
        self.lookback = lookback
        self.alpha = alpha
        self.z_window = z_window
        self.ewma_ret = None
        self.hist = []

    def update(self, prices):
        if len(prices) < self.lookback + 1:
            return None
        m_now  = prices[-1]
        m_then = prices[-(self.lookback+1)]
        r = np.log(m_now) - np.log(m_then)
        self.ewma_ret = r if self.ewma_ret is None else (1-self.alpha)*self.ewma_ret + self.alpha*r
        self.hist.append(self.ewma_ret)
        if len(self.hist) > self.z_window:
            self.hist = self.hist[-self.z_window:]
        mu = float(np.mean(self.hist))
        sd = float(np.std(self.hist)) if len(self.hist) > 1 else 1e-9
        return (self.ewma_ret - mu) / (sd if sd > 1e-9 else 1.0)

def tsmom_decision(z):
    if z is None: return 'HOLD'
    if z >= Z_ENTRY: return 'LONG'
    if ALLOW_SHORT and z <= -Z_ENTRY: return 'SHORT'
    return 'HOLD'

# ========= ORDER HELPERS =========
def make_bracket(ib: IB, side: str, qty: int, entry: float, tp_cents: float, sl_cents: float, order_ref: str):
    """
    Create a parent market + TP limit + SL stop bracket. All three are stamped with orderRef.
    """
    side = side.upper()
    opp  = opposite(side)
    tp_px = round(entry + (tp_cents/100.0)*(+1 if side=='BUY' else -1), 2)
    sl_px = round(entry - (sl_cents/100.0)*(+1 if side=='BUY' else -1), 2)

    parent   = MarketOrder(side, qty)
    takeProf = LimitOrder(opp, qty, lmtPrice=tp_px, tif='GTC')
    stopLoss = StopOrder(opp, qty, stopPrice=sl_px, tif='GTC')

    # ***** tag orders to this run *****
    for o in (parent, takeProf, stopLoss):
        o.orderRef = order_ref

    parent.orderId   = ib.client.getReqId()
    takeProf.orderId = ib.client.getReqId()
    stopLoss.orderId = ib.client.getReqId()
    takeProf.parentId = parent.orderId
    stopLoss.parentId = parent.orderId
    parent.transmit = False
    takeProf.transmit = False
    stopLoss.transmit = True
    return parent, takeProf, stopLoss

# ========= TURNOVER LIMITER =========
class TurnoverGuard:
    def __init__(self):
        self.by_sym = defaultdict(deque)
    def allow(self, symbol):
        dq = self.by_sym[symbol]
        now = time.time()
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= MAX_TRADES_PER_MIN_SYM:
            return False
        dq.append(now)
        return True

# ========= ROBUST HISTORICAL BACKFILL (HMDS) =========
def _req_hist(ib: IB, c: Contract, duration: str, bar_size: str, what: str) -> list[float]:
    bars = ib.reqHistoricalData(
        c, endDateTime='', durationStr=duration, barSizeSetting=bar_size,
        whatToShow=what, useRTH=False, formatDate=1, keepUpToDate=False
    )
    mids = []
    for b in bars:
        if getattr(b, 'high', None) and getattr(b, 'low', None):
            mids.append(float((b.high + b.low)/2.0))
        elif getattr(b, 'close', None):
            mids.append(float(b.close))
    return mids

def backfill_symbol(ib: IB, c: Contract, need_points: int) -> list[float]:
    attempts = [
        ('45 M',  '5 secs', 'MIDPOINT'),
        ('60 M',  '5 secs', 'TRADES'),
        ('6 H',   '1 min',  'MIDPOINT'),
        ('6 H',   '1 min',  'TRADES'),
        ('2 D',   '5 mins', 'MIDPOINT'),
        ('2 D',   '5 mins', 'TRADES'),
    ]
    for duration, bar_size, what in attempts:
        try:
            mids = _req_hist(ib, c, duration, bar_size, what)
            if mids:
                return mids[-need_points:]
        except Exception as e:
            print(f"[backfill warn] {c.symbol} {duration}/{bar_size}/{what}: {e}")
            time.sleep(0.2)
    print(f"[backfill fail] {c.symbol}: no historical mids.")
    return []

def backfill_all(ib: IB, contracts: dict[str, Contract], mid_buf: dict[str, deque], need_points: int):
    print(f"Backfilling ~{need_points} mids per symbol from HMDS…")
    for s, c in contracts.items():
        mids = backfill_symbol(ib, c, need_points)
        if mids:
            for px in mids:
                mid_buf[s].append(px)
            print(f"  • {s}: loaded {len(mids)} points")
        else:
            print(f"  • {s}: no data (will build from live)")

# ========= MARKET DATA (prefer RT → fallback to DELAYED if permitted) =========
class MarketData:
    def __init__(self, ib: IB, contracts: dict[str, Contract]):
        self.ib = ib
        self.contracts = contracts
        self.tickers: dict[str, Ticker] = {}
        self.haveDelayed = False
        ib.errorEvent += self._on_error

    def _on_error(self, reqId, code, msg, contract):
        # Switch to delayed when allowed and RT is not subscribed
        if code in (354, 10167, 10168):
            if not self.haveDelayed:
                print(f"⚠️ {code} → switching to DELAYED quotes.")
                self.ib.reqMarketDataType(3)
                self.haveDelayed = True
                for s in list(self.tickers.keys()):
                    try: self.ib.cancelMktData(self.tickers[s])
                    except Exception: pass
                    self.tickers[s] = self.ib.reqMktData(self.contracts[s], '', False, False)

    def start(self):
        # Try REALTIME first
        self.ib.reqMarketDataType(1)
        for s, c in self.contracts.items():
            self.tickers[s] = self.ib.reqMktData(c, '', False, False)
        # Brief wait for RT; if any symbol lacks bid/ask/last, try delayed
        t0 = time.time()
        while time.time() - t0 < 2.0:
            if all((t.bid and t.ask) or t.last or t.close for t in self.tickers.values()):
                return
            self.ib.sleep(0.1)
        print("No usable RT quotes — attempting DELAYED.")
        self.ib.reqMarketDataType(3)
        self.haveDelayed = True
        for s, c in self.contracts.items():
            try: self.ib.cancelMktData(self.tickers[s])
            except Exception: pass
            self.tickers[s] = self.ib.reqMktData(c, '', False, False)
        # one more short wait
        t1 = time.time()
        ok = False
        while time.time() - t1 < 2.0:
            if all(t.last or t.close or (t.bid and t.ask) for t in self.tickers.values()):
                ok = True; break
            self.ib.sleep(0.1)
        if not ok:
            # Delayed not enabled in account
            print("❌ Fatal: no usable market data (RT nor Delayed). Enable market data or Delayed quotes.")
            raise RuntimeError("No market data")

    def snapshot(self, symbol):
        t = self.tickers.get(symbol)
        if not t:
            return None, None, None
        # Prefer bid/ask
        if t.bid and t.ask and t.bid > 0 and t.ask > 0:
            bid, ask = float(t.bid), float(t.ask)
            return 0.5*(bid+ask), bid, ask
        # Fallback to last/close for delayed gaps
        if t.last and float(t.last) > 0:
            px = float(t.last); return px, None, None
        if t.close and float(t.close) > 0:
            px = float(t.close); return px, None, None
        return None, None, None

    def stop(self):
        for t in self.tickers.values():
            try: self.ib.cancelMktData(t)
            except Exception: pass

# ========= MAIN =========
def main():
    def _sigint(_sig, _frm): raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, _sigint)

    print(f"[{now_syd()}] Connecting to IBKR paper on {HOST}:{PORT} …")

    # Quiet repetitive perms errors (MarketData.start handles switching)
    def _quiet(reqId, code, msg, contract):
        if code in (10168, 354, 10167): return
        print(f"IB ERR {code} (reqId {reqId}): {msg}")

    ib = IB()
    ib.errorEvent += _quiet

    if not ib.connect(HOST, PORT, clientId=CLIENT_ID):
        print("Connect failed. Open TWS (paper), enable API, port 7497, localhost allowed.")
        return

    # Contracts
    contracts = {}
    for s in SYMBOLS:
        [qc] = ib.qualifyContracts(Stock(s, EXCHANGE, CCY))
        contracts[s] = qc

    # Buffers & models
    mid_buf  = {s: deque(maxlen=max(LOOKBACK_SAMPLES+2, Z_WINDOW+10)) for s in SYMBOLS}
    zmodel   = {s: ReturnMomentum() for s in SYMBOLS}
    # Position (DRY-RUN shadow only; in live you’d read from IB positions)
    position = {s: 0 for s in SYMBOLS}
    last_ts  = {s: 0.0 for s in SYMBOLS}
    guard    = TurnoverGuard()
    pnl_realized = 0.0
    last_log = 0.0

    # Market data (prefer RT → fallback delayed if available)
    md = MarketData(ib, contracts)
    try:
        md.start()
    except RuntimeError:
        md.stop()
        ib.disconnect()
        return

    # Backfill so TSMOM can compute immediately (≈ LOOKBACK + Z_WINDOW)
    need_points = LOOKBACK_SAMPLES + Z_WINDOW
    backfill_all(ib, contracts, mid_buf, need_points)

    print(f"RUN_TAG={RUN_TAG}")
    print("Running (Ctrl+C to stop)…")
    try:
        while True:
            ib.sleep(0.05)

            # heartbeat log
            if time.time() - last_log >= PRINT_EVERY:
                row = []
                for s in SYMBOLS:
                    z = zmodel[s].update(list(mid_buf[s])) if len(mid_buf[s]) else None
                    row.append(f"{s}:{(z if z is not None else float('nan')):+.2f}")
                print(f"[{now_syd()}] " + " | ".join(row) + f" | PnL={pnl_realized:.2f}")
                last_log = time.time()

            for s in SYMBOLS:
                mid, bid, ask = md.snapshot(s)
                if not mid:
                    continue
                mid_buf[s].append(mid)

                # spread filter (only if we have bid/ask)
                if bid and ask:
                    spread_ticks = (ask - bid)/TICK
                    if spread_ticks > SPREAD_SKIP_TICKS:
                        continue

                z = zmodel[s].update(list(mid_buf[s]))
                sig = tsmom_decision(z)

                # cooldown / turnover guard
                if time.time() - last_ts[s] < MIN_SECONDS_BETWEEN_TR:
                    continue
                if sig == 'HOLD':
                    # exit rule handled below if already in position
                    pass
                if not guard.allow(s):
                    continue

                # simple vol-based TP/SL (in cents)
                VOL_WIN = 80
                tp_cents, sl_cents = 4.0, 3.0
                if len(mid_buf[s]) >= VOL_WIN:
                    arr = np.array(list(mid_buf[s])[-VOL_WIN:])
                    vol_ticks = max(0.2, min(3.0, float(np.std(np.diff(arr)))/TICK))
                    tp_cents = max(1.0, min(6.0, 2.0 * vol_ticks))
                    sl_cents = max(1.0, min(5.0, 1.5 * vol_ticks))

                # shares from risk & notional cap
                sl_dollars = sl_cents/100.0
                shares_risk = size_for_risk_stock(PER_TRADE_RISK_USD, sl_dollars)
                shares_notional = max(1, int(PER_TRADE_NOTIONAL_MAX // max(mid, 0.01)))
                qty = max(1, min(shares_risk, shares_notional, MAX_SHARES_PER_TRADE))
                if qty < 5:
                    continue

                # sleeve exposure guard (naive per-symbol)
                open_notional = abs(position[s]) * mid
                if open_notional + qty*mid > OPEN_NOTIONAL_MAX:
                    continue

                state = 'LONG' if position[s] > 0 else ('SHORT' if position[s] < 0 else 'FLAT')
                order_ref_prefix = RUN_TAG                 # prefix matches our filter
                order_ref = f"{RUN_TAG}:{s}"               # per-symbol tag

                # Exit on neutrality band — only flatten *ours*
                if abs(position[s]) > 0 and z is not None and abs(z) <= Z_EXIT:
                    side_exit = 'SELL' if position[s] > 0 else 'BUY'
                    qty_exit  = abs(position[s])
                    flatten_ours(ib, contracts[s], qty_exit, side_exit, order_ref_prefix, DRY_RUN)
                    position[s] = 0  # DRY bookkeeping
                    last_ts[s] = time.time()
                    continue

                # Decide entry/flip; ensure we don't duplicate *our* same-side working orders
                if sig == 'LONG' and state != 'LONG':
                    # flip from short? flatten ours first (only our exposure)
                    if state == 'SHORT' and position[s] < 0:
                        flatten_ours(ib, contracts[s], qty=min(qty, -position[s]), side='BUY',
                                     order_ref_prefix=order_ref_prefix, dry_run=DRY_RUN)
                    # block duplicate BUYs from this run
                    if any(tr.order.action.upper() == 'BUY' for tr in our_trades_for(ib, contracts[s], order_ref_prefix)):
                        continue
                    side = 'BUY'

                elif sig == 'SHORT' and state != 'SHORT':
                    if not ALLOW_SHORT:
                        continue
                    # flip from long? flatten ours first (only our exposure)
                    if state == 'LONG' and position[s] > 0:
                        flatten_ours(ib, contracts[s], qty=min(qty, position[s]), side='SELL',
                                     order_ref_prefix=order_ref_prefix, dry_run=DRY_RUN)
                    # block duplicate SELLs from this run
                    if any(tr.order.action.upper() == 'SELL' for tr in our_trades_for(ib, contracts[s], order_ref_prefix)):
                        continue
                    side = 'SELL'
                else:
                    continue  # nothing to do

                # place (DRY RUN prints only)
                parent, tp, sl = make_bracket(ib, side, qty, mid, tp_cents, sl_cents, order_ref=order_ref)
                stop_px = getattr(sl, 'auxPrice', getattr(sl, 'stopPrice', None))
                print(f"✅ {s} {side} {qty} @~{mid:.2f} | TP {tp.lmtPrice} | SL {stop_px} (tp={tp_cents:.1f}c, sl={sl_cents:.1f}c) ref={order_ref}")
                if not DRY_RUN:
                    ib.placeOrder(contracts[s], parent); ib.placeOrder(contracts[s], tp); ib.placeOrder(contracts[s], sl)
                # DRY bookkeeping (simulation feel)
                position[s] += (qty if side=='BUY' else -qty)
                last_ts[s] = time.time()

            # demo daily loss stop (realized not tracked in DRY_RUN)
            if pnl_realized <= DAILY_LOSS_LIMIT:
                pass

    except KeyboardInterrupt:
        print("\nStopping… closing any open positions.")
        for s in SYMBOLS:
            if abs(position[s]) > 0:
                print(f"Closing {s} (simulated).")
                position[s] = 0
    finally:
        try:
            md.stop()
        except Exception:
            pass
        ib.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()
