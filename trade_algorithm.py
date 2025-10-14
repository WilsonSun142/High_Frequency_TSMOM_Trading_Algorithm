# trade_algorithm.py
# EDUCATIONAL PURPOSES ONLY — USE IBKR PAPER TRADING

from __future__ import annotations
from ib_insync import *
import numpy as np
import time, signal
from collections import deque, defaultdict
from datetime import datetime
import pytz, math

# ========= CONNECTION =========
HOST, PORT, CLIENT_ID = '127.0.0.1', 7497, 909

# ========= SLEEVE / RISK =========
HFT_SLEEVE_USD          = 150_000
PER_TRADE_RISK_USD      = 60
DAILY_LOSS_LIMIT        = -0.01 * HFT_SLEEVE_USD   # applies to *session* PnL once armed
OPEN_NOTIONAL_MAX       = 150_000

# ========= COST / TURNOVER GUARDS =========
DRY_RUN                 = False                     # set False to place paper orders
MIN_SECONDS_BETWEEN_TR  = 8.0                       # per symbol (entries only)
MAX_TRADES_PER_MIN_SYM  = 3                         # per symbol (entries only)
SPREAD_SKIP_TICKS       = 50.0                      # relaxed while on delayed quotes
DEFAULT_TICK            = 0.01
PER_TRADE_NOTIONAL_MAX  = 2_000                    # per-entry notional cap (tight for testing)
MAX_SHARES_PER_TRADE    = 2                    # hard cap ~ 1,000 shares

# ========= TSMOM SETTINGS =========
LOOKBACK_SAMPLES  = 120
EWMA_ALPHA        = 0.2
Z_WINDOW          = 300
Z_ENTRY           = 1.5
Z_EXIT            = 0.5
ALLOW_SHORT       = True                           # <— NO SHORTS
PRINT_EVERY       = 2.0
REQUIRE_RTH       = True                            # trade only US regular hours

# ========= UNIVERSE =========
SYMBOLS      = ['META', 'AMZN', 'AAPL']
EXCHANGE, CCY = 'SMART', 'USD'

# ========= TIMEZONE =========
syd = pytz.timezone('Australia/Sydney')
def now_syd(): return datetime.now(tz=syd)

# ========= RUN TAG (scopes all our orders) =========
RUN_TAG = f"TSMOM-{int(time.time())}"

# ========= HELPERS =========
def size_for_risk_stock(risk_usd, sl_dollars):
    if sl_dollars <= 0: return 0
    return max(1, int(risk_usd / sl_dollars))

def opposite(side): return 'SELL' if side.upper()=='BUY' else 'BUY'

def _is_working(status: str) -> bool:
    return status in {'Submitted','PreSubmitted','PendingSubmit','ApiPending','PendingCancel','Inactive'}

def our_trades_for(ib: IB, contract: Contract, order_ref_prefix: str):
    """Return working trades for this contract created by *this run* (orderRef startswith RUN_TAG)."""
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

def cancel_ours(ib: IB, contract: Contract, order_ref_prefix: str):
    """Cancel *our* working orders only (don’t touch manual orders)."""
    for tr in our_trades_for(ib, contract, order_ref_prefix):
        try:
            ib.cancelOrder(tr.order)
        except Exception:
            pass

# ========= OUR-POSITION TRACKING (by orderRef) =========
# We track *this run’s* filled quantity per symbol so we can exit only what we opened.
our_pos: dict[str, int] = defaultdict(int)

def _botorbuy(side: str) -> bool:
    s = (side or '').upper()
    return s in ('BOT', 'BUY')

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
        if m_now <= 0 or m_then <= 0:
            return None
        r = math.log(m_now) - math.log(m_then)
        self.ewma_ret = r if self.ewma_ret is None else (1-self.alpha)*self.ewma_ret + self.alpha*r
        self.hist.append(self.ewma_ret)
        if len(self.hist) > self.z_window:
            self.hist = self.hist[-self.z_window:]
        mu = float(np.mean(self.hist)) if self.hist else 0.0
        sd = float(np.std(self.hist)) if len(self.hist) > 1 else 1e-9
        z = (self.ewma_ret - mu) / (sd if sd > 1e-9 else 1.0)
        return z if np.isfinite(z) else None

def tsmom_decision(z):
    if z is None or not np.isfinite(z): return 'HOLD'
    if z >= Z_ENTRY: return 'LONG'
    return 'HOLD'  # no shorts

# ========= ORDER HELPERS =========
def make_bracket(ib: IB, side: str, qty: int, entry: float, tp_cents: float, sl_cents: float, order_ref: str):
    """Parent market + TP limit + SL stop bracket. All stamped with orderRef."""
    side = side.upper()
    opp  = opposite(side)
    tp_px = round(entry + (tp_cents/100.0)*(+1 if side=='BUY' else -1), 2)
    sl_px = round(entry - (sl_cents/100.0)*(+1 if side=='BUY' else -1), 2)

    parent   = MarketOrder(side, qty)
    takeProf = LimitOrder(opp, qty, lmtPrice=tp_px, tif='GTC')
    stopLoss = StopOrder(opp, qty, stopPrice=sl_px, tif='GTC')

    for o in (parent, takeProf, stopLoss):
        o.orderRef = order_ref

    parent.orderId    = ib.client.getReqId()
    takeProf.orderId  = ib.client.getReqId()
    stopLoss.orderId  = ib.client.getReqId()
    takeProf.parentId = parent.orderId
    stopLoss.parentId = parent.orderId
    parent.transmit   = False
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

# ========= TIME FILTER =========
def is_rth_newyork():
    et = pytz.timezone('America/New_York')
    t  = datetime.now(et)
    if t.weekday() >= 5:
        return False
    return (t.hour, t.minute) >= (9,30) and (t.hour, t.minute) < (16, 0)

# ========= HISTORICAL BACKFILL =========
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
        ('2700 S',  '5 secs', 'MIDPOINT'),  # 45 min
        ('3600 S',  '5 secs', 'TRADES'),    # 60 min
        ('21600 S', '1 min',  'MIDPOINT'),  # 6 hours
        ('21600 S', '1 min',  'TRADES'),
        ('2 D',     '5 mins', 'MIDPOINT'),
        ('2 D',     '5 mins', 'TRADES'),
        ('1 W',     '5 mins', 'TRADES'),
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

# ========= MARKET DATA =========
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
        t1 = time.time()
        ok = False
        while time.time() - t1 < 2.0:
            if all(t.last or t.close or (t.bid and t.ask) for t in self.tickers.values()):
                ok = True; break
            self.ib.sleep(0.1)
        if not ok:
            print("❌ Fatal: no usable market data (RT nor Delayed). Enable delayed data & API market data.")
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

# ========= TICK SIZE =========
def get_tick_size(ib: IB, c: Contract) -> float:
    try:
        di = ib.reqContractDetails(c)[0]
        mn = getattr(di, 'minTick', None)
        return float(mn) if mn else DEFAULT_TICK
    except Exception:
        return DEFAULT_TICK

# ========= MAIN =========
def main():
    def _sigint(_sig, _frm): raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, _sigint)

    print(f"[{now_syd()}] Connecting to IBKR paper on {HOST}:{PORT} …")

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

    # ======== LIVE PnL SUBSCRIPTION (robust) ========
    pnl_realized = 0.0
    pnl_unrealized = 0.0
    daily_pnl = 0.0

    accounts = ib.managedAccounts()
    if not accounts:
        try:
            accounts = [v.account for v in ib.accountValues()]
        except Exception:
            accounts = []
    account_id = accounts[0] if accounts else None

    def _pnl_update(realizedPnL=None, unrealizedPnL=None, dailyPnL=None):
        nonlocal pnl_realized, pnl_unrealized, daily_pnl
        if realizedPnL is not None:   pnl_realized = float(realizedPnL)
        if unrealizedPnL is not None: pnl_unrealized = float(unrealizedPnL)
        if dailyPnL is not None:      daily_pnl = float(dailyPnL)

    if account_id:
        def on_pnl(*args):
            if len(args) == 1 and hasattr(args[0], 'realizedPnL'):
                pnl_obj = args[0]
                _pnl_update(
                    realizedPnL=getattr(pnl_obj, 'realizedPnL', None),
                    unrealizedPnL=getattr(pnl_obj, 'unrealizedPnL', None),
                    dailyPnL=getattr(pnl_obj, 'dailyPnL', None),
                )
            elif len(args) >= 5:
                _, _, dailyPnL, unrealizedPnL, realizedPnL = args[:5]
                _pnl_update(realizedPnL=realizedPnL, unrealizedPnL=unrealizedPnL, dailyPnL=dailyPnL)
        ib.pnlEvent += on_pnl
        ib.reqPnL(account_id, '')  # '' => all model codes

    # Tick sizes
    TICKS = {s: (get_tick_size(ib, c) or DEFAULT_TICK) for s, c in contracts.items()}

    # Buffers & models
    mid_buf  = {s: deque(maxlen=max(LOOKBACK_SAMPLES+2, Z_WINDOW+10)) for s in SYMBOLS}
    zmodel   = {s: ReturnMomentum() for s in SYMBOLS}
    last_ts  = {s: 0.0 for s in SYMBOLS}
    guard    = TurnoverGuard()
    last_log = 0.0

    # Market data
    md = MarketData(ib, contracts)
    try:
        md.start()
    except RuntimeError:
        md.stop(); ib.disconnect(); return

    # Backfill
    need_points = LOOKBACK_SAMPLES + Z_WINDOW
    backfill_all(ib, contracts, mid_buf, need_points)

    # ---- Kill-switch baseline (armed only after first order/first fill) ----
    kill_armed = False
    daily_baseline = 0.0

    # Track our fills (inside main so we can arm kill-switch)
    def on_exec(_reqId, contract: Contract, execution: Execution):
        nonlocal kill_armed, daily_baseline
        ref = getattr(execution, 'orderRef', '') or ''
        if not ref.startswith(RUN_TAG):
            return
        sym = contract.symbol
        shares = int(execution.shares or 0)
        if _botorbuy(execution.side):
            our_pos[sym] += shares
        else:
            our_pos[sym] -= shares
        if not kill_armed:
            daily_baseline = float(daily_pnl)
            kill_armed = True
            print(f"🔒 Kill-switch armed. Session baseline set to {daily_baseline:.2f}")

    ib.execDetailsEvent += on_exec

    print(f"RUN_TAG={RUN_TAG}")
    print("Running (Ctrl+C to stop)…")
    try:
        while True:
            ib.sleep(0.05)

            if REQUIRE_RTH and not is_rth_newyork():
                continue

            # Heartbeat
            if time.time() - last_log >= PRINT_EVERY:
                row = []
                for s in SYMBOLS:
                    z = zmodel[s].update(list(mid_buf[s])) if len(mid_buf[s]) else None
                    row.append(f"{s}:{(z if z is not None else float('nan')):+.2f}")
                extra = ""
                if kill_armed:
                    session_daily = daily_pnl - daily_baseline
                    extra = f" | SessionDaily={session_daily:.2f}"
                print(f"[{now_syd()}] " + " | ".join(row) +
                      f" | AccountPnL: Real={pnl_realized:.2f} Unrl={pnl_unrealized:.2f} Daily={daily_pnl:.2f}{extra}")
                last_log = time.time()

            for s in SYMBOLS:
                # build price history
                mid, bid, ask = md.snapshot(s)
                if not mid:
                    continue
                mid_buf[s].append(mid)

                # spread guard only when realtime
                if bid and ask and not md.haveDelayed:
                    tick = TICKS.get(s, DEFAULT_TICK) or DEFAULT_TICK
                    spread_ticks = (ask - bid)/tick
                    if spread_ticks > SPREAD_SKIP_TICKS:
                        continue

                z = zmodel[s].update(list(mid_buf[s]))
                if z is None or len(zmodel[s].hist) < max(20, int(0.2*Z_WINDOW)):
                    continue

                contract = contracts[s]
                my_qty   = int(our_pos.get(s, 0))    # ONLY our book, not manual

                # === EXIT on neutrality: sell only what we opened ===
                if my_qty > 0 and abs(z) <= Z_EXIT:
                    cancel_ours(ib, contract, RUN_TAG)
                    if DRY_RUN:
                        print(f"[DRY] FLATTEN {s}: SELL {my_qty} (ours)")
                    else:
                        mkt = MarketOrder('SELL', my_qty); mkt.orderRef = f"{RUN_TAG}:FLAT"
                        ib.placeOrder(contract, mkt)
                    continue

                # === If model flips to short, just exit longs (no new shorts) ===
                if my_qty > 0 and z <= -Z_ENTRY:
                    cancel_ours(ib, contract, RUN_TAG)
                    if DRY_RUN:
                        print(f"[DRY] FLATTEN {s}: SELL {my_qty} (ours, flip)")
                    else:
                        mkt = MarketOrder('SELL', my_qty); mkt.orderRef = f"{RUN_TAG}:FLIP"
                        ib.placeOrder(contract, mkt)
                    continue

                # === Fresh entry only if we don't already hold from this run ===
                if my_qty != 0:
                    continue
                sig = tsmom_decision(z)
                if sig != 'LONG':   # no shorts
                    continue
                if time.time() - last_ts[s] < MIN_SECONDS_BETWEEN_TR:
                    continue
                if not guard.allow(s):
                    continue

                # Vol-based TP/SL (cents)
                VOL_WIN = 80
                tp_cents, sl_cents = 4.0, 3.0
                if len(mid_buf[s]) >= VOL_WIN + 1:
                    arr = np.array(list(mid_buf[s])[-VOL_WIN:])
                    dif = np.diff(arr)
                    tick = TICKS.get(s, DEFAULT_TICK)
                    vol_ticks = max(0.2, min(3.0, float(np.std(dif))/max(tick, 1e-6)))
                    tp_cents = max(1.0, min(6.0, 2.0 * vol_ticks))
                    sl_cents = max(1.0, min(5.0, 1.5 * vol_ticks))

                # Position sizing
                sl_dollars = sl_cents/100.0
                shares_risk = size_for_risk_stock(PER_TRADE_RISK_USD, sl_dollars)
                shares_notional = max(1, int(PER_TRADE_NOTIONAL_MAX // max(mid, 0.01)))
                qty = max(1, min(shares_risk, shares_notional, MAX_SHARES_PER_TRADE))
                if qty < 5:
                    continue

                # Sleeve exposure guard (only our book)
                open_notional = abs(my_qty) * mid
                if open_notional + qty*mid > OPEN_NOTIONAL_MAX:
                    continue

                # Place bracket (LONG only)
                side = 'BUY'
                order_ref = f"{RUN_TAG}:{s}"
                if any(tr.order.action.upper() == side for tr in our_trades_for(ib, contract, RUN_TAG)):
                    continue

                parent, tp, sl = make_bracket(ib, side, qty, mid, tp_cents, sl_cents, order_ref=order_ref)
                stop_px = getattr(sl, 'auxPrice', getattr(sl, 'stopPrice', None))
                print(f"✅ {s} {side} {qty} @~{mid:.2f} | TP {tp.lmtPrice} | SL {stop_px} "
                      f"(tp={tp_cents:.1f}c, sl={sl_cents:.1f}c) ref={order_ref}")
                if not DRY_RUN:
                    try:
                        ib.placeOrder(contract, parent)
                        ib.placeOrder(contract, tp)
                        ib.placeOrder(contract, sl)
                    except Exception as e:
                        print(f"❌ placeOrder failed for {s}: {e}. Cancelling staged children.")
                        try:
                            if tp.orderId: ib.cancelOrder(tp)
                            if sl.orderId: ib.cancelOrder(sl)
                        except Exception:
                            pass
                        continue

                # Arm kill-switch on first order if it isn’t already (covers very fast fills)
                nonlocal_kill = False
                if not kill_armed:
                    daily_baseline = float(daily_pnl)
                    kill_armed = True
                    print(f"🔒 Kill-switch armed. Session baseline set to {daily_baseline:.2f}")

                last_ts[s] = time.time()

            # Kill-switch: live only; active only once *armed*; flattens *our* positions
            if (not DRY_RUN) and kill_armed:
                session_daily = daily_pnl - daily_baseline
                if session_daily <= DAILY_LOSS_LIMIT:
                    print(f"⛔ Session loss limit hit (SessionDaily={session_daily:.2f} ≤ {DAILY_LOSS_LIMIT:.2f}). Flattening ours & stopping.")
                    for s, c in contracts.items():
                        my_qty = int(our_pos.get(s, 0))
                        if my_qty > 0:
                            cancel_ours(ib, c, RUN_TAG)
                            mkt = MarketOrder('SELL', my_qty); mkt.orderRef = f"{RUN_TAG}:KILL"
                            ib.placeOrder(c, mkt)
                    break

    except KeyboardInterrupt:
        print("\nStopping… CANCEL our orders & FLATTEN our positions.")
        for s, c in contracts.items():
            cancel_ours(ib, c, RUN_TAG)
            my_qty = int(our_pos.get(s, 0))
            if my_qty > 0:
                if DRY_RUN:
                    print(f"[DRY] FLATTEN {s}: SELL {my_qty} (ours, Ctrl+C)")
                else:
                    mkt = MarketOrder('SELL', my_qty); mkt.orderRef = f"{RUN_TAG}:STOP"
                    ib.placeOrder(c, mkt)
    finally:
        try:
            # cleanup PnL subscription
            try:
                if account_id:
                    ib.cancelPnL(account_id, '')
            except Exception:
                pass
            # final safety sweep for our positions only
            for s, c in contracts.items():
                cancel_ours(ib, c, RUN_TAG)
                my_qty = int(our_pos.get(s, 0))
                if my_qty > 0:
                    if DRY_RUN:
                        print(f"[DRY] FLATTEN {s}: SELL {my_qty} (ours, final)")
                    else:
                        mkt = MarketOrder('SELL', my_qty); mkt.orderRef = f"{RUN_TAG}:FINAL"
                        ib.placeOrder(c, mkt)
            md.stop()
        except Exception:
            pass
        ib.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()
