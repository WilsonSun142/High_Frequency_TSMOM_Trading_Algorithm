# --- fx_lob_net_demand_trader_guarded_v2.py ---
# EDUCATIONAL PURPOSES ONLY â€” USE IBKR PAPER TRADING

from ib_insync import *
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pytz

# ========= USER SETTINGS =========
HOST        = '127.0.0.1'
PORT        = 7497            # 7496=live, 7497=paper
CLIENT_ID   = 909

SYMBOL      = 'EURUSD'        # FX pair on IDEALPRO
AMOUNT      = 10_000          # base units, e.g., 10k EUR
LEVELS      = 10              # depth levels to aggregate (per side)
WEIGHTED    = True            # distance-weight by 1/(ticks_from_mid+1)

PIP         = 0.0001          # EUR.USD pip
TP_PIPS     = 2              # take-profit
SL_PIPS     = 1.5               # stop-loss

# Net-demand â†’ signal
ALPHA_EWMA  = 0.2             # EWMA smoothing
TH_LONG     = +2.0            # z-score to go/stay long
TH_SHORT    = -2.0            # z-score to go/stay short
EXIT_BAND   = 0.5             # prefer FLAT when |z| <= EXIT_BAND

# Behavior
ACCOUNT_ID  = None            # e.g., 'DU1234567' (sub-account), else None
DRY_RUN     = False            # True=print only, False=place orders
ALLOW_SHORT = True
AUTO_FLIP   = False           # if already long and short signal, flatten first
PRINT_EVERY = 2.0             # log cadence (sec)
MIN_PLACE_GAP = 10.0          # anti-spam between submissions (sec)
# =================================

syd = pytz.timezone('Australia/Sydney')
def now_syd(): return datetime.now(tz=syd)

# ---------- Trade logger ----------
class TradeLog:
    def __init__(self, want_symbol_prefix='EUR'):
        self.want_prefix = want_symbol_prefix  # Forex('EURUSD'): contract.symbol == 'EUR'
        self.fills = []
        self.position = 0.0
        self.avg_cost = 0.0
        self.realized = 0.0

    def on_exec(self, trade: Trade, fill: Fill):
        con = trade.contract
        if getattr(con, 'symbol', '')[:3] != self.want_prefix:
            return
        side = fill.execution.side  # 'BOT' / 'SLD'
        qty  = float(fill.execution.shares)
        px   = float(fill.execution.price)
        t    = pd.to_datetime(fill.time, utc=True).tz_convert(syd)

        self.fills.append({'time': t.strftime('%Y-%m-%d %H:%M:%S%z'),
                           'side': side, 'qty': qty, 'price': px,
                           'orderId': trade.order.orderId})

        signed = qty if side == 'BOT' else -qty
        if self.position == 0:
            self.position = signed
            self.avg_cost = px
        elif np.sign(self.position) == np.sign(signed):
            new_pos = self.position + signed
            self.avg_cost = (self.avg_cost*abs(self.position) + px*abs(signed)) / abs(new_pos)
            self.position = new_pos
        else:
            close_qty = min(abs(self.position), abs(signed))
            if self.position > 0:
                self.realized += (px - self.avg_cost) * close_qty
            else:
                self.realized += (self.avg_cost - px) * close_qty
            self.position += signed
            if self.position == 0:
                self.avg_cost = 0.0
            elif np.sign(self.position) != np.sign(signed):
                self.avg_cost = px

    def summary(self):
        side = 'LONG' if self.position>0 else ('SHORT' if self.position<0 else 'FLAT')
        return f"Realized P&L: {self.realized:.2f} | Position: {side} {abs(self.position):.0f} @ {self.avg_cost:.5f}"

    def fills_df(self):
        return pd.DataFrame(self.fills) if self.fills else pd.DataFrame(columns=['time','side','qty','price','orderId'])

# ---------- Order guards ----------
def _is_working(status: str) -> bool:
    return status in {'Submitted','PreSubmitted','PendingSubmit','Inactive'}

def count_working_for_side(ib: IB, contract: Contract, side: str) -> int:
    side = side.upper()
    n = 0
    for tr in ib.trades():
        try:
            if tr.contract.conId != contract.conId: continue
            if tr.orderStatus and _is_working(tr.orderStatus.status) and tr.order.action.upper() == side:
                n += 1
        except Exception:
            pass
    return n

def cancel_open_orders_for_contract(ib: IB, contract: Contract, side_filter=None):
    side_filter = None if side_filter is None else side_filter.upper()
    to_cancel = []
    for tr in ib.trades():
        try:
            if tr.contract.conId != contract.conId: continue
            st = tr.orderStatus.status if tr.orderStatus else ''
            if _is_working(st):
                if side_filter is None or tr.order.action.upper() == side_filter:
                    to_cancel.append(tr.order)
        except Exception:
            pass
    for o in to_cancel:
        ib.cancelOrder(o)
    if to_cancel:
        ib.sleep(0.7)  # let cancels propagate

def safe_flatten(ib: IB, contract: Contract, tlog: TradeLog, dry_run: bool) -> bool:
    """
    Flatten open position while respecting per-side order caps:
      - If SHORT, we need BUY to cover â†’ cancel BUY working orders first.
      - If LONG, we need SELL to exit â†’ cancel SELL working orders first.
    """
    pos = int(round(tlog.position))
    if pos == 0:
        print("Already flat.")
        return True

    side = 'SELL' if pos > 0 else 'BUY'
    print(f"Flattening via {side} {abs(pos)} â€” canceling {side} working orders firstâ€¦")
    cancel_open_orders_for_contract(ib, contract, side_filter=side)

    # Wait for that side to clear
    for _ in range(10):
        n = count_working_for_side(ib, contract, side)
        if n == 0:
            break
        print(f"â€¦waiting for {side} cancels to clear (still {n} working)â€¦")
        ib.sleep(0.5)

    if dry_run:
        print("DRY_RUN=True â€” not sending flatten order.")
        return False

    try:
        ib.placeOrder(contract, MarketOrder(side, abs(pos)))
        print("Flatten order sent.")
        return True
    except Exception as e:
        print(f"[flatten error] {e}")
        return False

# ---------- Bracket builder (StopOrder.stopPrice) ----------
def opposite(action): return 'SELL' if action.upper()=='BUY' else 'BUY'

def make_bracket(ib: IB, side: str, qty: int, entry: float, tp_pips: int, sl_pips: int, account_id=None):
    side = side.upper()
    opp  = opposite(side)
    if side == 'BUY':
        tp_px = round(entry + tp_pips*PIP, 5)
        sl_px = round(entry - sl_pips*PIP, 5)
    else:
        tp_px = round(entry - tp_pips*PIP, 5)
        sl_px = round(entry + sl_pips*PIP, 5)

    parent   = MarketOrder(side, qty)
    takeProf = LimitOrder(opp, qty, lmtPrice=tp_px, tif='GTC')
    stopLoss = StopOrder(opp, qty, stopPrice=sl_px, tif='GTC')  # use stopPrice

    if account_id:
        for o in (parent, takeProf, stopLoss):
            o.account = account_id

    parent.orderId   = ib.client.getReqId()
    takeProf.orderId = ib.client.getReqId()
    stopLoss.orderId = ib.client.getReqId()

    takeProf.parentId = parent.orderId
    stopLoss.parentId = parent.orderId

    parent.transmit   = False
    takeProf.transmit = False
    stopLoss.transmit = True
    return parent, takeProf, stopLoss

# ---------- LOB net-demand â†’ EWMA z-score ----------
class LOBSignal:
    def __init__(self, levels=10, weighted=True, alpha=0.2, z_window=300, tick_size=0.00005):
        self.levels    = levels
        self.weighted  = weighted
        self.alpha     = alpha
        self.tick_size = tick_size
        self.ewma = None
        self.hist = []
        self.z_window = z_window

    def _depth_sum(self, bids, asks, best_bid, best_ask):
        def w(price, ref):
            if not self.weighted: return 1.0
            ticks = max(0, int(round(abs(price-ref)/self.tick_size)))
            return 1.0/(ticks+1.0)
        nb = 0.0
        for price, size in sorted(bids, key=lambda x: -x[0])[:self.levels]:
            nb += size * w(price, best_bid)
        na = 0.0
        for price, size in sorted(asks, key=lambda x: x[0])[:self.levels]:
            na += size * w(price, best_ask)
        return nb - na

    def update(self, bids, asks):
        if not bids or not asks: return None, None, None
        best_bid = max(p for p,_ in bids)
        best_ask = min(p for p,_ in asks)
        nd = self._depth_sum(bids, asks, best_bid, best_ask)
        self.ewma = nd if self.ewma is None else (1-self.alpha)*self.ewma + self.alpha*nd
        self.hist.append(self.ewma)
        if len(self.hist) > self.z_window:
            self.hist = self.hist[-self.z_window:]
        mu = np.mean(self.hist)
        sd = np.std(self.hist) if len(self.hist) > 1 else 1.0
        z = (self.ewma - mu) / (sd if sd>1e-9 else 1.0)
        mid = 0.5*(best_bid+best_ask)
        return z, nd, mid

# ---------- Main ----------
def main():
    print(f"[{now_syd()}] Connecting to IBKR paper on {HOST}:{PORT} â€¦")
    ib = IB()
    if not ib.connect(HOST, PORT, clientId=CLIENT_ID):
        print("Connect failed. Open TWS/Gateway (paper), enable API, port 7497, localhost allowed.")
        return

    contract = Forex(SYMBOL)
    [contract] = ib.qualifyContracts(contract)
    print(f"Qualified FX contract: {contract}")

    print("Requesting market depth (Level-2)â€¦")
    try:
        ticker = ib.reqMktDepth(contract, numRows=LEVELS, isSmartDepth=True)
    except Exception as e:
        print(f"Cannot request depth: {e}")
        ib.disconnect()
        return

    tlog = TradeLog('EUR')
    ib.execDetailsEvent += tlog.on_exec

    lob = LOBSignal(levels=LEVELS, weighted=WEIGHTED, alpha=ALPHA_EWMA)

    last_place_ts = 0.0
    last_log = 0.0

    print("Running. Ctrl+C to stop.")
    try:
        while True:
            ib.sleep(0.05)

            bids = [(lvl.price, lvl.size) for lvl in ticker.domBids if lvl.price>0 and lvl.size>0]
            asks = [(lvl.price, lvl.size) for lvl in ticker.domAsks if lvl.price>0 and lvl.size>0]
            if not bids or not asks:
                nowt = time.time()
                if nowt - last_log > 1.5:
                    print("Waiting for depth data (check IDEALPRO market-depth permissions)â€¦")
                    last_log = nowt
                continue

            z, nd, mid = lob.update(bids, asks)
            if z is None:
                continue

            # Derive state from the actual position (not from attempted orders)
            state = 'LONG' if tlog.position > 0 else ('SHORT' if tlog.position < 0 else 'FLAT')

            nowt = time.time()
            if nowt - last_log >= PRINT_EVERY:
                print(f"[{now_syd()}] mid={mid:.5f} ND={nd:.1f} z={z:+.2f} State={state} | {tlog.summary()}")
                last_log = nowt

            # Desired state via hysteresis
            want = state
            if z >= TH_LONG:
                want = 'LONG'
            elif z <= TH_SHORT and ALLOW_SHORT:
                want = 'SHORT'
            elif abs(z) <= EXIT_BAND:
                want = 'FLAT'

            if want == state:
                continue

            # Anti-spam
            if nowt - last_place_ts < MIN_PLACE_GAP:
                continue
            last_place_ts = nowt

            # Switch logic
            if want == 'LONG':
                # ensure only one working BUY bracket
                if count_working_for_side(ib, contract, 'BUY') >= 1:
                    print("âš ï¸ Already have a working BUY bracket â€” skipping.")
                    continue
                cancel_open_orders_for_contract(ib, contract, side_filter='BUY')

                if state == 'SHORT' and AUTO_FLIP and not DRY_RUN and tlog.position < 0:
                    # flatten short first
                    safe_flatten(ib, contract, tlog, dry_run=False)
                    ib.sleep(0.5)

                print("ðŸš€ LOB signal â†’ BUY")
                parent, tp, sl = make_bracket(ib, 'BUY', AMOUNT, mid, TP_PIPS, SL_PIPS, account_id=ACCOUNT_ID)
                sl_price = getattr(sl, 'auxPrice', getattr(sl, 'stopPrice', None))
                print(f"Bracket BUY {AMOUNT} @~{mid:.5f} | TP {tp.lmtPrice} | SL {sl_price}")
                if DRY_RUN:
                    print("DRY_RUN=True â€” not sending.")
                else:
                    ib.placeOrder(contract, parent); ib.placeOrder(contract, tp); ib.placeOrder(contract, sl)

            elif want == 'SHORT':
                if not ALLOW_SHORT:
                    print("Shorting disabled.")
                    continue
                if count_working_for_side(ib, contract, 'SELL') >= 1:
                    print("âš ï¸ Already have a working SELL bracket â€” skipping.")
                    continue
                cancel_open_orders_for_contract(ib, contract, side_filter='SELL')

                if state == 'LONG' and AUTO_FLIP and not DRY_RUN and tlog.position > 0:
                    safe_flatten(ib, contract, tlog, dry_run=False)
                    ib.sleep(0.5)

                print("ðŸ“‰ LOB signal â†’ SELL")
                parent, tp, sl = make_bracket(ib, 'SELL', AMOUNT, mid, TP_PIPS, SL_PIPS, account_id=ACCOUNT_ID)
                sl_price = getattr(sl, 'auxPrice', getattr(sl, 'stopPrice', None))
                print(f"Bracket SELL {AMOUNT} @~{mid:.5f} | TP {tp.lmtPrice} | SL {sl_price}")
                if DRY_RUN:
                    print("DRY_RUN=True â€” not sending.")
                else:
                    ib.placeOrder(contract, parent); ib.placeOrder(contract, tp); ib.placeOrder(contract, sl)

            else:  # want == 'FLAT'
                print("âš“ LOB neutral â†’ prefer FLAT")
                safe_flatten(ib, contract, tlog, DRY_RUN)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt â€” flattening and summarizing â€¦")
        try:
            safe_flatten(ib, contract, tlog, DRY_RUN)
            ib.sleep(2.0)
            print("\n===== SESSION SUMMARY =====")
            print(tlog.summary())
            df = tlog.fills_df()
            if not df.empty:
                with pd.option_context('display.max_rows', None, 'display.width', 120):
                    print(df)
            else:
                print("(No fills recorded.)")
        except Exception as e:
            print(f"[cleanup error] {e}")
    finally:
        try:
            ib.cancelMktDepth(ticker)
        except Exception:
            pass
        ib.disconnect()
        print("Disconnected.")

if __name__ == '__main__':
    main()