# hft_momentum_trader.py
# EDUCATIONAL PURPOSES ONLY — USE IBKR PAPER TRADING

from ib_insync import *
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import deque
import pytz

# ========= CONNECTION =========
HOST        = '127.0.0.1'
PORT        = 7497              # 7496=live, 7497=paper
CLIENT_ID   = 909

# ========= HFT SLEEVE & RISK CAPS (for $150,000) =========
HFT_SLEEVE_USD     = 150_000
PER_TRADE_RISK_USD = 60          # ~0.04% of sleeve; (45–75) is sensible
DAILY_LOSS_LIMIT   = -0.01 * HFT_SLEEVE_USD   # -1.0% day stop
MAX_CONCURRENT     = 2           # max active brackets across sides
SPREAD_SKIP_PIPS   = 0.6         # skip trades above this spread
SPREAD_HALF_PIPS   = 0.3         # half size when in [0.3, 0.6]
MAX_NOTIONAL_BASE  = 300_000     # per-trade hard cap in base units (FX) or shares (equities cap below)

# ========= STRATEGY MODE =========
MOMENTUM_MODE = 'LOB'            # 'LOB' | 'TSMOM' | 'COMBO'
TREND_GATE    = False            # use SMA(50/200) gate
MA_SHORT      = 50               # samples (your loop sleeps 0.05s)
MA_LONG       = 200

# TSMOM controls
TSMOM_LOOKBACK    = 120          # samples back (~6s if sleep=0.05)
TSMOM_MIN_Z       = 1.5
TSMOM_EWMA_ALPHA  = 0.2
Z_WINDOW          = 300          # z-score lookback for both LOB/TSMOM

# ========= LOB / EXECUTION SETTINGS =========
LEVELS      = 10                 # depth levels
WEIGHTED    = True               # 1/(ticks_from_mid+1) weighting
ALPHA_EWMA  = 0.2                # LOB EWMA smoothing
TH_LONG     = +2.0               # z-score entry
TH_SHORT    = -2.0
EXIT_BAND   = 0.5
PRINT_EVERY = 2.0
MIN_PLACE_GAP = 10.0
AUTO_FLIP   = False
ALLOW_SHORT = True
DRY_RUN     = False
ACCOUNT_ID  = None               # e.g., 'DU1234567' if needed

# ========= INSTRUMENT SELECTION =========
# Choose exactly ONE of the blocks below.

INSTRUMENT = 'FX'                # 'FX' | 'STOCK'
FX_SYMBOL  = 'EURUSD'            # Majors recommended: EURUSD / USDJPY / GBPUSD

STOCK_SYMBOL = 'SPY'             # Example: 'SPY','QQQ','NVDA','TSLA'
STOCK_EXCHANGE = 'SMART'
STOCK_CCY   = 'USD'
STOCK_TICK  = 0.01               # $0.01
STOCK_MAX_SHARES = 3_000         # cap per trade if using equities

# ========= PIP/TICK CONFIG (auto from instrument) =========
def get_px_config(instrument):
    if instrument == 'FX':
        # IB quotes FX in decimal, typical pip = 0.0001 (EURUSD)
        return dict(PIP=0.0001, TICK_SIZE=0.00005, ROUND=lambda x: round(x, 5))
    else:  # STOCK
        return dict(PIP=STOCK_TICK, TICK_SIZE=STOCK_TICK, ROUND=lambda x: round(x, 2))

pxcfg = get_px_config(INSTRUMENT)
PIP = pxcfg['PIP']
ROUND = pxcfg['ROUND']
TICK_SIZE = pxcfg['TICK_SIZE']

# ========= TIMEZONE =========
syd = pytz.timezone('Australia/Sydney')
def now_syd(): return datetime.now(tz=syd)

# ========= HELPERS =========
def sma(buf, n):
    if len(buf) < n: return None
    return float(np.mean(list(buf)[-n:]))

def pip_value_per_unit_fx():
    # USD per pip per 1 base unit for majors quoted in USD (approx)
    return 0.0001

def size_for_risk_fx(per_trade_risk_usd, sl_pips):
    pv = pip_value_per_unit_fx()
    units = per_trade_risk_usd / (sl_pips * pv)
    return max(1, int(units))

def size_for_risk_stock(per_trade_risk_usd, sl_dollars):
    # shares = risk / stop ($)
    if sl_dollars <= 0:
        return 0
    return max(1, int(per_trade_risk_usd / sl_dollars))

def opposite(side): return 'SELL' if side.upper()=='BUY' else 'BUY'

# ========= TRADE LOG =========
class TradeLog:
    def __init__(self, symbol_prefix=''):
        self.want_prefix = symbol_prefix
        self.fills = []
        self.position = 0.0
        self.avg_cost = 0.0
        self.realized = 0.0
        self.session_pnl = 0.0
        self.submit_mid = {}      # parentId -> mid at submit
        self.fill_metrics = []

    def note_submit_mid(self, parent_id: int, mid: float):
        self.submit_mid[parent_id] = mid

    def on_exec(self, trade: Trade, fill: Fill):
        con = trade.contract
        # Symbol check is optional; keep broad to record all fills for selected contract
        side = fill.execution.side  # 'BOT' or 'SLD'
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

        # update session pnl
        self.session_pnl = self.realized

        # effective spread metric (map to parent group)
        parent_id = trade.order.parentId if getattr(trade.order, 'parentId', 0) else trade.order.orderId
        mid_submit = self.submit_mid.get(parent_id)
        if mid_submit is not None:
            eff_spread = 2.0 * abs(px - mid_submit)
            self.fill_metrics.append({
                'time': t,
                'orderId': trade.order.orderId,
                'parentId': parent_id,
                'exec_px': px,
                'mid_submit': mid_submit,
                'effective_spread': eff_spread
            })

    def summary(self):
        side = 'LONG' if self.position>0 else ('SHORT' if self.position<0 else 'FLAT')
        return f"Realized P&L: {self.realized:.2f} | Position: {side} {abs(self.position):.0f} @ {self.avg_cost:.5f}"

    def fills_df(self):
        return pd.DataFrame(self.fills) if self.fills else pd.DataFrame(columns=['time','side','qty','price','orderId'])

# ========= ORDER GUARDS =========
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
        ib.sleep(0.7)

def safe_flatten(ib: IB, contract: Contract, tlog: TradeLog, dry_run: bool) -> bool:
    pos = int(round(tlog.position))
    if pos == 0:
        print("Already flat.")
        return True
    side = 'SELL' if pos > 0 else 'BUY'
    print(f"Flatten via {side} {abs(pos)} — canceling {side} working first…")
    cancel_open_orders_for_contract(ib, contract, side_filter=side)
    for _ in range(20):
        n = count_working_for_side(ib, contract, side)
        if n == 0: break
        ib.sleep(0.75)
    if dry_run:
        print("DRY_RUN=True — not sending flatten.")
        return False
    try:
        ib.placeOrder(contract, MarketOrder(side, abs(pos)))
        print("Flatten order sent.")
        return True
    except Exception as e:
        print(f"[flatten error] {e}")
        return False

# ========= BRACKET BUILDER =========
def make_bracket(ib: IB, side: str, qty: int, entry: float, tp_pips: float, sl_pips: float, account_id=None):
    side = side.upper()
    opp  = opposite(side)
    if side == 'BUY':
        tp_px = ROUND(entry + tp_pips*PIP)
        sl_px = ROUND(entry - sl_pips*PIP)
    else:
        tp_px = ROUND(entry - tp_pips*PIP)
        sl_px = ROUND(entry + sl_pips*PIP)

    parent   = MarketOrder(side, qty)
    takeProf = LimitOrder(opp, qty, lmtPrice=tp_px, tif='GTC')
    stopLoss = StopOrder(opp, qty, stopPrice=sl_px, tif='GTC')

    if account_id:
        for o in (parent, takeProf, stopLoss):
            o.account = account_id

    parent.orderId   = ib.client.getReqId()
    takeProf.orderId = ib.client.getReqId()
    stopLoss.orderId = ib.client.getReqId()
    takeProf.parentId = parent.orderId
    stopLoss.parentId = parent.orderId
    parent.transmit = False
    takeProf.transmit = False
    stopLoss.transmit = True
    return parent, takeProf, stopLoss

# ========= SIGNALS =========
class LOBSignal:
    def __init__(self, levels=10, weighted=True, alpha=0.2, z_window=300, tick_size=TICK_SIZE):
        self.levels = levels
        self.weighted = weighted
        self.alpha = alpha
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

class ReturnMomentum:
    """EWMA + Z-score of log returns over a lookback window."""
    def __init__(self, lookback=TSMOM_LOOKBACK, alpha=TSMOM_EWMA_ALPHA, z_window=Z_WINDOW):
        self.lookback = lookback
        self.alpha = alpha
        self.z_window = z_window
        self.ewma_ret = None
        self.hist = []

    def update(self, mid_buf):
        if len(mid_buf) < self.lookback + 1:
            return None, None, None
        m_now = mid_buf[-1]
        m_then = mid_buf[-(self.lookback+1)]
        r = np.log(m_now) - np.log(m_then)
        self.ewma_ret = r if self.ewma_ret is None else (1-self.alpha)*self.ewma_ret + self.alpha*r
        self.hist.append(self.ewma_ret)
        if len(self.hist) > self.z_window:
            self.hist = self.hist[-self.z_window:]
        mu = np.mean(self.hist)
        sd = np.std(self.hist) if len(self.hist) > 1 else 1e-9
        z = (self.ewma_ret - mu)/(sd if sd>1e-9 else 1.0)
        return z, r, np.log(m_now)

def lob_decision(z):
    if z is None: return 'HOLD'
    if z >= TH_LONG: return 'LONG'
    if z <= TH_SHORT and ALLOW_SHORT: return 'SHORT'
    if abs(z) <= EXIT_BAND: return 'FLAT'
    return 'HOLD'

def tsmom_decision(z_ts):
    if z_ts is None: return 'HOLD'
    if z_ts >= TSMOM_MIN_Z: return 'LONG'
    if z_ts <= -TSMOM_MIN_Z and ALLOW_SHORT: return 'SHORT'
    return 'HOLD'

def gate_with_trend(sig, trend):
    if trend is None or sig in ('HOLD','FLAT'): return sig
    if sig == 'LONG' and trend != 'UP': return 'HOLD'
    if sig == 'SHORT' and trend != 'DOWN': return 'HOLD'
    return sig

# ========= MAIN =========
def main():
    print(f"[{now_syd()}] Connecting to IBKR paper on {HOST}:{PORT} …")
    ib = IB()
    if not ib.connect(HOST, PORT, clientId=CLIENT_ID):
        print("Connect failed. Open TWS/Gateway (paper), enable API, port 7497, localhost allowed.")
        return

    # --- Contract setup ---
    if INSTRUMENT == 'FX':
        contract = Forex(FX_SYMBOL)
    else:
        contract = Stock(STOCK_SYMBOL, STOCK_EXCHANGE, STOCK_CCY)

    [contract] = ib.qualifyContracts(contract)
    print(f"Qualified contract: {contract}")

    print("Requesting market depth (Level-2)…")
    try:
        ticker = ib.reqMktDepth(contract, numRows=LEVELS, isSmartDepth=True)
    except Exception as e:
        print(f"Cannot request depth: {e}")
        ib.disconnect()
        return

    tlog = TradeLog()
    ib.execDetailsEvent += tlog.on_exec

    lob = LOBSignal(levels=LEVELS, weighted=WEIGHTED, alpha=ALPHA_EWMA, z_window=Z_WINDOW)
    tsmom = ReturnMomentum()

    last_place_ts = 0.0
    last_log = 0.0
    mid_buf = deque(maxlen=max(MA_LONG, TSMOM_LOOKBACK+2))

    print("Running. Ctrl+C to stop.")
    try:
        while True:
            ib.sleep(0.05)

            bids = [(lvl.price, lvl.size) for lvl in ticker.domBids if lvl.price>0 and lvl.size>0]
            asks = [(lvl.price, lvl.size) for lvl in ticker.domAsks if lvl.price>0 and lvl.size>0]
            if not bids or not asks:
                nowt = time.time()
                if nowt - last_log > 1.5:
                    print("Waiting for depth data (check market-depth permissions)…")
                    last_log = nowt
                continue

            z, nd, mid = lob.update(bids, asks)
            if z is None:
                continue

            # Track mid & trend
            mid_buf.append(mid)
            trend = None
            if TREND_GATE:
                ma_s = sma(mid_buf, MA_SHORT)
                ma_l = sma(mid_buf, MA_LONG)
                if ma_s is not None and ma_l is not None:
                    if ma_s > ma_l: trend = 'UP'
                    elif ma_s < ma_l: trend = 'DOWN'

            # TSMOM signal
            z_ts, ret_ts, _ = tsmom.update(list(mid_buf))

            # Current state from actual position
            state = 'LONG' if tlog.position > 0 else ('SHORT' if tlog.position < 0 else 'FLAT')

            nowt = time.time()
            if nowt - last_log >= PRINT_EVERY:
                print(f"[{now_syd()}] mid={mid:.5f} ND={nd:.1f} zLOB={z:+.2f} zTS={z_ts if z_ts is not None else np.nan} "
                      f"State={state} | {tlog.summary()}")
                last_log = nowt

            # Daily loss stop
            if tlog.session_pnl <= DAILY_LOSS_LIMIT:
                if nowt - last_log >= PRINT_EVERY:
                    print(f"[{now_syd()}] Daily loss limit reached ({tlog.session_pnl:.2f} ≤ {DAILY_LOSS_LIMIT:.2f}) — halting entries.")
                    last_log = nowt
                continue

            # Spread filter
            best_bid = max(p for p,_ in bids)
            best_ask = min(p for p,_ in asks)
            spread_pips = (best_ask - best_bid) / PIP
            if spread_pips > SPREAD_SKIP_PIPS:
                continue

            # Decide desired action
            sig_lob = lob_decision(z)
            sig_ts = tsmom_decision(z_ts)

            if TREND_GATE:
                sig_lob = gate_with_trend(sig_lob, trend)
                sig_ts  = gate_with_trend(sig_ts, trend)

            if MOMENTUM_MODE == 'LOB':
                desired = sig_lob
            elif MOMENTUM_MODE == 'TSMOM':
                desired = sig_ts
            else:  # COMBO
                if sig_lob == sig_ts and sig_lob in ('LONG','SHORT'):
                    desired = sig_lob
                else:
                    desired = 'FLAT' if (sig_lob=='FLAT' and sig_ts=='FLAT') else 'HOLD'

            want = state
            if desired == 'LONG' and state != 'LONG':
                want = 'LONG'
            elif desired == 'SHORT' and state != 'SHORT':
                want = 'SHORT'
            elif desired == 'FLAT' and state != 'FLAT':
                want = 'FLAT'
            else:
                continue

            # Anti-spam
            if nowt - last_place_ts < MIN_PLACE_GAP:
                continue
            last_place_ts = nowt

            # Concurrency guard
            total_working = count_working_for_side(ib, contract, 'BUY') + count_working_for_side(ib, contract, 'SELL')
            if total_working >= MAX_CONCURRENT and want in ('LONG','SHORT'):
                print("⚠️ MAX_CONCURRENT reached — skipping new entry.")
                continue

            # Dynamic TP/SL scaling by short-term vol
            VOL_WIN = 100
            if len(mid_buf) >= VOL_WIN:
                arr = np.array(list(mid_buf)[-VOL_WIN:])
                vol_pips = float(np.std(np.diff(arr))) / PIP
                vol_pips = max(0.2, min(vol_pips, 3.0))
                tp_pips = max(1.0, min(4.0, 2.0 * (vol_pips / 1.0)))
                sl_pips = max(1.0, min(4.0, 1.5 * (vol_pips / 1.0)))
            else:
                tp_pips, sl_pips = 2.0, 1.5

            # Size from risk & spread
            if INSTRUMENT == 'FX':
                amt = size_for_risk_fx(PER_TRADE_RISK_USD, sl_pips)
                if SPREAD_HALF_PIPS <= spread_pips <= SPREAD_SKIP_PIPS:
                    amt = int(0.5 * amt)
                amt = min(amt, int(MAX_NOTIONAL_BASE))
                if amt < 10_000:
                    print("Size too small after caps — skipping.")
                    continue
            else:
                # For stocks, convert sl in $; use 1.5*PIP ($0.015) default if no better est
                sl_dollars = sl_pips * PIP
                shares = size_for_risk_stock(PER_TRADE_RISK_USD, sl_dollars)
                shares = min(shares, STOCK_MAX_SHARES)
                amt = shares
                if amt < 5:
                    print("Share size too small — skipping.")
                    continue

            # Switch logic
            if want == 'LONG':
                if count_working_for_side(ib, contract, 'BUY') >= 1:
                    print("⚠️ Already have a working BUY bracket — skipping.")
                    continue
                cancel_open_orders_for_contract(ib, contract, side_filter='BUY')
                if state == 'SHORT' and AUTO_FLIP and not DRY_RUN and tlog.position < 0:
                    safe_flatten(ib, contract, tlog, dry_run=False); ib.sleep(0.5)

                print("🚀 Signal → BUY")
                parent, tp, sl = make_bracket(ib, 'BUY', amt, mid, tp_pips, sl_pips, account_id=ACCOUNT_ID)
                tlog.note_submit_mid(parent.orderId, mid)
                sl_price = getattr(sl, 'auxPrice', getattr(sl, 'stopPrice', None))
                print(f"Bracket BUY {amt} @~{ROUND(mid)} | TP {tp.lmtPrice} | SL {sl_price}")
                if not DRY_RUN:
                    ib.placeOrder(contract, parent); ib.placeOrder(contract, tp); ib.placeOrder(contract, sl)
                else:
                    print("DRY_RUN=True — not sending.")

            elif want == 'SHORT':
                if not ALLOW_SHORT:
                    print("Shorting disabled."); continue
                if count_working_for_side(ib, contract, 'SELL') >= 1:
                    print("⚠️ Already have a working SELL bracket — skipping.")
                    continue
                cancel_open_orders_for_contract(ib, contract, side_filter='SELL')
                if state == 'LONG' and AUTO_FLIP and not DRY_RUN and tlog.position > 0:
                    safe_flatten(ib, contract, tlog, dry_run=False); ib.sleep(0.5)

                print("📉 Signal → SELL")
                parent, tp, sl = make_bracket(ib, 'SELL', amt, mid, tp_pips, sl_pips, account_id=ACCOUNT_ID)
                tlog.note_submit_mid(parent.orderId, mid)
                sl_price = getattr(sl, 'auxPrice', getattr(sl, 'stopPrice', None))
                print(f"Bracket SELL {amt} @~{ROUND(mid)} | TP {tp.lmtPrice} | SL {sl_price}")
                if not DRY_RUN:
                    ib.placeOrder(contract, parent); ib.placeOrder(contract, tp); ib.placeOrder(contract, sl)
                else:
                    print("DRY_RUN=True — not sending.")

            else:  # FLAT
                print("⏸ Neutral → prefer FLAT")
                safe_flatten(ib, contract, tlog, DRY_RUN)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — flattening and summarizing …")
        try:
            safe_flatten(ib, contract, tlog, DRY_RUN)
            ib.sleep(2.0)
            print("\n===== SESSION SUMMARY =====")
            print(tlog.summary())
            df = tlog.fills_df()
            if not df.empty:
                with pd.option_context('display.max_rows', None, 'display.width', 140):
                    print(df)
            else:
                print("(No fills recorded.)")

            if tlog.fill_metrics:
                met_df = pd.DataFrame(tlog.fill_metrics)
                if not met_df.empty:
                    met_df['effective_spread_pips'] = met_df['effective_spread'] / PIP
                    print("\n===== EFFECTIVE SPREAD METRICS =====")
                    with pd.option_context('display.max_rows', None, 'display.width', 160):
                        print(met_df[['time','parentId','exec_px','mid_submit','effective_spread_pips']])
                    ts = now_syd().strftime('%Y%m%d_%H%M%S')
                    met_df.to_csv(f'eff_spread_{contract.symbol}_{ts}.csv', index=False)

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
