# Systematic Time-Series Momentum (TSMOM) Trading Algorithm

An automated systematic trading system developed in Python to execute a Time-Series Momentum strategy on US equities via the **Interactive Brokers (IBKR) API**. The engine processes real-time market data to identify statistically significant price momentum, with execution and risk management treated as first-class design components rather than an afterthought.

### Key Technical Features

* **Time-Series Momentum Signal:** Computes an EWMA of rolling log-returns, converted into a Z-score against a historical distribution to identify statistically significant momentum, entries trigger only above a defined Z-score threshold.
* **Volatility-Adaptive Risk Sizing:** Take-profit and stop-loss distances scale dynamically based on realised price volatility over a rolling window, rather than using fixed thresholds.
* **Risk-Based Position Sizing:** Share quantity is calculated from a fixed dollar risk per trade divided by stop-loss distance, not from arbitrary notional caps alone.
* **Automated Risk Controls:** Generates atomic **Bracket Orders** (Entry + Take-Profit + Stop-Loss) with a session-level **Kill-Switch** that flattens all positions if session PnL breaches a defined loss limit.
* **Turnover and Rate Limiting:** A dedicated guard restricts order frequency per symbol to avoid excessive trading and reduce transaction cost drag.
* **Asynchronous Event Loop:** Built with `ib_insync` for non-blocking real-time market data ingestion, historical backfill with progressive fallback across multiple data granularities, and live PnL subscription for real-time session monitoring.

### The Stack

* **Language:** Python 3.x
* **Mathematics:** `NumPy`, `Pandas` (vectorized signal processing)
* **Brokerage Infrastructure:** Interactive Brokers (TWS / IB Gateway) via `ib_insync`

### Universe

Currently configured for **META, AMZN, AAPL** as a proof-of-concept universe, easily extensible to additional symbols.

---

### Execution Workflow

The algorithm monitors real-time price data and evaluates entry conditions on every market data update.

1. **Signal Calculation:** Computes EWMA log-return momentum and converts it to a rolling Z-score.
2. **Entry Decision:** Triggers a `LONG` signal when Z-score exceeds the entry threshold; long-only in the current configuration.
3. **Exit Decision:** Flattens the position when the Z-score reverts toward neutral or flips against the position.
4. **Risk Enforcement:** Every entry is placed as a bracket order with volatility-adjusted take-profit and stop-loss levels; a session kill-switch monitors realised and unrealised PnL and force-flattens all positions if the daily loss limit is breached.

All trades operate in a strict `DRY_RUN` mode by default, no live orders are placed until this flag is explicitly disabled after review.

```text
[Signal → LONG]
Bracket BUY 12 AAPL @ 187.42 | TP 187.46 | SL 187.39
Status: DRY_RUN=True — Logic Validated.
```

### Risk Management Philosophy

The majority of the design effort in this project went into risk controls rather than the signal itself. Session-level kill-switches, per-trade dynamic stop-losses, and turnover limiting were treated as first-class components of the system, not an afterthought layered on top of a working signal.
