# FX High-Frequency Trading (HFT) Algorithm
An automated high-frequency trading (HFT) system developed in Python to execute Forex strategies via the **Interactive Brokers (IBKR) API**. The engine processes real-time Market Depth (L2) to capitalize on order book imbalances and price momentum.

### Key Technical Features
* **Asynchronous Event Loop:** Built with `ib_insync` for non-blocking market data ingestion and high-concurrency execution.
* **L2 Market Depth Analysis:** Computes real-time Net Density (ND) and `zLOB` (Limit Order Book) scores to identify liquidity skews.
* **Signal Integration:** Multi-factor logic combining LOB imbalances with `zTS` (Time-Series Momentum) using EWMA log-returns.
* **Automated Risk Controls:** Generates atomic **Bracket Orders** (Entry + Take-Profit + Stop-Loss) with a global "Kill-Switch" for downside protection.

### The Stack
* **Language:** Python 3.x
* **Mathematics:** `NumPy`, `Pandas` (vectorized signal processing)
* **Brokerage Infrastructure:** Interactive Brokers (TWS / IB Gateway)

---

### Execution Workflow
The algorithm monitors the market state and triggers signals when Z-score thresholds align across both liquidity and trend factors:

1.  **Calculation:** Computes real-time Z-scores for order book skew and trend velocity.
2.  **Execution:** Triggers `BUY`/`SELL` signals; bypasses live markets if `DRY_RUN=True`.

```text
[Signal → SELL] 
Bracket SELL 300,000 @ 1.17354 | TP 1.17344 | SL 1.17364
Status: DRY_RUN=True — Logic Validated.
