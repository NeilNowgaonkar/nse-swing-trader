# NSE Swing Trader — Neil's Personal System

> **Priority: STABILITY OVER PROFITABILITY. Single breadwinner.**

A Python-based swing trading signal system for NSE stocks.
All execution is 100% manual in Groww. This system only generates alerts.

---

## Phase Status

| Phase | Module | Status |
|-------|--------|--------|
| 1 | Foundation (config, DB, fetch) | ✅ Complete |
| 2 | Factor scoring + regime engine | 🔜 Next |
| 3 | Telegram bot + portfolio memory | 🔜 Upcoming |
| 4 | Streamlit dashboard | 🔜 Upcoming |

---

## Quick Start (Local)

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/nse_swing_trader.git
cd nse_swing_trader

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create `.env` file (Telegram secrets — Phase 3)

```bash
# .env
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 3. Run Phase 1 verification

```bash
# Step 1: Verify config loads correctly
python config.py

# Step 2: Create the database and all 4 tables
python create_db.py

# Step 3: Fetch data for all 28 stocks (takes ~2 minutes)
python fetch_data.py
```

---

## Test Commands (copy-paste to verify each module)

```bash
# Verify config — should show 28 stocks, capital settings
python config.py

# Verify DB — should show 4 tables with correct column counts
python create_db.py

# Verify fetch — should show latest close prices for all 28 stocks
python fetch_data.py

# Quick DB query — check row counts per symbol
python -c "
import sqlite3
from config import DB_PATH
conn = sqlite3.connect(DB_PATH)
rows = conn.execute(\"SELECT symbol, COUNT(*) as rows FROM daily_prices GROUP BY symbol ORDER BY symbol\").fetchall()
for r in rows: print(f'  {r[0]:>14}: {r[1]} rows')
conn.close()
"
```

---

## System Architecture

```
GitHub Actions (08:00 IST, Mon–Fri)
        │
        ▼
  fetch_data.py          ← pulls OHLCV via yfinance
        │
        ▼
  score_stocks.py        ← Phase 2: 6-factor model (0–100)
        │
        ▼
  regime_engine.py       ← Phase 2: 5-state market regime
        │
        ▼
  telegram_bot.py        ← Phase 3: alerts + /bought /sold commands
        │
        ▼
  streamlit_app.py       ← Phase 4: dashboard
```

---

## Risk Rules (Never Override These)

- Risk per trade: **1.5%** of capital
- Max open trades: **4**
- Daily loss limit: **2.5%** → no new trades that day
- Drawdown ≥ **8%** → 10-day mandatory pause
- Trade only if score ≥ **68/100** AND all 3 hard gates pass
- Regime = BEAR → no new entries

---

## Universe (28 Stocks)

| Bucket | Stocks |
|--------|--------|
| Large-Cap (8) | RELIANCE, HDFCBANK, INFY, TCS, ICICIBANK, KOTAKBANK, AXISBANK, HINDUNILVR |
| Momentum (8) | TATAMOTORS, BAJFINANCE, ADANIPORTS, SUNPHARMA, MARUTI, WIPRO, HCLTECH, LT |
| Midcap Alpha (6) | PERSISTENT, TRENT, VOLTAS, ABCAPITAL, SUNDARMFIN, LAURUSLABS |
| Defensive (6) | NESTLEIND, BRITANNIA, DRREDDY, CIPLA, POWERGRID, NTPC |
