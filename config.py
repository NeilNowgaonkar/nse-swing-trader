# config.py
# NSE Swing Trader — Master Configuration
# Author: Neil
# Priority: STABILITY OVER PROFITABILITY

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# UNIVERSE: 28 STOCKS
# ─────────────────────────────────────────

# Each entry: "NSE_SYMBOL": "yfinance_ticker"
# yfinance uses .NS suffix for NSE stocks

LARGE_CAP = {
    "RELIANCE":   "RELIANCE.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "INFY":       "INFY.NS",
    "TCS":        "TCS.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "KOTAKBANK":  "KOTAKBANK.NS",
    "AXISBANK":   "AXISBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
}

MOMENTUM = {
    "MM": "M&M.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "SUNPHARMA":  "SUNPHARMA.NS",
    "MARUTI":     "MARUTI.NS",
    "WIPRO":      "WIPRO.NS",
    "HCLTECH":    "HCLTECH.NS",
    "LT":         "LT.NS",
}

MIDCAP_ALPHA = {
    "PERSISTENT": "PERSISTENT.NS",
    "TRENT":      "TRENT.NS",
    "VOLTAS":     "VOLTAS.NS",
    "ABCAPITAL":  "ABCAPITAL.NS",
    "SUNDARMFIN": "SUNDARMFIN.NS",
    "LAURUSLABS":  "LAURUSLABS.NS",
}

DEFENSIVE = {
    "NESTLEIND":  "NESTLEIND.NS",
    "BRITANNIA":  "BRITANNIA.NS",
    "DRREDDY":    "DRREDDY.NS",
    "CIPLA":      "CIPLA.NS",
    "POWERGRID":  "POWERGRID.NS",
    "NTPC":       "NTPC.NS",
}

# Combined universe — used by fetch_data.py and all scoring modules
ALL_STOCKS = {}
ALL_STOCKS.update(LARGE_CAP)
ALL_STOCKS.update(MOMENTUM)
ALL_STOCKS.update(MIDCAP_ALPHA)
ALL_STOCKS.update(DEFENSIVE)

# Bucket labels — used for position sizing logic later
BUCKET_LABELS = {}
for sym in LARGE_CAP:    BUCKET_LABELS[sym] = "large_cap"
for sym in MOMENTUM:     BUCKET_LABELS[sym] = "momentum"
for sym in MIDCAP_ALPHA: BUCKET_LABELS[sym] = "midcap_alpha"
for sym in DEFENSIVE:    BUCKET_LABELS[sym] = "defensive"

# ─────────────────────────────────────────
# INDEX TICKERS (for regime detection)
# ─────────────────────────────────────────
NIFTY50_TICKER  = "^NSEI"
INDIA_VIX_TICKER = "^INDIAVIX"

# ─────────────────────────────────────────
# CAPITAL & RISK SETTINGS
# ─────────────────────────────────────────
STARTING_CAPITAL      = 5000       # INR
RISK_PER_TRADE_PCT    = 1.5          # % of capital risked per trade
MAX_OPEN_TRADES       = 4            # hard cap on simultaneous positions
DAILY_LOSS_LIMIT_PCT  = 2.5          # % — triggers no-new-trades for the day
DRAWDOWN_PAUSE_PCT    = 8.0          # % drawdown → 10-day trading pause
DRAWDOWN_PAUSE_DAYS   = 10

# ─────────────────────────────────────────
# SCORING & SIGNAL THRESHOLDS
# ─────────────────────────────────────────
TRADE_SIGNAL_THRESHOLD = 68          # minimum score (out of 100) to generate alert
MAX_FACTOR_SCORE       = 100

# ─────────────────────────────────────────
# REGIME STATES
# ─────────────────────────────────────────
REGIMES = ["BULL", "NEUTRAL", "SIDEWAYS", "HIGH_VIX", "BEAR"]

# VIX thresholds
VIX_HIGH_THRESHOLD   = 20.0          # above this → HIGH_VIX regime
VIX_EXTREME_THRESHOLD = 25.0         # above this → BEAR bias

# ─────────────────────────────────────────
# DATA SETTINGS
# ─────────────────────────────────────────
DATA_LOOKBACK_DAYS  = 365            # how many calendar days of history to fetch
TIMEZONE            = "Asia/Kolkata"

# ─────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
DB_PATH    = os.path.join(DATA_DIR, "trades.db")

# ─────────────────────────────────────────
# TELEGRAM (loaded from .env / GitHub Secrets)
# ─────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ─────────────────────────────────────────
# SANITY CHECK (run this file directly to verify)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print(f"✅ Universe loaded: {len(ALL_STOCKS)} stocks")
    for bucket, label in [
        (LARGE_CAP,    "Large-Cap"),
        (MOMENTUM,     "Momentum"),
        (MIDCAP_ALPHA, "Midcap Alpha"),
        (DEFENSIVE,    "Defensive"),
    ]:
        syms = ", ".join(bucket.keys())
        print(f"   [{label:>14}] ({len(bucket)}) → {syms}")

    print(f"\n💰 Capital: ₹{STARTING_CAPITAL:,}")
    print(f"⚠️  Risk per trade: {RISK_PER_TRADE_PCT}%")
    print(f"🔒 Max open trades: {MAX_OPEN_TRADES}")
    print(f"📉 Daily loss limit: {DAILY_LOSS_LIMIT_PCT}%")
    print(f"🛑 Drawdown pause: {DRAWDOWN_PAUSE_PCT}% → {DRAWDOWN_PAUSE_DAYS} days")
    print(f"🎯 Signal threshold: {TRADE_SIGNAL_THRESHOLD}/100")
    print(f"\n📁 DB path: {DB_PATH}")
    print(f"📁 Log dir: {LOG_DIR}")
    print(f"\n🔑 Telegram token set: {'YES' if TELEGRAM_BOT_TOKEN else 'NO — add to .env'}")
    print(f"🔑 Telegram chat ID set: {'YES' if TELEGRAM_CHAT_ID else 'NO — add to .env'}")
