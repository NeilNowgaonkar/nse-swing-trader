# regime_engine.py
# 5-State Market Regime Engine
# States: BULL | NEUTRAL | SIDEWAYS | HIGH_VIX | BEAR

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import pytz
import os

from config import (
    DB_PATH, LOG_DIR, TIMEZONE,
    VIX_HIGH_THRESHOLD, VIX_EXTREME_THRESHOLD,
)
from create_db import get_connection


# ─────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)

logger.add(
    os.path.join(LOG_DIR, "regime_{time:YYYY-MM-DD}.log"),
    rotation="1 day",
    retention="14 days",
    level="INFO"
)

IST = pytz.timezone(TIMEZONE)


# ─────────────────────────────────────────────────────────────
# REGIME RULES
# ─────────────────────────────────────────────────────────────

REGIME_TRADING_RULES = {
    "BULL":      {"max_trades": 4, "min_score": 68,  "allowed_buckets": "ALL"},
    "NEUTRAL":   {"max_trades": 4, "min_score": 68,  "allowed_buckets": "ALL"},
    "SIDEWAYS":  {"max_trades": 3, "min_score": 75,  "allowed_buckets": ["large_cap", "defensive"]},
    "HIGH_VIX":  {"max_trades": 2, "min_score": 80,  "allowed_buckets": ["defensive"]},
    "BEAR":      {"max_trades": 0, "min_score": 999, "allowed_buckets": []},
}


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _safe_fmt(val, digits=1):
    """
    Safe number formatting for logs.
    Prevents crashes when values are None or NaN.
    """
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.{digits}f}"


def _sma(series: pd.Series, period: int):
    """Simple moving average."""
    if len(series) < period:
        return np.nan
    return series.tail(period).mean()


def _slope_pct(series: pd.Series, period: int = 10):
    """
    % slope over last N days.
    Helps detect sideways markets.
    """
    if len(series) < period:
        return 0.0

    start = series.iloc[-period]
    end   = series.iloc[-1]

    if start == 0:
        return 0.0

    return ((end - start) / start) * 100


# ─────────────────────────────────────────────────────────────
# CORE REGIME DETECTOR
# ─────────────────────────────────────────────────────────────

def detect_regime(conn: sqlite3.Connection):

    today = datetime.now(IST).strftime("%Y-%m-%d")

    # ── Load Nifty data ──────────────────────────────────────

    nifty_df = pd.read_sql_query("""
        SELECT date, close
        FROM daily_prices
        WHERE symbol = 'NIFTY50'
        ORDER BY date DESC
        LIMIT 220
    """, conn)

    # ── Load VIX data ────────────────────────────────────────

    vix_df = pd.read_sql_query("""
        SELECT date, close
        FROM daily_prices
        WHERE symbol = 'INDIA_VIX'
        ORDER BY date DESC
        LIMIT 10
    """, conn)

    if nifty_df.empty:
        logger.error("No NIFTY50 data in DB")
        return _regime_result("NEUTRAL", today, None, None, None, None, "Missing Nifty data")

    nifty_df = nifty_df.sort_values("date")
    nifty_series = nifty_df["close"]

    nifty_close = float(nifty_series.iloc[-1])

    # ── VIX value ────────────────────────────────────────────

    india_vix = None
    if not vix_df.empty:
        india_vix = float(vix_df.sort_values("date").iloc[-1]["close"])

    # ── Moving averages ──────────────────────────────────────

    sma50  = _sma(nifty_series, 50)
    sma200 = _sma(nifty_series, 200)

    slope = _slope_pct(nifty_series, 20)

    logger.info(
        f"Regime inputs | "
        f"Nifty={_safe_fmt(nifty_close)} | "
        f"SMA50={_safe_fmt(sma50)} | "
        f"SMA200={_safe_fmt(sma200)} | "
        f"VIX={_safe_fmt(india_vix)} | "
        f"Slope={_safe_fmt(slope,2)}%"
    )

    # ─────────────────────────────────────────────
    # REGIME LOGIC
    # Order matters (most restrictive first)
    # ─────────────────────────────────────────────

    if not pd.isna(sma200) and nifty_close < sma200:
        return _regime_result("BEAR", today, nifty_close, sma50, sma200, india_vix,
                              "Nifty below SMA-200")

    if india_vix is not None and india_vix >= VIX_EXTREME_THRESHOLD:
        return _regime_result("BEAR", today, nifty_close, sma50, sma200, india_vix,
                              f"VIX extreme ({india_vix:.1f})")

    if india_vix is not None and india_vix >= VIX_HIGH_THRESHOLD:
        return _regime_result("HIGH_VIX", today, nifty_close, sma50, sma200, india_vix,
                              f"VIX elevated ({india_vix:.1f})")

    if not pd.isna(sma50) and nifty_close < sma50:
        return _regime_result("SIDEWAYS", today, nifty_close, sma50, sma200, india_vix,
                              "Below SMA-50")

    if abs(slope) < 1.5:
        return _regime_result("SIDEWAYS", today, nifty_close, sma50, sma200, india_vix,
                              f"Flat slope ({slope:.2f}%)")

    sma_aligned = (
        not pd.isna(sma50) and
        not pd.isna(sma200) and
        sma50 > sma200
    )

    vix_calm = india_vix is None or india_vix < 15

    if nifty_close > sma50 and sma_aligned and vix_calm and slope > 2:
        return _regime_result("BULL", today, nifty_close, sma50, sma200, india_vix,
                              "Trend aligned")

    return _regime_result("NEUTRAL", today, nifty_close, sma50, sma200, india_vix,
                          "Mixed conditions")


# ─────────────────────────────────────────────────────────────
# RESULT BUILDER
# ─────────────────────────────────────────────────────────────

def _regime_result(regime, date, nifty, sma50, sma200, vix, notes):

    rules = REGIME_TRADING_RULES[regime]

    result = {
        "date": date,
        "regime": regime,
        "nifty_close": round(nifty, 2) if nifty else None,
        "nifty_sma50": round(sma50, 2) if sma50 and not pd.isna(sma50) else None,
        "nifty_sma200": round(sma200, 2) if sma200 and not pd.isna(sma200) else None,
        "india_vix": round(vix, 2) if vix else None,
        "notes": notes,
        "max_trades": rules["max_trades"],
        "min_score": rules["min_score"],
        "allowed_buckets": rules["allowed_buckets"]
    }

    logger.info(f"Regime detected: {regime} | {notes}")

    return result


# ─────────────────────────────────────────────────────────────
# SAVE REGIME
# ─────────────────────────────────────────────────────────────

def save_regime(conn, regime_data):

    try:

        conn.execute("""
            INSERT OR REPLACE INTO regime_log
            (date, regime, nifty_close, nifty_sma50, nifty_sma200, india_vix, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            regime_data["date"],
            regime_data["regime"],
            regime_data["nifty_close"],
            regime_data["nifty_sma50"],
            regime_data["nifty_sma200"],
            regime_data["india_vix"],
            regime_data["notes"]
        ))

        conn.commit()

        logger.info(f"Regime saved: {regime_data['regime']}")

    except Exception as e:
        logger.error(f"Failed to save regime: {e}")

# ─────────────────────────────────────────────────────────────
# GET LATEST REGIME
# Used by score_stocks.py
# ─────────────────────────────────────────────────────────────

def get_latest_regime(conn: sqlite3.Connection) -> str:
    """
    Returns the most recent market regime stored in regime_log.

    If no regime exists yet (first run / cleared DB),
    it will automatically detect and save one.
    """

    row = conn.execute("""
        SELECT regime
        FROM regime_log
        ORDER BY date DESC
        LIMIT 1
    """).fetchone()

    # If a regime already exists, return it
    if row:
        return row[0]

    # Otherwise detect and store a new regime
    logger.warning("No regime found in DB. Detecting new regime.")

    regime_data = detect_regime(conn)
    save_regime(conn, regime_data)

    return regime_data["regime"]


# ─────────────────────────────────────────────────────────────
# GET REGIME RULES
# Used later by scoring and Telegram alerts
# ─────────────────────────────────────────────────────────────

def get_regime_rules(regime: str) -> dict:
    """
    Returns trading rules for a given regime.

    Example:
    BULL → 4 trades allowed
    BEAR → 0 trades allowed
    """

    return REGIME_TRADING_RULES.get(
        regime,
        REGIME_TRADING_RULES["NEUTRAL"]
    )

# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    conn = get_connection()

    regime_data = detect_regime(conn)

    save_regime(conn, regime_data)

    print("\nMarket Regime:", regime_data["regime"])
    print("Notes:", regime_data["notes"])

    conn.close()