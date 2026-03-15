# create_db.py
# Creates all 4 SQLite tables for the NSE Swing Trader
# Safe to run multiple times — uses IF NOT EXISTS
# Run once before anything else: python create_db.py

import sqlite3
import os
from config import DB_PATH, DATA_DIR
from loguru import logger

# ── Ensure data directory exists ──────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row   # lets you access columns by name
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    # ── TABLE 1: daily_prices ──────────────────────────────────────────────────
    # Stores OHLCV for all 28 stocks. Fetched daily by fetch_data.py
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            date        TEXT    NOT NULL,          -- YYYY-MM-DD
            open        REAL    NOT NULL,
            high        REAL    NOT NULL,
            low         REAL    NOT NULL,
            close       REAL    NOT NULL,
            volume      INTEGER NOT NULL,
            adj_close   REAL,                      -- adjusted close from yfinance
            created_at  TEXT    DEFAULT (datetime('now','localtime')),
            UNIQUE(symbol, date)                   -- no duplicate rows
        )
    """)

    # ── TABLE 2: trades ───────────────────────────────────────────────────────
    # Portfolio memory — every open and closed trade.
    # Updated via /bought and /sold Telegram commands (Phase 3).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT    NOT NULL,
            bucket          TEXT    NOT NULL,      -- large_cap / momentum / midcap_alpha / defensive
            direction       TEXT    NOT NULL DEFAULT 'LONG',   -- LONG only for now
            status          TEXT    NOT NULL DEFAULT 'OPEN',   -- OPEN / CLOSED / PAUSED
            entry_price     REAL    NOT NULL,
            entry_date      TEXT    NOT NULL,      -- YYYY-MM-DD
            qty             INTEGER NOT NULL,
            stop_loss       REAL    NOT NULL,
            target          REAL    NOT NULL,
            exit_price      REAL,                  -- filled on /sold
            exit_date       TEXT,                  -- YYYY-MM-DD
            pnl             REAL,                  -- calculated on close
            pnl_pct         REAL,                  -- % gain/loss
            exit_reason     TEXT,                  -- TARGET / STOPLOSS / MANUAL / TIMEOUT
            score_at_entry  REAL,                  -- factor score that triggered the trade
            regime_at_entry TEXT,                  -- market regime when entered
            notes           TEXT,                  -- free-text from Telegram
            created_at      TEXT DEFAULT (datetime('now','localtime')),
            updated_at      TEXT DEFAULT (datetime('now','localtime'))
        )
    """)

    # ── TABLE 3: daily_scores ─────────────────────────────────────────────────
    # Factor model output for each stock each day.
    # Populated by score_stocks.py (Phase 2).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_scores (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT    NOT NULL,
            date            TEXT    NOT NULL,       -- YYYY-MM-DD
            total_score     REAL,                   -- 0–100
            momentum_score  REAL,                   -- individual factor scores
            trend_score     REAL,
            volume_score    REAL,
            rsi_score       REAL,
            macd_score      REAL,
            bb_score        REAL,
            signal          TEXT,                   -- BUY / WATCH / IGNORE
            regime          TEXT,                   -- market regime at scoring time
            gate_passed     INTEGER DEFAULT 0,      -- 1 if all 3 hard gates passed
            created_at      TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(symbol, date)
        )
    """)

    # ── TABLE 4: regime_log ───────────────────────────────────────────────────
    # Daily market regime snapshots. Used for filtering trades and backtesting.
    # Populated by regime_engine.py (Phase 2).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS regime_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT    NOT NULL UNIQUE,  -- YYYY-MM-DD
            regime          TEXT    NOT NULL,          -- BULL/NEUTRAL/SIDEWAYS/HIGH_VIX/BEAR
            nifty_close     REAL,
            nifty_sma50     REAL,
            nifty_sma200    REAL,
            india_vix       REAL,
            adv_decline     REAL,                     -- breadth ratio (Phase 2)
            notes           TEXT,
            created_at      TEXT DEFAULT (datetime('now','localtime'))
        )
    """)

    conn.commit()
    logger.info("All 4 tables created (or already exist).")


def verify_tables(conn: sqlite3.Connection) -> None:
    """Print table names and column counts as a quick sanity check."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()

    print("\n✅ Database verified at:", DB_PATH)
    print(f"   Tables found: {len(tables)}\n")

    expected = {"daily_prices", "trades", "daily_scores", "regime_log"}
    found    = set()

    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = cursor.fetchall()
        print(f"   📋 {table_name:<20} — {len(cols)} columns")
        found.add(table_name)

    missing = expected - found
    if missing:
        print(f"\n⚠️  Missing tables: {missing}")
    else:
        print("\n   All expected tables present. ✅")


if __name__ == "__main__":
    logger.info(f"Initialising database at: {DB_PATH}")
    conn = get_connection()
    create_tables(conn)
    verify_tables(conn)
    conn.close()
    logger.info("Database setup complete.")
