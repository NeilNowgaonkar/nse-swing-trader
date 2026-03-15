# fetch_data.py
# NSE OHLCV fetcher — GitHub Actions compatible
#
# Root causes of previous failures:
#   - yfinance 0.2.51 has YFTzMissingError bug on Linux for .NS tickers
#   - stooq dropped Indian .IN symbol support in 2024
#   - Yahoo REST v8 blocks GitHub Actions IPs outright
#
# Solution:
#   - yfinance 0.2.61+ (timezone bug fixed, works on Linux/GitHub Actions)
#   - Download tickers one by one with a delay — more reliable than batch
#     for .NS symbols on the fixed version
#   - Retry logic with exponential backoff
#   - NSE-Python REST as final fallback for any remaining failures

import os
import time
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests
import yfinance as yf
from loguru import logger

from config import (
    ALL_STOCKS, DATA_DIR, DB_PATH,
    INDIA_VIX_TICKER, LOG_DIR, NIFTY50_TICKER, TIMEZONE,
)
from create_db import get_connection, create_tables

os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logger.add(
    os.path.join(LOG_DIR, "fetch_{time:YYYY-MM-DD}.log"),
    rotation="1 day", retention="14 days", level="INFO",
)

IST      = pytz.timezone(TIMEZONE)
START_DT = (datetime.now(IST) - timedelta(days=730)).strftime("%Y-%m-%d")
END_DT   = datetime.now(IST).strftime("%Y-%m-%d")


def today_ist():
    return datetime.now(IST).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN DATAFRAME HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df_raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalise a raw yfinance DataFrame into our standard schema.
    Handles both single-ticker (flat columns) and the edge cases
    where yfinance wraps columns in a MultiIndex.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    # Flatten MultiIndex if present
    if isinstance(df_raw.columns, pd.MultiIndex):
        # MultiIndex: (metric, ticker) — extract just this ticker
        try:
            df_raw.columns = df_raw.columns.get_level_values(0)
        except Exception:
            return pd.DataFrame()

    # Rename to lowercase
    df_raw.columns = [c.lower().replace(" ", "_") for c in df_raw.columns]

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df_raw.columns)):
        logger.warning(f"{ticker}: missing columns {required - set(df_raw.columns)}")
        return pd.DataFrame()

    df = pd.DataFrame()
    df["date"]      = df_raw.index.strftime("%Y-%m-%d")
    df["open"]      = pd.to_numeric(df_raw["open"],   errors="coerce")
    df["high"]      = pd.to_numeric(df_raw["high"],   errors="coerce")
    df["low"]       = pd.to_numeric(df_raw["low"],    errors="coerce")
    df["close"]     = pd.to_numeric(df_raw["close"],  errors="coerce")
    df["volume"]    = pd.to_numeric(df_raw["volume"], errors="coerce").fillna(0).astype(int)
    df["adj_close"] = pd.to_numeric(
        df_raw.get("adj_close", df_raw.get("adjclose", df_raw["close"])),
        errors="coerce"
    )

    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — yfinance single-ticker with retry
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance(nse_symbol: str, yf_ticker: str,
                   retries: int = 3) -> pd.DataFrame:
    """
    Fetch one ticker using yfinance.Ticker.history().
    Uses .history() instead of .download() — more stable for .NS tickers.
    """
    for attempt in range(1, retries + 1):
        try:
            ticker_obj = yf.Ticker(yf_ticker)
            raw = ticker_obj.history(
                start=START_DT,
                end=END_DT,
                interval="1d",
                auto_adjust=True,   # adj_close baked into Close
                actions=False,
            )
            if raw is None or raw.empty:
                logger.warning(f"  yfinance empty for {nse_symbol} attempt {attempt}")
                time.sleep(2 * attempt)
                continue

            # .history() with auto_adjust=True: Close is already adjusted
            raw["adj_close"] = raw["Close"]
            df = _clean(raw, nse_symbol)
            if not df.empty:
                return df

            time.sleep(2 * attempt)

        except Exception as e:
            logger.warning(f"  yfinance {nse_symbol} attempt {attempt}: {e}")
            time.sleep(3 * attempt)

    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — NSE unofficial JSON API (fallback, no rate limits from GH IPs)
# ─────────────────────────────────────────────────────────────────────────────

# NSE's public bhavcopy CSV — works reliably from GitHub Actions.
# We use the NSE India public API for historical OHLCV.
# This is the same data NSE publishes daily for free.

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":         "application/json, text/plain, */*",
    "Accept-Language":"en-US,en;q=0.9",
    "Referer":        "https://www.nseindia.com/",
    "Origin":         "https://www.nseindia.com",
    "Connection":     "keep-alive",
}

def _nse_session() -> requests.Session:
    """
    NSE requires a session cookie obtained by hitting the homepage first.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Get cookies by hitting the main page
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
    except Exception:
        pass
    return session


_nse_sess = None   # module-level singleton so we only initialise once

def fetch_nse_api(nse_symbol: str, retries: int = 2) -> pd.DataFrame:
    """
    Fetch 1-year OHLCV from NSE India's public historical data API.
    Endpoint: /api/historical/cm/equity
    """
    global _nse_sess
    if _nse_sess is None:
        logger.info("  Initialising NSE session …")
        _nse_sess = _nse_session()

    end_date   = datetime.now(IST)
    start_date = end_date - timedelta(days=365)

    url = "https://www.nseindia.com/api/historical/cm/equity"
    params = {
        "symbol":   nse_symbol,
        "series":   '["EQ"]',
        "from":     start_date.strftime("%d-%m-%Y"),
        "to":       end_date.strftime("%d-%m-%Y"),
        "csv":      "true",
    }

    for attempt in range(1, retries + 1):
        try:
            r = _nse_sess.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(10)
                continue
            if r.status_code != 200:
                logger.warning(f"  NSE API {nse_symbol}: HTTP {r.status_code}")
                time.sleep(3)
                continue

            # Response is CSV text
            from io import StringIO
            raw = pd.read_csv(StringIO(r.text))

            if raw.empty:
                return pd.DataFrame()

            # NSE column names vary — normalise
            raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

            # Map NSE column names to our schema
            col_map = {
                "date": "date",
                "open_price": "open",   "open": "open",
                "high_price": "high",   "high": "high",
                "low_price":  "low",    "low":  "low",
                "close_price":"close",  "close":"close",
                "prev_close": "adj_close",
                "total_traded_quantity": "volume",
                "tottrdqty": "volume",
            }
            raw = raw.rename(columns=col_map)

            needed = ["date","open","high","low","close","volume"]
            missing = [c for c in needed if c not in raw.columns]
            if missing:
                logger.warning(f"  NSE API {nse_symbol}: missing cols {missing}")
                return pd.DataFrame()

            raw["date"] = pd.to_datetime(
                raw["date"], dayfirst=True, errors="coerce"
            ).dt.strftime("%Y-%m-%d")

            if "adj_close" not in raw.columns:
                raw["adj_close"] = raw["close"]

            df = raw[["date","open","high","low","close","volume","adj_close"]].copy()
            for col in ["open","high","low","close","adj_close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            logger.warning(f"  NSE API {nse_symbol} attempt {attempt}: {e}")
            time.sleep(3)

    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# DB UPSERT
# ─────────────────────────────────────────────────────────────────────────────

def upsert_prices(conn: sqlite3.Connection,
                  symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cursor   = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO daily_prices
                    (symbol, date, open, high, low, close, volume, adj_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                str(row["date"]),
                float(row["open"])      if pd.notna(row["open"])      else 0.0,
                float(row["high"])      if pd.notna(row["high"])      else 0.0,
                float(row["low"])       if pd.notna(row["low"])       else 0.0,
                float(row["close"]),
                int(row["volume"]),
                float(row["adj_close"]) if pd.notna(row["adj_close"]) else float(row["close"]),
            ))
            inserted += cursor.rowcount
        except Exception as e:
            logger.error(f"DB insert {symbol} {row.get('date')}: {e}")
    conn.commit()
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_fetch(conn: sqlite3.Connection) -> dict:
    summary = {
        "date":          today_ist(),
        "total":         0,
        "success":       0,
        "failed":        [],
        "rows_inserted": 0,
        "sources":       {},
    }

    all_tickers = dict(ALL_STOCKS)
    all_tickers["NIFTY50"]   = NIFTY50_TICKER    # ^NSEI
    all_tickers["INDIA_VIX"] = INDIA_VIX_TICKER  # ^INDIAVIX
    summary["total"] = len(all_tickers)

    logger.info(f"=== Fetch start | {today_ist()} | {len(all_tickers)} tickers ===")

    # Split: equity vs indices (indices don't exist on NSE API)
    equity_symbols = {s: t for s, t in all_tickers.items() if not t.startswith("^")}
    index_symbols  = {s: t for s, t in all_tickers.items() if     t.startswith("^")}

    # ── Equities: yfinance first, NSE API fallback ───────────────────────────
    for nse_sym, yf_ticker in equity_symbols.items():
        logger.info(f"  Fetching {nse_sym} …")
        df = fetch_yfinance(nse_sym, yf_ticker)

        if not df.empty:
            n = upsert_prices(conn, nse_sym, df)
            summary["rows_inserted"] += n
            summary["success"]       += 1
            summary["sources"][nse_sym] = "yfinance"
            logger.info(f"  ✅ {nse_sym:>14}  rows={len(df):>4}  new={n:>4}  [yfinance]")
        else:
            # Fallback: NSE India public API
            logger.warning(f"  yfinance failed {nse_sym} — trying NSE API …")
            df = fetch_nse_api(nse_sym)
            if not df.empty:
                n = upsert_prices(conn, nse_sym, df)
                summary["rows_inserted"] += n
                summary["success"]       += 1
                summary["sources"][nse_sym] = "nse_api"
                logger.info(f"  ✅ {nse_sym:>14}  rows={len(df):>4}  new={n:>4}  [nse_api]")
            else:
                summary["failed"].append(nse_sym)
                logger.error(f"  ❌ {nse_sym:>14}  both sources failed")

        time.sleep(0.8)   # polite delay between tickers

    # ── Indices: yfinance only ───────────────────────────────────────────────
    logger.info("  Fetching indices …")
    for nse_sym, yf_ticker in index_symbols.items():
        time.sleep(2)
        df = fetch_yfinance(nse_sym, yf_ticker)
        if not df.empty:
            n = upsert_prices(conn, nse_sym, df)
            summary["rows_inserted"] += n
            summary["success"]       += 1
            summary["sources"][nse_sym] = "yfinance"
            logger.info(f"  ✅ {nse_sym:>14}  rows={len(df):>4}  new={n:>4}  [yfinance]")
        else:
            summary["failed"].append(nse_sym)
            logger.warning(
                f"  ⚠️  {nse_sym}: index fetch failed — "
                f"regime engine will use last cached value"
            )

    return summary


def print_summary(s: dict) -> None:
    print("\n" + "─" * 58)
    print(f"  Date          : {s['date']}")
    print(f"  Total tickers : {s['total']}")
    print(f"  Successful    : {s['success']}")
    print(f"  Failed        : {len(s['failed'])}")
    print(f"  Rows inserted : {s['rows_inserted']}")
    if s["failed"]:
        print(f"  Failed list   : {', '.join(s['failed'])}")
    src_counts: dict = {}
    for src in s["sources"].values():
        src_counts[src] = src_counts.get(src, 0) + 1
    for src, n in src_counts.items():
        print(f"  Source [{src}]: {n} tickers")
    print("─" * 58)


def get_latest_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT symbol, date, close, volume FROM daily_prices
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) FROM daily_prices GROUP BY symbol
        )
        ORDER BY symbol
    """, conn)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conn = get_connection()
    create_tables(conn)

    summary = run_fetch(conn)
    print_summary(summary)

    df = get_latest_prices(conn)
    if not df.empty:
        print("\n📊 Latest prices:\n")
        print(df.to_string(index=False))
    else:
        print("No data found in DB.")

    conn.close()

    # Fail the pipeline only if majority of equities failed
    equity_failures = [f for f in summary["failed"]
                       if f not in ("NIFTY50", "INDIA_VIX")]
    if len(equity_failures) > len(ALL_STOCKS) * 0.5:
        logger.error("Over 50% equity failures — marking pipeline failed.")
        raise SystemExit(1)

    logger.info("fetch_data.py complete.")
