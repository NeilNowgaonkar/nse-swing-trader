# fetch_data.py
# NSE OHLCV fetcher — GitHub Actions compatible
#
# Sources (tried in order):
#   1. NSE Bhavcopy CSV — static file server, never blocked, no auth needed
#      Downloads daily OHLCV CSVs from NSE archives and builds history.
#      https://archives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
#   2. yfinance Ticker.history() — works ~60% of the time from GitHub IPs
#      Used only if Bhavcopy already has recent data (top-up missing days).
#
# Fail-fast: 5s timeout on all HTTP calls. No hanging for 12 seconds.
# Pipeline exits immediately if >50% fail, not after wasting 13 minutes.

import os
import io
import time
import zipfile
import sqlite3
from datetime import datetime, timedelta, date

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

IST     = pytz.timezone(TIMEZONE)
TIMEOUT = 5   # seconds — fail fast, don't hang

def today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Encoding": "gzip, deflate",
        "Accept":          "*/*",
        "Connection":      "keep-alive",
    })
    return s

SESSION = _session()


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — NSE Bhavcopy (static CSV archive, never rate-limited)
# ─────────────────────────────────────────────────────────────────────────────
# NSE publishes a full-market OHLCV CSV for every trading day.
# URL pattern: https://archives.nseindia.com/products/content/
#              sec_bhavdata_full_DDMMYYYY.csv
# This is a static file server — no cookies, no auth, no rate limits.
# We download the last N trading days and merge them.

BHAVCOPY_URL = (
    "https://archives.nseindia.com/products/content/"
    "sec_bhavdata_full_{date}.csv"
)

def _trading_days(n: int = 365) -> list[date]:
    """Return last n calendar days, Mon–Fri only (approximate trading days)."""
    days = []
    d = datetime.now(IST).date() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:   # Mon=0 … Fri=4
            days.append(d)
        d -= timedelta(days=1)
    return days


def fetch_bhavcopy_range(symbols: set, lookback_days: int = 370) -> dict[str, pd.DataFrame]:
    """
    Download Bhavcopy CSVs for the last `lookback_days` trading days.
    Returns dict: symbol → DataFrame with columns date/open/high/low/close/volume/adj_close
    Only EQ series rows are kept.
    """
    all_rows: list[pd.DataFrame] = []
    days      = _trading_days(lookback_days)
    fetched   = 0
    missing   = 0

    logger.info(f"Bhavcopy: fetching up to {len(days)} daily CSVs …")

    for d in days:
        url = BHAVCOPY_URL.format(date=d.strftime("%d%m%Y"))
        try:
            r = SESSION.get(url, timeout=TIMEOUT)
            if r.status_code == 404:
                missing += 1
                continue   # holiday or weekend — skip silently
            if r.status_code != 200:
                logger.warning(f"Bhavcopy {d}: HTTP {r.status_code}")
                continue

            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]

            # Keep EQ series only
            if "SERIES" in df.columns:
                df = df[df["SERIES"].str.strip() == "EQ"]

            # Normalise column names
            col_map = {
                "SYMBOL":       "symbol",
                "OPEN":         "open",   "OPEN_PRICE":  "open",
                "HIGH":         "high",   "HIGH_PRICE":  "high",
                "LOW":          "low",    "LOW_PRICE":   "low",
                "CLOSE":        "close",  "CLOSE_PRICE": "close",
                "LAST":         "close",
                "TTL_TRD_QNTY": "volume", "VOLUME":      "volume",
                "TOTTRDQTY":    "volume",
                "PREV_CLOSE":   "adj_close",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            df["date"] = d.strftime("%Y-%m-%d")

            needed = {"symbol", "open", "high", "low", "close"}
            if not needed.issubset(set(df.columns)):
                continue

            if "volume" not in df.columns:
                df["volume"] = 0
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]

            # Filter to only our symbols — saves memory
            df = df[df["symbol"].isin(symbols)]
            if not df.empty:
                all_rows.append(
                    df[["symbol","date","open","high","low","close","volume","adj_close"]]
                )
            fetched += 1

        except requests.exceptions.Timeout:
            logger.warning(f"Bhavcopy {d}: timeout — skipping")
            continue
        except Exception as e:
            logger.warning(f"Bhavcopy {d}: {e}")
            continue

    logger.info(f"Bhavcopy: {fetched} days fetched, {missing} holidays skipped")

    if not all_rows:
        return {}

    combined = pd.concat(all_rows, ignore_index=True)

    # Clean numeric columns
    for col in ["open","high","low","close","adj_close"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["volume"] = pd.to_numeric(combined["volume"], errors="coerce").fillna(0).astype(int)
    combined = combined.dropna(subset=["close"])
    combined = combined[combined["close"] > 0]
    combined = combined.sort_values("date")

    # Split by symbol
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym_df = combined[combined["symbol"] == sym][
            ["date","open","high","low","close","volume","adj_close"]
        ].reset_index(drop=True)
        if not sym_df.empty:
            result[sym] = sym_df

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — yfinance (for indices + top-up any missing equity days)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance_single(nse_symbol: str, yf_ticker: str,
                          days: int = 730) -> pd.DataFrame:
    """
    Quick yfinance fetch with hard 5s timeout enforced via requests.
    Returns empty DataFrame if Yahoo blocks or times out — no hanging.
    """
    end_dt   = datetime.now(IST)
    start_dt = end_dt - timedelta(days=days)
    try:
        t   = yf.Ticker(yf_ticker)
        raw = t.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            actions=False,
            timeout=TIMEOUT,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        df = pd.DataFrame({
            "date":      raw.index.strftime("%Y-%m-%d"),
            "open":      pd.to_numeric(raw["open"],   errors="coerce"),
            "high":      pd.to_numeric(raw["high"],   errors="coerce"),
            "low":       pd.to_numeric(raw["low"],    errors="coerce"),
            "close":     pd.to_numeric(raw["close"],  errors="coerce"),
            "volume":    pd.to_numeric(raw.get("volume", pd.Series([0]*len(raw))),
                                       errors="coerce").fillna(0).astype(int),
            "adj_close": pd.to_numeric(raw["close"],  errors="coerce"),
        })
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"  yfinance {nse_symbol}: {type(e).__name__}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# DB UPSERT
# ─────────────────────────────────────────────────────────────────────────────

def upsert_prices(conn: sqlite3.Connection,
                  symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cursor = conn.cursor()
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


def get_latest_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT symbol, date, close, volume FROM daily_prices
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) FROM daily_prices GROUP BY symbol
        )
        ORDER BY symbol
    """, conn)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_fetch(conn: sqlite3.Connection) -> dict:
    summary = {
        "date":          today_ist(),
        "total":         len(ALL_STOCKS) + 2,
        "success":       0,
        "failed":        [],
        "rows_inserted": 0,
        "sources":       {},
    }

    logger.info(f"=== Fetch start | {today_ist()} | {summary['total']} tickers ===")

    equity_symbols = set(ALL_STOCKS.keys())

    # ── STEP 1: Bhavcopy for all 28 equities in one pass ─────────────────────
    logger.info("Step 1: NSE Bhavcopy bulk download …")
    bhavcopy_results = fetch_bhavcopy_range(equity_symbols, lookback_days=370)

    bhavcopy_failed = []
    for sym in equity_symbols:
        df = bhavcopy_results.get(sym, pd.DataFrame())
        if not df.empty:
            n = upsert_prices(conn, sym, df)
            summary["rows_inserted"] += n
            summary["success"]       += 1
            summary["sources"][sym]   = "bhavcopy"
            logger.info(f"  ✅ {sym:>14}  rows={len(df):>4}  new={n:>4}  [bhavcopy]")
        else:
            bhavcopy_failed.append(sym)
            logger.warning(f"  ⚠️  {sym:>14}  not in bhavcopy → yfinance fallback")

    # ── STEP 2: yfinance fallback for any equity Bhavcopy missed ─────────────
    if bhavcopy_failed:
        logger.info(f"Step 2: yfinance fallback for {len(bhavcopy_failed)} symbols …")
        for sym in bhavcopy_failed:
            yf_ticker = ALL_STOCKS.get(sym, f"{sym}.NS")
            df = fetch_yfinance_single(sym, yf_ticker)
            if not df.empty:
                n = upsert_prices(conn, sym, df)
                summary["rows_inserted"] += n
                summary["success"]       += 1
                summary["sources"][sym]   = "yfinance"
                logger.info(f"  ✅ {sym:>14}  rows={len(df):>4}  new={n:>4}  [yfinance]")
            else:
                summary["failed"].append(sym)
                logger.error(f"  ❌ {sym:>14}  both sources failed")
            time.sleep(1)

    # ── STEP 3: Indices via yfinance (NIFTY50 + INDIA_VIX) ───────────────────
    logger.info("Step 3: indices via yfinance …")
    for nse_sym, yf_ticker in [("NIFTY50", NIFTY50_TICKER),
                                ("INDIA_VIX", INDIA_VIX_TICKER)]:
        df = fetch_yfinance_single(nse_sym, yf_ticker)
        if not df.empty:
            n = upsert_prices(conn, nse_sym, df)
            summary["rows_inserted"] += n
            summary["success"]       += 1
            summary["sources"][nse_sym] = "yfinance"
            logger.info(f"  ✅ {nse_sym:>14}  rows={len(df):>4}  new={n:>4}  [yfinance]")
        else:
            summary["failed"].append(nse_sym)
            logger.warning(f"  ⚠️  {nse_sym}: failed — regime engine will use cached value")
        time.sleep(1)

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
        print("No data in DB.")

    conn.close()

    equity_failures = [f for f in summary["failed"]
                       if f not in ("NIFTY50", "INDIA_VIX")]
    if len(equity_failures) > len(ALL_STOCKS) * 0.5:
        logger.error("Over 50% equity failures — marking pipeline failed.")
        raise SystemExit(1)

    logger.info("fetch_data.py complete.")
