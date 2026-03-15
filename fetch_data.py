# fetch_data.py
# NSE data fetcher (fast + retry + safe DB inserts)

import sqlite3
import os
import time
import requests
import pandas as pd
import pytz
from datetime import datetime
from loguru import logger

from config import (
    ALL_STOCKS,
    DATA_DIR,
    DB_PATH,
    INDIA_VIX_TICKER,
    LOG_DIR,
    NIFTY50_TICKER,
    TIMEZONE
)

from create_db import get_connection, create_tables


IST = pytz.timezone(TIMEZONE)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logger.add(
    os.path.join(LOG_DIR, "fetch_{time:YYYY-MM-DD}.log"),
    rotation="1 day",
    retention="14 days",
    level="INFO"
)

# -------------------------------------------
# Helpers
# -------------------------------------------

def today_ist():
    return datetime.now(IST).strftime("%Y-%m-%d")


def fetch_from_yahoo(ticker: str, retries=2):

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for attempt in range(retries):

        try:

            r = requests.get(url, headers=headers, timeout=10)

            if r.status_code != 200:
                logger.warning(f"{ticker} HTTP {r.status_code} (attempt {attempt+1})")
                time.sleep(1)
                continue

            data = r.json()

            result = data["chart"]["result"][0]

            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]

            adjclose = None
            if "adjclose" in result["indicators"]:
                adjclose = result["indicators"]["adjclose"][0]["adjclose"]

            df = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s"),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"]
            })

            if adjclose:
                df["adj_close"] = adjclose
            else:
                df["adj_close"] = df["close"]

            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            df = df.dropna(subset=["close"])

            return df.reset_index(drop=True)

        except Exception as e:

            logger.warning(f"{ticker} failed attempt {attempt+1}: {e}")
            time.sleep(1)

    return pd.DataFrame()


def upsert_prices(conn, symbol, df):

    if df.empty:
        return 0

    cursor = conn.cursor()

    inserted = 0

    for _, row in df.iterrows():

        cursor.execute("""
        INSERT OR IGNORE INTO daily_prices
        (symbol, date, open, high, low, close, volume, adj_close)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            row["date"],
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
            row["adj_close"]
        ))

        inserted += cursor.rowcount

    conn.commit()

    return inserted


# -------------------------------------------
# Main pipeline
# -------------------------------------------

def run_fetch(conn):

    summary = {
        "date": today_ist(),
        "total_tickers": 0,
        "success": 0,
        "failed": [],
        "rows_inserted": 0
    }

    all_tickers = dict(ALL_STOCKS)
    all_tickers["NIFTY50"] = NIFTY50_TICKER
    all_tickers["INDIA_VIX"] = INDIA_VIX_TICKER

    summary["total_tickers"] = len(all_tickers)

    logger.info(f"Starting fetch for {len(all_tickers)} tickers")

    failed_retry = []

    for symbol, ticker in all_tickers.items():

        logger.info(f"Fetching {symbol} ({ticker})")

        df = fetch_from_yahoo(ticker)

        if df.empty:

            logger.warning(f"{symbol} failed — will retry later")
            failed_retry.append((symbol, ticker))
            continue

        inserted = upsert_prices(conn, symbol, df)

        summary["rows_inserted"] += inserted
        summary["success"] += 1

        logger.info(f"{symbol} rows={len(df)} inserted={inserted}")

        time.sleep(0.3)

    # retry failed tickers once more
    if failed_retry:

        logger.info("Retrying failed tickers...")

        for symbol, ticker in failed_retry:

            df = fetch_from_yahoo(ticker)

            if df.empty:

                summary["failed"].append(symbol)
                logger.error(f"{symbol} permanently failed")
                continue

            inserted = upsert_prices(conn, symbol, df)

            summary["rows_inserted"] += inserted
            summary["success"] += 1

    return summary


def print_summary(summary):

    print("\n----------------------------------------")
    print("Date:", summary["date"])
    print("Total tickers:", summary["total_tickers"])
    print("Successful:", summary["success"])
    print("Failed:", len(summary["failed"]))
    print("Rows inserted:", summary["rows_inserted"])

    if summary["failed"]:
        print("Failed tickers:", summary["failed"])

    print("----------------------------------------\n")


def get_latest_prices(conn):

    query = """
    SELECT symbol, date, close, volume
    FROM daily_prices
    WHERE (symbol, date) IN (
        SELECT symbol, MAX(date)
        FROM daily_prices
        GROUP BY symbol
    )
    ORDER BY symbol
    """

    return pd.read_sql_query(query, conn)


# -------------------------------------------
# Entry point
# -------------------------------------------

if __name__ == "__main__":

    conn = get_connection()

    create_tables(conn)

    summary = run_fetch(conn)

    print_summary(summary)

    print("Latest prices:\n")

    df = get_latest_prices(conn)

    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No data found")

    conn.close()

    logger.info("Fetch completed")