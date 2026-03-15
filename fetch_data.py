# fetch_data.py
# Dual-source OHLCV fetcher — survives GitHub Actions IP rate limits
#
# Strategy:
#   1. PRIMARY  — yfinance.download() in ONE batch call (all tickers at once)
#                 One batch request is far less blocked than 30 individual calls.
#   2. FALLBACK — pandas_datareader stooq for any ticker that failed.
#                 Stooq has no rate limits and works reliably from GitHub IPs.
#   3. INDICES  — Yahoo REST for NIFTY50 + INDIA_VIX (stooq doesn't carry them).
#                 VIX failure is non-fatal — regime engine has safe defaults.

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

IST = pytz.timezone(TIMEZONE)

# Stooq symbol mapping — NSE stocks use .IN suffix
_STOOQ_OVERRIDE = {
    "MM":        "mm.in",   # M&M — ampersand breaks auto-mapping
    "NIFTY50":   None,      # Yahoo only
    "INDIA_VIX": None,      # Yahoo only
}

def _stooq_sym(nse_symbol):
    if nse_symbol in _STOOQ_OVERRIDE:
        return _STOOQ_OVERRIDE[nse_symbol]
    return f"{nse_symbol.lower()}.in"


def _yf_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def today_ist():
    return datetime.now(IST).strftime("%Y-%m-%d")


# SOURCE 1: yfinance batch
def fetch_batch_yfinance(tickers, lookback_days=730):
    end_dt   = datetime.now(IST)
    start_dt = end_dt - timedelta(days=lookback_days)
    yf_list  = list(tickers.values())
    nse_by_yf = {v: k for k, v in tickers.items()}
    result   = {}

    logger.info(f"yfinance batch: {len(yf_list)} tickers in one call")
    try:
        raw = yf.download(
            tickers=yf_list,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            session=_yf_session(),
        )
        if raw.empty:
            logger.warning("yfinance batch: empty response")
            return {s: pd.DataFrame() for s in tickers}

        for yft in yf_list:
            nse_sym = nse_by_yf.get(yft, yft)
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    closes = raw[("Close", yft)]
                    opens  = raw[("Open",  yft)]
                    highs  = raw[("High",  yft)]
                    lows   = raw[("Low",   yft)]
                    vols   = raw[("Volume",yft)]
                    adj    = raw[("Adj Close", yft)] if ("Adj Close", yft) in raw.columns else closes
                else:
                    closes = raw["Close"]
                    opens  = raw["Open"]
                    highs  = raw["High"]
                    lows   = raw["Low"]
                    vols   = raw["Volume"]
                    adj    = raw.get("Adj Close", closes)

                df = pd.DataFrame({
                    "date":      raw.index.strftime("%Y-%m-%d"),
                    "open":      opens.values,
                    "high":      highs.values,
                    "low":       lows.values,
                    "close":     closes.values,
                    "volume":    vols.fillna(0).astype(int).values,
                    "adj_close": adj.values,
                })
                df = df.dropna(subset=["close"])
                df = df[df["close"] > 0]
                result[nse_sym] = df.reset_index(drop=True)
            except Exception as e:
                logger.warning(f"  parse error {nse_sym}: {e}")
                result[nse_sym] = pd.DataFrame()

    except Exception as e:
        logger.error(f"yfinance batch exception: {e}")
        return {s: pd.DataFrame() for s in tickers}

    return result


# SOURCE 2: stooq fallback (per ticker, no rate limits)
def fetch_single_stooq(nse_symbol, lookback_days=730):
    stooq_sym = _stooq_sym(nse_symbol)
    if not stooq_sym:
        return pd.DataFrame()
    try:
        from pandas_datareader import data as pdr
        end_dt   = datetime.now(IST).date()
        start_dt = end_dt - timedelta(days=lookback_days)
        raw = pdr.get_data_stooq(stooq_sym, start=start_dt, end=end_dt)
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = raw.sort_index()
        df = pd.DataFrame({
            "date":      raw.index.strftime("%Y-%m-%d"),
            "open":      raw["Open"].values,
            "high":      raw["High"].values,
            "low":       raw["Low"].values,
            "close":     raw["Close"].values,
            "volume":    raw["Volume"].fillna(0).astype(int).values,
            "adj_close": raw["Close"].values,
        })
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"stooq failed {nse_symbol}: {e}")
        return pd.DataFrame()


# SOURCE 3: Yahoo REST for indices
def fetch_single_yahoo_rest(yf_ticker, retries=3):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}?range=2y&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 429:
                logger.warning(f"{yf_ticker}: rate limited, sleeping 15s")
                time.sleep(15)
                continue
            if r.status_code != 200:
                time.sleep(3)
                continue
            data   = r.json()["chart"]["result"][0]
            ts     = data["timestamp"]
            quote  = data["indicators"]["quote"][0]
            adj    = data["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
            df = pd.DataFrame({
                "date":      pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d"),
                "open":      quote["open"],
                "high":      quote["high"],
                "low":       quote["low"],
                "close":     quote["close"],
                "volume":    [int(v) if v else 0 for v in quote["volume"]],
                "adj_close": adj if adj else quote["close"],
            })
            df = df.dropna(subset=["close"])
            return df[df["close"] > 0].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"{yf_ticker} REST attempt {attempt}: {e}")
            time.sleep(3)
    return pd.DataFrame()


def upsert_prices(conn, symbol, df):
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
                symbol, str(row["date"]),
                float(row["open"])  if not pd.isna(row["open"])  else 0,
                float(row["high"])  if not pd.isna(row["high"])  else 0,
                float(row["low"])   if not pd.isna(row["low"])   else 0,
                float(row["close"]),
                int(row["volume"]),
                float(row["adj_close"]) if not pd.isna(row["adj_close"]) else float(row["close"]),
            ))
            inserted += cursor.rowcount
        except Exception as e:
            logger.error(f"DB insert {symbol} {row.get('date')}: {e}")
    conn.commit()
    return inserted


def get_latest_prices(conn):
    return pd.read_sql_query("""
        SELECT symbol, date, close, volume FROM daily_prices
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) FROM daily_prices GROUP BY symbol
        )
        ORDER BY symbol
    """, conn)


def run_fetch(conn):
    summary = {"date": today_ist(), "total": 0, "success": 0,
               "failed": [], "rows_inserted": 0, "sources": {}}

    all_tickers = dict(ALL_STOCKS)
    all_tickers["NIFTY50"]   = NIFTY50_TICKER
    all_tickers["INDIA_VIX"] = INDIA_VIX_TICKER
    summary["total"] = len(all_tickers)

    logger.info(f"=== Fetch start | {today_ist()} | {len(all_tickers)} tickers ===")

    # Separate equities from indices (^ tickers go to Yahoo REST only)
    equity_tickers = {s: t for s, t in all_tickers.items() if not t.startswith("^")}
    index_tickers  = {s: t for s, t in all_tickers.items() if     t.startswith("^")}

    # Step 1 — batch yfinance for equities
    batch = fetch_batch_yfinance(equity_tickers)
    yf_failed = {}
    for sym, df in batch.items():
        if not df.empty:
            n = upsert_prices(conn, sym, df)
            summary["rows_inserted"] += n
            summary["success"]       += 1
            summary["sources"][sym]   = "yfinance_batch"
            logger.info(f"  ✅ {sym:>14}  rows={len(df):>4}  new={n:>4}  [batch]")
        else:
            yf_failed[sym] = equity_tickers[sym]
            logger.warning(f"  ⚠️  {sym:>14}  batch empty → stooq fallback")

    # Step 2 — stooq fallback
    if yf_failed:
        logger.info(f"Step 2: stooq fallback for {len(yf_failed)} tickers")
        for sym in yf_failed:
            time.sleep(0.5)
            df = fetch_single_stooq(sym)
            if not df.empty:
                n = upsert_prices(conn, sym, df)
                summary["rows_inserted"] += n
                summary["success"]       += 1
                summary["sources"][sym]   = "stooq"
                logger.info(f"  ✅ {sym:>14}  rows={len(df):>4}  new={n:>4}  [stooq]")
            else:
                summary["failed"].append(sym)
                logger.error(f"  ❌ {sym:>14}  both sources failed")

    # Step 3 — Yahoo REST for indices
    logger.info("Step 3: indices via Yahoo REST")
    for sym, ticker in index_tickers.items():
        time.sleep(1)
        df = fetch_single_yahoo_rest(ticker)
        if not df.empty:
            n = upsert_prices(conn, sym, df)
            summary["rows_inserted"] += n
            summary["success"]       += 1
            summary["sources"][sym]   = "yahoo_rest"
            logger.info(f"  ✅ {sym:>14}  rows={len(df):>4}  new={n:>4}  [yahoo_rest]")
        else:
            summary["failed"].append(sym)
            logger.warning(f"  ⚠️  {sym}: index fetch failed — regime will use cached value")

    return summary


def print_summary(s):
    print("\n" + "─" * 58)
    print(f"  Date          : {s['date']}")
    print(f"  Total tickers : {s['total']}")
    print(f"  Successful    : {s['success']}")
    print(f"  Failed        : {len(s['failed'])}")
    print(f"  Rows inserted : {s['rows_inserted']}")
    if s["failed"]:
        print(f"  Failed list   : {', '.join(s['failed'])}")
    src_counts = {}
    for src in s["sources"].values():
        src_counts[src] = src_counts.get(src, 0) + 1
    for src, n in src_counts.items():
        print(f"  Source [{src}]: {n} tickers")
    print("─" * 58)


if __name__ == "__main__":
    conn = get_connection()
    create_tables(conn)
    summary = run_fetch(conn)
    print_summary(summary)

    df = get_latest_prices(conn)
    if not df.empty:
        print("\n📊 Latest prices:\n")
        print(df.to_string(index=False))

    conn.close()

    equity_failures = [f for f in summary["failed"] if f not in ("NIFTY50", "INDIA_VIX")]
    if len(equity_failures) > len(ALL_STOCKS) * 0.5:
        logger.error("Over 50% equity failures — marking run as failed")
        raise SystemExit(1)

    logger.info("fetch_data.py complete.")
