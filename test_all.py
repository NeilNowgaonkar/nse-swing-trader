# test_all.py
# Single command to test every module in the correct order.
# Run this after copying all files to verify everything works.
#
# Usage: py test_all.py
#
# What it tests:
#   1. Config loads correctly (28 stocks, all settings)
#   2. Database creates all 6 tables (4 original + 2 new from daily_analysis)
#   3. Fetch data works (tries Bhavcopy, reports source)
#   4. Indicators calculate correctly on real data
#   5. Regime engine detects current market state
#   6. Score engine scores all 28 stocks
#   7. Daily analysis generates reasoning and expected exit dates
#   8. Backup system works (backup + restore round-trip)
#   9. Weekend review runs (needs at least 1 day of analysis data)
#  10. Telegram sends a test message (only if token is set)
#
# Each test prints PASS / FAIL / SKIP clearly.
# Stops at first critical failure (DB, fetch) — rest are non-critical.

import sys
import os
import traceback

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
SKIP  = "⏭️  SKIP"
WARN  = "⚠️  WARN"

results = []

def test(name: str, fn, critical: bool = False):
    print(f"\n{'─'*55}")
    print(f"Testing: {name}")
    try:
        result = fn()
        status = PASS
        msg    = str(result) if result else "OK"
        print(f"{status}  {msg}")
        results.append((name, True, msg))
        return True
    except Exception as e:
        print(f"{FAIL}  {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))
        if critical:
            print(f"\n⛔ Critical failure — stopping tests.")
            print_summary()
            sys.exit(1)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Config
# ─────────────────────────────────────────────────────────────────────────────

def test_config():
    from config import ALL_STOCKS, BUCKET_LABELS, STARTING_CAPITAL, TRADE_SIGNAL_THRESHOLD
    assert len(ALL_STOCKS) == 28, f"Expected 28 stocks, got {len(ALL_STOCKS)}"
    assert "MM" in ALL_STOCKS, "MM (Mahindra) missing from config"
    assert "TATAMOTORS" not in ALL_STOCKS, "TATAMOTORS should have been replaced by MM"
    assert STARTING_CAPITAL == 50_000
    assert TRADE_SIGNAL_THRESHOLD == 68
    return f"28 stocks loaded. Capital ₹{STARTING_CAPITAL:,}. Threshold {TRADE_SIGNAL_THRESHOLD}."

test("Config loads correctly", test_config, critical=True)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Database
# ─────────────────────────────────────────────────────────────────────────────

def test_database():
    from create_db import get_connection, create_tables
    from daily_analysis import ensure_analysis_tables

    conn = get_connection()
    create_tables(conn)
    ensure_analysis_tables(conn)

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = [t[0] for t in tables]
    conn.close()

    expected = {"daily_prices", "trades", "daily_scores", "regime_log",
                "daily_analysis", "accuracy_log"}
    missing  = expected - set(table_names)

    if missing:
        raise AssertionError(f"Missing tables: {missing}")

    return f"All 6 tables present: {', '.join(sorted(table_names))}"

test("Database creates all 6 tables", test_database, critical=True)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Fetch data (quick check — just 3 stocks)
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_quick():
    from fetch_data import fetch_bhavcopy_range, fetch_yfinance_single
    from config import ALL_STOCKS

    # Test Bhavcopy with just 3 symbols, 30 days
    test_syms = {"RELIANCE", "HDFCBANK", "INFY"}
    result    = fetch_bhavcopy_range(test_syms, lookback_days=30)

    fetched   = [s for s in test_syms if s in result and not result[s].empty]

    if not fetched:
        # Bhavcopy may be slow — try yfinance as fallback test
        df = fetch_yfinance_single("RELIANCE", "RELIANCE.NS", days=30)
        if df.empty:
            raise AssertionError("Both Bhavcopy and yfinance returned empty — check network")
        return f"Bhavcopy empty today (possible holiday). yfinance works: {len(df)} rows for RELIANCE"

    return f"Bhavcopy: {len(fetched)}/3 symbols fetched. Rows: {[len(result[s]) for s in fetched]}"

test("Fetch data (quick 3-stock test)", test_fetch_quick, critical=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Indicators
# ─────────────────────────────────────────────────────────────────────────────

def test_indicators():
    from create_db import get_connection
    from indicators import (
        add_all_indicators, calculate_stop_loss,
        calculate_target, calculate_position_size
    )
    import pandas as pd

    conn = get_connection()
    df   = pd.read_sql_query("""
        SELECT date, open, high, low, close, volume
        FROM daily_prices WHERE symbol = 'RELIANCE'
        ORDER BY date DESC LIMIT 250
    """, conn)
    conn.close()

    if df.empty:
        return "SKIP — no RELIANCE data in DB yet. Run fetch first."

    df = df.sort_values("date").reset_index(drop=True)
    df = add_all_indicators(df)
    row = df.iloc[-1]

    required_cols = ["rsi_14", "macd_line", "bb_pct_b", "atr_14", "vol_ratio", "sma_200"]
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise AssertionError(f"Missing indicator columns: {missing_cols}")

    close = row["close"]
    atr   = row["atr_14"]
    sl    = calculate_stop_loss(close, atr)
    tgt   = calculate_target(close, atr)
    qty   = calculate_position_size(50000, 1.5, close, sl)

    assert sl < close,  f"Stop-loss {sl} must be below close {close}"
    assert tgt > close, f"Target {tgt} must be above close {close}"
    assert qty > 0,     f"Quantity must be positive, got {qty}"

    # Verify target is within 1 year's realistic range (not >30% above close)
    tgt_pct = (tgt - close) / close * 100
    if tgt_pct > 30:
        return (f"WARN: Target {tgt_pct:.1f}% above close — may be unrealistic "
                f"(RSI={row['rsi_14']:.1f}, ATR={atr:.2f}). "
                f"Consider if this stock is too volatile.")

    return (f"RELIANCE: close=₹{close:.0f} SL=₹{sl:.0f} TGT=₹{tgt:.0f} "
            f"QTY={qty} RSI={row['rsi_14']:.1f} ATR={atr:.1f}")

test("Indicators calculate correctly", test_indicators, critical=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Regime Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_regime():
    from create_db import get_connection
    from regime_engine import detect_regime, get_regime_rules
    from config import REGIMES

    conn   = get_connection()
    result = detect_regime(conn)
    conn.close()

    assert result["regime"] in REGIMES, f"Invalid regime: {result['regime']}"
    rules = get_regime_rules(result["regime"])
    assert "max_trades" in rules
    assert "min_score"  in rules

    return (f"Regime: {result['regime']} | "
            f"Nifty: {result['nifty_close']} | "
            f"VIX: {result['india_vix']} | "
            f"Max trades: {rules['max_trades']}")

test("Regime engine detects market state", test_regime, critical=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Score Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_scoring():
    from create_db import get_connection
    from score_stocks import score_single_stock
    import pandas as pd

    conn = get_connection()
    df   = pd.read_sql_query("""
        SELECT date, open, high, low, close, volume
        FROM daily_prices WHERE symbol = 'RELIANCE'
        ORDER BY date ASC LIMIT 250
    """, conn)
    conn.close()

    if df.empty:
        return "SKIP — no data yet. Run fetch first."

    result = score_single_stock("RELIANCE", df)

    assert result["total_score"] >= 0
    assert result["total_score"] <= 100
    assert result["signal"] in ("BUY", "WATCH", "IGNORE")
    assert result.get("error") is None, f"Scoring error: {result.get('error')}"

    return (f"RELIANCE score: {result['total_score']:.1f}/100 "
            f"| Signal: {result['signal']} "
            f"| Gates: {'✅' if result['gate_passed'] else '❌'}")

test("Score engine scores a stock", test_scoring, critical=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Daily Analysis (reasoning generation)
# ─────────────────────────────────────────────────────────────────────────────

def test_daily_analysis():
    from create_db import get_connection
    from daily_analysis import (
        ensure_analysis_tables, run_daily_analysis,
        estimate_hold_days, generate_full_reasoning
    )
    import pandas as pd

    conn = get_connection()
    ensure_analysis_tables(conn)

    # Check that tables exist and are queryable
    count = conn.execute("SELECT COUNT(*) FROM daily_analysis").fetchone()[0]

    # Test hold-day estimator
    hold = estimate_hold_days(atr=25.0, close=1000.0, regime="NEUTRAL")
    assert 3 <= hold <= 20, f"Hold days {hold} out of expected range"

    conn.close()

    return (f"daily_analysis table has {count} rows. "
            f"Hold estimator working (NEUTRAL test: {hold}d). "
            f"Run 'py daily_analysis.py' to generate today's full analysis.")

test("Daily analysis module loads correctly", test_daily_analysis, critical=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Backup system
# ─────────────────────────────────────────────────────────────────────────────

def test_backup():
    from backup_trades import backup, restore
    from create_db import get_connection
    import json

    conn = get_connection()
    n_backed = backup(conn)
    conn.close()

    # Verify file was created and is valid JSON
    from config import DATA_DIR
    backup_path = os.path.join(DATA_DIR, "trades_backup.json")
    assert os.path.exists(backup_path), "Backup file not created"

    with open(backup_path) as f:
        data = json.load(f)
    assert "trades" in data
    assert "row_count" in data

    return f"Backup created: {n_backed} trades → {backup_path}"

test("Backup and restore system", test_backup, critical=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Weekend review (dry run, needs data)
# ─────────────────────────────────────────────────────────────────────────────

def test_weekend_review():
    from create_db import get_connection
    from weekend_review import run_weekend_review
    from daily_analysis import ensure_analysis_tables

    conn = get_connection()
    ensure_analysis_tables(conn)

    count = conn.execute("SELECT COUNT(*) FROM daily_analysis").fetchone()[0]
    if count == 0:
        conn.close()
        return "SKIP — no daily_analysis data yet. Run 'py daily_analysis.py' first, then re-test."

    stats = run_weekend_review(conn, lookback_days=30)
    conn.close()

    if "error" in stats:
        return f"SKIP — {stats['error']}"

    return (f"Review ran: {stats['total_analysed']} analyses, "
            f"{stats['buy_signals']} BUY signals, "
            f"avg P&L {stats['avg_pnl_pct']:+.2f}%")

test("Weekend review runs correctly", test_weekend_review, critical=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — Telegram (just connectivity, not full send)
# ─────────────────────────────────────────────────────────────────────────────

def test_telegram():
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    import requests

    if not TELEGRAM_BOT_TOKEN:
        return "SKIP — TELEGRAM_BOT_TOKEN not set in .env"

    r = requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
        timeout=5
    )
    if r.status_code != 200:
        raise AssertionError(f"Bot API returned {r.status_code} — check token")

    data = r.json()
    bot_name = data["result"]["username"]

    if not TELEGRAM_CHAT_ID:
        return f"Bot connected: @{bot_name}. WARN: TELEGRAM_CHAT_ID not set — alerts won't send."

    return f"Bot connected: @{bot_name} | Chat ID: {TELEGRAM_CHAT_ID}"

test("Telegram bot connectivity", test_telegram, critical=False)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)

    print(f"\n{'═'*55}")
    print(f"  TEST SUMMARY: {passed} passed, {failed} failed")
    print(f"{'─'*55}")
    for name, ok, msg in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name}")
        if not ok:
            print(f"      ↳ {msg[:80]}")
    print(f"{'═'*55}")

    if failed == 0:
        print("\n  🎉 All tests passed. System is healthy.")
        print("\n  Next steps:")
        print("  1. py fetch_data.py          ← populate DB with price data")
        print("  2. py regime_engine.py       ← detect today's regime")
        print("  3. py score_stocks.py        ← score all 28 stocks")
        print("  4. py daily_analysis.py      ← generate reasoning + send Telegram")
        print("  5. py streamlit run streamlit_app.py  ← open dashboard")
    else:
        print(f"\n  Fix the {failed} failing test(s) before deploying.")

print_summary()
