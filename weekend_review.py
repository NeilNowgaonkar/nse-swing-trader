# weekend_review.py
# Every Saturday: reviews last week's analyses against what actually happened.
# Asks: was the reasoning correct? Did price move as expected?
# Builds an accuracy log that improves your system over time.
#
# Run automatically via cron every Saturday at 10:00 AM IST.
# Also run manually anytime: py weekend_review.py
#
# Output:
#   - accuracy_log table updated in DB
#   - Telegram summary: hit rate, best calls, worst calls, factor accuracy
#   - Console report

import sqlite3
from typing import Optional
import os
import asyncio
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import pytz
from loguru import logger

from config import (
    ALL_STOCKS, LOG_DIR, TIMEZONE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
)
from create_db import get_connection
from daily_analysis import ensure_analysis_tables

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(
    os.path.join(LOG_DIR, "review_{time:YYYY-MM-DD}.log"),
    rotation="1 week", retention="12 weeks", level="INFO",
)

IST = pytz.timezone(TIMEZONE)


def today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def get_actual_price(conn: sqlite3.Connection, symbol: str,
                     target_date: str) -> Optional[float]:
    """
    Get the closing price on or just after a given date.
    Looks up to 5 days forward (skips holidays/weekends).
    """
    row = conn.execute("""
        SELECT close FROM daily_prices
        WHERE symbol = ? AND date >= ?
        ORDER BY date ASC LIMIT 1
    """, (symbol, target_date)).fetchone()
    return row[0] if row else None


def compute_accuracy_for_analysis(conn: sqlite3.Connection,
                                   analysis: dict,
                                   review_date: str) -> dict:
    """
    For a single analysis record, check what actually happened.
    Returns accuracy metrics.
    """
    symbol         = analysis["symbol"]
    entry_price    = analysis["close_price"]
    target         = analysis["target"]
    stop_loss      = analysis["stop_loss"]
    analysis_date  = analysis["date"]
    expected_exit  = analysis["expected_exit_date"]
    signal         = analysis["signal"]

    # Get price at expected exit date
    actual_at_exit = get_actual_price(conn, symbol, expected_exit)

    # Also check if price hit target or SL at any point between analysis and now
    prices_since = pd.read_sql_query("""
        SELECT date, high, low, close FROM daily_prices
        WHERE symbol = ? AND date > ? AND date <= ?
        ORDER BY date ASC
    """, conn, params=(symbol, analysis_date, review_date))

    hit_target   = False
    hit_stoploss = False

    if not prices_since.empty and entry_price > 0:
        for _, prow in prices_since.iterrows():
            if target > 0 and prow["high"] >= target:
                hit_target   = True
            if stop_loss > 0 and prow["low"] <= stop_loss:
                hit_stoploss = True

    # P&L at review date vs entry price
    latest_price = get_actual_price(conn, symbol, review_date)
    pnl_pct = None
    if latest_price and entry_price > 0:
        pnl_pct = round((latest_price - entry_price) / entry_price * 100, 2)

    # Reasoning score (auto-computed):
    # 5 = target hit without SL hit first
    # 4 = price moved in right direction > 2%
    # 3 = price flat ±2%
    # 2 = price moved wrong direction
    # 1 = SL hit
    if signal == "BUY" and entry_price > 0:
        if hit_stoploss and not hit_target:
            reasoning_score = 10
        elif hit_target and not hit_stoploss:
            reasoning_score = 100
        elif hit_target and hit_stoploss:
            reasoning_score = 50   # ambiguous
        elif pnl_pct is not None:
            # 0-100 score based on how far price moved toward target vs stop
            # 100 = at target, 50 = flat, 10 = at stop-loss, 0 = below stop
            if entry_price > 0 and target and stop_loss:
                range_up   = target    - entry_price
                range_down = entry_price - stop_loss
                if pnl_pct > 0 and range_up > 0:
                    # Positive: scale 50-99 based on % of the way to target
                    pct_to_target  = min((latest_price - entry_price) / range_up, 1.0)
                    reasoning_score = int(50 + pct_to_target * 49)
                elif pnl_pct < 0 and range_down > 0:
                    # Negative: scale 10-49 based on how close to stop
                    pct_to_stop    = min((entry_price - latest_price) / range_down, 1.0)
                    reasoning_score = int(50 - pct_to_stop * 40)
                else:
                    reasoning_score = 50
            else:
                reasoning_score = 50 if pnl_pct >= 0 else 30
        else:
            reasoning_score = None
    else:
        reasoning_score = None

    return {
        "symbol":                symbol,
        "analysis_date":         analysis_date,
        "review_date":           review_date,
        "signal":                signal,
        "entry_price":           entry_price,
        "expected_exit_date":    expected_exit,
        "actual_price_at_review": latest_price,
        "target":                target,
        "stop_loss":             stop_loss,
        "hit_target":            1 if hit_target   else 0,
        "hit_stoploss":          1 if hit_stoploss else 0,
        "pnl_pct_so_far":        pnl_pct,
        "reasoning_score":       reasoning_score,
    }


def save_accuracy(conn: sqlite3.Connection, record: dict) -> None:
    try:
        conn.execute("""
            INSERT OR REPLACE INTO accuracy_log (
                symbol, analysis_date, review_date, signal,
                entry_price, expected_exit_date, actual_price_at_review,
                target, stop_loss, hit_target, hit_stoploss,
                pnl_pct_so_far, reasoning_score
            ) VALUES (
                :symbol, :analysis_date, :review_date, :signal,
                :entry_price, :expected_exit_date, :actual_price_at_review,
                :target, :stop_loss, :hit_target, :hit_stoploss,
                :pnl_pct_so_far, :reasoning_score
            )
        """, record)
        conn.commit()
    except Exception as e:
        logger.error(f"accuracy_log write error {record['symbol']}: {e}")


def run_weekend_review(conn: sqlite3.Connection,
                       lookback_days: int = 7) -> dict:
    """
    Review all analyses from the past `lookback_days` days.
    Returns a summary dict with accuracy stats.
    """
    ensure_analysis_tables(conn)
    today      = today_ist()
    start_date = (datetime.now(IST) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    logger.info(f"=== Weekend Review | {today} | Reviewing {start_date} → {today} ===")

    analyses = pd.read_sql_query("""
        SELECT symbol, date, total_score, signal, regime, gate_passed,
               close_price, stop_loss, target, expected_hold_days,
               expected_exit_date, reasoning
        FROM daily_analysis
        WHERE date >= ? AND date < ?
        ORDER BY date ASC, total_score DESC
    """, conn, params=(start_date, today))

    if analyses.empty:
        logger.warning("No analyses found for review period.")
        return {"error": "No analyses found"}

    logger.info(f"Reviewing {len(analyses)} analysis records …")

    accuracy_records = []
    for _, row in analyses.iterrows():
        rec = compute_accuracy_for_analysis(conn, row.to_dict(), today)
        save_accuracy(conn, rec)
        accuracy_records.append(rec)
        logger.info(
            f"  {row['symbol']:>14}  signal={row['signal']:>6}  "
            f"pnl={rec['pnl_pct_so_far'] or 0:+.1f}%  "
            f"score={rec['reasoning_score'] or '?'}/100  "
            f"tgt={'✅' if rec['hit_target'] else '—'}  "
            f"sl={'❌' if rec['hit_stoploss'] else '—'}"
        )

    # ── Compute summary stats ─────────────────────────────────────────────────
    df = pd.DataFrame(accuracy_records)
    buy_signals = df[df["signal"] == "BUY"]

    stats = {
        "review_date":         today,
        "period":              f"{start_date} → {today}",
        "total_analysed":      len(df),
        "buy_signals":         len(buy_signals),
        "targets_hit":         int(buy_signals["hit_target"].sum())   if not buy_signals.empty else 0,
        "stops_hit":           int(buy_signals["hit_stoploss"].sum()) if not buy_signals.empty else 0,
        "avg_pnl_pct":         round(buy_signals["pnl_pct_so_far"].dropna().mean(), 2) if not buy_signals.empty else 0,
        "avg_reasoning_score": round(df["reasoning_score"].dropna().mean(), 2),
        "best_call":           None,
        "worst_call":          None,
        "factor_insights":     [],
    }

    if not buy_signals.empty:
        best_idx  = buy_signals["pnl_pct_so_far"].idxmax()
        worst_idx = buy_signals["pnl_pct_so_far"].idxmin()
        if pd.notna(best_idx):
            stats["best_call"]  = buy_signals.loc[best_idx, "symbol"]
        if pd.notna(worst_idx):
            stats["worst_call"] = buy_signals.loc[worst_idx, "symbol"]

    # ── Factor insight: which regime signals proved most accurate ─────────────
    regime_perf = df[df["signal"]=="BUY"].groupby(
        df[df["signal"]=="BUY"]["analysis_date"].apply(
            lambda d: conn.execute(
                "SELECT regime FROM daily_analysis WHERE date=? LIMIT 1", (d,)
            ).fetchone() or ("UNKNOWN",)
        ).apply(lambda x: x[0])
    )["pnl_pct_so_far"].mean().to_dict()

    if regime_perf:
        best_regime = max(regime_perf, key=regime_perf.get)
        stats["factor_insights"].append(
            f"Best performing regime this week: {best_regime} "
            f"(avg {regime_perf[best_regime]:.1f}%)"
        )

    # Score distribution insight
    score_corr = df[df["pnl_pct_so_far"].notna()]["reasoning_score"].corr(
        df[df["pnl_pct_so_far"].notna()]["pnl_pct_so_far"]
    )
    if not pd.isna(score_corr):
        if score_corr > 0.5:
            stats["factor_insights"].append(
                f"Strong correlation ({score_corr:.2f}) between reasoning score and actual P&L — model is working."
            )
        elif score_corr > 0.2:
            stats["factor_insights"].append(
                f"Moderate correlation ({score_corr:.2f}) — model has predictive value but needs refinement."
            )
        else:
            stats["factor_insights"].append(
                f"Weak correlation ({score_corr:.2f}) — review factor weights, current market may not suit this model."
            )

    return stats


def print_review_report(stats: dict) -> None:
    if "error" in stats:
        print(f"\n⚠️  {stats['error']}")
        return

    print(f"\n{'═'*60}")
    print(f"  📋 WEEKEND REVIEW — {stats['review_date']}")
    print(f"  Period: {stats['period']}")
    print(f"{'─'*60}")
    print(f"  Total analyses reviewed : {stats['total_analysed']}")
    print(f"  BUY signals reviewed    : {stats['buy_signals']}")
    print(f"  Targets hit             : {stats['targets_hit']}")
    print(f"  Stop-losses hit         : {stats['stops_hit']}")
    print(f"  Avg P&L (BUY signals)   : {stats['avg_pnl_pct']:+.2f}%")
    print(f"  Avg reasoning score     : {stats['avg_reasoning_score']:.1f}/5")
    if stats["best_call"]:
        print(f"  Best call this week     : {stats['best_call']}")
    if stats["worst_call"]:
        print(f"  Worst call this week    : {stats['worst_call']}")
    print(f"{'─'*60}")
    for insight in stats["factor_insights"]:
        print(f"  💡 {insight}")
    print(f"{'═'*60}\n")


async def send_review_telegram(stats: dict) -> None:
    import httpx
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    if "error" in stats:
        return

    tgt_rate = (stats["targets_hit"] / stats["buy_signals"] * 100) if stats["buy_signals"] > 0 else 0
    sl_rate  = (stats["stops_hit"]   / stats["buy_signals"] * 100) if stats["buy_signals"] > 0 else 0

    msg = (
        f"📋 *Weekend Review — {stats['review_date']}*\n"
        f"Period: _{stats['period']}_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Analyses reviewed : `{stats['total_analysed']}`\n"
        f"🔔 BUY signals       : `{stats['buy_signals']}`\n"
        f"🎯 Targets hit       : `{stats['targets_hit']}` ({tgt_rate:.0f}%)\n"
        f"🛑 Stops hit         : `{stats['stops_hit']}` ({sl_rate:.0f}%)\n"
        f"💰 Avg P&L           : `{stats['avg_pnl_pct']:+.2f}%`\n"
        f"🧠 Reasoning score   : `{stats['avg_reasoning_score']:.0f}/100`\n"
    )
    if stats["best_call"]:
        msg += f"⭐ Best call : `{stats['best_call']}`\n"
    if stats["worst_call"]:
        msg += f"💀 Worst call: `{stats['worst_call']}`\n"

    if stats["factor_insights"]:
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        for insight in stats["factor_insights"]:
            msg += f"💡 _{insight}_\n"

    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
        )


if __name__ == "__main__":
    conn  = get_connection()
    ensure_analysis_tables(conn)
    stats = run_weekend_review(conn, lookback_days=7)
    print_review_report(stats)
    asyncio.run(send_review_telegram(stats))
    conn.close()
    logger.info("weekend_review.py complete.")
