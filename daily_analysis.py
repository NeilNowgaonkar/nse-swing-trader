# daily_analysis.py
# Records daily analysis with human-readable reasoning for every stock.
# This is the "memory" of the system — every decision logged with WHY.
#
# What it does:
#   1. Reads today's scores from daily_scores table
#   2. Reads latest price + indicators for each stock
#   3. Generates plain-English reasoning for each factor score
#   4. Estimates expected hold period and target exit date
#   5. Records everything in daily_analysis table
#   6. Sends a full scorecard to Telegram (even for IGNORE stocks)
#
# Why this matters:
#   Every weekend, weekend_review.py reads these analyses and checks
#   whether the reasoning proved correct. Over time you'll see which
#   factors are actually predictive for your specific universe.

import sqlite3
import os
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import pytz
from loguru import logger

from config import (
    ALL_STOCKS, BUCKET_LABELS, DB_PATH, LOG_DIR, TIMEZONE,
    STARTING_CAPITAL, RISK_PER_TRADE_PCT, TRADE_SIGNAL_THRESHOLD,
)
from create_db import get_connection
from indicators import add_all_indicators, calculate_stop_loss, calculate_target, calculate_position_size
from regime_engine import get_latest_regime, get_regime_rules

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(
    os.path.join(LOG_DIR, "analysis_{time:YYYY-MM-DD}.log"),
    rotation="1 day", retention="30 days", level="INFO",
)

IST = pytz.timezone(TIMEZONE)


# ─────────────────────────────────────────────────────────────────────────────
# DB SETUP — new table for daily_analysis
# ─────────────────────────────────────────────────────────────────────────────

def ensure_analysis_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily_analysis (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol            TEXT NOT NULL,
            date              TEXT NOT NULL,
            total_score       REAL,
            signal            TEXT,
            regime            TEXT,
            gate_passed       INTEGER DEFAULT 0,
            close_price       REAL,
            stop_loss         REAL,
            target            REAL,
            qty               INTEGER,
            expected_hold_days INTEGER,
            expected_exit_date TEXT,
            reasoning         TEXT,   -- full plain-English explanation
            momentum_reason   TEXT,
            trend_reason      TEXT,
            rsi_reason        TEXT,
            macd_reason       TEXT,
            volume_reason     TEXT,
            bb_reason         TEXT,
            gate_reason       TEXT,
            atr               REAL,
            rsi_value         REAL,
            macd_hist         REAL,
            vol_ratio         REAL,
            bb_pct_b          REAL,
            price_vs_sma200   REAL,   -- % above/below SMA-200
            created_at        TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(symbol, date)
        );

        CREATE TABLE IF NOT EXISTS accuracy_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol            TEXT NOT NULL,
            analysis_date     TEXT NOT NULL,
            review_date       TEXT NOT NULL,
            signal            TEXT,
            entry_price       REAL,
            expected_exit_date TEXT,
            actual_price_at_review REAL,
            target            REAL,
            stop_loss         REAL,
            hit_target        INTEGER DEFAULT 0,
            hit_stoploss      INTEGER DEFAULT 0,
            pnl_pct_so_far    REAL,
            reasoning_score   INTEGER,  -- 1–5: was reasoning actually correct?
            notes             TEXT,
            created_at        TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(symbol, analysis_date, review_date)
        );
    """)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# REASONING GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def reason_momentum(row: pd.Series, score: float) -> str:
    close  = row.get("close", 0)
    ema20  = row.get("ema_20", np.nan)
    ema50  = row.get("ema_50", np.nan)

    parts = []
    if not pd.isna(ema20) and close > 0:
        gap = (close - ema20) / ema20 * 100
        if close > ema20:
            parts.append(f"price is {gap:.1f}% above EMA-20 (bullish)")
        else:
            parts.append(f"price is {abs(gap):.1f}% below EMA-20 (bearish)")

    if not pd.isna(ema50):
        if close > ema50:
            parts.append("price above EMA-50 (medium-term strength)")
        else:
            parts.append("price below EMA-50 (medium-term weakness)")

    if not parts:
        return "Insufficient data for momentum assessment."

    verdict = "Strong" if score >= 16 else "Moderate" if score >= 10 else "Weak"
    return f"{verdict} momentum: {'; '.join(parts)}."


def reason_trend(row: pd.Series, score: float) -> str:
    sma20  = row.get("sma_20",  np.nan)
    sma50  = row.get("sma_50",  np.nan)
    sma200 = row.get("sma_200", np.nan)
    ema9   = row.get("ema_9",   np.nan)
    ema20  = row.get("ema_20",  np.nan)

    parts = []
    if not (pd.isna(sma20) or pd.isna(sma50)):
        if sma20 > sma50:
            parts.append("SMA-20 > SMA-50 (short-term trend up)")
        else:
            parts.append("SMA-20 < SMA-50 (short-term trend down)")

    if not (pd.isna(sma50) or pd.isna(sma200)):
        if sma50 > sma200:
            parts.append("SMA-50 > SMA-200 (long-term bull structure)")
        else:
            parts.append("SMA-50 < SMA-200 (long-term bear structure)")

    if not (pd.isna(ema9) or pd.isna(ema20)):
        if ema9 > ema20:
            parts.append("EMA-9 > EMA-20 (immediate momentum positive)")
        else:
            parts.append("EMA-9 < EMA-20 (immediate momentum negative)")

    if not parts:
        return "Insufficient data for trend assessment."

    verdict = "Aligned bull" if score >= 16 else "Mixed" if score >= 8 else "Bearish"
    return f"{verdict} trend structure: {'; '.join(parts)}."


def reason_rsi(row: pd.Series, score: float) -> str:
    rsi = row.get("rsi_14", np.nan)
    if pd.isna(rsi):
        return "RSI-14 not available."

    if rsi > 70:
        zone = "overbought — risky entry, momentum may reverse"
    elif rsi > 65:
        zone = "extended but not overbought — watch for pullback"
    elif rsi > 55:
        zone = "sweet spot — trending up with room to run"
    elif rsi > 50:
        zone = "above midline — mild bullish bias"
    elif rsi > 40:
        zone = "below midline — weak, momentum declining"
    elif rsi > 30:
        zone = "approaching oversold — not a buy yet"
    else:
        zone = "oversold — potential bounce but downtrend intact"

    return f"RSI-14 at {rsi:.1f}: {zone}."


def reason_macd(row: pd.Series, score: float) -> str:
    line   = row.get("macd_line",   np.nan)
    signal = row.get("macd_signal", np.nan)
    hist   = row.get("macd_hist",   np.nan)

    if pd.isna(line) or pd.isna(signal):
        return "MACD not available."

    parts = []
    if line > signal:
        parts.append("MACD line above signal (bullish crossover active)")
    else:
        parts.append("MACD line below signal (bearish)")

    if not pd.isna(hist):
        if hist > 0:
            parts.append(f"histogram positive ({hist:.3f}) — momentum accelerating")
        else:
            parts.append(f"histogram negative ({hist:.3f}) — momentum fading")

    if line > 0:
        parts.append("MACD above zero line (broader bull trend)")
    else:
        parts.append("MACD below zero line (broader bear trend)")

    return "; ".join(parts) + "."


def reason_volume(row: pd.Series, score: float) -> str:
    vol_ratio = row.get("vol_ratio", np.nan)
    close     = row.get("close", np.nan)
    open_     = row.get("open",  np.nan)

    if pd.isna(vol_ratio):
        return "Volume data unavailable."

    is_green = (not pd.isna(close) and not pd.isna(open_) and close > open_)
    candle   = "green (bullish)" if is_green else "red (bearish)"

    if vol_ratio >= 2.0:
        strength = "very high volume (2x+ average)"
    elif vol_ratio >= 1.5:
        strength = "above-average volume (1.5x+)"
    elif vol_ratio >= 1.0:
        strength = "average volume"
    elif vol_ratio >= 0.7:
        strength = "below-average volume"
    else:
        strength = "very thin volume (possible trap)"

    return f"Volume {vol_ratio:.2f}x 20-day average on a {candle} candle — {strength}."


def reason_bollinger(row: pd.Series, score: float) -> str:
    pct_b    = row.get("bb_pct_b", np.nan)
    bb_width = row.get("bb_width", np.nan)

    if pd.isna(pct_b):
        return "Bollinger Band data unavailable."

    if pct_b > 0.85:
        position = "at upper band — extended, do not chase"
    elif pct_b > 0.70:
        position = "in upper zone — slightly extended"
    elif pct_b > 0.55:
        position = "above mid-band — healthy uptrend position"
    elif pct_b > 0.35:
        position = "near mid-band — good entry zone with room to run"
    elif pct_b > 0.20:
        position = "below mid-band — slight weakness"
    elif pct_b > 0.10:
        position = "near lower band — potential support, watch closely"
    else:
        position = "at lower band — could be breakdown or oversold bounce"

    squeeze = ""
    if not pd.isna(bb_width) and bb_width < 0.02:
        squeeze = " Bands are very tight (volatility squeeze) — breakout imminent but direction unclear."

    return f"Price at {pct_b:.1%} of Bollinger Band ({position}).{squeeze}"


def reason_gates(row: pd.Series, gate_passed: bool, gate_failures: list) -> str:
    if gate_passed:
        return "All 3 hard gates passed: price above SMA-200, volume adequate, ATR within limits."
    else:
        return f"Gate failures: {'; '.join(gate_failures)}"


def estimate_hold_days(atr: float, close: float, regime: str) -> int:
    """
    Estimate expected hold period based on ATR volatility and regime.
    Higher volatility = faster moves = shorter hold.
    Conservative regime = longer hold required.
    """
    if pd.isna(atr) or atr <= 0 or pd.isna(close) or close <= 0:
        return 7

    atr_pct = (atr / close) * 100

    if atr_pct > 3:
        base_days = 4   # very volatile — target/SL hit quickly
    elif atr_pct > 2:
        base_days = 6
    elif atr_pct > 1:
        base_days = 9
    else:
        base_days = 12  # low volatility — needs more time

    regime_mult = {
        "BULL":     0.8,   # moves faster in bull
        "NEUTRAL":  1.0,
        "SIDEWAYS": 1.4,   # choppy — takes longer
        "HIGH_VIX": 0.7,   # fast but dangerous
        "BEAR":     1.5,   # if you somehow enter in bear
    }.get(regime, 1.0)

    return max(3, int(base_days * regime_mult))


def generate_full_reasoning(
    symbol: str, row: pd.Series,
    scores: dict, gate_passed: bool,
    gate_failures: list, regime: str, signal: str
) -> str:
    """
    Full plain-English reasoning paragraph for a trade decision.
    Written as if explaining to yourself why you should or shouldn't take this trade.
    """
    close  = row.get("close", 0)
    sma200 = row.get("sma_200", np.nan)
    total  = scores["total"]

    # Opening assessment
    if signal == "BUY":
        opening = f"{symbol} is generating a BUY signal with a score of {total:.0f}/100."
    elif signal == "WATCH":
        opening = f"{symbol} is on WATCH with a score of {total:.0f}/100 — close to threshold but not quite there."
    else:
        opening = f"{symbol} scores {total:.0f}/100 — not actionable today."

    # Strongest and weakest factors
    factor_scores = {
        "Momentum": scores["momentum"],
        "Trend":    scores["trend"],
        "RSI":      scores["rsi"],
        "MACD":     scores["macd"],
        "Volume":   scores["volume"],
        "Bollinger":scores["bb"],
    }
    max_weights = {"Momentum":20,"Trend":20,"RSI":15,"MACD":15,"Volume":15,"Bollinger":15}
    pct_scores  = {k: v/max_weights[k]*100 for k, v in factor_scores.items()}
    strongest   = max(pct_scores, key=pct_scores.get)
    weakest     = min(pct_scores, key=pct_scores.get)

    factor_line = (
        f"Strongest factor: {strongest} ({pct_scores[strongest]:.0f}% of max). "
        f"Weakest factor: {weakest} ({pct_scores[weakest]:.0f}% of max)."
    )

    # Gate and regime context
    if not gate_passed:
        gate_line = f"GATES FAILED — trade blocked: {'; '.join(gate_failures)}."
    else:
        gate_line = "All hard gates passed."

    regime_rules = get_regime_rules(regime)
    if regime == "BEAR":
        regime_line = "BEAR regime — no new entries regardless of score."
    elif regime == "HIGH_VIX":
        regime_line = f"HIGH_VIX regime — only defensive stocks allowed, min score {regime_rules['min_score']}."
    elif regime == "SIDEWAYS":
        regime_line = f"SIDEWAYS regime — only large-cap/defensive, min score {regime_rules['min_score']}."
    else:
        regime_line = f"{regime} regime — min score {regime_rules['min_score']}, all buckets eligible."

    # Price structure context
    if not pd.isna(sma200) and sma200 > 0 and close > 0:
        vs_200 = (close - sma200) / sma200 * 100
        if vs_200 > 10:
            price_line = f"Price is {vs_200:.1f}% above SMA-200 — strong long-term position."
        elif vs_200 > 0:
            price_line = f"Price is {vs_200:.1f}% above SMA-200 — marginally above long-term trend."
        else:
            price_line = f"Price is {abs(vs_200):.1f}% below SMA-200 — below long-term trend (gate should catch this)."
    else:
        price_line = "SMA-200 not yet calculated (insufficient history)."

    return " ".join([opening, factor_line, gate_line, regime_line, price_line])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_analysis(conn: sqlite3.Connection) -> list[dict]:
    ensure_analysis_tables(conn)

    today  = datetime.now(IST).strftime("%Y-%m-%d")
    regime = get_latest_regime(conn)
    results = []

    logger.info(f"=== Daily Analysis | {today} | Regime: {regime} ===")

    for symbol in ALL_STOCKS:
        df_raw = pd.read_sql_query("""
            SELECT date, open, high, low, close, volume
            FROM daily_prices WHERE symbol = ?
            ORDER BY date DESC LIMIT 250
        """, conn, params=(symbol,))

        if df_raw.empty or len(df_raw) < 50:
            logger.warning(f"{symbol}: insufficient data ({len(df_raw)} rows)")
            continue

        df = add_all_indicators(df_raw.sort_values("date").reset_index(drop=True))
        row = df.iloc[-1]
        close = row.get("close", 0)
        atr   = row.get("atr_14", close * 0.02)

        # Get today's score from DB (scored by score_stocks.py earlier)
        score_row = conn.execute("""
            SELECT total_score, momentum_score, trend_score, rsi_score,
                   macd_score, volume_score, bb_score, signal, gate_passed
            FROM daily_scores WHERE symbol = ? ORDER BY date DESC LIMIT 1
        """, (symbol,)).fetchone()

        if score_row:
            total, mom, trend, rsi, macd, vol, bb, signal, gate_int = score_row
        else:
            # Score not available yet — use defaults
            total, mom, trend, rsi, macd, vol, bb, signal, gate_int = 0,0,0,0,0,0,0,"IGNORE",0

        gate_passed = bool(gate_int)
        scores = {
            "total": total or 0, "momentum": mom or 0, "trend": trend or 0,
            "rsi": rsi or 0, "macd": macd or 0, "volume": vol or 0, "bb": bb or 0,
        }

        # Gate failure reasons
        gate_failures = []
        sma200 = row.get("sma_200", np.nan)
        vol_ratio = row.get("vol_ratio", np.nan)
        atr_pct   = row.get("atr_pct",   np.nan)

        if pd.isna(sma200) or close <= sma200:
            gate_failures.append(f"price below SMA-200 (SMA200={sma200:.0f} close={close:.0f})" if not pd.isna(sma200) else "SMA-200 not yet calculated")
        if pd.isna(vol_ratio) or vol_ratio < 0.7:
            gate_failures.append(f"thin volume (ratio={vol_ratio:.2f})" if not pd.isna(vol_ratio) else "no volume data")
        if not pd.isna(atr_pct) and atr_pct > 5.0:
            gate_failures.append(f"ATR too high ({atr_pct:.1f}%)")

        # Individual factor reasons
        m_reason  = reason_momentum(row, scores["momentum"])
        t_reason  = reason_trend(row, scores["trend"])
        r_reason  = reason_rsi(row, scores["rsi"])
        ma_reason = reason_macd(row, scores["macd"])
        v_reason  = reason_volume(row, scores["volume"])
        b_reason  = reason_bollinger(row, scores["bb"])
        g_reason  = reason_gates(row, gate_passed, gate_failures)

        # Trade levels
        sl  = calculate_stop_loss(close, atr) if close > 0 else 0
        tgt = calculate_target(close, atr)    if close > 0 else 0
        qty = calculate_position_size(STARTING_CAPITAL, RISK_PER_TRADE_PCT, close, sl)

        # Expected hold and exit date
        hold_days    = estimate_hold_days(atr, close, regime)
        today_date   = datetime.now(IST).date()
        # Skip weekends for exit date
        exit_date    = today_date
        bdays_added  = 0
        while bdays_added < hold_days:
            exit_date += timedelta(days=1)
            if exit_date.weekday() < 5:
                bdays_added += 1

        # Full reasoning
        full_reasoning = generate_full_reasoning(
            symbol, row, scores, gate_passed, gate_failures, regime, signal or "IGNORE"
        )

        vs_200 = ((close - sma200) / sma200 * 100) if not pd.isna(sma200) and sma200 > 0 else None

        record = {
            "symbol":             symbol,
            "date":               today,
            "total_score":        round(scores["total"], 2),
            "signal":             signal or "IGNORE",
            "regime":             regime,
            "gate_passed":        1 if gate_passed else 0,
            "close_price":        round(close, 2),
            "stop_loss":          round(sl, 2),
            "target":             round(tgt, 2),
            "qty":                qty,
            "expected_hold_days": hold_days,
            "expected_exit_date": exit_date.strftime("%Y-%m-%d"),
            "reasoning":          full_reasoning,
            "momentum_reason":    m_reason,
            "trend_reason":       t_reason,
            "rsi_reason":         r_reason,
            "macd_reason":        ma_reason,
            "volume_reason":      v_reason,
            "bb_reason":          b_reason,
            "gate_reason":        g_reason,
            "atr":                round(atr, 2) if not pd.isna(atr) else None,
            "rsi_value":          round(row.get("rsi_14", np.nan), 2) if not pd.isna(row.get("rsi_14", np.nan)) else None,
            "macd_hist":          round(row.get("macd_hist", np.nan), 4) if not pd.isna(row.get("macd_hist", np.nan)) else None,
            "vol_ratio":          round(vol_ratio, 2) if not pd.isna(vol_ratio) else None,
            "bb_pct_b":           round(row.get("bb_pct_b", np.nan), 4) if not pd.isna(row.get("bb_pct_b", np.nan)) else None,
            "price_vs_sma200":    round(vs_200, 2) if vs_200 is not None else None,
        }

        # Upsert into DB
        try:
            conn.execute("""
                INSERT OR REPLACE INTO daily_analysis (
                    symbol, date, total_score, signal, regime, gate_passed,
                    close_price, stop_loss, target, qty,
                    expected_hold_days, expected_exit_date,
                    reasoning, momentum_reason, trend_reason, rsi_reason,
                    macd_reason, volume_reason, bb_reason, gate_reason,
                    atr, rsi_value, macd_hist, vol_ratio, bb_pct_b, price_vs_sma200
                ) VALUES (
                    :symbol, :date, :total_score, :signal, :regime, :gate_passed,
                    :close_price, :stop_loss, :target, :qty,
                    :expected_hold_days, :expected_exit_date,
                    :reasoning, :momentum_reason, :trend_reason, :rsi_reason,
                    :macd_reason, :volume_reason, :bb_reason, :gate_reason,
                    :atr, :rsi_value, :macd_hist, :vol_ratio, :bb_pct_b, :price_vs_sma200
                )
            """, record)
        except Exception as e:
            logger.error(f"{symbol} DB write error: {e}")

        results.append(record)
        logger.info(
            f"  {symbol:>14}  score={scores['total']:>5.1f}  "
            f"signal={signal:>6}  hold={hold_days}d  exit={exit_date}"
        )

    conn.commit()
    logger.info(f"Analysis complete. {len(results)} stocks recorded.")
    return sorted(results, key=lambda x: x["total_score"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM SCORECARD SENDER
# ─────────────────────────────────────────────────────────────────────────────

async def send_daily_scorecard(results: list[dict]) -> None:
    """
    Send full scorecard to Telegram:
    - One summary message with all 28 stocks ranked by score
    - Separate detailed messages for BUY signals only
    """
    import httpx
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — skipping scorecard send")
        return

    async def push(text: str) -> None:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
            })
            if r.status_code != 200:
                logger.error(f"Telegram error: {r.text}")

    today   = results[0]["date"] if results else datetime.now(IST).strftime("%Y-%m-%d")
    regime  = results[0]["regime"] if results else "UNKNOWN"
    buys    = [r for r in results if r["signal"] == "BUY"   and r["gate_passed"]]
    watches = [r for r in results if r["signal"] == "WATCH"]

    regime_icon = {"BULL":"🟢","NEUTRAL":"🔵","SIDEWAYS":"🟡","HIGH_VIX":"🟠","BEAR":"🔴"}.get(regime,"⚪")

    # ── Message 1: Full scorecard table ──────────────────────────────────────
    scorecard = f"📊 *Daily Scorecard — {today}*\n{regime_icon} Regime: `{regime}`\n"
    scorecard += f"🟢 BUY: {len(buys)}  👀 WATCH: {len(watches)}\n"
    scorecard += "```\n"
    scorecard += f"{'SYMBOL':>12}  {'SCR':>4}  {'SIG':>5}  {'GATE':>4}\n"
    scorecard += "─" * 32 + "\n"
    for r in results:
        gate  = "✅" if r["gate_passed"] else "❌"
        scorecard += f"{r['symbol']:>12}  {r['total_score']:>4.0f}  {r['signal']:>5}  {gate:>4}\n"
    scorecard += "```"
    await push(scorecard)

    # ── Message 2+: Detailed card per BUY signal ─────────────────────────────
    for r in buys:
        sl_pct  = ((r["stop_loss"]  - r["close_price"]) / r["close_price"] * 100) if r["close_price"] else 0
        tgt_pct = ((r["target"]     - r["close_price"]) / r["close_price"] * 100) if r["close_price"] else 0

        detail = (
            f"🔔 *BUY — {r['symbol']}* | Score `{r['total_score']:.0f}/100`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Entry     : `₹{r['close_price']:.2f}`\n"
            f"🛑 Stop-Loss : `₹{r['stop_loss']:.2f}` ({sl_pct:.1f}%)\n"
            f"🎯 Target    : `₹{r['target']:.2f}` ({tgt_pct:+.1f}%)\n"
            f"📦 Qty       : `{r['qty']} shares`\n"
            f"📅 Hold ~{r['expected_hold_days']}d, exit by `{r['expected_exit_date']}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📝 _{r['reasoning'][:280]}_\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Momentum: `{r['momentum_reason'][:100]}`\n"
            f"Trend:    `{r['trend_reason'][:100]}`\n"
            f"RSI:      `{r['rsi_reason'][:100]}`\n"
            f"Gates:    `{r['gate_reason'][:120]}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ _Execute manually in Groww. Stop-loss is mandatory._"
        )
        await push(detail)

    if not buys:
        await push(
            f"📭 *No actionable BUY signals today.*\n"
            f"Regime: `{regime}` | "
            f"Highest score: `{results[0]['total_score']:.0f}` ({results[0]['symbol']})"
            if results else "No analysis data available."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    conn    = get_connection()
    results = run_daily_analysis(conn)

    # Print scorecard to console
    print(f"\n{'─'*70}")
    print(f"  {'SYMBOL':>14}  {'SCORE':>6}  {'SIGNAL':>6}  {'GATES':>5}  "
          f"{'CLOSE':>8}  {'HOLD':>5}  {'EXIT DATE':>10}")
    print(f"{'─'*70}")
    for r in results:
        gate = "✅" if r["gate_passed"] else "❌"
        print(f"  {r['symbol']:>14}  {r['total_score']:>6.1f}  "
              f"{r['signal']:>6}  {gate:>5}  "
              f"₹{r['close_price']:>7.1f}  "
              f"{r['expected_hold_days']:>4}d  "
              f"{r['expected_exit_date']:>10}")

    buy_count = sum(1 for r in results if r["signal"]=="BUY" and r["gate_passed"])
    print(f"{'─'*70}")
    print(f"  BUY signals: {buy_count}  |  Total analysed: {len(results)}")
    print(f"{'─'*70}\n")

    # Send to Telegram
    asyncio.run(send_daily_scorecard(results))
    conn.close()
    logger.info("daily_analysis.py complete.")
