# telegram_bot.py
# NSE Swing Trader — Telegram Bot
# Phase 3: Alerts + Portfolio Memory + Risk Guards
#
# TWO modes:
#   1. --send-alerts  → run from GitHub Actions after scoring. Sends BUY signals. No polling.
#   2. (no args)      → starts interactive bot. Listens for /commands from your phone.
#
# Commands:
#   /start    — welcome + help
#   /signals  — today's BUY signals with entry, SL, target, qty
#   /status   — open trades, unrealised P&L, free slots
#   /bought SYMBOL PRICE QTY [NOTES]  — log a new manual trade
#   /sold SYMBOL PRICE [REASON]       — close a trade, calculates P&L
#   /risk     — daily loss check, drawdown status, pause state
#   /regime   — current market regime + trading rules
#   /history  — last 10 closed trades
#
# DOWNSIDE FIRST:
#   This bot has no execution power. It cannot place orders.
#   All trades are manual in Groww. The bot is memory + alerts only.
#   Every alert contains the stop-loss prominently — never trade without it.

import argparse
import sqlite3
import sys
import os
from datetime import datetime, timedelta, date
from loguru import logger
import pytz

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    STARTING_CAPITAL,
    RISK_PER_TRADE_PCT,
    MAX_OPEN_TRADES,
    DAILY_LOSS_LIMIT_PCT,
    DRAWDOWN_PAUSE_PCT,
    DRAWDOWN_PAUSE_DAYS,
    TRADE_SIGNAL_THRESHOLD,
    BUCKET_LABELS,
    LOG_DIR,
    TIMEZONE,
)
from create_db import get_connection
from regime_engine import get_latest_regime, get_regime_rules

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(
    os.path.join(LOG_DIR, "bot_{time:YYYY-MM-DD}.log"),
    rotation="1 day", retention="14 days", level="INFO"
)

IST = pytz.timezone(TIMEZONE)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M")


def get_current_capital(conn: sqlite3.Connection) -> float:
    """
    Estimate current capital:
    Starting capital + sum of all closed trade P&L
    """
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status = 'CLOSED'"
    ).fetchone()
    realised_pnl = row[0] if row else 0.0
    return STARTING_CAPITAL + realised_pnl


def get_open_trades(conn: sqlite3.Connection) -> list:
    rows = conn.execute("""
        SELECT id, symbol, entry_price, entry_date, qty, stop_loss, target,
               score_at_entry, regime_at_entry, bucket, notes
        FROM trades
        WHERE status = 'OPEN'
        ORDER BY entry_date DESC
    """).fetchall()
    return rows


def get_today_pnl(conn: sqlite3.Connection) -> float:
    """Sum of P&L from trades closed today."""
    row = conn.execute("""
        SELECT COALESCE(SUM(pnl), 0)
        FROM trades
        WHERE status = 'CLOSED' AND exit_date = ?
    """, (today_ist(),)).fetchone()
    return row[0] if row else 0.0


def get_drawdown(conn: sqlite3.Connection) -> tuple[float, float]:
    """
    Returns (drawdown_pct, peak_capital).
    Drawdown = how far we are from the highest capital we've ever had.
    """
    current = get_current_capital(conn)

    # Build equity curve from closed trades ordered by date
    rows = conn.execute("""
        SELECT pnl FROM trades
        WHERE status = 'CLOSED'
        ORDER BY exit_date ASC
    """).fetchall()

    equity = STARTING_CAPITAL
    peak   = STARTING_CAPITAL
    for (pnl,) in rows:
        equity += (pnl or 0)
        if equity > peak:
            peak = equity

    dd_pct = ((peak - current) / peak * 100) if peak > 0 else 0.0
    return round(dd_pct, 2), round(peak, 2)


def is_in_pause(conn: sqlite3.Connection) -> tuple[bool, str]:
    """
    Returns (paused: bool, reason: str).
    Check if we're in a 10-day drawdown pause.
    """
    # Find the most recent CLOSED trade that triggered drawdown
    # We store this in a simple approach: check current drawdown
    dd_pct, _ = get_drawdown(conn)
    if dd_pct >= DRAWDOWN_PAUSE_PCT:
        # Find when drawdown was first breached (approximate: last close date)
        row = conn.execute("""
            SELECT exit_date FROM trades
            WHERE status = 'CLOSED'
            ORDER BY exit_date DESC LIMIT 1
        """).fetchone()
        if row:
            last_exit = datetime.strptime(row[0], "%Y-%m-%d").date()
            pause_end = last_exit + timedelta(days=DRAWDOWN_PAUSE_DAYS)
            today     = date.today()
            if today <= pause_end:
                days_left = (pause_end - today).days
                return True, f"Drawdown {dd_pct:.1f}% ≥ {DRAWDOWN_PAUSE_PCT}%. Pause until {pause_end} ({days_left}d left)"
    return False, ""


def check_daily_loss_limit(conn: sqlite3.Connection) -> tuple[bool, float]:
    """Returns (limit_hit: bool, today_pnl_pct: float)"""
    capital   = get_current_capital(conn)
    today_pnl = get_today_pnl(conn)
    today_pct = (today_pnl / capital * 100) if capital > 0 else 0.0
    return today_pct <= -DAILY_LOSS_LIMIT_PCT, round(today_pct, 2)


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_signal_message(row: dict) -> str:
    """Format a single BUY signal for Telegram."""
    regime_icon = {
        "BULL": "🟢", "NEUTRAL": "🔵", "SIDEWAYS": "🟡",
        "HIGH_VIX": "🟠", "BEAR": "🔴"
    }.get(row.get("regime", "NEUTRAL"), "⚪")

    sl_pct  = ""
    tgt_pct = ""
    if row.get("close") and row.get("stop_loss"):
        sl_pct  = f" ({((row['stop_loss']  - row['close']) / row['close'] * 100):.1f}%)"
    if row.get("close") and row.get("target"):
        tgt_pct = f" ({((row['target']     - row['close']) / row['close'] * 100):+.1f}%)"

    msg = (
        f"🔔 *BUY SIGNAL — {row['symbol']}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Score     : `{row.get('total_score', 0):.1f}/100`\n"
        f"💰 Entry     : `₹{row.get('close', 0):.2f}`\n"
        f"🛑 Stop-Loss : `₹{row.get('stop_loss', 0):.2f}`{sl_pct}\n"
        f"🎯 Target    : `₹{row.get('target', 0):.2f}`{tgt_pct}\n"
        f"📦 Qty       : `{row.get('qty', 0)} shares`\n"
        f"💸 Max Risk  : `₹{round((row.get('close', 0) - row.get('stop_loss', 0)) * row.get('qty', 0), 0):.0f}`\n"
        f"🏷️ Bucket    : `{BUCKET_LABELS.get(row['symbol'], 'unknown')}`\n"
        f"{regime_icon} Regime   : `{row.get('regime', 'NEUTRAL')}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _Execute manually in Groww. Set stop-loss immediately after buying._"
    )
    return msg


def fmt_open_trade(row) -> str:
    """Format one open trade row for /status."""
    (tid, symbol, entry, entry_date, qty, sl, tgt,
     score, regime, bucket, notes) = row
    return (
        f"  *{symbol}* ({bucket})\n"
        f"  Entry ₹{entry:.2f} × {qty} | {entry_date}\n"
        f"  SL ₹{sl:.2f}  →  TGT ₹{tgt:.2f}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "👋 *NSE Swing Trader Bot*\n\n"
        "Manual execution in Groww. This bot = alerts + memory.\n\n"
        "*Commands:*\n"
        "/signals — today's BUY signals\n"
        "/status  — open trades + P&L\n"
        "/risk    — daily loss + drawdown\n"
        "/regime  — market regime\n"
        "/history — last 10 closed trades\n\n"
        "*Portfolio commands:*\n"
        "`/bought SYMBOL PRICE QTY`\n"
        "`/sold SYMBOL PRICE REASON`\n\n"
        "e.g. `/bought RELIANCE 1250.50 10`\n"
        "e.g. `/sold RELIANCE 1320 TARGET`"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show today's BUY signals from daily_scores."""
    conn   = get_connection()
    today  = today_ist()
    regime = get_latest_regime(conn)
    rules  = get_regime_rules(regime)

    if rules["max_trades"] == 0:
        await update.message.reply_text(
            f"🔴 *BEAR MARKET — No new trades today.*\n"
            f"Regime: `{regime}`. Capital preservation only.\n"
            f"Monitor your open stops.",
            parse_mode=ParseMode.MARKDOWN
        )
        conn.close()
        return

    # Check pause / daily loss
    paused, pause_reason = is_in_pause(conn)
    if paused:
        await update.message.reply_text(
            f"🛑 *Trading Paused*\n`{pause_reason}`\n\nNo new entries.",
            parse_mode=ParseMode.MARKDOWN
        )
        conn.close()
        return

    limit_hit, today_pct = check_daily_loss_limit(conn)
    if limit_hit:
        await update.message.reply_text(
            f"⛔ *Daily loss limit hit* ({today_pct:.1f}% vs -{DAILY_LOSS_LIMIT_PCT}% limit)\n"
            f"No new entries today.",
            parse_mode=ParseMode.MARKDOWN
        )
        conn.close()
        return

    rows = conn.execute("""
        SELECT symbol, total_score, signal, regime, gate_passed
        FROM daily_scores
        WHERE date = ? AND signal = 'BUY' AND gate_passed = 1
        ORDER BY total_score DESC
    """, (today,)).fetchall()

    if not rows:
        # Also check yesterday's scores in case pipeline ran overnight
        rows = conn.execute("""
            SELECT symbol, total_score, signal, regime, gate_passed
            FROM daily_scores
            WHERE signal = 'BUY' AND gate_passed = 1
            ORDER BY date DESC, total_score DESC
            LIMIT 10
        """).fetchall()
        date_note = "_(latest available — run pipeline for today's scores)_"
    else:
        date_note = f"_{today}_"

    conn.close()

    if not rows:
        await update.message.reply_text(
            f"📭 No BUY signals today.\n{date_note}\n\nRegime: `{regime}`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    # Filter by regime's allowed buckets
    allowed = rules["allowed_buckets"]
    min_score = rules["min_score"]

    header = (
        f"📋 *Signals — {date_note}*\n"
        f"Regime: `{regime}` | Min score: `{min_score}`\n"
        f"─────────────────────\n"
    )
    await update.message.reply_text(header, parse_mode=ParseMode.MARKDOWN)

    sent = 0
    for (symbol, score, signal, sig_regime, gate) in rows:
        bucket = BUCKET_LABELS.get(symbol, "unknown")
        if allowed != "ALL" and bucket not in allowed:
            continue
        if score < min_score:
            continue

        # Get latest price data for the message
        conn2 = get_connection()
        price_row = conn2.execute("""
            SELECT close, date FROM daily_prices
            WHERE symbol = ? ORDER BY date DESC LIMIT 1
        """, (symbol,)).fetchone()

        score_row = conn2.execute("""
            SELECT momentum_score, trend_score, rsi_score, macd_score,
                   volume_score, bb_score
            FROM daily_scores WHERE symbol = ? AND signal = 'BUY'
            ORDER BY date DESC LIMIT 1
        """, (symbol,)).fetchone()
        conn2.close()

        close = price_row[0] if price_row else 0

        from indicators import calculate_stop_loss, calculate_target, calculate_position_size
        conn3 = get_connection()
        atr_row = conn3.execute("""
            SELECT close FROM daily_prices WHERE symbol = ?
            ORDER BY date DESC LIMIT 15
        """, (symbol,)).fetchall()
        conn3.close()

        # Simple ATR estimate from recent closes (fallback)
        if len(atr_row) >= 14:
            closes = [r[0] for r in atr_row]
            atr = abs(closes[0] - closes[-1]) / len(closes)
        else:
            atr = close * 0.02

        sl  = calculate_stop_loss(close, atr)
        tgt = calculate_target(close, atr)
        qty = calculate_position_size(STARTING_CAPITAL, RISK_PER_TRADE_PCT, close, sl)

        signal_data = {
            "symbol": symbol, "total_score": score,
            "close": close, "stop_loss": sl, "target": tgt,
            "qty": qty, "regime": sig_regime or regime,
        }

        factor_line = ""
        if score_row:
            factor_line = (
                f"\n📐 _Factors: Mom={score_row[0]:.0f} Trend={score_row[1]:.0f} "
                f"RSI={score_row[2]:.0f} MACD={score_row[3]:.0f} "
                f"Vol={score_row[4]:.0f} BB={score_row[5]:.0f}_"
            )

        msg = fmt_signal_message(signal_data) + factor_line
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        sent += 1

    if sent == 0:
        await update.message.reply_text(
            f"No signals pass regime filter (`{regime}` allows: {allowed})",
            parse_mode=ParseMode.MARKDOWN
        )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all open trades + capital + P&L summary."""
    conn        = get_connection()
    capital     = get_current_capital(conn)
    open_trades = get_open_trades(conn)
    today_pnl   = get_today_pnl(conn)
    dd_pct, peak = get_drawdown(conn)
    regime      = get_latest_regime(conn)
    paused, pause_reason = is_in_pause(conn)

    total_closed = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(pnl),0) FROM trades WHERE status='CLOSED'"
    ).fetchone()
    total_winners = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl > 0"
    ).fetchone()[0]

    conn.close()

    free_slots = MAX_OPEN_TRADES - len(open_trades)
    regime_icon = {"BULL":"🟢","NEUTRAL":"🔵","SIDEWAYS":"🟡","HIGH_VIX":"🟠","BEAR":"🔴"}.get(regime,"⚪")

    msg = (
        f"📊 *Portfolio Status — {now_ist()}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 Capital     : `₹{capital:,.0f}`\n"
        f"📈 Today P&L   : `₹{today_pnl:+,.0f}`\n"
        f"📉 Drawdown    : `{dd_pct:.1f}%` (peak ₹{peak:,.0f})\n"
        f"{regime_icon} Regime     : `{regime}`\n"
        f"🔓 Free slots  : `{free_slots}/{MAX_OPEN_TRADES}`\n"
    )

    if paused:
        msg += f"\n🛑 *PAUSED: {pause_reason}*\n"

    if total_closed[0] > 0:
        win_rate = total_winners / total_closed[0] * 100
        msg += (
            f"\n📋 *Closed Trades*\n"
            f"  Total  : {total_closed[0]}\n"
            f"  P&L    : ₹{total_closed[1]:+,.0f}\n"
            f"  Win%   : {win_rate:.0f}%\n"
        )

    if open_trades:
        msg += f"\n🟡 *Open Trades ({len(open_trades)})*\n"
        msg += "─────────────────────\n"
        for row in open_trades:
            msg += fmt_open_trade(row)
    else:
        msg += "\n_No open trades._"

    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_bought(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /bought SYMBOL PRICE QTY [NOTES]
    Logs a new manual trade into the DB.
    """
    args = context.args
    if len(args) < 3:
        await update.message.reply_text(
            "❌ Usage: `/bought SYMBOL PRICE QTY [notes]`\n"
            "e.g. `/bought RELIANCE 1250.50 10 strong breakout`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    symbol = args[0].upper()
    try:
        entry_price = float(args[1])
        qty         = int(args[2])
    except ValueError:
        await update.message.reply_text("❌ Price and qty must be numbers.")
        return

    notes = " ".join(args[3:]) if len(args) > 3 else ""

    conn   = get_connection()
    today  = today_ist()
    regime = get_latest_regime(conn)
    bucket = BUCKET_LABELS.get(symbol, "unknown")

    # Risk guards — warn but don't block (it's already a done trade)
    open_count = len(get_open_trades(conn))
    if open_count >= MAX_OPEN_TRADES:
        await update.message.reply_text(
            f"⚠️ *Warning:* You now have {open_count + 1} open trades — above the {MAX_OPEN_TRADES} max.\n"
            f"Logging anyway since you already bought.",
            parse_mode=ParseMode.MARKDOWN
        )

    # Check if already have an open position in this symbol
    existing = conn.execute("""
        SELECT id FROM trades WHERE symbol = ? AND status = 'OPEN'
    """, (symbol,)).fetchone()
    if existing:
        await update.message.reply_text(
            f"⚠️ Already have an open trade for {symbol} (ID {existing[0]}).\n"
            f"Use /status to check. Logging anyway.",
            parse_mode=ParseMode.MARKDOWN
        )

    # Get today's score and SL/target from daily_scores if available
    score_row = conn.execute("""
        SELECT total_score FROM daily_scores
        WHERE symbol = ? ORDER BY date DESC LIMIT 1
    """, (symbol,)).fetchone()
    score_at_entry = score_row[0] if score_row else None

    # Calculate stop-loss and target using ATR
    from indicators import calculate_stop_loss, calculate_target
    atr_rows = conn.execute("""
        SELECT close FROM daily_prices WHERE symbol = ?
        ORDER BY date DESC LIMIT 15
    """, (symbol,)).fetchall()

    if len(atr_rows) >= 14:
        closes = [r[0] for r in atr_rows]
        atr = abs(closes[0] - closes[-1]) / len(closes)
    else:
        atr = entry_price * 0.02

    sl  = calculate_stop_loss(entry_price, atr)
    tgt = calculate_target(entry_price, atr)

    conn.execute("""
        INSERT INTO trades
        (symbol, bucket, direction, status, entry_price, entry_date,
         qty, stop_loss, target, score_at_entry, regime_at_entry, notes)
        VALUES (?, ?, 'LONG', 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, bucket, entry_price, today, qty, sl, tgt,
          score_at_entry, regime, notes))
    conn.commit()

    trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    max_loss  = round((entry_price - sl) * qty, 0)
    invested  = round(entry_price * qty, 0)

    conn.close()

    msg = (
        f"✅ *Trade Logged — {symbol}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🆔 Trade ID  : `#{trade_id}`\n"
        f"💰 Entry     : `₹{entry_price:.2f} × {qty} = ₹{invested:,.0f}`\n"
        f"🛑 Stop-Loss : `₹{sl:.2f}` (auto-calculated)\n"
        f"🎯 Target    : `₹{tgt:.2f}`\n"
        f"⚠️ Max Loss  : `₹{max_loss:,.0f}`\n"
        f"🏷️ Bucket   : `{bucket}`\n"
        f"📅 Date      : `{today}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Set your stop-loss in Groww at ₹{sl:.2f} immediately._"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_sold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /sold SYMBOL PRICE [REASON]
    Closes the most recent open trade for SYMBOL. Calculates P&L.
    REASON: TARGET / STOPLOSS / MANUAL (default: MANUAL)
    """
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "❌ Usage: `/sold SYMBOL PRICE [REASON]`\n"
            "e.g. `/sold RELIANCE 1320 TARGET`\n"
            "Reasons: TARGET / STOPLOSS / MANUAL",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    symbol = args[0].upper()
    try:
        exit_price = float(args[1])
    except ValueError:
        await update.message.reply_text("❌ Price must be a number.")
        return

    reason = args[2].upper() if len(args) > 2 else "MANUAL"

    conn = get_connection()
    today = today_ist()

    trade = conn.execute("""
        SELECT id, entry_price, qty, stop_loss, target
        FROM trades
        WHERE symbol = ? AND status = 'OPEN'
        ORDER BY entry_date DESC LIMIT 1
    """, (symbol,)).fetchone()

    if not trade:
        conn.close()
        await update.message.reply_text(
            f"❌ No open trade found for *{symbol}*.\nUse /status to check.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    tid, entry_price, qty, sl, tgt = trade
    pnl     = round((exit_price - entry_price) * qty, 2)
    pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

    conn.execute("""
        UPDATE trades
        SET status='CLOSED', exit_price=?, exit_date=?,
            pnl=?, pnl_pct=?, exit_reason=?,
            updated_at=datetime('now','localtime')
        WHERE id=?
    """, (exit_price, today, pnl, pnl_pct, reason, tid))
    conn.commit()

    new_capital = get_current_capital(conn)
    conn.close()

    icon = "🟢" if pnl >= 0 else "🔴"
    msg = (
        f"{icon} *Trade Closed — {symbol}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🆔 Trade ID  : `#{tid}`\n"
        f"💰 Entry     : `₹{entry_price:.2f}`\n"
        f"🏁 Exit      : `₹{exit_price:.2f}`\n"
        f"📦 Qty       : `{qty}`\n"
        f"📊 P&L       : `₹{pnl:+,.2f} ({pnl_pct:+.2f}%)`\n"
        f"📋 Reason    : `{reason}`\n"
        f"💼 New Capital: `₹{new_capital:,.0f}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )

    if pnl < 0 and abs(pnl_pct) >= 3:
        msg += f"⚠️ _Significant loss. Review trade setup._"

    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show risk dashboard: daily loss, drawdown, pause status."""
    conn = get_connection()
    capital   = get_current_capital(conn)
    today_pnl = get_today_pnl(conn)
    dd_pct, peak = get_drawdown(conn)
    paused, pause_reason = is_in_pause(conn)
    limit_hit, today_pct  = check_daily_loss_limit(conn)
    conn.close()

    daily_used  = abs(min(today_pct, 0))
    daily_left  = max(DAILY_LOSS_LIMIT_PCT - daily_used, 0)
    dd_left     = max(DRAWDOWN_PAUSE_PCT - dd_pct, 0)

    daily_bar = "🟥" * int(daily_used / DAILY_LOSS_LIMIT_PCT * 5) + \
                "⬜" * (5 - int(daily_used / DAILY_LOSS_LIMIT_PCT * 5))
    dd_bar    = "🟥" * int(dd_pct / DRAWDOWN_PAUSE_PCT * 5) + \
                "⬜" * (5 - int(dd_pct / DRAWDOWN_PAUSE_PCT * 5))

    msg = (
        f"🛡️ *Risk Dashboard — {today_ist()}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 Capital      : `₹{capital:,.0f}`\n"
        f"📅 Today P&L    : `₹{today_pnl:+,.0f} ({today_pct:+.2f}%)`\n"
        f"\n*Daily Loss Limit ({DAILY_LOSS_LIMIT_PCT}%)*\n"
        f"{daily_bar} `{daily_used:.1f}% used / {daily_left:.1f}% left`\n"
        f"\n*Drawdown Limit ({DRAWDOWN_PAUSE_PCT}%)*\n"
        f"{dd_bar} `{dd_pct:.1f}% used / {dd_left:.1f}% left`\n"
    )

    if paused:
        msg += f"\n🛑 *STATUS: TRADING PAUSED*\n`{pause_reason}`"
    elif limit_hit:
        msg += f"\n⛔ *Daily limit hit — no new trades today.*"
    else:
        msg += f"\n✅ *All risk checks: OK*"

    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current market regime and trading rules."""
    conn = get_connection()
    regime = get_latest_regime(conn)
    rules  = get_regime_rules(regime)

    row = conn.execute("""
        SELECT date, nifty_close, nifty_sma50, nifty_sma200, india_vix, notes
        FROM regime_log ORDER BY date DESC LIMIT 1
    """).fetchone()
    conn.close()

    icon = {"BULL":"🟢","NEUTRAL":"🔵","SIDEWAYS":"🟡","HIGH_VIX":"🟠","BEAR":"🔴"}.get(regime,"⚪")

    msg = f"{icon} *Market Regime: {regime}*\n━━━━━━━━━━━━━━━━━━━━\n"

    if row:
        date_, nifty, sma50, sma200, vix, notes = row
        msg += (
            f"📅 Date       : `{date_}`\n"
            f"📈 Nifty50    : `{nifty}`\n"
            f"〰️ SMA-50     : `{sma50}`\n"
            f"〰️ SMA-200    : `{sma200}`\n"
            f"😨 India VIX  : `{vix}`\n"
            f"📝 Reason     : _{notes}_\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
        )

    allowed = rules["allowed_buckets"]
    buckets_str = "All buckets" if allowed == "ALL" else (", ".join(allowed) if allowed else "❌ NONE")

    msg += (
        f"*Trading Rules for {regime}:*\n"
        f"  Max trades : `{rules['max_trades']}`\n"
        f"  Min score  : `{rules['min_score']}/100`\n"
        f"  Allowed    : `{buckets_str}`\n"
    )

    if regime == "BEAR":
        msg += "\n⛔ *NO NEW ENTRIES. Capital preservation only.*"
    elif regime == "HIGH_VIX":
        msg += "\n⚠️ *Defensive stocks only. Tighten stops.*"

    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show last 10 closed trades."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT symbol, entry_price, exit_price, qty, pnl, pnl_pct,
               exit_reason, entry_date, exit_date
        FROM trades
        WHERE status = 'CLOSED'
        ORDER BY exit_date DESC
        LIMIT 10
    """).fetchall()
    conn.close()

    if not rows:
        await update.effective_message.reply_text("📭 No closed trades yet.")
        return

    total_pnl  = sum(r[4] for r in rows if r[4])
    msg = f"📋 *Last {len(rows)} Closed Trades*\n━━━━━━━━━━━━━━━━━━━━\n"

    for (sym, entry, exit_, qty, pnl, pnl_pct, reason, edate, xdate) in rows:
        icon = "🟢" if (pnl or 0) >= 0 else "🔴"
        msg += (
            f"{icon} *{sym}* | ₹{pnl:+,.0f} ({pnl_pct:+.1f}%)\n"
            f"   ₹{entry:.2f}→₹{exit_:.2f} × {qty} | {reason} | {xdate}\n"
        )

    msg += f"━━━━━━━━━━━━━━━━━━━━\n💼 Total P&L (shown): `₹{total_pnl:+,.0f}`"
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


# ─────────────────────────────────────────────────────────────────────────────
# DAILY ALERT SENDER  (called from GitHub Actions, not polling)
# ─────────────────────────────────────────────────────────────────────────────

async def send_daily_alerts() -> None:
    """
    Standalone async function.
    Reads today's BUY signals and pushes them to Telegram.
    Called with: python telegram_bot.py --send-alerts
    No polling, no bot start. Just send and exit.
    """
    import httpx

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Skipping alerts.")
        return

    conn   = get_connection()
    today  = today_ist()
    regime = get_latest_regime(conn)
    rules  = get_regime_rules(regime)

    async def push(text: str) -> None:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       text,
            "parse_mode": "Markdown",
        }
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                logger.error(f"Telegram push failed: {r.text}")

    # ── Risk checks first ────────────────────────────────────────────────────
    paused, pause_reason = is_in_pause(conn)
    limit_hit, today_pct = check_daily_loss_limit(conn)
    dd_pct, _ = get_drawdown(conn)

    regime_icon = {"BULL":"🟢","NEUTRAL":"🔵","SIDEWAYS":"🟡","HIGH_VIX":"🟠","BEAR":"🔴"}.get(regime,"⚪")

    # Morning summary header
    capital     = get_current_capital(conn)
    open_trades = get_open_trades(conn)
    free_slots  = MAX_OPEN_TRADES - len(open_trades)

    header = (
        f"🌅 *Morning Report — {today}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{regime_icon} Regime   : `{regime}`\n"
        f"💼 Capital  : `₹{capital:,.0f}`\n"
        f"📉 Drawdown : `{dd_pct:.1f}%`\n"
        f"🔓 Slots    : `{free_slots}/{MAX_OPEN_TRADES} free`\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    await push(header)

    if rules["max_trades"] == 0:
        await push("🔴 *BEAR MARKET — No new entries today. Monitor open stops.*")
        conn.close()
        return

    if paused:
        await push(f"🛑 *Trading Paused*\n{pause_reason}")
        conn.close()
        return

    if limit_hit:
        await push(f"⛔ *Daily loss limit hit ({today_pct:.1f}%). No new trades.*")
        conn.close()
        return

    # ── Fetch signals ────────────────────────────────────────────────────────
    rows = conn.execute("""
        SELECT symbol, total_score, momentum_score, trend_score,
               rsi_score, macd_score, volume_score, bb_score, regime
        FROM daily_scores
        WHERE date = ? AND signal = 'BUY' AND gate_passed = 1
        ORDER BY total_score DESC
    """, (today,)).fetchall()

    allowed   = rules["allowed_buckets"]
    min_score = rules["min_score"]

    # Filter by regime rules
    filtered = []
    for row in rows:
        sym = row[0]
        score = row[1]
        bucket = BUCKET_LABELS.get(sym, "unknown")
        if allowed != "ALL" and bucket not in allowed:
            continue
        if score < min_score:
            continue
        filtered.append(row)

    if not filtered:
        await push(f"📭 No BUY signals today after regime filter (`{regime}`).")
        conn.close()
        return

    await push(f"🔔 *{len(filtered)} BUY Signal(s) Today*\n_Score ≥ {min_score}, Gates ✅, Regime: {regime}_")

    from indicators import calculate_stop_loss, calculate_target, calculate_position_size

    for row in filtered:
        sym = row[0]

        price_row = conn.execute("""
            SELECT close FROM daily_prices WHERE symbol = ?
            ORDER BY date DESC LIMIT 1
        """, (sym,)).fetchone()
        close = price_row[0] if price_row else 0

        atr_rows = conn.execute("""
            SELECT close FROM daily_prices WHERE symbol = ?
            ORDER BY date DESC LIMIT 15
        """, (sym,)).fetchall()

        if len(atr_rows) >= 14:
            closes = [r[0] for r in atr_rows]
            atr = abs(closes[0] - closes[-1]) / len(closes)
        else:
            atr = close * 0.02

        sl  = calculate_stop_loss(close, atr)
        tgt = calculate_target(close, atr)
        qty = calculate_position_size(STARTING_CAPITAL, RISK_PER_TRADE_PCT, close, sl)

        signal_data = {
            "symbol": sym, "total_score": row[1],
            "close": close, "stop_loss": sl, "target": tgt,
            "qty": qty, "regime": row[8] or regime,
        }
        msg = fmt_signal_message(signal_data)
        await push(msg)

    conn.close()
    logger.info(f"Daily alerts sent: {len(filtered)} signals.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — polling bot OR one-shot alert sender
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send-alerts", action="store_true",
                        help="Send today's alerts and exit (for GitHub Actions)")
    args = parser.parse_args()

    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set. Add to .env or GitHub Secrets.")
        sys.exit(1)

    if args.send_alerts:
        import asyncio
        asyncio.run(send_daily_alerts())
        return

    # Interactive bot (run locally or on a server)
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("bought",  cmd_bought))
    app.add_handler(CommandHandler("sold",    cmd_sold))
    app.add_handler(CommandHandler("risk",    cmd_risk))
    app.add_handler(CommandHandler("regime",  cmd_regime))
    app.add_handler(CommandHandler("history", cmd_history))

    print(f"🤖 Bot started — waiting for commands...")
    print(f"   Chat ID: {TELEGRAM_CHAT_ID or 'NOT SET'}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
