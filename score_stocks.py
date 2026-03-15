# score_stocks.py
# 6-Factor scoring model (0–100) + 3 Hard Gates
# Reads from daily_prices → writes to daily_scores
# Run after fetch_data.py and regime_engine.py
#
# DOWNSIDE FIRST:
#   A score of 68+ is necessary but NOT sufficient to trade.
#   All 3 hard gates must also pass.
#   Even then, confirm the regime is not BEAR or HIGH_VIX before entering.

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import pytz
import os

from config import (
    ALL_STOCKS,
    BUCKET_LABELS,
    DB_PATH,
    LOG_DIR,
    TIMEZONE,
    TRADE_SIGNAL_THRESHOLD,
    STARTING_CAPITAL,
    RISK_PER_TRADE_PCT,
)
from create_db import get_connection
from indicators import (
    add_all_indicators,
    calculate_stop_loss,
    calculate_target,
    calculate_position_size,
)

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(
    os.path.join(LOG_DIR, "score_{time:YYYY-MM-DD}.log"),
    rotation="1 day", retention="14 days", level="INFO"
)

IST = pytz.timezone(TIMEZONE)


# ─────────────────────────────────────────────────────────────────────────────
# FACTOR WEIGHTS  (must sum to 100)
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS = {
    "momentum": 20,   # Price momentum — is it trending up?
    "trend":    20,   # EMA/SMA structure — is the trend healthy?
    "rsi":      15,   # RSI — not overbought, has room to run
    "macd":     15,   # MACD — momentum turning positive
    "volume":   15,   # Volume confirmation — smart money involved?
    "bb":       15,   # Bollinger Band position — not extended
}
assert sum(WEIGHTS.values()) == 100, "Weights must sum to 100"


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL FACTOR SCORERS  (each returns 0.0–1.0)
# Multiply by weight to get points contribution.
# ─────────────────────────────────────────────────────────────────────────────

def score_momentum(row: pd.Series) -> float:
    """
    Price vs EMA-20 and EMA-50.
    Best: price clearly above both EMAs (strong uptrend).
    Worst: price below both EMAs.
    """
    score = 0.0
    close = row.get("close", np.nan)

    ema20 = row.get("ema_20", np.nan)
    ema50 = row.get("ema_50", np.nan)

    if pd.isna(close) or close <= 0:
        return 0.0

    if not pd.isna(ema20) and close > ema20:
        score += 0.5
        # bonus: how far above EMA-20?
        gap_pct = (close - ema20) / ema20 * 100
        if gap_pct > 3:
            score += 0.2
        elif gap_pct > 1:
            score += 0.1

    if not pd.isna(ema50) and close > ema50:
        score += 0.3

    return min(score, 1.0)


def score_trend(row: pd.Series) -> float:
    """
    SMA alignment: SMA-20 > SMA-50 > SMA-200 = perfect bull structure.
    Also checks EMA-9 > EMA-20 (short-term momentum within trend).
    """
    score = 0.0

    sma20  = row.get("sma_20",  np.nan)
    sma50  = row.get("sma_50",  np.nan)
    sma200 = row.get("sma_200", np.nan)
    ema9   = row.get("ema_9",   np.nan)
    ema20  = row.get("ema_20",  np.nan)

    if not (pd.isna(sma20) or pd.isna(sma50)) and sma20 > sma50:
        score += 0.4

    if not (pd.isna(sma50) or pd.isna(sma200)) and sma50 > sma200:
        score += 0.4

    if not (pd.isna(ema9) or pd.isna(ema20)) and ema9 > ema20:
        score += 0.2

    return min(score, 1.0)


def score_rsi(row: pd.Series) -> float:
    """
    RSI-14 scoring.
    Sweet spot: 50–65 (trending, not overbought).
    Above 70: overbought — penalise.
    Below 40: weak — low score.
    30–50: recovering — partial credit.
    """
    rsi = row.get("rsi_14", np.nan)
    if pd.isna(rsi):
        return 0.0

    if 55 <= rsi <= 65:    return 1.0    # ideal zone
    if 50 <= rsi < 55:     return 0.8    # good
    if 65 < rsi <= 70:     return 0.6    # slightly extended but ok
    if 45 <= rsi < 50:     return 0.5    # neutral
    if 40 <= rsi < 45:     return 0.3    # weakening
    if rsi > 70:           return 0.2    # overbought — risky entry
    if 30 <= rsi < 40:     return 0.2    # oversold — not our setup
    return 0.0


def score_macd(row: pd.Series) -> float:
    """
    MACD: line above signal + positive histogram = bullish.
    Histogram growing = momentum accelerating.
    """
    macd_line   = row.get("macd_line",   np.nan)
    macd_signal = row.get("macd_signal", np.nan)
    macd_hist   = row.get("macd_hist",   np.nan)

    if pd.isna(macd_line) or pd.isna(macd_signal):
        return 0.0

    score = 0.0

    if macd_line > macd_signal:
        score += 0.5    # bullish crossover or continuation

    if not pd.isna(macd_hist):
        if macd_hist > 0:
            score += 0.3
        # Is the histogram growing (momentum increasing)?
        # We'll handle multi-row logic in the wrapper below
        # For now: positive histogram gets full credit

    if macd_line > 0:
        score += 0.2    # MACD above zero line = broader bull trend

    return min(score, 1.0)


def score_volume(row: pd.Series) -> float:
    """
    Volume ratio vs 20-day average.
    We want above-average volume on a bullish day.
    If close > open (green candle) AND high volume → full score.
    """
    vol_ratio = row.get("vol_ratio", np.nan)
    close     = row.get("close",     np.nan)
    open_     = row.get("open",      np.nan)

    if pd.isna(vol_ratio):
        return 0.5   # no volume data → neutral (don't punish)

    if vol_ratio < 0.5:
        return 0.1   # suspiciously low volume

    is_green = (not pd.isna(close) and not pd.isna(open_) and close > open_)

    if vol_ratio >= 2.0:
        return 1.0 if is_green else 0.6
    if vol_ratio >= 1.5:
        return 0.85 if is_green else 0.5
    if vol_ratio >= 1.0:
        return 0.65 if is_green else 0.4
    if vol_ratio >= 0.7:
        return 0.4
    return 0.2


def score_bollinger(row: pd.Series) -> float:
    """
    Bollinger Band %B.
    Best entry: price in lower-to-mid band (0.2–0.55) = room to run.
    Avoid: price at top of band (>0.85) = extended.
    Also penalise very narrow bands (bb_width < 0.03) = consolidation, no trend.
    """
    pct_b    = row.get("bb_pct_b", np.nan)
    bb_width = row.get("bb_width", np.nan)

    if pd.isna(pct_b):
        return 0.5

    # Very narrow band = low volatility squeeze — not ideal for entry
    if not pd.isna(bb_width) and bb_width < 0.02:
        return 0.3

    if 0.35 <= pct_b <= 0.55:   return 1.0   # mid-band: price has room both ways
    if 0.20 <= pct_b < 0.35:    return 0.85  # slightly below mid: good value entry
    if 0.55 < pct_b <= 0.70:    return 0.65  # slightly above mid: ok
    if 0.10 <= pct_b < 0.20:    return 0.50  # near lower band: could be weak
    if 0.70 < pct_b <= 0.85:    return 0.40  # extended
    if pct_b > 0.85:            return 0.10  # at/above upper band — do not chase
    if pct_b < 0.10:            return 0.20  # at lower band — could be breakdown
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# HARD GATES  (must ALL pass — no exceptions)
# ─────────────────────────────────────────────────────────────────────────────

def check_hard_gates(row: pd.Series) -> tuple[bool, list[str]]:
    """
    3 mandatory conditions. If any fail → no trade, regardless of score.

    DOWNSIDE: These gates exist specifically to protect capital.
    Never override them. If a stock looks amazing but fails a gate,
    it's telling you something.

    Returns (passed: bool, reasons: list[str])
    """
    failures = []

    close  = row.get("close",   np.nan)
    sma200 = row.get("sma_200", np.nan)
    vol_ratio = row.get("vol_ratio", np.nan)
    atr_pct   = row.get("atr_pct",   np.nan)

    # Gate 1: Price must be above SMA-200
    # Trading below 200 SMA = fighting the long-term trend. Don't.
    if pd.isna(sma200) or pd.isna(close) or close <= sma200:
        failures.append("GATE1_FAIL: price below SMA-200")

    # Gate 2: Volume must be at least 70% of 20-day average
    # Thin volume = no institutional participation = trap
    if pd.isna(vol_ratio) or vol_ratio < 0.7:
        failures.append(f"GATE2_FAIL: low volume ratio ({vol_ratio:.2f})" if not pd.isna(vol_ratio) else "GATE2_FAIL: no volume data")

    # Gate 3: ATR% must not be extreme (> 5% of price = too volatile for our size)
    # On a ₹50k account, extreme volatility means stop-loss eats too much capital
    if not pd.isna(atr_pct) and atr_pct > 5.0:
        failures.append(f"GATE3_FAIL: ATR too high ({atr_pct:.1f}%)")

    passed = len(failures) == 0
    return passed, failures


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORER
# ─────────────────────────────────────────────────────────────────────────────

def score_single_stock(symbol: str, df: pd.DataFrame) -> dict:
    """
    Takes full OHLCV history, computes indicators, scores latest row.
    Returns a dict with all factor scores and signal.
    """
    result = {
        "symbol":         symbol,
        "date":           None,
        "total_score":    0.0,
        "momentum_score": 0.0,
        "trend_score":    0.0,
        "rsi_score":      0.0,
        "macd_score":     0.0,
        "volume_score":   0.0,
        "bb_score":       0.0,
        "gate_passed":    0,
        "gate_failures":  [],
        "signal":         "IGNORE",
        "stop_loss":      None,
        "target":         None,
        "qty":            0,
        "close":          None,
        "atr":            None,
        "error":          None,
    }

    if df.empty or len(df) < 50:
        result["error"] = f"Insufficient data: {len(df)} rows"
        return result

    try:
        df = add_all_indicators(df)
    except Exception as e:
        result["error"] = f"Indicator error: {e}"
        return result

    latest = df.iloc[-1]
    result["date"]  = latest.get("date", datetime.now(IST).strftime("%Y-%m-%d"))
    result["close"] = latest.get("close", None)
    result["atr"]   = latest.get("atr_14", None)

    # ── Score each factor ────────────────────────────────────────────────────
    f_momentum = score_momentum(latest)
    f_trend    = score_trend(latest)
    f_rsi      = score_rsi(latest)
    f_macd     = score_macd(latest)
    f_volume   = score_volume(latest)
    f_bb       = score_bollinger(latest)

    result["momentum_score"] = round(f_momentum * WEIGHTS["momentum"], 2)
    result["trend_score"]    = round(f_trend    * WEIGHTS["trend"],    2)
    result["rsi_score"]      = round(f_rsi      * WEIGHTS["rsi"],      2)
    result["macd_score"]     = round(f_macd     * WEIGHTS["macd"],     2)
    result["volume_score"]   = round(f_volume   * WEIGHTS["volume"],   2)
    result["bb_score"]       = round(f_bb       * WEIGHTS["bb"],       2)

    result["total_score"] = round(
        result["momentum_score"] + result["trend_score"] +
        result["rsi_score"]      + result["macd_score"]  +
        result["volume_score"]   + result["bb_score"],
        2
    )

    # ── Hard gates ───────────────────────────────────────────────────────────
    gate_passed, gate_failures = check_hard_gates(latest)
    result["gate_passed"]   = 1 if gate_passed else 0
    result["gate_failures"] = gate_failures

    # ── Signal determination ─────────────────────────────────────────────────
    if result["total_score"] >= TRADE_SIGNAL_THRESHOLD and gate_passed:
        result["signal"] = "BUY"
    elif result["total_score"] >= TRADE_SIGNAL_THRESHOLD * 0.88:
        result["signal"] = "WATCH"   # close — monitor tomorrow
    else:
        result["signal"] = "IGNORE"

    # ── Pre-calculate trade levels for BUY signals ───────────────────────────
    if result["signal"] == "BUY" and result["close"] and result["atr"]:
        close = result["close"]
        atr   = result["atr"]
        result["stop_loss"] = calculate_stop_loss(close, atr, multiplier=2.0)
        result["target"]    = calculate_target(close, atr, multiplier=3.0)
        result["qty"]       = calculate_position_size(
            STARTING_CAPITAL, RISK_PER_TRADE_PCT, close, result["stop_loss"]
        )

    return result


def run_scoring(conn: sqlite3.Connection, regime: str = "NEUTRAL") -> list[dict]:
    """
    Score all 28 stocks and save results to daily_scores table.
    Returns list of results sorted by total_score descending.
    """
    today = datetime.now(IST).strftime("%Y-%m-%d")
    results = []

    logger.info(f"Scoring {len(ALL_STOCKS)} stocks | regime={regime} | date={today}")

    for symbol in ALL_STOCKS:
        # Fetch last 250 trading days (enough for SMA-200)
        query = """
            SELECT date, open, high, low, close, volume
            FROM daily_prices
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 250
        """
        df = pd.read_sql_query(query, conn, params=(symbol,))

        if df.empty:
            logger.warning(f"{symbol}: no price data in DB")
            continue

        # Sort ascending for indicator calculation
        df = df.sort_values("date").reset_index(drop=True)

        scored = score_single_stock(symbol, df)
        scored["regime"] = regime

        if scored.get("error"):
            logger.warning(f"{symbol}: {scored['error']}")
        else:
            logger.info(
                f"{symbol:>14} | score={scored['total_score']:>5.1f} "
                f"| signal={scored['signal']:>6} | gates={'✅' if scored['gate_passed'] else '❌'}"
            )

        results.append(scored)
        _upsert_score(conn, scored, today, regime)

    results.sort(key=lambda x: x["total_score"], reverse=True)
    logger.info(f"Scoring complete. BUY signals: {sum(1 for r in results if r['signal']=='BUY')}")
    return results


def _upsert_score(conn: sqlite3.Connection, scored: dict, date: str, regime: str) -> None:
    """Write one score row to daily_scores (upsert)."""
    try:
        conn.execute("""
            INSERT OR REPLACE INTO daily_scores
            (symbol, date, total_score, momentum_score, trend_score,
             volume_score, rsi_score, macd_score, bb_score,
             signal, regime, gate_passed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scored["symbol"], date,
            scored["total_score"],
            scored["momentum_score"], scored["trend_score"],
            scored["volume_score"],   scored["rsi_score"],
            scored["macd_score"],     scored["bb_score"],
            scored["signal"],         regime,
            scored["gate_passed"],
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"DB write error for {scored['symbol']}: {e}")


def print_scoreboard(results: list[dict]) -> None:
    """Pretty-print the day's scores to console."""
    print("\n" + "═" * 75)
    print(f"  {'SYMBOL':>14}  {'SCORE':>6}  {'SIGNAL':>6}  {'GATES':>5}  "
          f"{'CLOSE':>8}  {'SL':>8}  {'TGT':>8}  {'QTY':>4}")
    print("─" * 75)

    for r in results:
        if r.get("error"):
            continue
        gates = "✅" if r["gate_passed"] else "❌"
        close = f"₹{r['close']:.1f}"     if r["close"] else "—"
        sl    = f"₹{r['stop_loss']:.1f}" if r["stop_loss"] else "—"
        tgt   = f"₹{r['target']:.1f}"    if r["target"]   else "—"
        qty   = str(r["qty"])             if r["qty"] > 0  else "—"

        print(f"  {r['symbol']:>14}  {r['total_score']:>6.1f}  "
              f"{r['signal']:>6}  {gates:>5}  "
              f"{close:>8}  {sl:>8}  {tgt:>8}  {qty:>4}")

    print("═" * 75)
    buys   = [r for r in results if r["signal"] == "BUY"]
    watches = [r for r in results if r["signal"] == "WATCH"]
    print(f"  🟢 BUY: {len(buys)}   👀 WATCH: {len(watches)}   "
          f"Threshold: {TRADE_SIGNAL_THRESHOLD}/100\n")


if __name__ == "__main__":
    from regime_engine import get_latest_regime

    conn   = get_connection()
    regime = get_latest_regime(conn)
    print(f"📊 Current regime: {regime}")

    results = run_scoring(conn, regime=regime)
    print_scoreboard(results)
    conn.close()
