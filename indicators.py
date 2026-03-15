# indicators.py
# Pure calculation functions — no DB reads, no side effects.
# Takes a pandas DataFrame (date, open, high, low, close, volume)
# Returns the same DataFrame with indicator columns appended.
#
# All functions are defensive — they return NaN columns on bad input
# rather than crashing the whole pipeline.

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MOVING AVERAGES
# ─────────────────────────────────────────────────────────────────────────────

def add_ema(df: pd.DataFrame, periods: list[int] = [9, 20, 50]) -> pd.DataFrame:
    """Exponential Moving Averages."""
    df = df.copy()
    for p in periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean().round(2)
    return df


def add_sma(df: pd.DataFrame, periods: list[int] = [20, 50, 200]) -> pd.DataFrame:
    """Simple Moving Averages."""
    df = df.copy()
    for p in periods:
        df[f"sma_{p}"] = df["close"].rolling(window=p).mean().round(2)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI (Relative Strength Index) using Wilder's smoothing.
    Column added: rsi_14
    """
    df = df.copy()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{period}"] = (100 - (100 / (1 + rs))).round(2)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────────────────────────

def add_macd(df: pd.DataFrame,
             fast: int = 12,
             slow: int = 26,
             signal: int = 9) -> pd.DataFrame:
    """
    MACD line, Signal line, and Histogram.
    Columns added: macd_line, macd_signal, macd_hist
    """
    df = df.copy()
    ema_fast   = df["close"].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df["close"].ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    macd_sig   = macd_line.ewm(span=signal,   adjust=False).mean()

    df["macd_line"]   = macd_line.round(4)
    df["macd_signal"] = macd_sig.round(4)
    df["macd_hist"]   = (macd_line - macd_sig).round(4)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BOLLINGER BANDS
# ─────────────────────────────────────────────────────────────────────────────

def add_bollinger(df: pd.DataFrame,
                  period: int = 20,
                  std_dev: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands + %B (where price sits within the band).
    Columns added: bb_upper, bb_mid, bb_lower, bb_pct_b, bb_width
    bb_pct_b = 0 → at lower band, 0.5 → midline, 1 → upper band
    """
    df = df.copy()
    rolling        = df["close"].rolling(window=period)
    bb_mid         = rolling.mean()
    bb_std         = rolling.std()

    df["bb_upper"]   = (bb_mid + std_dev * bb_std).round(2)
    df["bb_mid"]     = bb_mid.round(2)
    df["bb_lower"]   = (bb_mid - std_dev * bb_std).round(2)

    band_range       = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"]   = ((df["close"] - df["bb_lower"]) / band_range.replace(0, np.nan)).round(4)
    df["bb_width"]   = (band_range / bb_mid.replace(0, np.nan)).round(4)   # normalised width
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ATR (Average True Range) — used for stop-loss sizing
# ─────────────────────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    ATR using Wilder's smoothing.
    Columns added: atr_14, atr_pct (ATR as % of close — volatility normaliser)
    """
    df = df.copy()
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    df[f"atr_{period}"] = tr.ewm(alpha=1/period, adjust=False).mean().round(2)
    df["atr_pct"]       = (df[f"atr_{period}"] / df["close"].replace(0, np.nan) * 100).round(2)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def add_volume_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Volume ratio vs moving average + OBV.
    Columns added: vol_sma_20, vol_ratio, obv
    vol_ratio > 1.5 = high volume confirmation
    vol_ratio < 0.7 = low volume = weak signal
    """
    df = df.copy()
    df["vol_sma_20"] = df["volume"].rolling(window=period).mean().round(0)
    df["vol_ratio"]  = (df["volume"] / df["vol_sma_20"].replace(0, np.nan)).round(2)

    # OBV (On-Balance Volume)
    direction    = np.sign(df["close"].diff()).fillna(0)
    df["obv"]    = (direction * df["volume"]).cumsum().astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: add all indicators at once
# ─────────────────────────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all indicator functions in the correct order.
    Minimum required columns: date, open, high, low, close, volume
    Minimum rows needed: 200 (for SMA-200 to be valid)
    """
    if df.empty or len(df) < 30:
        return df   # not enough data — return as-is, scorer will handle NaN

    df = df.sort_values("date").reset_index(drop=True)
    df = add_sma(df,             periods=[20, 50, 200])
    df = add_ema(df,             periods=[9, 20, 50])
    df = add_rsi(df,             period=14)
    df = add_macd(df)
    df = add_bollinger(df,       period=20)
    df = add_atr(df,             period=14)
    df = add_volume_indicators(df, period=20)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STOP-LOSS CALCULATOR (ATR-based)
# Used by score_stocks.py to generate trade alerts with pre-calculated stops
# ─────────────────────────────────────────────────────────────────────────────

def calculate_stop_loss(close: float, atr: float, multiplier: float = 2.0) -> float:
    """
    ATR-based stop-loss below current price.
    Default: 2× ATR below close.
    Conservative — gives trade room to breathe without excessive risk.
    """
    if pd.isna(atr) or atr <= 0:
        return round(close * 0.95, 2)   # fallback: 5% below close
    return round(close - (multiplier * atr), 2)


def calculate_target(close: float, atr: float, multiplier: float = 3.0) -> float:
    """
    ATR-based target above current price.
    Default: 3× ATR above close → gives 1.5 R:R ratio vs 2× ATR stop.
    """
    if pd.isna(atr) or atr <= 0:
        return round(close * 1.08, 2)   # fallback: 8% above close
    return round(close + (multiplier * atr), 2)


def calculate_position_size(capital: float,
                             risk_pct: float,
                             entry: float,
                             stop_loss: float) -> int:
    """
    How many shares to buy given capital, risk%, entry, and stop.
    Formula: qty = (capital × risk%) / (entry - stop_loss)
    Returns 0 if stop >= entry (invalid setup).

    DOWNSIDE FIRST: This caps your loss to risk_pct of capital.
    On a ₹50,000 account at 1.5% risk, max loss per trade = ₹750.
    """
    risk_amount = capital * (risk_pct / 100)
    per_share_risk = entry - stop_loss

    if per_share_risk <= 0:
        return 0

    qty = int(risk_amount / per_share_risk)
    return max(qty, 1)   # minimum 1 share


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sqlite3
    from config import DB_PATH

    conn  = sqlite3.connect(DB_PATH)
    query = """
        SELECT date, open, high, low, close, volume
        FROM daily_prices
        WHERE symbol = 'RELIANCE'
        ORDER BY date DESC
        LIMIT 250
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("❌ No data found. Run fetch_data.py first.")
    else:
        df = add_all_indicators(df)
        latest = df.iloc[-1]
        print(f"\n✅ Indicators for RELIANCE (latest row: {latest['date']})\n")
        cols = [
            "close", "sma_20", "sma_50", "sma_200",
            "ema_20", "rsi_14", "macd_line", "macd_signal",
            "macd_hist", "bb_pct_b", "atr_14", "atr_pct",
            "vol_ratio"
        ]
        for c in cols:
            val = latest.get(c, "N/A")
            print(f"  {c:<15}: {val}")

        sl  = calculate_stop_loss(latest["close"], latest["atr_14"])
        tgt = calculate_target(latest["close"], latest["atr_14"])
        qty = calculate_position_size(50000, 1.5, latest["close"], sl)
        print(f"\n  Stop-loss  : ₹{sl}")
        print(f"  Target     : ₹{tgt}")
        print(f"  Qty (₹50k) : {qty} shares")
        print(f"  Max loss   : ₹{round((latest['close'] - sl) * qty, 2)}")
