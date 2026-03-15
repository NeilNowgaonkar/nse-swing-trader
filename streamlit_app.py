# streamlit_app.py
# NSE Swing Trader — Phase 4 Dashboard
# Run locally: streamlit run streamlit_app.py
# Design: Dark terminal aesthetic. Data-dense. No fluff.
#
# Pages:
#   📊 Dashboard   — morning snapshot, regime, capital, open trades
#   🔔 Signals     — today's BUY/WATCH signals with full factor breakdown
#   📂 Portfolio   — open + closed trades, equity curve
#   🛡️  Risk        — daily loss, drawdown meter, position size calculator
#   📈 Charts      — price + indicators for any of the 28 stocks
#   🗃️  Trade Log   — full sortable history

import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st
import pytz
import os
import sys

# ── path so imports work when run from any directory ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ALL_STOCKS, BUCKET_LABELS,
    STARTING_CAPITAL, RISK_PER_TRADE_PCT,
    MAX_OPEN_TRADES, DAILY_LOSS_LIMIT_PCT,
    DRAWDOWN_PAUSE_PCT, DRAWDOWN_PAUSE_DAYS,
    TRADE_SIGNAL_THRESHOLD, DB_PATH, TIMEZONE,
    VIX_HIGH_THRESHOLD, VIX_EXTREME_THRESHOLD,
)
from create_db import get_connection
from regime_engine import get_latest_regime, get_regime_rules
from indicators import (
    add_all_indicators,
    calculate_stop_loss,
    calculate_target,
    calculate_position_size,
)

IST = pytz.timezone(TIMEZONE)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Swing Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — dark terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0A0E1A;
    color: #E2E8F0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D1220;
    border-right: 1px solid #1E2D45;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #94A3B8;
    padding: 6px 0;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: #00D4AA;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1E2D45;
    border-radius: 10px;
    padding: 16px 20px;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover {
    border-color: #00D4AA40;
}
div[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    color: #64748B !important;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #E2E8F0;
}

/* ── Tables ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #1E2D45;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Section headers ── */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00D4AA;
    margin: 24px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1E2D45;
}

/* ── Signal cards ── */
.signal-card {
    background: #111827;
    border: 1px solid #1E2D45;
    border-left: 3px solid #00D4AA;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.8;
}
.signal-card.watch {
    border-left-color: #F59E0B;
}
.signal-card.bear {
    border-left-color: #EF4444;
}

/* ── Regime badge ── */
.regime-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 4px 14px;
    border-radius: 20px;
}
.regime-BULL     { background:#00D4AA20; color:#00D4AA; border:1px solid #00D4AA40; }
.regime-NEUTRAL  { background:#3B82F620; color:#60A5FA; border:1px solid #3B82F640; }
.regime-SIDEWAYS { background:#F59E0B20; color:#FBBF24; border:1px solid #F59E0B40; }
.regime-HIGH_VIX { background:#F9731620; color:#FB923C; border:1px solid #F9731640; }
.regime-BEAR     { background:#EF444420; color:#F87171; border:1px solid #EF444440; }

/* ── Progress bar ── */
.risk-bar-wrap { margin: 6px 0 16px 0; }
.risk-bar-bg {
    background: #1E2D45;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.risk-bar-fill {
    height: 8px;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* ── Score circle ── */
.score-circle {
    width: 64px; height: 64px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px; font-weight: 700;
    flex-shrink: 0;
}

/* ── Footer ── */
.footer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #334155;
    text-align: center;
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #1E2D45;
}

/* ── Divider ── */
hr { border-color: #1E2D45; }

/* ── Input/select ── */
div[data-baseweb="select"] {
    background: #111827 !important;
    border-color: #1E2D45 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.05em;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E2D45; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2D4060; }

/* ── Plotly chart bg ── */
.js-plotly-plot { border-radius: 8px; }

/* ── Positive / Negative ── */
.pos { color: #00D4AA; }
.neg { color: #F87171; }
.neu { color: #94A3B8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DATA HELPERS  (cached so they don't re-query on every widget interaction)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_open_trades() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT id, symbol, bucket, entry_price, entry_date, qty,
               stop_loss, target, score_at_entry, regime_at_entry, notes
        FROM trades WHERE status='OPEN'
        ORDER BY entry_date DESC
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_closed_trades() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT id, symbol, bucket, entry_price, exit_price, entry_date,
               exit_date, qty, pnl, pnl_pct, exit_reason,
               score_at_entry, regime_at_entry, notes
        FROM trades WHERE status='CLOSED'
        ORDER BY exit_date DESC
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_today_scores() -> pd.DataFrame:
    conn = get_connection()
    # Get latest available scores (today or most recent run)
    latest_date = conn.execute(
        "SELECT MAX(date) FROM daily_scores"
    ).fetchone()[0]
    if not latest_date:
        conn.close()
        return pd.DataFrame()
    df = pd.read_sql_query("""
        SELECT symbol, total_score, momentum_score, trend_score,
               volume_score, rsi_score, macd_score, bb_score,
               signal, regime, gate_passed, date
        FROM daily_scores
        WHERE date = ?
        ORDER BY total_score DESC
    """, conn, params=(latest_date,))
    conn.close()
    return df


@st.cache_data(ttl=600)
def load_regime_history(days: int = 60) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT date, regime, nifty_close, nifty_sma50, nifty_sma200,
               india_vix, notes
        FROM regime_log
        ORDER BY date DESC
        LIMIT ?
    """, conn, params=(days,))
    conn.close()
    return df.sort_values("date")


@st.cache_data(ttl=300)
def load_price_history(symbol: str, limit: int = 250) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT date, open, high, low, close, volume
        FROM daily_prices
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT ?
    """, conn, params=(symbol, limit))
    conn.close()
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300)
def get_latest_prices_all() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT symbol, date, close, volume
        FROM daily_prices
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) FROM daily_prices GROUP BY symbol
        )
        ORDER BY symbol
    """, conn)
    conn.close()
    return df


def get_capital_stats() -> dict:
    conn = get_connection()
    closed = conn.execute(
        "SELECT COALESCE(SUM(pnl),0), COUNT(*) FROM trades WHERE status='CLOSED'"
    ).fetchone()
    winners = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl > 0"
    ).fetchone()[0]
    today_ist_str = datetime.now(IST).strftime("%Y-%m-%d")
    today_pnl = conn.execute(
        "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE status='CLOSED' AND exit_date=?",
        (today_ist_str,)
    ).fetchone()[0]
    open_count = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status='OPEN'"
    ).fetchone()[0]
    conn.close()

    realised = closed[0]
    total_trades = closed[1]
    capital = STARTING_CAPITAL + realised
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0

    # Drawdown
    conn2 = get_connection()
    all_pnl = conn2.execute(
        "SELECT pnl FROM trades WHERE status='CLOSED' ORDER BY exit_date"
    ).fetchall()
    conn2.close()
    equity = STARTING_CAPITAL
    peak   = STARTING_CAPITAL
    for (p,) in all_pnl:
        equity += (p or 0)
        peak    = max(peak, equity)
    drawdown_pct = ((peak - capital) / peak * 100) if peak > 0 else 0.0

    return {
        "capital":       round(capital, 2),
        "realised_pnl":  round(realised, 2),
        "today_pnl":     round(today_pnl, 2),
        "total_trades":  total_trades,
        "win_rate":      round(win_rate, 1),
        "drawdown_pct":  round(drawdown_pct, 2),
        "peak_capital":  round(peak, 2),
        "open_count":    open_count,
        "free_slots":    MAX_OPEN_TRADES - open_count,
    }


def today_ist_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def fmt_inr(val: float) -> str:
    return f"₹{val:,.0f}"


def fmt_pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def regime_color(regime: str) -> str:
    return {
        "BULL":     "#00D4AA",
        "NEUTRAL":  "#60A5FA",
        "SIDEWAYS": "#FBBF24",
        "HIGH_VIX": "#FB923C",
        "BEAR":     "#F87171",
    }.get(regime, "#94A3B8")


def score_color(score: float) -> str:
    if score >= 75: return "#00D4AA"
    if score >= 68: return "#FBBF24"
    if score >= 55: return "#FB923C"
    return "#F87171"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 24px 0;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:18px;
                    font-weight:700;color:#00D4AA;letter-spacing:0.05em;'>
            NSE SWING
        </div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:10px;
                    color:#334155;letter-spacing:0.12em;margin-top:2px;'>
            TRADER  v2.0  ·  NEIL
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "📊  Dashboard",
            "🔔  Signals",
            "📂  Portfolio",
            "🛡️   Risk",
            "📈  Charts",
            "🗃️   Trade Log",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Quick regime badge in sidebar
    conn_sb = get_connection()
    regime_now = get_latest_regime(conn_sb)
    conn_sb.close()
    rc = regime_color(regime_now)
    st.markdown(f"""
    <div style='margin:8px 0;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:10px;
                    color:#334155;letter-spacing:0.12em;margin-bottom:6px;'>
            MARKET REGIME
        </div>
        <span class='regime-badge regime-{regime_now}'>{regime_now}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"""
    <div class='footer'>
        STABILITY OVER PROFITABILITY<br/>
        Capital ₹{STARTING_CAPITAL:,} · Risk {RISK_PER_TRADE_PCT}%/trade
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

if page == "📊  Dashboard":

    stats = get_capital_stats()

    # ── Header ───────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"""
        <div style='padding:8px 0 20px 0;'>
            <div style='font-family:"JetBrains Mono",monospace;font-size:22px;
                        font-weight:700;color:#E2E8F0;'>
                Morning Dashboard
            </div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:12px;
                        color:#475569;margin-top:4px;'>
                {datetime.now(IST).strftime("%A, %d %B %Y  ·  %I:%M %p IST")}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        rc = regime_color(regime_now)
        rules = get_regime_rules(regime_now)
        st.markdown(f"""
        <div style='text-align:right;padding-top:12px;'>
            <span class='regime-badge regime-{regime_now}'>{regime_now}</span>
            <div style='font-family:"JetBrains Mono",monospace;font-size:10px;
                        color:#475569;margin-top:6px;'>
                max {rules["max_trades"]} trades · score ≥ {rules["min_score"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key Metrics Row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    pnl_delta = f"{fmt_pct(stats['realised_pnl']/STARTING_CAPITAL*100)}"
    today_sign = "+" if stats['today_pnl'] >= 0 else ""

    with c1:
        st.metric("Portfolio Value", fmt_inr(stats['capital']), pnl_delta)
    with c2:
        st.metric("Today's P&L", f"₹{stats['today_pnl']:+,.0f}",
                  f"{today_sign}{stats['today_pnl']/stats['capital']*100:.2f}%")
    with c3:
        st.metric("Open Trades", f"{stats['open_count']}/{MAX_OPEN_TRADES}",
                  f"{stats['free_slots']} slots free")
    with c4:
        st.metric("Win Rate", f"{stats['win_rate']:.0f}%",
                  f"{stats['total_trades']} trades")
    with c5:
        dd_delta = f"-{stats['drawdown_pct']:.1f}% from peak"
        st.metric("Drawdown", f"{stats['drawdown_pct']:.1f}%", dd_delta,
                  delta_color="inverse")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Regime Row + Signals Summary ─────────────────────────────────────────
    col_r, col_s = st.columns([1, 1])

    with col_r:
        st.markdown("<div class='section-header'>Regime History (60d)</div>",
                    unsafe_allow_html=True)
        rh = load_regime_history(60)
        if not rh.empty:
            regime_map = {"BULL":5,"NEUTRAL":4,"SIDEWAYS":3,"HIGH_VIX":2,"BEAR":1}
            rh["regime_num"] = rh["regime"].map(regime_map)
            rh_colors = rh["regime"].map({
                "BULL":"#00D4AA","NEUTRAL":"#60A5FA","SIDEWAYS":"#FBBF24",
                "HIGH_VIX":"#FB923C","BEAR":"#F87171"
            })

            fig_r = go.Figure()
            for r_name, r_color in [
                ("BULL","#00D4AA"),("NEUTRAL","#60A5FA"),("SIDEWAYS","#FBBF24"),
                ("HIGH_VIX","#FB923C"),("BEAR","#F87171")
            ]:
                mask = rh["regime"] == r_name
                if mask.any():
                    fig_r.add_trace(go.Scatter(
                        x=rh.loc[mask, "date"],
                        y=rh.loc[mask, "regime_num"],
                        mode="markers",
                        marker=dict(color=r_color, size=8, symbol="square"),
                        name=r_name,
                        hovertemplate=f"<b>{r_name}</b><br>%{{x}}<extra></extra>",
                    ))

            fig_r.update_layout(
                height=200,
                margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(
                    orientation="h", y=-0.25,
                    font=dict(family="JetBrains Mono", size=10, color="#64748B")
                ),
                xaxis=dict(
                    showgrid=False, zeroline=False,
                    tickfont=dict(family="JetBrains Mono", size=10, color="#475569"),
                ),
                yaxis=dict(
                    showticklabels=False, showgrid=False,
                    zeroline=False, range=[0.5, 5.5],
                ),
            )
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

            latest_regime_row = rh.iloc[-1]
            st.markdown(f"""
            <div style='font-family:"JetBrains Mono",monospace;font-size:12px;color:#64748B;'>
                Nifty50: <span style='color:#E2E8F0;'>{latest_regime_row.get('nifty_close','—')}</span> &nbsp;·&nbsp;
                VIX: <span style='color:#E2E8F0;'>{latest_regime_row.get('india_vix','—')}</span> &nbsp;·&nbsp;
                SMA200: <span style='color:#E2E8F0;'>{latest_regime_row.get('nifty_sma200','—')}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No regime data. Run `py regime_engine.py` first.")

    with col_s:
        st.markdown("<div class='section-header'>Today's Signal Summary</div>",
                    unsafe_allow_html=True)
        scores_df = load_today_scores()
        if not scores_df.empty:
            buy_df   = scores_df[scores_df["signal"] == "BUY"]
            watch_df = scores_df[scores_df["signal"] == "WATCH"]

            s1, s2, s3 = st.columns(3)
            s1.metric("BUY Signals",   len(buy_df[buy_df["gate_passed"]==1]))
            s2.metric("WATCH Signals", len(watch_df))
            s3.metric("Avg Score",     f"{scores_df['total_score'].mean():.1f}")

            # Score distribution mini chart
            fig_hist = go.Figure(go.Histogram(
                x=scores_df["total_score"],
                nbinsx=20,
                marker_color="#1E3A5F",
                marker_line_color="#00D4AA",
                marker_line_width=0.5,
            ))
            fig_hist.add_vline(
                x=TRADE_SIGNAL_THRESHOLD, line_dash="dash",
                line_color="#00D4AA", line_width=1.5,
                annotation_text=f"Threshold {TRADE_SIGNAL_THRESHOLD}",
                annotation_font=dict(family="JetBrains Mono", size=10, color="#00D4AA"),
            )
            fig_hist.update_layout(
                height=160,
                margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(family="JetBrains Mono",size=10,color="#475569"),
                    title=dict(text="Score", font=dict(size=10, color="#475569")),
                ),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_hist, use_container_width=True,
                            config={"displayModeBar": False})

            st.markdown(f"""
            <div style='font-family:"JetBrains Mono",monospace;font-size:10px;color:#475569;'>
                Scores as of: {scores_df['date'].iloc[0] if not scores_df.empty else '—'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No scores yet. Run `py score_stocks.py` first.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Open Trades Table ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Open Positions</div>",
                unsafe_allow_html=True)
    open_df = load_open_trades()

    if open_df.empty:
        st.markdown("""
        <div style='font-family:"JetBrains Mono",monospace;font-size:13px;
                    color:#334155;padding:20px 0;text-align:center;'>
            No open trades. Use /bought in Telegram to log a trade.
        </div>
        """, unsafe_allow_html=True)
    else:
        prices_df = get_latest_prices_all()
        prices_map = dict(zip(prices_df["symbol"], prices_df["close"]))

        display_rows = []
        for _, row in open_df.iterrows():
            sym   = row["symbol"]
            cmp   = prices_map.get(sym, row["entry_price"])
            entry = row["entry_price"]
            unr   = round((cmp - entry) * row["qty"], 2)
            unr_p = round((cmp - entry) / entry * 100, 2) if entry > 0 else 0

            display_rows.append({
                "Symbol":    sym,
                "Bucket":    row["bucket"],
                "Entry ₹":   f"{entry:.2f}",
                "CMP ₹":     f"{cmp:.2f}",
                "SL ₹":      f"{row['stop_loss']:.2f}",
                "Target ₹":  f"{row['target']:.2f}",
                "Qty":        row["qty"],
                "Unr. P&L":  f"₹{unr:+,.0f} ({unr_p:+.1f}%)",
                "Entry Date": row["entry_date"],
                "Score":      f"{row['score_at_entry']:.0f}" if row["score_at_entry"] else "—",
            })

        disp = pd.DataFrame(display_rows)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Equity mini-curve ─────────────────────────────────────────────────────
    closed_df = load_closed_trades()
    if not closed_df.empty and len(closed_df) >= 2:
        st.markdown("<div class='section-header'>Equity Curve</div>",
                    unsafe_allow_html=True)

        eq = closed_df.sort_values("exit_date").copy()
        eq["cumulative_pnl"] = eq["pnl"].cumsum()
        eq["equity"]         = STARTING_CAPITAL + eq["cumulative_pnl"]

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq["exit_date"], y=eq["equity"],
            mode="lines",
            line=dict(color="#00D4AA", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.05)",
            hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
        ))
        fig_eq.add_hline(
            y=STARTING_CAPITAL, line_dash="dot",
            line_color="#334155", line_width=1,
            annotation_text="Start ₹50,000",
            annotation_font=dict(family="JetBrains Mono", size=10, color="#475569"),
        )
        fig_eq.update_layout(
            height=200,
            margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(
                showgrid=False, zeroline=False,
                tickfont=dict(family="JetBrains Mono",size=10,color="#475569"),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#1E2D45",
                tickfont=dict(family="JetBrains Mono",size=10,color="#475569"),
                tickprefix="₹",
                tickformat=",.0f",
            ),
        )
        st.plotly_chart(fig_eq, use_container_width=True,
                        config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔔  Signals":
    st.markdown("""
    <div style='padding:8px 0 20px 0;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:22px;
                    font-weight:700;color:#E2E8F0;'>Signal Scanner</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:12px;
                    color:#475569;margin-top:4px;'>
            6-factor model · Gates: SMA-200, Volume, ATR
        </div>
    </div>
    """, unsafe_allow_html=True)

    scores_df = load_today_scores()

    if scores_df.empty:
        st.warning("No scores found. Run `py score_stocks.py` to generate signals.")
        st.stop()

    rules       = get_regime_rules(regime_now)
    allowed     = rules["allowed_buckets"]
    min_score_r = rules["min_score"]

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        sig_filter = st.selectbox("Signal", ["ALL", "BUY", "WATCH", "IGNORE"],
                                  index=0)
    with f2:
        bucket_filter = st.selectbox(
            "Bucket", ["ALL"] + ["large_cap","momentum","midcap_alpha","defensive"]
        )
    with f3:
        min_score_f = st.slider("Min Score", 0, 100, 50)

    # Apply filters
    df = scores_df.copy()
    df["bucket"] = df["symbol"].map(BUCKET_LABELS)
    if sig_filter != "ALL":
        df = df[df["signal"] == sig_filter]
    if bucket_filter != "ALL":
        df = df[df["bucket"] == bucket_filter]
    df = df[df["total_score"] >= min_score_f]

    st.markdown(f"""
    <div style='font-family:"JetBrains Mono",monospace;font-size:11px;
                color:#475569;margin:-10px 0 16px 0;'>
        Showing {len(df)} stocks · Regime rules: min score {min_score_r},
        allowed: {"all buckets" if allowed=="ALL" else ", ".join(allowed) if allowed else "NONE"}
        · Scores from: {scores_df['date'].iloc[0]}
    </div>
    """, unsafe_allow_html=True)

    # ── Score heatmap (all 28 stocks) ─────────────────────────────────────────
    st.markdown("<div class='section-header'>Score Heatmap — All 28 Stocks</div>",
                unsafe_allow_html=True)

    heat_data = scores_df.copy()
    heat_data["bucket"] = heat_data["symbol"].map(BUCKET_LABELS)
    heat_data = heat_data.sort_values(["bucket","total_score"], ascending=[True,False])

    factors = ["momentum_score","trend_score","rsi_score",
               "macd_score","volume_score","bb_score"]
    factor_labels = ["Momentum","Trend","RSI","MACD","Volume","BB"]

    z_vals  = heat_data[factors].values.tolist()
    y_syms  = heat_data["symbol"].tolist()
    x_labs  = factor_labels

    fig_heat = go.Figure(go.Heatmap(
        z=z_vals,
        x=x_labs,
        y=y_syms,
        colorscale=[
            [0.0,  "#1A1F2E"],
            [0.3,  "#1E3A5F"],
            [0.6,  "#1B6B4A"],
            [1.0,  "#00D4AA"],
        ],
        zmin=0, zmax=20,
        hovertemplate="<b>%{y} · %{x}</b><br>Score: %{z:.1f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            thickness=10, len=0.8,
            tickfont=dict(family="JetBrains Mono",size=9,color="#475569"),
            title=dict(text="pts", font=dict(family="JetBrains Mono",size=10)),
        ),
    ))
    fig_heat.update_layout(
        height=420,
        margin=dict(l=0,r=40,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(family="JetBrains Mono",size=11,color="#94A3B8"),
            side="top",
        ),
        yaxis=dict(
            tickfont=dict(family="JetBrains Mono",size=10,color="#94A3B8"),
            autorange="reversed",
        ),
    )
    st.plotly_chart(fig_heat, use_container_width=True,
                    config={"displayModeBar": False})

    # ── Signal Cards ──────────────────────────────────────────────────────────
    if sig_filter in ["ALL","BUY","WATCH"]:
        st.markdown("<div class='section-header'>Signal Detail Cards</div>",
                    unsafe_allow_html=True)

        prices_df = get_latest_prices_all()
        prices_map = dict(zip(prices_df["symbol"], prices_df["close"]))

        buy_stocks   = df[df["signal"]=="BUY"].head(8)
        watch_stocks = df[df["signal"]=="WATCH"].head(5)
        show_stocks  = pd.concat([buy_stocks, watch_stocks])

        if show_stocks.empty:
            st.info("No BUY/WATCH signals with current filters.")
        else:
            for _, row in show_stocks.iterrows():
                sym    = row["symbol"]
                score  = row["total_score"]
                signal = row["signal"]
                bucket = row.get("bucket", BUCKET_LABELS.get(sym, "—"))
                gates  = "✅ Gates Pass" if row["gate_passed"] else "❌ Gates Fail"
                close  = prices_map.get(sym, 0)

                sc = score_color(score)
                card_class = "signal-card" if signal == "BUY" else "signal-card watch"

                # Regime bucket warning
                regime_warn = ""
                if allowed != "ALL" and bucket not in (allowed or []):
                    regime_warn = f"<br/><span style='color:#F59E0B;font-size:11px;'>⚠️ Regime ({regime_now}) limits this bucket</span>"

                # Factor bars
                factor_html = ""
                for fname, flabel in zip(
                    ["momentum_score","trend_score","rsi_score",
                     "macd_score","volume_score","bb_score"],
                    ["Momentum","Trend","RSI","MACD","Volume","BB"]
                ):
                    pts   = row.get(fname, 0)
                    maxw  = {"Momentum":20,"Trend":20,"RSI":15,
                             "MACD":15,"Volume":15,"BB":15}[flabel]
                    pct   = min(pts/maxw*100, 100)
                    fc    = "#00D4AA" if pct >= 70 else "#FBBF24" if pct >= 40 else "#F87171"
                    factor_html += f"""
                    <div style='margin:3px 0;display:flex;align-items:center;gap:10px;'>
                        <span style='width:70px;font-size:10px;color:#475569;'>{flabel}</span>
                        <div style='flex:1;background:#1E2D45;border-radius:3px;height:6px;'>
                            <div style='width:{pct}%;background:{fc};height:6px;border-radius:3px;'></div>
                        </div>
                        <span style='width:30px;font-size:10px;color:#94A3B8;text-align:right;'>{pts:.0f}</span>
                    </div>"""

                atr  = close * 0.02
                sl   = calculate_stop_loss(close, atr)
                tgt  = calculate_target(close, atr)
                qty  = calculate_position_size(STARTING_CAPITAL, RISK_PER_TRADE_PCT, close, sl)
                risk = round((close - sl) * qty, 0)

                sl_pct  = f"{(sl-close)/close*100:.1f}%" if close > 0 else "—"
                tgt_pct = f"+{(tgt-close)/close*100:.1f}%" if close > 0 else "—"

                st.markdown(f"""
                <div class='{card_class}'>
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                        <div>
                            <span style='font-size:16px;font-weight:700;color:#E2E8F0;'>{sym}</span>
                            <span style='font-size:10px;color:#475569;margin-left:10px;'>{bucket}</span>
                            <span style='font-size:10px;color:#475569;margin-left:8px;'>{gates}</span>
                            {regime_warn}
                        </div>
                        <div style='text-align:right;'>
                            <div style='font-size:22px;font-weight:700;color:{sc};'>{score:.0f}</div>
                            <div style='font-size:10px;color:#475569;'>/ 100</div>
                        </div>
                    </div>
                    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0 8px 0;'>
                        <div>
                            <div style='font-size:10px;color:#475569;'>CMP</div>
                            <div style='font-size:14px;color:#E2E8F0;'>₹{close:.2f}</div>
                        </div>
                        <div>
                            <div style='font-size:10px;color:#475569;'>Stop-Loss</div>
                            <div style='font-size:14px;color:#F87171;'>₹{sl:.2f} <span style='font-size:10px;'>({sl_pct})</span></div>
                        </div>
                        <div>
                            <div style='font-size:10px;color:#475569;'>Target</div>
                            <div style='font-size:14px;color:#00D4AA;'>₹{tgt:.2f} <span style='font-size:10px;'>({tgt_pct})</span></div>
                        </div>
                        <div>
                            <div style='font-size:10px;color:#475569;'>Qty · Max Risk</div>
                            <div style='font-size:14px;color:#E2E8F0;'>{qty} <span style='font-size:11px;color:#475569;'>· ₹{risk:,.0f}</span></div>
                        </div>
                    </div>
                    <div style='margin-top:10px;'>{factor_html}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Full Score Table ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Full Score Table</div>",
                unsafe_allow_html=True)
    disp = df[["symbol","bucket","total_score","signal","gate_passed",
               "momentum_score","trend_score","rsi_score",
               "macd_score","volume_score","bb_score"]].copy()
    disp.columns = ["Symbol","Bucket","Total","Signal","Gates",
                    "Momentum","Trend","RSI","MACD","Volume","BB"]
    st.dataframe(disp.reset_index(drop=True), use_container_width=True,
                 hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📂  Portfolio":
    st.markdown("""
    <div style='padding:8px 0 20px 0;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:22px;
                    font-weight:700;color:#E2E8F0;'>Portfolio</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:12px;
                    color:#475569;margin-top:4px;'>
            Open positions · Closed trades · Equity curve
        </div>
    </div>
    """, unsafe_allow_html=True)

    stats     = get_capital_stats()
    open_df   = load_open_trades()
    closed_df = load_closed_trades()

    # ── Capital metrics ───────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Capital",       fmt_inr(stats["capital"]),
              fmt_pct(stats["realised_pnl"]/STARTING_CAPITAL*100))
    c2.metric("Realised P&L",  f"₹{stats['realised_pnl']:+,.0f}")
    c3.metric("Win Rate",      f"{stats['win_rate']:.0f}%",
              f"{stats['total_trades']} total")
    c4.metric("Drawdown",      f"{stats['drawdown_pct']:.1f}%",
              f"Peak ₹{stats['peak_capital']:,.0f}", delta_color="inverse")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Equity Curve ──────────────────────────────────────────────────────────
    if not closed_df.empty:
        st.markdown("<div class='section-header'>Equity Curve</div>",
                    unsafe_allow_html=True)
        eq = closed_df.sort_values("exit_date").copy()
        eq["equity"]    = STARTING_CAPITAL + eq["pnl"].cumsum()
        eq["drawdown"]  = eq["equity"].cummax() - eq["equity"]
        eq["dd_pct"]    = eq["drawdown"] / eq["equity"].cummax() * 100

        fig_eq = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.04,
        )
        fig_eq.add_trace(go.Scatter(
            x=eq["exit_date"], y=eq["equity"],
            mode="lines+markers",
            line=dict(color="#00D4AA", width=2.5),
            marker=dict(size=5, color="#00D4AA"),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.04)",
            name="Equity",
            hovertemplate="₹%{y:,.0f}<extra>%{x}</extra>",
        ), row=1, col=1)
        fig_eq.add_hline(
            y=STARTING_CAPITAL, line_dash="dot",
            line_color="#334155", line_width=1, row=1, col=1,
        )
        fig_eq.add_trace(go.Bar(
            x=eq["exit_date"], y=eq["dd_pct"],
            marker_color="#F87171", opacity=0.6,
            name="Drawdown %",
            hovertemplate="%{y:.1f}%<extra>DD</extra>",
        ), row=2, col=1)

        axis_style = dict(
            showgrid=True, gridcolor="#1E2D45", zeroline=False,
            tickfont=dict(family="JetBrains Mono",size=10,color="#475569"),
        )
        fig_eq.update_layout(
            height=380,
            margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig_eq.update_xaxes(**{k:v for k,v in axis_style.items() if k!="tickprefix"})
        fig_eq.update_yaxes(
            **axis_style, tickprefix="₹", tickformat=",.0f", row=1, col=1
        )
        fig_eq.update_yaxes(
            **axis_style, ticksuffix="%", row=2, col=1
        )
        st.plotly_chart(fig_eq, use_container_width=True,
                        config={"displayModeBar": False})

    # ── Open positions ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Open Positions</div>",
                unsafe_allow_html=True)
    if open_df.empty:
        st.info("No open trades.")
    else:
        prices_df  = get_latest_prices_all()
        prices_map = dict(zip(prices_df["symbol"], prices_df["close"]))
        rows = []
        for _, r in open_df.iterrows():
            cmp   = prices_map.get(r["symbol"], r["entry_price"])
            unr   = round((cmp - r["entry_price"]) * r["qty"], 2)
            unr_p = round((cmp - r["entry_price"]) / r["entry_price"] * 100, 2)
            rows.append({
                "Symbol":    r["symbol"],
                "Bucket":    r["bucket"],
                "Entry":     f"₹{r['entry_price']:.2f}",
                "CMP":       f"₹{cmp:.2f}",
                "SL":        f"₹{r['stop_loss']:.2f}",
                "Target":    f"₹{r['target']:.2f}",
                "Qty":       r["qty"],
                "Unr P&L":   f"₹{unr:+,.0f}",
                "Unr %":     f"{unr_p:+.1f}%",
                "Since":     r["entry_date"],
                "Score":     f"{r['score_at_entry']:.0f}" if r["score_at_entry"] else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── P&L by bucket ─────────────────────────────────────────────────────────
    if not closed_df.empty:
        st.markdown("<div class='section-header'>P&L by Bucket</div>",
                    unsafe_allow_html=True)

        bkt_stats = closed_df.groupby("bucket").agg(
            total_pnl=("pnl","sum"),
            trades=("pnl","count"),
            wins=("pnl", lambda x: (x > 0).sum()),
        ).reset_index()
        bkt_stats["win_rate"] = bkt_stats["wins"]/bkt_stats["trades"]*100

        fig_bkt = go.Figure()
        colors = {
            "large_cap":    "#00D4AA",
            "momentum":     "#60A5FA",
            "midcap_alpha": "#FBBF24",
            "defensive":    "#A78BFA",
        }
        for _, b in bkt_stats.iterrows():
            col = colors.get(b["bucket"], "#94A3B8")
            fig_bkt.add_trace(go.Bar(
                x=[b["bucket"]], y=[b["total_pnl"]],
                marker_color=col,
                text=[f"₹{b['total_pnl']:+,.0f}"],
                textposition="outside",
                textfont=dict(family="JetBrains Mono",size=11,color=col),
                name=b["bucket"],
                hovertemplate=f"<b>{b['bucket']}</b><br>P&L: ₹{b['total_pnl']:+,.0f}<br>Trades: {b['trades']}<br>Win: {b['win_rate']:.0f}%<extra></extra>",
            ))

        fig_bkt.update_layout(
            height=220,
            margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                tickfont=dict(family="JetBrains Mono",size=11,color="#94A3B8"),
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#1E2D45", zeroline=True,
                zerolinecolor="#334155", tickprefix="₹",
                tickfont=dict(family="JetBrains Mono",size=10,color="#475569"),
            ),
        )
        st.plotly_chart(fig_bkt, use_container_width=True,
                        config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — RISK MONITOR
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🛡️   Risk":
    st.markdown("""
    <div style='padding:8px 0 20px 0;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:22px;
                    font-weight:700;color:#E2E8F0;'>Risk Monitor</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:12px;
                    color:#475569;margin-top:4px;'>
            Daily loss · Drawdown · Position sizing calculator
        </div>
    </div>
    """, unsafe_allow_html=True)

    stats = get_capital_stats()
    capital = stats["capital"]

    # ── Risk Gauges ────────────────────────────────────────────────────────────
    col_d, col_dd = st.columns(2)

    with col_d:
        today_pnl_pct = stats["today_pnl"] / capital * 100
        daily_used    = abs(min(today_pnl_pct, 0))
        daily_pct_of_limit = min(daily_used / DAILY_LOSS_LIMIT_PCT * 100, 100)
        bar_color = "#00D4AA" if daily_pct_of_limit < 50 else \
                    "#FBBF24" if daily_pct_of_limit < 80 else "#F87171"
        limit_hit = today_pnl_pct <= -DAILY_LOSS_LIMIT_PCT

        st.markdown(f"""
        <div style='background:#111827;border:1px solid #1E2D45;border-radius:10px;
                    padding:20px 24px;'>
            <div style='font-family:"JetBrains Mono",monospace;font-size:10px;
                        letter-spacing:0.12em;color:#64748B;text-transform:uppercase;
                        margin-bottom:8px;'>Daily Loss Limit</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:28px;
                        font-weight:700;color:{bar_color};'>{daily_used:.2f}%</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:11px;
                        color:#475569;margin-bottom:14px;'>
                of {DAILY_LOSS_LIMIT_PCT}% limit · ₹{stats['today_pnl']:+,.0f} today
            </div>
            <div style='background:#1E2D45;border-radius:4px;height:10px;'>
                <div style='width:{daily_pct_of_limit:.0f}%;background:{bar_color};
                            height:10px;border-radius:4px;'></div>
            </div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:11px;
                        color:#475569;margin-top:8px;'>
                {"⛔ LIMIT HIT — No new trades today" if limit_hit
                 else f"✅ {DAILY_LOSS_LIMIT_PCT - daily_used:.2f}% remaining"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_dd:
        dd_pct = stats["drawdown_pct"]
        dd_pct_of_limit = min(dd_pct / DRAWDOWN_PAUSE_PCT * 100, 100)
        dd_color = "#00D4AA" if dd_pct_of_limit < 50 else \
                   "#FBBF24" if dd_pct_of_limit < 80 else "#F87171"
        paused = dd_pct >= DRAWDOWN_PAUSE_PCT

        st.markdown(f"""
        <div style='background:#111827;border:1px solid #1E2D45;border-radius:10px;
                    padding:20px 24px;'>
            <div style='font-family:"JetBrains Mono",monospace;font-size:10px;
                        letter-spacing:0.12em;color:#64748B;text-transform:uppercase;
                        margin-bottom:8px;'>Drawdown</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:28px;
                        font-weight:700;color:{dd_color};'>{dd_pct:.2f}%</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:11px;
                        color:#475569;margin-bottom:14px;'>
                of {DRAWDOWN_PAUSE_PCT}% pause threshold ·
                Peak ₹{stats['peak_capital']:,.0f}
            </div>
            <div style='background:#1E2D45;border-radius:4px;height:10px;'>
                <div style='width:{dd_pct_of_limit:.0f}%;background:{dd_color};
                            height:10px;border-radius:4px;'></div>
            </div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:11px;
                        color:#475569;margin-top:8px;'>
                {"🛑 PAUSED — 10-day pause triggered" if paused
                 else f"✅ {DRAWDOWN_PAUSE_PCT - dd_pct:.2f}% remaining before pause"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Position Size Calculator ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>Position Size Calculator</div>",
                unsafe_allow_html=True)

    px1, px2, px3, px4 = st.columns(4)
    with px1:
        calc_capital = st.number_input(
            "Capital (₹)", value=int(capital), step=1000, min_value=1000
        )
    with px2:
        calc_risk = st.number_input(
            "Risk %", value=RISK_PER_TRADE_PCT,
            step=0.1, min_value=0.1, max_value=5.0
        )
    with px3:
        calc_entry = st.number_input("Entry Price (₹)", value=1000.0, step=0.5)
    with px4:
        calc_sl = st.number_input(
            "Stop-Loss (₹)", value=round(1000.0 * 0.96, 2), step=0.5
        )

    if calc_entry > calc_sl > 0:
        qty       = calculate_position_size(calc_capital, calc_risk, calc_entry, calc_sl)
        invested  = round(qty * calc_entry, 2)
        max_loss  = round((calc_entry - calc_sl) * qty, 2)
        sl_pct    = round((calc_entry - calc_sl) / calc_entry * 100, 2)
        tgt_atr   = calculate_target(calc_entry, (calc_entry - calc_sl) / 2)
        rr        = round((tgt_atr - calc_entry) / (calc_entry - calc_sl), 2)

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Shares to Buy",    qty)
        r2.metric("Capital Deployed", fmt_inr(invested))
        r3.metric("Max Loss",         fmt_inr(max_loss),
                  f"{calc_risk}% of capital")
        r4.metric("SL Distance",      f"{sl_pct:.1f}%")
        r5.metric("R:R Ratio",        f"1 : {rr:.1f}")

        st.markdown(f"""
        <div style='font-family:"JetBrains Mono",monospace;font-size:11px;
                    color:#475569;margin-top:8px;background:#111827;
                    border:1px solid #1E2D45;border-radius:8px;padding:12px 16px;'>
            On a ₹{calc_capital:,} account at {calc_risk}% risk:
            Buy {qty} shares at ₹{calc_entry:.2f} ·
            Stop at ₹{calc_sl:.2f} ({sl_pct:.1f}% below) ·
            Max loss ₹{max_loss:,.0f} ·
            Auto-target ₹{tgt_atr:.2f}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Stop-loss must be below entry price.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Risk Per Slot View ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Capital Allocation — All 4 Slots</div>",
                unsafe_allow_html=True)

    risk_per_trade = round(capital * RISK_PER_TRADE_PCT / 100, 0)
    max_risk_total = round(risk_per_trade * MAX_OPEN_TRADES, 0)
    open_count     = stats["open_count"]

    slot_cols = st.columns(MAX_OPEN_TRADES)
    for i, sc in enumerate(slot_cols):
        used  = i < open_count
        color = "#00D4AA" if used else "#1E2D45"
        label = "OPEN" if used else "FREE"
        sc.markdown(f"""
        <div style='background:#111827;border:2px solid {color};border-radius:10px;
                    padding:16px;text-align:center;'>
            <div style='font-family:"JetBrains Mono",monospace;font-size:10px;
                        color:{color};letter-spacing:0.1em;'>{label}</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:20px;
                        font-weight:700;color:#E2E8F0;margin:6px 0;'>
                Slot {i+1}</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:11px;
                        color:#475569;'>Max risk<br>₹{risk_per_trade:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-family:"JetBrains Mono",monospace;font-size:11px;color:#475569;
                margin-top:12px;'>
        Total max risk across all 4 slots: ₹{max_risk_total:,.0f}
        ({round(max_risk_total/capital*100,1)}% of ₹{capital:,.0f} capital)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈  Charts":
    st.markdown("""
    <div style='padding:8px 0 20px 0;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:22px;
                    font-weight:700;color:#E2E8F0;'>Price Charts</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:12px;
                    color:#475569;margin-top:4px;'>
            OHLCV + EMA + RSI + MACD + Volume · All 28 stocks
        </div>
    </div>
    """, unsafe_allow_html=True)

    ch1, ch2, ch3 = st.columns([2, 1, 1])
    with ch1:
        sym_list = sorted(ALL_STOCKS.keys())
        selected = st.selectbox("Select Stock", sym_list)
    with ch2:
        lookback = st.selectbox("Period",
                                ["3 months","6 months","1 year"], index=1)
    with ch3:
        show_signals = st.checkbox("Show Score Band", value=True)

    limit_map = {"3 months": 65, "6 months": 130, "1 year": 252}
    limit     = limit_map[lookback]

    df_raw = load_price_history(selected, limit=limit)

    if df_raw.empty:
        st.warning(f"No price data for {selected}. Run `py fetch_data.py` first.")
    else:
        df_chart = add_all_indicators(df_raw.copy())
        latest   = df_chart.iloc[-1]

        # Quick stat pills
        s1,s2,s3,s4,s5 = st.columns(5)
        s1.metric("Close",   f"₹{latest['close']:.2f}")
        s2.metric("RSI-14",  f"{latest.get('rsi_14',0):.1f}")
        s3.metric("MACD",    f"{latest.get('macd_hist',0):.2f}",
                  "▲ Bullish" if latest.get('macd_hist',0) > 0 else "▼ Bearish")
        s4.metric("ATR%",    f"{latest.get('atr_pct',0):.1f}%")
        s5.metric("Vol/Avg", f"{latest.get('vol_ratio',1):.2f}x")

        # Main chart (3 rows: OHLCV + RSI + MACD)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.03,
        )

        # ── Candlesticks ──────────────────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=df_chart["date"],
            open=df_chart["open"], high=df_chart["high"],
            low=df_chart["low"],   close=df_chart["close"],
            name=selected,
            increasing_line_color="#00D4AA",
            decreasing_line_color="#F87171",
        ), row=1, col=1)

        # ── EMAs ─────────────────────────────────────────────────────────────
        for period, color, width in [
            (20, "#60A5FA", 1.5),
            (50, "#FBBF24", 1.5),
        ]:
            col_name = f"ema_{period}"
            if col_name in df_chart.columns:
                fig.add_trace(go.Scatter(
                    x=df_chart["date"], y=df_chart[col_name],
                    mode="lines",
                    line=dict(color=color, width=width, dash="solid"),
                    name=f"EMA-{period}",
                    hovertemplate=f"EMA{period}: ₹%{{y:.2f}}<extra></extra>",
                ), row=1, col=1)

        if "sma_200" in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart["date"], y=df_chart["sma_200"],
                mode="lines",
                line=dict(color="#A78BFA", width=1.5, dash="dot"),
                name="SMA-200",
                hovertemplate="SMA200: ₹%{y:.2f}<extra></extra>",
            ), row=1, col=1)

        # ── Volume bars ───────────────────────────────────────────────────────
        vol_colors = [
            "rgba(0,212,170,0.27)" if c >= o else "rgba(248,113,113,0.27)"
            for c, o in zip(df_chart["close"], df_chart["open"])
        ]
        fig.add_trace(go.Bar(
            x=df_chart["date"], y=df_chart["volume"],
            marker_color=vol_colors,
            name="Volume",
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
            yaxis="y2",
        ), row=1, col=1)

        # ── RSI ───────────────────────────────────────────────────────────────
        if "rsi_14" in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart["date"], y=df_chart["rsi_14"],
                mode="lines",
                line=dict(color="#60A5FA", width=1.5),
                name="RSI-14",
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,113,113,0.38)",
                         line_width=1, row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,212,170,0.38)",
                         line_width=1, row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="rgba(51,65,85,0.38)",
                         line_width=1, row=2, col=1)
            fig.add_hrect(y0=55, y1=65, fillcolor="rgba(0,212,170,0.03)",
                         line_width=0, row=2, col=1)

        # ── MACD ─────────────────────────────────────────────────────────────
        if "macd_line" in df_chart.columns:
            macd_colors = [
                "#00D4AA" if h >= 0 else "#F87171"
                for h in df_chart["macd_hist"].fillna(0)
            ]
            fig.add_trace(go.Bar(
                x=df_chart["date"], y=df_chart["macd_hist"],
                marker_color=macd_colors, opacity=0.7,
                name="MACD Hist",
                hovertemplate="Hist: %{y:.4f}<extra></extra>",
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df_chart["date"], y=df_chart["macd_line"],
                mode="lines", line=dict(color="#60A5FA", width=1.5),
                name="MACD", hovertemplate="MACD: %{y:.4f}<extra></extra>",
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df_chart["date"], y=df_chart["macd_signal"],
                mode="lines", line=dict(color="#FBBF24", width=1.5, dash="dash"),
                name="Signal", hovertemplate="Signal: %{y:.4f}<extra></extra>",
            ), row=3, col=1)

        axis_common = dict(
            showgrid=True, gridcolor="rgba(30,45,69,0.19)", zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=10, color="#475569"),
        )
        fig.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h", y=1.02, x=0,
                font=dict(family="JetBrains Mono", size=10, color="#64748B"),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )
        fig.update_xaxes(**axis_common)
        fig.update_yaxes(
            row=1, col=1, tickprefix="₹", tickformat=",.0f", **axis_common
        )
        fig.update_yaxes(row=2, col=1, title_text="RSI", range=[0,100], **axis_common)
        fig.update_yaxes(row=3, col=1, title_text="MACD", **axis_common)

        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": True, "displaylogo": False})

        # ── Score for this stock ───────────────────────────────────────────────
        if show_signals:
            conn_c = get_connection()
            score_rows = conn_c.execute("""
                SELECT date, total_score, signal, gate_passed
                FROM daily_scores WHERE symbol=?
                ORDER BY date DESC LIMIT 60
            """, (selected,)).fetchall()
            conn_c.close()

            if score_rows:
                sc_df = pd.DataFrame(score_rows,
                                     columns=["date","score","signal","gates"])
                sc_df = sc_df.sort_values("date")

                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=sc_df["date"], y=sc_df["score"],
                    mode="lines+markers",
                    line=dict(color="#00D4AA", width=1.5),
                    marker=dict(size=4),
                    fill="tozeroy", fillcolor="rgba(0,212,170,0.05)",
                    hovertemplate="%{x}<br>Score: %{y:.0f}<extra></extra>",
                ))
                fig_sc.add_hline(
                    y=TRADE_SIGNAL_THRESHOLD,
                    line_dash="dash", line_color="#FBBF24", line_width=1.5,
                    annotation_text=f"BUY threshold {TRADE_SIGNAL_THRESHOLD}",
                    annotation_font=dict(family="JetBrains Mono",size=10,color="#FBBF24"),
                )
                fig_sc.update_layout(
                    height=160, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    xaxis=dict(showgrid=False, tickfont=dict(
                        family="JetBrains Mono",size=10,color="#475569")),
                    yaxis=dict(showgrid=True, gridcolor="#1E2D45",
                               range=[0,105],
                               tickfont=dict(family="JetBrains Mono",size=10,color="#475569")),
                )
                st.markdown(
                    "<div class='section-header'>Factor Score History (60d)</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig_sc, use_container_width=True,
                                config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — TRADE LOG
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🗃️   Trade Log":
    st.markdown("""
    <div style='padding:8px 0 20px 0;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:22px;
                    font-weight:700;color:#E2E8F0;'>Trade Log</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:12px;
                    color:#475569;margin-top:4px;'>
            Full history · Filter · Export
        </div>
    </div>
    """, unsafe_allow_html=True)

    closed_df = load_closed_trades()

    if closed_df.empty:
        st.info("No closed trades yet. Use /sold in Telegram to close a trade.")
        st.stop()

    # ── Summary Stats ─────────────────────────────────────────────────────────
    total_pnl  = closed_df["pnl"].sum()
    avg_win    = closed_df.loc[closed_df["pnl"]>0,"pnl"].mean() if (closed_df["pnl"]>0).any() else 0
    avg_loss   = closed_df.loc[closed_df["pnl"]<0,"pnl"].mean() if (closed_df["pnl"]<0).any() else 0
    winners    = (closed_df["pnl"] > 0).sum()
    losers     = (closed_df["pnl"] < 0).sum()
    win_rate   = winners / len(closed_df) * 100
    expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss) if len(closed_df)>0 else 0

    t1,t2,t3,t4,t5,t6 = st.columns(6)
    t1.metric("Total P&L",   f"₹{total_pnl:+,.0f}")
    t2.metric("Win Rate",    f"{win_rate:.0f}%",
              f"{winners}W / {losers}L")
    t3.metric("Avg Win",     f"₹{avg_win:,.0f}")
    t4.metric("Avg Loss",    f"₹{avg_loss:,.0f}")
    t5.metric("Expectancy",  f"₹{expectancy:+,.0f}")
    t6.metric("Total Trades",len(closed_df))

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        buckets = ["ALL"] + sorted(closed_df["bucket"].dropna().unique().tolist())
        filt_bkt = st.selectbox("Bucket", buckets)
    with f2:
        reasons = ["ALL"] + sorted(closed_df["exit_reason"].dropna().unique().tolist())
        filt_rsn = st.selectbox("Exit Reason", reasons)
    with f3:
        filt_result = st.selectbox("Result", ["ALL", "Winners", "Losers"])
    with f4:
        filt_regime = st.selectbox(
            "Entry Regime",
            ["ALL"] + sorted(closed_df["regime_at_entry"].dropna().unique().tolist())
        )

    fdf = closed_df.copy()
    if filt_bkt != "ALL":   fdf = fdf[fdf["bucket"] == filt_bkt]
    if filt_rsn != "ALL":   fdf = fdf[fdf["exit_reason"] == filt_rsn]
    if filt_result == "Winners": fdf = fdf[fdf["pnl"] > 0]
    if filt_result == "Losers":  fdf = fdf[fdf["pnl"] < 0]
    if filt_regime != "ALL": fdf = fdf[fdf["regime_at_entry"] == filt_regime]

    st.markdown(f"""
    <div style='font-family:"JetBrains Mono",monospace;font-size:11px;
                color:#475569;margin:-6px 0 12px 0;'>
        {len(fdf)} trades · P&L: ₹{fdf['pnl'].sum():+,.0f}
    </div>
    """, unsafe_allow_html=True)

    # ── Trade Table ───────────────────────────────────────────────────────────
    disp = fdf[[
        "symbol","bucket","entry_price","exit_price","qty",
        "pnl","pnl_pct","exit_reason","entry_date","exit_date",
        "score_at_entry","regime_at_entry"
    ]].copy()
    disp.columns = [
        "Symbol","Bucket","Entry ₹","Exit ₹","Qty",
        "P&L ₹","P&L %","Reason","Entry","Exit",
        "Score","Regime"
    ]
    disp["P&L ₹"] = disp["P&L ₹"].apply(lambda x: f"₹{x:+,.0f}")
    disp["P&L %"] = disp["P&L %"].apply(lambda x: f"{x:+.2f}%")
    disp["Entry ₹"] = disp["Entry ₹"].apply(lambda x: f"₹{x:.2f}")
    disp["Exit ₹"]  = disp["Exit ₹"].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "—")
    disp["Score"]   = disp["Score"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")

    st.dataframe(disp.reset_index(drop=True), use_container_width=True,
                 hide_index=True)

    # ── P&L waterfall ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Trade P&L Waterfall</div>",
                unsafe_allow_html=True)

    wf = fdf.sort_values("exit_date").copy()
    wf["label"] = wf["symbol"] + "\n" + wf["exit_date"].astype(str).str[5:]
    wf_colors = ["#00D4AA" if p >= 0 else "#F87171" for p in wf["pnl"]]

    fig_wf = go.Figure(go.Bar(
        x=wf["label"], y=wf["pnl"],
        marker_color=wf_colors,
        text=[f"₹{p:+,.0f}" for p in wf["pnl"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10),
        hovertemplate="<b>%{x}</b><br>P&L: ₹%{y:+,.0f}<extra></extra>",
    ))
    fig_wf.add_hline(y=0, line_color="#334155", line_width=1)
    fig_wf.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(family="JetBrains Mono", size=9, color="#475569"),
            tickangle=-45,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1E2D45", zeroline=False,
            tickprefix="₹",
            tickfont=dict(family="JetBrains Mono", size=10, color="#475569"),
        ),
    )
    st.plotly_chart(fig_wf, use_container_width=True,
                    config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    NSE SWING TRADER · STABILITY OVER PROFITABILITY · FOR EDUCATIONAL USE ONLY<br/>
    All execution is manual in Groww. This dashboard does not place orders.
</div>
""", unsafe_allow_html=True)
