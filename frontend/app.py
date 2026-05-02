import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from backend.model import predict_intervals
from streamlit_autorefresh import st_autorefresh

URL = "https://data-api.binance.vision/api/v3"

st.set_page_config(
    page_title="Alphinance | BTC Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 10-second live refresh
st_autorefresh(interval=10_000, limit=10000, key="btc_live_refresh")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
    border: 1px solid rgba(240,165,0,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #f0a500;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1;
}
.metric-sub {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.35);
    margin-top: 4px;
}
.price-hero {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #f0a500, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.range-card {
    background: linear-gradient(135deg, #0d1f2d, #0a1628);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
}
.range-lower { color: #ef4444; font-size: 1.5rem; font-weight: 800; }
.range-upper { color: #22c55e; font-size: 1.5rem; font-weight: 800; }
.range-sep { color: rgba(255,255,255,0.3); font-size: 1.3rem; margin: 0 16px; }
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f0a500;
    border-left: 3px solid #f0a500;
    padding-left: 10px;
    margin: 28px 0 14px 0;
}
.live-badge {
    display: inline-block;
    border: 1px solid #00c89633;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: #00c896;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=10)
def fetch_ticker() -> tuple:
    r = requests.get(f"{URL}/ticker/24hr",
                     params={"symbol": "BTCUSDT"}, timeout=10)
    r.raise_for_status()
    d = r.json()
    return float(d["lastPrice"]), float(d["priceChangePercent"])


@st.cache_data(ttl=300)
def fetch_ohlcv(limit=70) -> pd.DataFrame:
    r = requests.get(f"{URL}/klines",
                     params={"symbol": "BTCUSDT", "interval": "1h", "limit": limit},
                     timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())[[0, 4]]
    df.columns = ["open_time", "close"]
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


@st.cache_data(ttl=3600)
def run_backtest():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "data", "btc_1h.csv")
    closes = pd.read_csv(path)["close"]
    alpha = 0.05
    results = []
    for t in range(21, 720):
        L, U = predict_intervals(closes, t)
        actual = closes.iloc[t]
        results.append({"L": L, "U": U, "actual": actual})
    df = pd.DataFrame(results)
    inside = (df["actual"] >= df["L"]) & (df["actual"] <= df["U"])
    coverage = inside.mean()
    avg_width = (df["U"] - df["L"]).mean()
    width = df["U"] - df["L"]
    winkler = np.where(
        df["actual"] < df["L"],
        width + (2 / alpha) * (df["L"] - df["actual"]),
        np.where(
            df["actual"] > df["U"],
            width + (2 / alpha) * (df["actual"] - df["U"]),
            width
        )
    )
    return float(coverage), float(avg_width), float(np.mean(winkler))


def update_predictions(L, U, current_price):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "backend", "data", "predictions.csv"
    )
    if os.path.exists(path):
        df_pred = pd.read_csv(path, dtype={"inside": str})
    else:
        df_pred = pd.DataFrame(columns=["timestamp", "L", "U", "actual", "inside"])
    if len(df_pred) > 0:
        df_pred.loc[df_pred.index[-1], "actual"] = current_price
        hit = df_pred.loc[df_pred.index[-1], "L"] <= current_price <= df_pred.loc[df_pred.index[-1], "U"]
        df_pred.loc[df_pred.index[-1], "inside"] = "Yes" if hit else "No"
    new_row = pd.DataFrame([{
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "L": round(L, 2),
        "U": round(U, 2),
        "actual": None,
        "inside": None
    }])
    df_pred = pd.concat([df_pred, new_row], ignore_index=True)
    df_pred.to_csv(path, index=False)
    return df_pred


# ── Fetch data ────────────────────────────────────────────────────────────────
try:
    df = fetch_ohlcv(limit=70)
    current_price, change_pct = fetch_ticker()
    closes_live = df["close"].reset_index(drop=True)
    L, U = predict_intervals(closes_live, t=len(closes_live))
    last_updated = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.stop()

width_val = U - L
pct_up = ((U - current_price) / current_price) * 100
pct_dn = ((current_price - L) / current_price) * 100
price_up = current_price >= closes_live.iloc[-2] if len(closes_live) >= 2 else True

df_pred = update_predictions(L, U, current_price)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 28px 0 6px 0;">
    <span style="font-size:2.6rem; font-weight:900;
        background:linear-gradient(90deg,#f0a500,#ff6b35);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        ₿ Alphinance
    </span>
    <div style="font-size:0.85rem; color:rgba(255,255,255,0.4); margin-top:5px;">
        BTC/USDT &nbsp;·&nbsp; 1-Hour Prediction &nbsp;·&nbsp; AlphaI × Polaris Build Challenge
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Live Price ────────────────────────────────────────────────────────────────
arrow = "▲" if change_pct >= 0 else "▼"
change_color = "#22c55e" if change_pct >= 0 else "#ef4444"

col_price, col_refresh = st.columns([5, 1])
with col_price:
    st.markdown(f"""
    <div style="padding: 4px 0 8px 0;">
        <div style="font-size:0.72rem; font-weight:600; letter-spacing:.12em;
                    text-transform:uppercase; color:#f0a500; margin-bottom:6px;">
            Current BTC Price
        </div>
        <div style="display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;">
            <span class="price-hero">${current_price:,.2f}</span>
            <span style="font-size:1rem; color:{change_color}; font-weight:700;">
                {arrow} {abs(change_pct):.2f}% 24h
            </span>
            <span class="live-badge">● LIVE (10s)</span>
        </div>
        <div style="font-size:0.72rem; color:rgba(255,255,255,0.3); margin-top:5px;">
            ⚡ Refreshes every 10s · Last: {last_updated}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_refresh:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("↻ Refresh", use_container_width=True):
        fetch_ohlcv.clear()
        fetch_ticker.clear()
        st.rerun()

st.divider()

# ── Prediction Range ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🎯 Next 1-Hour Candle — 95% Confidence Range</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="range-card">
    <div style="font-size:0.75rem; color:rgba(255,255,255,0.35); margin-bottom:12px;
                letter-spacing:.1em; text-transform:uppercase;">
        Predicted range for the next 1-hour bar
    </div>
    <span class="range-lower">Lower: ${L:,.2f}</span>
    <span class="range-sep">|</span>
    <span class="range-upper">Upper: ${U:,.2f}</span>
    <div style="margin-top:10px; font-size:0.78rem; color:rgba(255,255,255,0.3);">
        Width: ${width_val:,.2f} &nbsp;·&nbsp; Midpoint: ${(U+L)/2:,.2f}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Next Hour Low</div>
        <div class="metric-value">${L:,.2f}</div>
        <div class="metric-sub">-{pct_dn:.2f}% from now</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Next Hour High</div>
        <div class="metric-value">${U:,.2f}</div>
        <div class="metric-sub">+{pct_up:.2f}% from now</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Range Width</div>
        <div class="metric-value">${width_val:,.2f}</div>
        <div class="metric-sub">95% confidence</div>
    </div>""", unsafe_allow_html=True)
with c4:
    direction = "▲ Up" if price_up else "▼ Down"
    dir_color = "#22c55e" if price_up else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Trend</div>
        <div class="metric-value" style="color:{dir_color};">{direction}</div>
        <div class="metric-sub">vs prev close</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Price Chart — Last 50 Bars + Forecast</div>', unsafe_allow_html=True)

display_df = df.tail(50).copy().reset_index(drop=True)
next_time   = display_df["open_time"].iloc[-1] + pd.Timedelta(hours=1)
pred_x0     = next_time - pd.Timedelta(minutes=20)
pred_x1     = next_time + pd.Timedelta(minutes=20)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=display_df["open_time"],
    y=display_df["close"],
    mode="lines",
    name="BTC/USDT",
    line=dict(color="#f0a500", width=2),
    hovertemplate="<b>%{x}</b><br>Close: $%{y:,.2f}<extra></extra>",
))

fig.add_vrect(x0=pred_x0, x1=pred_x1,
    fillcolor="rgba(34,197,94,0.08)", line_width=0)

fig.add_shape(type="line",
    x0=pred_x0, x1=pred_x1, y0=U, y1=U,
    line=dict(color="#22c55e", width=2, dash="dot"))

fig.add_shape(type="line",
    x0=pred_x0, x1=pred_x1, y0=L, y1=L,
    line=dict(color="#ef4444", width=2, dash="dot"))

fig.update_layout(
    height=380,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e5e7eb"),
    margin=dict(l=8, r=8, t=16, b=8),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickprefix="$",
               tickformat=",.0f", tickfont=dict(size=10)),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Backtest ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 30-Day Backtest · 720 Hourly Bars</div>', unsafe_allow_html=True)

with st.spinner("Running backtest..."):
    coverage, avg_width_bt, winkler_score = run_backtest()

b1, b2, b3 = st.columns(3)
with b1:
    ok = coverage >= 0.95
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Coverage</div>
        <div class="metric-value" style="color:{'#22c55e' if ok else '#ef4444'};">
            {coverage*100:.2f}%
        </div>
        <div class="metric-sub">{'✓ Meets 95% target' if ok else '✗ Below target'}</div>
    </div>""", unsafe_allow_html=True)
with b2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Range Width</div>
        <div class="metric-value">${avg_width_bt:,.2f}</div>
        <div class="metric-sub">Lower is better</div>
    </div>""", unsafe_allow_html=True)
with b3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Winkler Score</div>
        <div class="metric-value">{winkler_score:,.2f}</div>
        <div class="metric-sub">Lower is better</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Prediction History ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Prediction History</div>', unsafe_allow_html=True)
st.dataframe(df_pred[df_pred["actual"].notna()], use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; margin-top:40px; padding-top:14px;
            border-top:1px solid rgba(255,255,255,0.08);
            font-size:0.72rem; color:rgba(255,255,255,0.25);">
    Alphinance · AlphaI × Polaris Challenge · {last_updated} ·
    <a href="https://huggingface.co/spaces/Manthan6683/Alphinance"
       style="color:rgba(255,255,255,0.35);">Live Dashboard</a> ·
    <a href="https://github.com/ManthanNimodiya/Alphinance"
       style="color:rgba(255,255,255,0.35);">GitHub</a>
</div>
""", unsafe_allow_html=True)
