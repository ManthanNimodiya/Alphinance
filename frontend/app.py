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

st_autorefresh(interval=10_000, limit=10000, key="btc_live_refresh")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Wordmark ── */
.wordmark {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.wordmark span {
    color: #38bdf8;
}

/* ── Pill badge ── */
.pill {
    display: inline-block;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    color: #38bdf8;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 999px;
}
.pill-green {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    color: #22c55e;
}

/* ── Big price number ── */
.price-display {
    font-family: 'DM Mono', monospace;
    font-size: 3.4rem;
    font-weight: 500;
    color: #f1f5f9;
    letter-spacing: -0.03em;
    line-height: 1;
}
.price-delta-up   { color: #22c55e; font-size: 0.9rem; font-weight: 600; }
.price-delta-down { color: #f87171; font-size: 0.9rem; font-weight: 600; }

/* ── Stat block ── */
.stat-block {
    border-left: 2px solid #1e293b;
    padding: 0 0 0 16px;
}
.stat-label {
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 4px;
}
.stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: #e2e8f0;
    letter-spacing: -0.02em;
}
.stat-sub {
    font-size: 0.68rem;
    color: #475569;
    margin-top: 2px;
}

/* ── Interval bar ── */
.interval-wrap {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 20px 28px;
}
.interval-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 14px;
}
.interval-row {
    display: flex;
    align-items: center;
    gap: 12px;
}
.bound-low  {
    font-family: 'DM Mono', monospace;
    font-size: 1.7rem;
    font-weight: 500;
    color: #f87171;
    letter-spacing: -0.02em;
}
.bound-high {
    font-family: 'DM Mono', monospace;
    font-size: 1.7rem;
    font-weight: 500;
    color: #4ade80;
    letter-spacing: -0.02em;
}
.bound-sep {
    flex: 1;
    height: 1px;
    background: #334155;
    position: relative;
}
.bound-sep::after {
    content: '95% CI';
    position: absolute;
    top: -9px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.6rem;
    color: #475569;
    background: #1e293b;
    padding: 0 6px;
    letter-spacing: 0.06em;
}

/* ── Metric tile ── */
.tile {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 20px;
}
.tile-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 6px;
}
.tile-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #e2e8f0;
    letter-spacing: -0.02em;
}
.tile-sub {
    font-size: 0.65rem;
    color: #475569;
    margin-top: 3px;
}
.tile-pass { border-top: 2px solid #22c55e; }
.tile-neutral { border-top: 2px solid #38bdf8; }

/* ── Section heading ── */
.sec-head {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin: 32px 0 12px 0;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; margin: 20px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data fetching ─────────────────────────────────────────────────────────────

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
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "backend", "data", "btc_1h.csv")
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
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "backend", "data", "predictions.csv")
    if os.path.exists(path):
        df_pred = pd.read_csv(path, dtype={"inside": str})
    else:
        df_pred = pd.DataFrame(columns=["timestamp", "L", "U", "actual", "inside"])
    if len(df_pred) > 0:
        df_pred.loc[df_pred.index[-1], "actual"] = current_price
        hit = (df_pred.loc[df_pred.index[-1], "L"]
               <= current_price
               <= df_pred.loc[df_pred.index[-1], "U"])
        df_pred.loc[df_pred.index[-1], "inside"] = "Yes" if hit else "No"
    new_row = pd.DataFrame([{
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "L": round(L, 2), "U": round(U, 2),
        "actual": None, "inside": None
    }])
    df_pred = pd.concat([df_pred, new_row], ignore_index=True)
    df_pred.to_csv(path, index=False)
    return df_pred


# ── Fetch ─────────────────────────────────────────────────────────────────────
try:
    df          = fetch_ohlcv(limit=70)
    price, chg  = fetch_ticker()
    closes_live = df["close"].reset_index(drop=True)
    L, U        = predict_intervals(closes_live, t=len(closes_live))
    now_str     = datetime.utcnow().strftime("%H:%M:%S UTC")
    today_str   = datetime.utcnow().strftime("%d %b %Y")
except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.stop()

width_val = U - L
pct_up    = ((U - price) / price) * 100
pct_dn    = ((price - L) / price) * 100
price_up  = price >= closes_live.iloc[-2] if len(closes_live) >= 2 else True
df_pred   = update_predictions(L, U, price)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Top bar ──────────────────────────────────────────────────────────────────
col_logo, col_meta, col_btn = st.columns([3, 5, 1])

with col_logo:
    st.markdown("""
    <div style="padding-top:6px;">
        <span class="wordmark">Alpha<span>nance</span></span>
    </div>""", unsafe_allow_html=True)

with col_meta:
    st.markdown(f"""
    <div style="display:flex; gap:8px; align-items:center; padding-top:10px; flex-wrap:wrap;">
        <span class="pill">BTC / USDT</span>
        <span class="pill">1-Hour</span>
        <span class="pill-green pill">● Live · {now_str}</span>
        <span style="font-size:0.68rem; color:#475569;">{today_str}</span>
    </div>""", unsafe_allow_html=True)

with col_btn:
    st.markdown("<div style='padding-top:4px;'>", unsafe_allow_html=True)
    if st.button("↻", use_container_width=True, help="Clear cache and refresh"):
        fetch_ohlcv.clear()
        fetch_ticker.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Price + stats row ─────────────────────────────────────────────────────────
delta_class = "price-delta-up" if chg >= 0 else "price-delta-down"
arrow       = "▲" if chg >= 0 else "▼"

col_p, col_s1, col_s2, col_s3 = st.columns([4, 2, 2, 2])

with col_p:
    st.markdown(f"""
    <div>
        <div style="font-size:0.68rem; text-transform:uppercase;
                    letter-spacing:0.1em; color:#475569; margin-bottom:8px;">
            Current Price
        </div>
        <div class="price-display">${price:,.2f}</div>
        <div style="margin-top:6px;">
            <span class="{delta_class}">{arrow} {abs(chg):.2f}% &nbsp;24h</span>
            &nbsp;
            <span style="font-size:0.68rem; color:#475569;">
                ⚡ refreshes every 10s
            </span>
        </div>
    </div>""", unsafe_allow_html=True)

with col_s1:
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-label">Next Low</div>
        <div class="stat-value" style="color:#f87171;">${L:,.2f}</div>
        <div class="stat-sub">−{pct_dn:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col_s2:
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-label">Next High</div>
        <div class="stat-value" style="color:#4ade80;">${U:,.2f}</div>
        <div class="stat-sub">+{pct_up:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col_s3:
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-label">Range Width</div>
        <div class="stat-value">${width_val:,.2f}</div>
        <div class="stat-sub">95% confidence</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction interval bar ───────────────────────────────────────────────────
st.markdown(f"""
<div class="interval-wrap">
    <div class="interval-label">Next 1-hour candle · predicted interval</div>
    <div class="interval-row">
        <span class="bound-low">${L:,.2f}</span>
        <div class="bound-sep"></div>
        <span class="bound-high">${U:,.2f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Price chart · last 50 bars</div>', unsafe_allow_html=True)

display_df = df.tail(50).copy().reset_index(drop=True)
next_time  = display_df["open_time"].iloc[-1] + pd.Timedelta(hours=1)
pred_x0    = next_time - pd.Timedelta(minutes=25)
pred_x1    = next_time + pd.Timedelta(minutes=25)

fig = go.Figure()

# Candle close line
fig.add_trace(go.Scatter(
    x=display_df["open_time"],
    y=display_df["close"],
    mode="lines",
    name="Close",
    line=dict(color="#38bdf8", width=1.5),
    hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
))

# Shaded forecast zone
fig.add_trace(go.Scatter(
    x=[display_df["open_time"].iloc[-1], pred_x1, pred_x1,
       display_df["open_time"].iloc[-1]],
    y=[U, U, L, L],
    fill="toself",
    fillcolor="rgba(56,189,248,0.06)",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip",
))

# Upper / lower dashed bounds
fig.add_shape(type="line", x0=pred_x0, x1=pred_x1, y0=U, y1=U,
              line=dict(color="#4ade80", width=1.5, dash="dot"))
fig.add_shape(type="line", x0=pred_x0, x1=pred_x1, y0=L, y1=L,
              line=dict(color="#f87171", width=1.5, dash="dot"))

# "Now" vertical rule
fig.add_vline(
    x=display_df["open_time"].iloc[-1].timestamp() * 1000,
    line_width=1,
    line_dash="dot",
    line_color="rgba(255,255,255,0.1)",
)

fig.update_layout(
    height=340,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=11, color="#64748b"),
    margin=dict(l=0, r=0, t=8, b=0),
    xaxis=dict(
        showgrid=True, gridcolor="#1e293b",
        zeroline=False, tickfont=dict(size=10),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#1e293b",
        zeroline=False, tickprefix="$", tickformat=",.0f",
        tickfont=dict(size=10),
    ),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Backtest metrics ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Backtest · 720 hourly bars · 30 days</div>',
            unsafe_allow_html=True)

with st.spinner(""):
    coverage, avg_width_bt, winkler_score = run_backtest()

ok = coverage >= 0.95
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown(f"""
    <div class="tile {'tile-pass' if ok else 'tile-neutral'}">
        <div class="tile-label">Coverage</div>
        <div class="tile-value" style="color:{'#4ade80' if ok else '#f87171'};">
            {coverage*100:.2f}%
        </div>
        <div class="tile-sub">{'✓ meets 95% target' if ok else '✗ below target'}</div>
    </div>""", unsafe_allow_html=True)

with b2:
    st.markdown(f"""
    <div class="tile tile-neutral">
        <div class="tile-label">Avg Width</div>
        <div class="tile-value">${avg_width_bt:,.2f}</div>
        <div class="tile-sub">lower is better</div>
    </div>""", unsafe_allow_html=True)

with b3:
    st.markdown(f"""
    <div class="tile tile-neutral">
        <div class="tile-label">Winkler Score</div>
        <div class="tile-value">{winkler_score:,.2f}</div>
        <div class="tile-sub">lower is better</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Prediction history ────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Prediction history</div>', unsafe_allow_html=True)
history = df_pred[df_pred["actual"].notna()].copy()
if history.empty:
    st.markdown(
        '<p style="font-size:0.8rem; color:#475569;">No resolved predictions yet.</p>',
        unsafe_allow_html=True)
else:
    st.dataframe(history, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:40px; padding-top:16px; border-top:1px solid #1e293b;
            display:flex; justify-content:space-between; align-items:center;
            flex-wrap:wrap; gap:8px;">
    <span style="font-size:0.68rem; color:#334155;">
        Alphinance · Student-t rolling interval · AlphaI × Polaris Challenge
    </span>
    <span style="font-size:0.68rem; color:#334155;">
        <a href="https://huggingface.co/spaces/Manthan6683/Alphinance"
           style="color:#38bdf8; text-decoration:none;">Live Dashboard</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/ManthanNimodiya/Alphinance"
           style="color:#38bdf8; text-decoration:none;">GitHub</a>
    </span>
</div>
""", unsafe_allow_html=True)
