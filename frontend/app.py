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
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Kill Streamlit's default top padding ── */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 1.2rem !important;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Logo mark ── */
.logo-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
}
.logo-icon {
    width: 32px;
    height: 32px;
    background: #1d1d27;
    border: 1px solid #2e2e3e;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    line-height: 1;
}
.logo-text {
    font-size: 1.15rem;
    font-weight: 600;
    color: #f4f4f5;
    letter-spacing: -0.03em;
}
.logo-text sup {
    font-size: 0.55rem;
    font-weight: 500;
    color: #52525b;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    vertical-align: super;
    margin-left: 4px;
}

/* ── Pill badge ── */
.pill {
    display: inline-block;
    background: #1c1c28;
    border: 1px solid #2a2a3a;
    color: #a1a1aa;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 3px 10px;
    border-radius: 999px;
}
.pill-live {
    background: #0f2318;
    border: 1px solid #166534;
    color: #4ade80;
}

/* ── Price ── */
.price-number {
    font-family: 'DM Mono', monospace;
    font-size: 3rem;
    font-weight: 400;
    color: #fafafa;
    letter-spacing: -0.04em;
    line-height: 1;
}
.delta-up   { font-size: 0.82rem; color: #4ade80; font-weight: 500; }
.delta-down { font-size: 0.82rem; color: #f87171; font-weight: 500; }
.meta-text  { font-size: 0.68rem; color: #3f3f46; }

/* ── Stat cols ── */
.stat {
    padding-left: 20px;
    border-left: 1px solid #27272a;
}
.stat-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #52525b;
    margin-bottom: 4px;
}
.stat-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.35rem;
    font-weight: 400;
    color: #e4e4e7;
    letter-spacing: -0.02em;
}
.stat-sub { font-size: 0.65rem; color: #3f3f46; margin-top: 2px; }

/* ── Interval bar ── */
.iv-wrap {
    background: #111118;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 14px 22px;
    margin-top: 12px;
}
.iv-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3f3f46;
    margin-bottom: 10px;
}
.iv-row { display:flex; align-items:center; gap:10px; }
.iv-low  {
    font-family: 'DM Mono', monospace;
    font-size: 1.55rem;
    color: #fca5a5;
    font-weight: 400;
    letter-spacing: -0.02em;
}
.iv-high {
    font-family: 'DM Mono', monospace;
    font-size: 1.55rem;
    color: #86efac;
    font-weight: 400;
    letter-spacing: -0.02em;
}
.iv-rule {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}
.iv-rule::before {
    content: '';
    display: block;
    width: 100%;
    height: 1px;
    background: #27272a;
}
.iv-ci {
    position: absolute;
    font-size: 0.58rem;
    letter-spacing: 0.06em;
    color: #3f3f46;
    background: #111118;
    padding: 0 6px;
}

/* ── Metric tile ── */
.tile {
    background: #111118;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 14px 18px;
}
.tile-label { font-size:0.62rem; text-transform:uppercase; letter-spacing:0.09em; color:#52525b; margin-bottom:5px; }
.tile-val   { font-family:'DM Mono',monospace; font-size:1.4rem; font-weight:400; color:#e4e4e7; letter-spacing:-0.02em; }
.tile-sub   { font-size:0.62rem; color:#3f3f46; margin-top:2px; }
.tile-pass  { border-top: 2px solid #16a34a; }
.tile-dim   { border-top: 2px solid #27272a; }

/* ── Section label ── */
.sec {
    font-size:0.62rem;
    font-weight:600;
    text-transform:uppercase;
    letter-spacing:0.1em;
    color:#3f3f46;
    margin: 18px 0 8px 0;
}

/* ── Rule ── */
.rule { border:none; border-top:1px solid #1c1c1c; margin: 14px 0; }
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def fetch_ticker() -> tuple:
    r = requests.get(f"{URL}/ticker/24hr", params={"symbol": "BTCUSDT"}, timeout=10)
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
    alpha, results = 0.05, []
    for t in range(21, 720):
        L, U = predict_intervals(closes, t)
        results.append({"L": L, "U": U, "actual": closes.iloc[t]})
    df = pd.DataFrame(results)
    inside  = (df["actual"] >= df["L"]) & (df["actual"] <= df["U"])
    width   = df["U"] - df["L"]
    winkler = np.where(df["actual"] < df["L"],
                       width + (2/alpha)*(df["L"]-df["actual"]),
                       np.where(df["actual"] > df["U"],
                                width + (2/alpha)*(df["actual"]-df["U"]), width))
    return float(inside.mean()), float(width.mean()), float(np.mean(winkler))


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
        "L": round(L, 2), "U": round(U, 2), "actual": None, "inside": None
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
    date_str    = datetime.utcnow().strftime("%d %b %Y")
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
col_logo, col_pills, col_btn = st.columns([2, 6, 1])

with col_logo:
    st.markdown("""
    <div class="logo-wrap" style="padding-top:4px;">
        <div class="logo-icon">◈</div>
        <div class="logo-text">Alphinance<sup>beta</sup></div>
    </div>""", unsafe_allow_html=True)

with col_pills:
    live_color = "#4ade80"
    st.markdown(f"""
    <div style="display:flex;gap:6px;align-items:center;padding-top:8px;flex-wrap:wrap;">
        <span class="pill">BTC / USDT</span>
        <span class="pill">1H</span>
        <span class="pill-live pill">● {now_str}</span>
        <span class="meta-text">{date_str}</span>
    </div>""", unsafe_allow_html=True)

with col_btn:
    st.markdown("<div style='padding-top:2px;'>", unsafe_allow_html=True)
    if st.button("↻ Refresh", use_container_width=True):
        fetch_ohlcv.clear()
        fetch_ticker.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ── Price row ────────────────────────────────────────────────────────────────
arrow       = "▲" if chg >= 0 else "▼"
delta_class = "delta-up" if chg >= 0 else "delta-down"

col_p, col_low, col_high, col_w = st.columns([4, 2, 2, 2])

with col_p:
    st.markdown(f"""
    <div style="padding:2px 0 0 0;">
        <div class="meta-text" style="margin-bottom:6px;">CURRENT PRICE</div>
        <div class="price-number">${price:,.2f}</div>
        <div style="margin-top:5px;display:flex;gap:10px;align-items:center;">
            <span class="{delta_class}">{arrow} {abs(chg):.2f}%&thinsp;24h</span>
            <span class="meta-text">⚡ refreshes every 10s</span>
        </div>
    </div>""", unsafe_allow_html=True)

with col_low:
    st.markdown(f"""
    <div class="stat">
        <div class="stat-label">Next Low</div>
        <div class="stat-val" style="color:#fca5a5;">${L:,.2f}</div>
        <div class="stat-sub">−{pct_dn:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col_high:
    st.markdown(f"""
    <div class="stat">
        <div class="stat-label">Next High</div>
        <div class="stat-val" style="color:#86efac;">${U:,.2f}</div>
        <div class="stat-sub">+{pct_up:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col_w:
    st.markdown(f"""
    <div class="stat">
        <div class="stat-label">Range Width</div>
        <div class="stat-val">${width_val:,.2f}</div>
        <div class="stat-sub">95% confidence</div>
    </div>""", unsafe_allow_html=True)

# ── Interval bar ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="iv-wrap">
    <div class="iv-label">Next 1-hour candle · predicted interval</div>
    <div class="iv-row">
        <span class="iv-low">${L:,.2f}</span>
        <div class="iv-rule"><span class="iv-ci">95% CI</span></div>
        <span class="iv-high">${U:,.2f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec" style="margin-top:16px;">Price chart · last 50 bars</div>',
            unsafe_allow_html=True)

display_df = df.tail(50).copy().reset_index(drop=True)
next_time  = display_df["open_time"].iloc[-1] + pd.Timedelta(hours=1)
pred_x0    = next_time - pd.Timedelta(minutes=25)
pred_x1    = next_time + pd.Timedelta(minutes=25)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=display_df["open_time"], y=display_df["close"],
    mode="lines", name="Close",
    line=dict(color="#60a5fa", width=1.5),
    hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[display_df["open_time"].iloc[-1], pred_x1, pred_x1,
       display_df["open_time"].iloc[-1]],
    y=[U, U, L, L],
    fill="toself", fillcolor="rgba(96,165,250,0.05)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_shape(type="line", x0=pred_x0, x1=pred_x1, y0=U, y1=U,
              line=dict(color="#86efac", width=1.2, dash="dot"))
fig.add_shape(type="line", x0=pred_x0, x1=pred_x1, y0=L, y1=L,
              line=dict(color="#fca5a5", width=1.2, dash="dot"))
fig.add_vline(
    x=display_df["open_time"].iloc[-1].timestamp() * 1000,
    line_width=1, line_dash="dot", line_color="#27272a",
)
fig.update_layout(
    height=320,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=10, color="#52525b"),
    margin=dict(l=0, r=0, t=4, b=0),
    xaxis=dict(showgrid=True, gridcolor="#18181b", zeroline=False,
               tickfont=dict(size=9)),
    yaxis=dict(showgrid=True, gridcolor="#18181b", zeroline=False,
               tickprefix="$", tickformat=",.0f", tickfont=dict(size=9)),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ── Backtest ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Backtest · 720 hourly bars · 30 days</div>',
            unsafe_allow_html=True)

with st.spinner(""):
    coverage, avg_width_bt, winkler_score = run_backtest()

ok = coverage >= 0.95
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown(f"""
    <div class="tile {'tile-pass' if ok else 'tile-dim'}">
        <div class="tile-label">Coverage</div>
        <div class="tile-val" style="color:{'#4ade80' if ok else '#f87171'};">
            {coverage*100:.2f}%
        </div>
        <div class="tile-sub">{'✓ meets 95% target' if ok else '✗ below target'}</div>
    </div>""", unsafe_allow_html=True)

with b2:
    st.markdown(f"""
    <div class="tile tile-dim">
        <div class="tile-label">Avg Width</div>
        <div class="tile-val">${avg_width_bt:,.2f}</div>
        <div class="tile-sub">lower is better</div>
    </div>""", unsafe_allow_html=True)

with b3:
    st.markdown(f"""
    <div class="tile tile-dim">
        <div class="tile-label">Winkler Score</div>
        <div class="tile-val">{winkler_score:,.2f}</div>
        <div class="tile-sub">lower is better</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ── History ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Prediction history</div>', unsafe_allow_html=True)
history = df_pred[df_pred["actual"].notna()].copy()
if history.empty:
    st.markdown('<p style="font-size:0.78rem;color:#3f3f46;">No resolved predictions yet.</p>',
                unsafe_allow_html=True)
else:
    st.dataframe(history, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:24px;padding-top:12px;border-top:1px solid #18181b;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
    <span style="font-size:0.62rem;color:#27272a;">
        Alphinance · Student-t rolling interval · AlphaI × Polaris Challenge · {date_str}
    </span>
    <span style="font-size:0.62rem;">
        <a href="https://huggingface.co/spaces/Manthan6683/Alphinance"
           style="color:#3f3f46;text-decoration:none;">Live Dashboard</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/ManthanNimodiya/Alphinance"
           style="color:#3f3f46;text-decoration:none;">GitHub</a>
    </span>
</div>
""", unsafe_allow_html=True)
