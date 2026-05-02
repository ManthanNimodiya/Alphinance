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
    page_title="Alphinance",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st_autorefresh(interval=10_000, limit=10000, key="btc_live_refresh")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Playfair+Display:ital,wght@0,700;1,400&display=swap');

/* ── Kill Streamlit padding ── */
.block-container,
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0.9rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: #000 !important;
}

/* ── Top nav ── */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 0;
}
.nav-left-top {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    color: #fff;
    text-transform: uppercase;
}
.nav-left-sub {
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    color: #444;
    text-transform: uppercase;
    margin-top: 2px;
}
.nav-right {
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    color: #444;
    text-transform: uppercase;
    text-align: right;
}

/* ── Divider ── */
.rule { border: none; border-top: 1px solid #1a1a1a; margin: 12px 0; }

/* ── Section label ── */
.sec-label {
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* ── Price hero ── */
.price-hero {
    font-family: 'DM Mono', monospace;
    font-size: 4rem;
    font-weight: 300;
    color: #fff;
    letter-spacing: -0.04em;
    line-height: 1;
}
.price-delta-up   { font-size: 0.72rem; color: #fff; letter-spacing: 0.06em; }
.price-delta-down { font-size: 0.72rem; color: #555; letter-spacing: 0.06em; }
.price-meta       { font-size: 0.58rem; color: #333; letter-spacing: 0.08em; }

/* ── Stat ── */
.stat {
    border-left: 1px solid #1a1a1a;
    padding-left: 20px;
}
.stat-label { font-size: 0.55rem; letter-spacing: 0.14em; color: #444; text-transform: uppercase; margin-bottom: 4px; }
.stat-val   { font-family: 'DM Mono', monospace; font-size: 1.4rem; font-weight: 300; color: #fff; letter-spacing: -0.02em; }
.stat-sub   { font-size: 0.58rem; color: #333; margin-top: 2px; }

/* ── Interval ── */
.iv-wrap {
    border: 1px solid #1a1a1a;
    padding: 16px 22px;
    margin-top: 14px;
}
.iv-row { display: flex; align-items: center; gap: 12px; }
.iv-low  { font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 300; color: #777; letter-spacing: -0.02em; }
.iv-high { font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 300; color: #fff; letter-spacing: -0.02em; }
.iv-sep  {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    color: #222;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
}
.iv-sep::before,
.iv-sep::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1a1a;
}

/* ── Metric tile ── */
.tile {
    border: 1px solid #1a1a1a;
    padding: 14px 18px;
}
.tile-label { font-size: 0.55rem; letter-spacing: 0.14em; color: #444; text-transform: uppercase; margin-bottom: 6px; }
.tile-val   { font-family: 'DM Mono', monospace; font-size: 1.5rem; font-weight: 300; color: #fff; letter-spacing: -0.02em; }
.tile-sub   { font-size: 0.58rem; color: #333; margin-top: 3px; }
.tile-pass  .tile-val { color: #fff; }

/* ── Comment style ── */
.comment { font-size: 0.6rem; color: #2a2a2a; letter-spacing: 0.06em; }
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
    inside = (df["actual"] >= df["L"]) & (df["actual"] <= df["U"])
    width  = df["U"] - df["L"]
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

# ── Nav ───────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([6, 1])
with col_l:
    st.markdown(f"""
    <div>
        <div class="nav-left-top">Alphinance</div>
        <div class="nav-left-sub">Prediction Engine &nbsp;·&nbsp; BTC/USDT &nbsp;·&nbsp; 1H</div>
    </div>""", unsafe_allow_html=True)
with col_r:
    st.markdown(f"""
    <div class="nav-right" style="padding-top:2px;">
        <div style="color:#fff;">● LIVE</div>
        <div>{now_str}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ── Price + stats ─────────────────────────────────────────────────────────────
arrow       = "▲" if chg >= 0 else "▼"
delta_class = "price-delta-up" if chg >= 0 else "price-delta-down"

col_p, col_low, col_high, col_w = st.columns([4, 2, 2, 2])

with col_p:
    st.markdown(f"""
    <div class="sec-label">// current price</div>
    <div class="price-hero">${price:,.2f}</div>
    <div style="margin-top:6px;display:flex;gap:14px;align-items:center;">
        <span class="{delta_class}">{arrow} {abs(chg):.2f}%&thinsp;24h</span>
        <span class="price-meta">refreshes every 10s</span>
    </div>""", unsafe_allow_html=True)

with col_low:
    st.markdown(f"""
    <div class="stat">
        <div class="stat-label">// floor</div>
        <div class="stat-val">${L:,.2f}</div>
        <div class="stat-sub">−{pct_dn:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col_high:
    st.markdown(f"""
    <div class="stat">
        <div class="stat-label">// ceiling</div>
        <div class="stat-val">${U:,.2f}</div>
        <div class="stat-sub">+{pct_up:.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col_w:
    st.markdown(f"""
    <div class="stat">
        <div class="stat-label">// width</div>
        <div class="stat-val">${width_val:,.2f}</div>
        <div class="stat-sub">95% confidence</div>
    </div>""", unsafe_allow_html=True)

# ── Interval bar ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="iv-wrap">
    <div class="sec-label" style="margin-bottom:10px;">// next 1-hour candle · predicted interval · 95% CI</div>
    <div class="iv-row">
        <span class="iv-low">${L:,.2f}</span>
        <div class="iv-sep">95% CI</div>
        <span class="iv-high">${U:,.2f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label" style="margin-top:16px;">// price · last 50 bars</div>',
            unsafe_allow_html=True)

display_df = df.tail(50).copy().reset_index(drop=True)
next_time  = display_df["open_time"].iloc[-1] + pd.Timedelta(hours=1)
pred_x0    = next_time - pd.Timedelta(minutes=25)
pred_x1    = next_time + pd.Timedelta(minutes=25)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=display_df["open_time"], y=display_df["close"],
    mode="lines", name="close",
    line=dict(color="#ffffff", width=1),
    hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[display_df["open_time"].iloc[-1], pred_x1, pred_x1,
       display_df["open_time"].iloc[-1]],
    y=[U, U, L, L],
    fill="toself", fillcolor="rgba(255,255,255,0.03)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_shape(type="line", x0=pred_x0, x1=pred_x1, y0=U, y1=U,
              line=dict(color="#555", width=1, dash="dot"))
fig.add_shape(type="line", x0=pred_x0, x1=pred_x1, y0=L, y1=L,
              line=dict(color="#333", width=1, dash="dot"))
fig.add_vline(
    x=display_df["open_time"].iloc[-1].timestamp() * 1000,
    line_width=1, line_dash="dot", line_color="#1a1a1a",
)
fig.update_layout(
    height=300,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono", size=9, color="#444"),
    margin=dict(l=0, r=0, t=4, b=0),
    xaxis=dict(showgrid=True, gridcolor="#0d0d0d", zeroline=False,
               tickfont=dict(size=8), linecolor="#1a1a1a"),
    yaxis=dict(showgrid=True, gridcolor="#0d0d0d", zeroline=False,
               tickprefix="$", tickformat=",.0f", tickfont=dict(size=8)),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9), x=0, y=1),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ── Backtest ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">// backtest · 720 bars · 30 days</div>',
            unsafe_allow_html=True)

with st.spinner(""):
    coverage, avg_width_bt, winkler_score = run_backtest()

ok = coverage >= 0.95
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown(f"""
    <div class="tile">
        <div class="tile-label">coverage</div>
        <div class="tile-val">{coverage*100:.2f}%</div>
        <div class="tile-sub">{'target met ✓' if ok else 'below target'}</div>
    </div>""", unsafe_allow_html=True)

with b2:
    st.markdown(f"""
    <div class="tile">
        <div class="tile-label">avg width</div>
        <div class="tile-val">${avg_width_bt:,.2f}</div>
        <div class="tile-sub">lower is better</div>
    </div>""", unsafe_allow_html=True)

with b3:
    st.markdown(f"""
    <div class="tile">
        <div class="tile-label">winkler score</div>
        <div class="tile-val">{winkler_score:,.2f}</div>
        <div class="tile-sub">lower is better</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ── History ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">// prediction history</div>', unsafe_allow_html=True)
history = df_pred[df_pred["actual"].notna()].copy()
if history.empty:
    st.markdown('<span class="comment">// no resolved predictions yet</span>',
                unsafe_allow_html=True)
else:
    st.dataframe(history, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:20px;padding-top:10px;border-top:1px solid #111;
            display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px;">
    <span style="font-size:0.55rem;color:#222;letter-spacing:0.1em;text-transform:uppercase;">
        Alphinance · Student-t · AlphaI × Polaris · {date_str}
    </span>
    <span style="font-size:0.55rem;letter-spacing:0.08em;">
        <a href="https://huggingface.co/spaces/Manthan6683/Alphinance"
           style="color:#333;text-decoration:none;">Live Dashboard</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/ManthanNimodiya/Alphinance"
           style="color:#333;text-decoration:none;">GitHub</a>
    </span>
</div>
""", unsafe_allow_html=True)
