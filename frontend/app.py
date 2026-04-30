import sys
import os
# path fix so Python can find backend/model.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
from backend.model import predict_intervals

st.set_page_config(
    page_title="Alphinance | BTC Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_data(ttl=300)
def fetch_ohlcv(limit=70) -> pd.DataFrame:
    df = yf.download("BTC-USD", period="10d", interval="1h",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].tail(limit).reset_index()
    df.columns = ["open_time", "close"]
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"]).dt.tz_localize(None)
    return df

@st.cache_data(ttl=60)
def fetch_ticker() -> float:
    df = yf.download("BTC-USD", period="1d", interval="1m",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return float(df["Close"].iloc[-1])

@st.cache_data(ttl=3600)
def run_backtest():
    # 1. load the csv
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "data", "btc_1h.csv")
    closes = pd.read_csv(path)["close"]
    # 2. set alpha = 0.05
    alpha = 0.05
    # 3. results = []
    results = []
    # 4. loop t from 21 to 719:
    #       call predict_intervals(closes, t) → get L, U
    #       get actual = closes.iloc[t]
    #       append {"L": L, "U": U, "actual": actual}
    for t in range(21,720):
        L,U =predict_intervals(closes,t)
        actual = closes.iloc[t]
        results.append({"L":L,"U":U,"actual":actual})
    # 5. convert to DataFrame
    df = pd.DataFrame(results)
    # 6. compute coverage, avg_width, winkler
    
    # Coverage
    inside = (df["actual"] >= df["L"]) & (df["actual"] <= df["U"])
    coverage = inside.mean()

    # avg_width
    avg_width=(df["U"]-df["L"]).mean()
    # winkler
    width = df["U"]-df["L"]

    winkler = np.where(
        df["actual"] < df["L"],
        width + (2/alpha)*(df["L"] - df["actual"]),
        np.where(
            df["actual"] > df["U"],
            width + (2/alpha)*(df["actual"] - df["U"]),
            width
        )
    )

    winkler_score = np.mean(winkler)
    # 7. return all three

    return float(coverage), float(avg_width), float(winkler_score)





# Fetch all data
try:
    df = fetch_ohlcv(limit=70)
    current_price = fetch_ticker()
    closes_live = df["close"].reset_index(drop=True)
    L, U = predict_intervals(closes_live, t=len(closes_live))
    last_updated = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.stop()

width_val = U - L
pct_up = ((U - current_price) / current_price) * 100
pct_dn = ((current_price - L) / current_price) * 100
price_up = current_price >= closes_live.iloc[-2]


col_title, col_refresh = st.columns([6, 1])

with col_title:
    st.markdown("## ₿ Alphinance")
    st.caption(f"BTC/USDT · 1-Hour Prediction · {last_updated}")

with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↻ Refresh", use_container_width=True):
        fetch_ohlcv.clear()
        fetch_ticker.clear()
        st.rerun()

st.divider()



# Metrics Section
st.markdown("##### Live")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label="BTC Price",
        value=f"${current_price:,.2f}",
        delta="▲ Up" if price_up else "▼ Down"
    )
with c2:
    st.metric(label="Next Hour Low",  value=f"${L:,.2f}", delta=f"-{pct_dn:.2f}%")
with c3:
    st.metric(label="Next Hour High", value=f"${U:,.2f}", delta=f"+{pct_up:.2f}%")
with c4:
    st.metric(label="Range Width",    value=f"${width_val:,.2f}", delta="95% confidence")



st.divider()

# prepare chart data
display_df = df.tail(50).copy().reset_index(drop=True)
next_time   = display_df["open_time"].iloc[-1] + pd.Timedelta(hours=1)
pred_x0     = next_time - pd.Timedelta(minutes=20)
pred_x1     = next_time + pd.Timedelta(minutes=20)

fig = go.Figure()

# price line
fig.add_trace(go.Scatter(
    x=display_df["open_time"],
    y=display_df["close"],
    mode="lines",
    name="BTC/USDT",
    line=dict(color="#111111", width=2),
))

# prediction shaded zone
fig.add_vrect(x0=pred_x0, x1=pred_x1,
    fillcolor="rgba(59,130,246,0.1)", line_width=0)

# upper bound
fig.add_shape(type="line",
    x0=pred_x0, x1=pred_x1, y0=U, y1=U,
    line=dict(color="green", width=2, dash="dot"))

# lower bound
fig.add_shape(type="line",
    x0=pred_x0, x1=pred_x1, y0=L, y1=L,
    line=dict(color="red", width=2, dash="dot"))

fig.update_layout(
    height=400,
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    margin=dict(l=8, r=8, t=32, b=8),
    xaxis=dict(gridcolor="#f0f0f0"),
    yaxis=dict(gridcolor="#f0f0f0", tickprefix="$", tickformat=",.0f"),
)

st.plotly_chart(fig, use_container_width=True)


# backtest metrics
st.divider()
st.markdown("##### 30-Day Backtest · 720 Hourly Bars")

with st.spinner("Running backtest..."):
    coverage, avg_width_bt, winkler_score = run_backtest()

b1, b2, b3 = st.columns(3)

with b1:
    st.metric(
        label="Coverage",
        value=f"{coverage*100:.2f}%",
        delta="✓ Meets target" if coverage >= 0.95 else "✗ Below target"
    )
with b2:
    st.metric(
        label="Avg Range Width",
        value=f"${avg_width_bt:,.2f}",
        delta="Lower is better"
    )
with b3:
    st.metric(
        label="Winkler Score",
        value=f"{winkler_score:,.2f}",
        delta="Lower is better"
    )

st.caption(f"Alphinance · AlphaI × Polaris Challenge · {last_updated}")