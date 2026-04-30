---
title: Alphinance
emoji: ₿
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.43.0"
app_file: frontend/app.py
pinned: false
---

# ₿ Alphinance
> BTC/USDT 1-Hour Prediction Interval System · AlphaI × Polaris Challenge

## What It Does
Predicts the price **range** where BTC will land one hour ahead — not a single price, but a `[Low, High]` interval with 95% confidence. Built from scratch using statistical methods, no ML frameworks.

## Live Dashboard
> Deployed on Streamlit Community Cloud

Shows:
- Current BTC price (live)
- Next-hour predicted range `[L, U]`
- 50-bar chart with prediction band
- 30-day backtest performance metrics

## Results
| Metric | Value |
|---|---|
| Coverage | 95.28% ✅ |
| Avg Range Width | $1,470 |
| Winkler Score | 1818 |

## How It Works
1. Fetch 720 hourly BTC bars from Binance API
2. Compute log returns: `r_t = ln(P_t / P_{t-1})`
3. Estimate rolling mean (μ) and std (σ) over last 20 returns
4. Build 95% prediction interval using Student-t distribution (df=4.2)
5. Convert log-return bounds back to price space

## Key Concepts
- **Log returns** — scale-invariant, stationary, additive across time
- **Rolling window (N=20)** — captures volatility clustering
- **Student-t (df=4.2)** — fat tails for BTC's extreme moves
- **No data leakage** — predictions only use past data
- **Winkler Score** — penalises both being wrong and being wide

## Stack
| Layer | Tech |
|---|---|
| Data | Binance Public API |
| Model | Python · NumPy · SciPy |
| Backtest | Pandas · NumPy |
| Dashboard | Streamlit · Plotly |
| Deploy | Streamlit Community Cloud |

## Run Locally
```bash
git clone https://github.com/ManthanNimodiya/Alphinance.git
cd Alphinance
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python backend/fetch_data.py      # fetch fresh BTC data
streamlit run frontend/app.py     # launch dashboard
```

## Project Structure
```
Alphinance/
├── backend/
│   ├── data/btc_1h.csv       ← 720 hourly BTC bars
│   ├── fetch_data.py          ← Binance API → CSV
│   ├── model.py               ← predict_intervals()
│   └── backtest.py            ← coverage, width, Winkler
├── frontend/
│   └── app.py                 ← Streamlit dashboard
└── requirements.txt
```

---
*Built for AlphaI × Polaris Build Challenge*
