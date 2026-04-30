# ₿ Alphinance — Full Project Notes
> AlphaI × Polaris Build Challenge  
> Built: April 29, 2026  
> Stack: Python · Pandas · Numpy · Scipy · Streamlit · Plotly

---

## What Are We Building?

A system that predicts **where BTC price will land one hour from now** — not a single price, but a **range** `[L, U]`.

The challenge scores you on two things simultaneously:
- **Coverage** — does the real price fall inside your range? (target ≥ 95%)
- **Width** — how narrow is the range? (narrower = better)

These two goals fight each other. Wider range = easier to be right. Narrow range = harder. The **Winkler Score** combines both into one number. Lower is better.

---

## The Winkler Score Formula

For each prediction:

```
width = U - L

if actual is INSIDE [L, U]:
    score = width                                      ← you still pay for being wide

if actual is BELOW L:
    score = width + (2 / alpha) * (L - actual)        ← width + 40x the miss distance

if actual is ABOVE U:
    score = width + (2 / alpha) * (actual - U)        ← width + 40x the miss distance
```

Final score = **mean of all individual scores** across all predictions.

### Why 40x penalty?
> **Q (Manthan): Why are we multiplying by 2/alpha? Why 2? Why distance? To whom are we paying the penalty?**

Nobody literally. It's a scoring metric, like marks in an exam. Lower = better rank.

- `alpha = 0.05` means you claim 95% confidence
- `2 / 0.05 = 40` — the more confident you claimed to be, the harder you're punished for being wrong
- The `2` is a convention from Winkler's 1972 paper — ensures penalty always exceeds width alone
- `distance` = how far outside the range the actual price landed

Even when correct, you pay `width` — because being right with a $10,000 range is trivial. Being right with a $500 range is impressive.

---

## Core Concepts — From Zero

### 1. Why Log Returns (not raw prices)?

> **Q (Manthan): What do we compute first from prices, and why? Is it just the randomness?**

Close — raw prices are **non-stationary**. Their mean and variance change over time. BTC was $20k in 2020, $60k in 2024. Models assume statistical properties are stable — raw prices violate that.

**Log returns** fix this:
```
r_t = ln(P_t / P_{t-1})
```

> **Q (Manthan): Why log and not simple returns? Or even absolute price differences? In the example (100→200→100), all methods show zero profit.**

You're right for that example. But absolute changes break across different price levels:

```
2020: BTC moves $100 → $110   → absolute change = +$10  (10% move)
2024: BTC moves $60,000 → $60,010 → absolute change = +$10  (0.017% move)
```

Same absolute change. Completely different events. Your model would treat them identically. **Wrong.**

Log returns fix this — they're **scale-invariant**:
```
ln(110/100)   = 0.0953   ← correctly shows 9.5% move
ln(60010/60000) = 0.00017 ← correctly shows 0.017% move
```

Also, log returns are **additive across time**:
```
3-hour log return = hour1 + hour2 + hour3   ✓
```
Simple and absolute returns don't add cleanly.

---

### 2. What is a Model?

> **Q (Manthan): What is a model? Go deeper into basics.**

You flip a coin 10 times. You get H H T H T T H H T H. You want to predict the next flip. You can't know for sure. But you say — "I believe this coin has 60% chance of heads." That belief, expressed as math, is your model.

**A model = a set of assumptions about how your data behaves, written as math.**

In our case:
- Data = BTC log returns
- Assumption = "returns follow a Student-t shaped distribution, centered at mean μ, with spread σ"
- That assumption lets us say: "95% of future returns will fall within this range"

No training. No neural networks. Just: **"I assume the data has this shape. I use that shape to predict."**

---

### 3. Mean and Standard Deviation

> **Q (Manthan): What are the two numbers that describe a distribution? (Guessed: mean and spread)**

Exactly right.

**Mean (μ)** — the center. Average of all values.
```
returns: 0.01, -0.02, 0.03, -0.01, 0.02
mean = (0.01 - 0.02 + 0.03 - 0.01 + 0.02) / 5 = 0.006
→ on average, BTC moved up 0.6% per hour
```

**Standard Deviation (σ)** — the spread. How far values typically wander from the mean.
```
Small σ = calm market
Large σ = wild market
```

Your prediction interval is literally:
```
Lower = μ - k*σ
Upper = μ + k*σ
```
Where `k` controls the width. For 95% confidence, `k ≈ 2`.

---

### 4. Rolling Window

> **Q (Manthan): Should we use all 720 hours or last 20 hours for μ and σ? (Guessed: 720 because more data = more accuracy)**

Opposite. Using all 720 hours brings in stale data. If BTC was calm for 600 hours then wild for 120 hours — the 600 calm hours drag σ down, making your interval too narrow, missing the recent wild moves.

**Rolling window** = always use the most recent N values. Window slides forward as time moves:
```
Predict hour 25 → use hours 5–24   (last 20)
Predict hour 26 → use hours 6–25   (last 20)
Predict hour 27 → use hours 7–26   (last 20)
```

N = 20 — not too stale, not too noisy.

---

### 5. Volatility Clustering

> **Q (Manthan): What is the behavior called when the model adapts to the current situation? (Guessed: "befitting according to situation")**

Close. The exact term is **volatility clustering**.

Volatility doesn't appear randomly. Wild hours cluster together. Calm hours cluster together.

```
Calm: 0.001, 0.002, -0.001, 0.003  ← small moves follow small moves
Wild: 0.05, -0.08, 0.06, -0.04     ← big moves follow big moves
```

Your rolling window captures this automatically. When wild period starts → last 20 returns fill with large values → σ grows → interval widens. When calm returns → σ shrinks → interval narrows. You didn't program this — it **emerges** from the rolling window.

---

### 6. Student-t Distribution

> **Q (Manthan): What is Student-t? What is it really?**

Start with normal distribution — the bell curve. It assumes you know the true σ of the population.

But you're computing σ from only 20 data points. That estimate could be wrong. You're uncertain about your own uncertainty. Normal doesn't account for that. It's overconfident.

**Student-t** adds a knob — **degrees of freedom (df / ν)**:
```
df = 1    → extremely fat tails (almost anything can happen)
df = 5    → moderately fat tails (good for BTC)
df = 30   → almost identical to normal
df = ∞    → exactly normal
```

Fatter tails = more probability given to extreme events.

> **Q (Manthan): So Student-t has higher k value compared to normal?**

Exactly:
```
Normal distribution:    k = 1.96  for 95%
Student-t (df=5):       k = 2.57  for 95%
```

Student-t gives a wider interval — which is what you need for a market that crashes and pumps more than a normal curve expects.

> **Q (Manthan): Why scipy? Was it made for this problem?**

No. Scipy is a general scientific computing library for physicists, engineers, mathematicians. `scipy.stats` has every major probability distribution built in. Computing Student-t intervals from scratch requires gamma functions and complex integrals. Scipy already solved all of that. You just call `.interval()`.

---

### 7. Alpha

> **Q (Manthan): What is alpha? Why is it a default value in the function?**

Alpha = the error rate you're willing to accept.
```
confidence = 95%
alpha      = 5% = 0.05
confidence = 1 - alpha
```

Alpha = fraction of the time you're okay with being wrong. Default value `0.05` means unless told otherwise, assume 95% confidence.

`scipy.stats.t.interval(confidence, df, loc, scale)`:
- `confidence` → 0.95
- `df` → degrees of freedom (controls tail fatness)
- `loc` → μ (mean of last 20 log returns)
- `scale` → σ (std of last 20 log returns)

---

## Code — Full Walkthrough

### `backend/fetch_data.py`

```python
import requests
import pandas as pd
import numpy as np

URL = "https://api.binance.com/api/v3/klines"

params = {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "limit": 720        # 30 days of hourly data
}

response = requests.get(URL, params=params)

# Why raise_for_status()?
# Q (Manthan): What can go wrong with an API call?
# → Binance could be down (500), your request could be wrong (400), internet could drop
# → raise_for_status() crashes loudly if status != 200
# → "Fail loudly, never silently"
response.raise_for_status()

raw = response.json()
# raw is now a list of 720 lists. Each inner list has 12 elements.
# We only need index [0] = timestamp, [4] = close price.

df = pd.DataFrame(raw)
df = df[[0, 4]]
df.columns = ["open_time", "close"]

# Why astype(float)?
# Q (Manthan): What happens if you do math on a string column in pandas?
# → Tried: pd.Series(["63200.00"]) + 1 → Error: cannot concatenate
# → JSON comes in as strings. Pandas doesn't auto-convert. Must be explicit.
df["close"] = df["close"].astype(float)

# Why pd.to_datetime(..., unit="ms")?
# Binance gives timestamp as 1714521600000 (milliseconds since Jan 1, 1970)
# Converting to 2024-05-01 00:00:00 — human readable and sortable
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

# Computing log returns: r_t = ln(P_t / P_{t-1})
# Why .shift(1)?
# Q (Manthan): What does .shift(1) do?
# → Shifts the entire column DOWN by 1. Index 0 becomes NaN. Length stays the same.
# → Gives you P_{t-1} at the same row as P_t — so you can divide row by row.
# Q (Manthan): Why not just use df["close"].shift(1)[last_index]?
# → Because you're not getting one value. You're computing 720 log returns at once.
# → Pandas does this row by row across the entire column (vectorization). No loop needed.
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
# Row 0 = NaN (no previous price). Everything else = small decimals near zero.

# Why index=False?
# Default pandas behaviour adds row numbers (0,1,2...) as a column in CSV. Not needed.
df.to_csv("backend/data/btc_1h.csv", index=False)

print(df.head(10))
print(f"Total rows: {len(df)}")
```

---

### `backend/model.py`

```python
import numpy as np
import pandas as pd
from scipy import stats

def predict_intervals(closes, t, alpha=0.05):
    # STEP 1 — Slice. Only use data before index t.
    # This is the no-leakage guarantee.
    # If you used closes[t:] you'd be peeking at the future. That's cheating.
    closes = closes[:t]

    # STEP 2 — Compute log returns from the slice
    log_return = np.log(closes / closes.shift(1))

    # STEP 3 — Take last 20 log returns. Compute μ and σ.
    # Q (Manthan): Why last 20 and not all data?
    # → Using all data brings in stale information. Recent volatility is the best
    #   predictor of near-future volatility. 20 = recent enough, stable enough.
    last_20 = log_return[-20:]
    mean = np.mean(last_20)   # μ — center of the distribution
    std  = np.std(last_20)    # σ — spread of the distribution

    # STEP 4 — Student-t interval in log-return space
    # Q (Manthan): Why df=4.2 specifically?
    # → Tuned by running backtest with combinations of df and window size.
    # → df=4.2, window=20 gave coverage=95.28% and lowest Winkler among passing runs.
    # → df controls tail fatness. Lower df = fatter tails = wider interval = better coverage
    #   but worse (higher) Winkler score. 4.2 is the sweet spot.
    lower, upper = stats.t.interval(
        confidence=0.95,   # 1 - alpha
        df=4.2,            # degrees of freedom — tail fatness knob
        loc=mean,          # center the interval at μ
        scale=std          # scale by σ
    )
    # lower and upper are log-return bounds, e.g. (-0.012, 0.014)

    # STEP 5 — Convert from log-return space back to price space
    # Why closes.iloc[-1]?
    # After slicing closes[:t], the last element is P_{t-1} (previous price).
    # Log return is relative to previous price, so:
    # L = P_{t-1} * e^(lower_log_return)
    # U = P_{t-1} * e^(upper_log_return)
    L = closes.iloc[-1] * np.exp(lower)
    U = closes.iloc[-1] * np.exp(upper)

    return L, U


if __name__ == "__main__":
    df = pd.read_csv("backend/data/btc_1h.csv")
    closes = df["close"]
    L, U = predict_intervals(closes, t=100)
    print(f"Lower: ${L:,.2f}")
    print(f"Upper: ${U:,.2f}")
    print(f"Width: ${U-L:,.2f}")
```

---

### `backend/backtest.py`

```python
import pandas as pd
from model import predict_intervals
import numpy as np

df_data = pd.read_csv("backend/data/btc_1h.csv")
closes = df_data["close"]
alpha = 0.05

# Loop from t=21 to t=719
# Why start at 21? Because we need at least 20 data points for the rolling window.
# At t=21, closes[:21] gives us 21 prices → 20 log returns → last 20 = all of them.
results = []
for t in range(21, 720):
    L, U = predict_intervals(closes, t)
    actual = closes.iloc[t]      # the real price at hour t — what we were predicting
    results.append({
        "t": t,
        "L": L,
        "U": U,
        "actual": actual
    })

# Q (Manthan): Why list of dicts, not append to DataFrame directly?
# → Appending rows to DataFrame inside a loop is very slow.
# → Build a list first, convert once at the end. Much faster.
df_results = pd.DataFrame(results)

# ── Coverage ─────────────────────────────────────────────────────
# Fraction of hours where actual fell inside [L, U]
# True/False per row. True = 1, False = 0. Mean = fraction correct.
inside = (df_results["actual"] >= df_results["L"]) & (df_results["actual"] <= df_results["U"])
coverage = inside.mean()
print(coverage)   # target ≥ 0.95

# ── Avg Width ────────────────────────────────────────────────────
# Mean of (U - L) across all predictions. Lower = tighter model.
avg_width = (df_results["U"] - df_results["L"]).mean()
print(avg_width)

# ── Winkler Score ────────────────────────────────────────────────
# Implemented using np.where — applies the formula row by row across the entire DataFrame.
# np.where(condition, if_true, if_false) — nested for three cases.
width = df_results["U"] - df_results["L"]

winkler = np.where(
    df_results["actual"] < df_results["L"],
    width + (2 / alpha) * (df_results["L"] - df_results["actual"]),   # below lower bound
    np.where(
        df_results["actual"] > df_results["U"],
        width + (2 / alpha) * (df_results["actual"] - df_results["U"]),  # above upper bound
        width   # inside — just pay the width
    )
)

winkler_score = np.mean(winkler)
print(winkler_score)
```

---

## Tuning Results

Ran the backtest with different combinations of `df` and rolling window `N`:

| df  | Window | Coverage | Avg Width | Winkler |
|-----|--------|----------|-----------|---------|
| 5.0 | 20     | 94.56%   | $1,386    | 1801    |
| 4.0 | 20     | 95.57%   | $1,497    | 1826    |
| 5.0 | 30     | 95.99%   | —         | 1854    |
| 4.0 | 30     | 96.28%   | $1,540    | 1900    |
| 4.5 | 20     | 94.85%   | $1,434    | 1809    |
| 4.2 | 20     | **95.28%** | $1,470  | **1818** ✅ |

**Winner: `df=4.2, window=20`**  
Coverage ≥ 95% ✅ + lowest Winkler among all passing runs.

**Pattern observed:**
- Lower df → fatter tails → wider interval → better coverage, worse Winkler
- Larger window → more stable σ → wider interval → better coverage, worse Winkler
- Every direction that helps coverage hurts Winkler — fundamental tradeoff

---

## Project Structure

```
Alphinance/
├── backend/
│   ├── data/
│   │   └── btc_1h.csv          ← 720 hourly BTC bars from Binance
│   ├── fetch_data.py            ← hits Binance API, saves CSV
│   ├── model.py                 ← predict_intervals() function
│   ├── backtest.py              ← 700-iteration backtest loop + 3 metrics
│   └── utils.py                 ← (unused)
├── requirements.txt
└── .gitignore
```

---

## APIs Used

```
# Historical data (no API key needed)
GET https://api.binance.com/api/v3/klines
    ?symbol=BTCUSDT&interval=1h&limit=720

# Live price (no API key needed)
GET https://api.binance.com/api/v3/ticker/price
    ?symbol=BTCUSDT
```

Response from klines: list of 12-element arrays.  
You only need index `[0]` (timestamp) and `[4]` (close price).

---

## Libraries

| Library | Purpose |
|---|---|
| `requests` | HTTP calls to Binance API |
| `pandas` | DataFrame — store, slice, compute on tabular data |
| `numpy` | Log returns, rolling math, np.where for Winkler |
| `scipy.stats` | `t.interval()` — Student-t prediction intervals |
| `streamlit` | Dashboard UI — Python renders as a web page |
| `plotly` | Interactive charts inside Streamlit |

---

---

*Alphinance · AlphaI × Polaris Challenge · Notes from April 29, 2026*
