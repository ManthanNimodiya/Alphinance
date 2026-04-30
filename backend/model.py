import numpy as np
import pandas as pd
from scipy import stats
def predict_intervals(closes, t, alpha=0.05):
    # step 1 - slice closes till the index t
    closes = closes[:t]
    # Step 2 — Compute log returns using the log(p(t)/p(t-1))
    log_return = np.log(closes/closes.shift(1))
    # Step 3 — Compute μ and σ for last 20 values
    last_20 = log_return[-20:]
    mean = np.mean(last_20)
    std = np.std(last_20)
    # Step 4 — Build the interval in log-return space
    lower, upper = stats.t.interval(confidence=0.95, df=4.2, loc=mean, scale=std)
    # Step 5 — Convert back to price space
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