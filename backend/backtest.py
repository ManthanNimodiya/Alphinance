import pandas as pd
from model import predict_intervals
import numpy as np

df_data = pd.read_csv("backend/data/btc_1h.csv")
closes = df_data["close"]
alpha = 0.05

results = []
for t in range(21,720):
    L,U = predict_intervals(closes, t)
    actual = closes.iloc[t]
    results.append({
        "t": t,
        "L": L,
        "U": U,
        "actual": actual
    })

df_results = pd.DataFrame(results)

# Coverage
inside = (df_results["actual"] >= df_results["L"]) & (df_results["actual"] <= df_results["U"])
coverage = inside.mean()
print(coverage)

# Avg_Width
avg_width = (df_results["U"] - df_results["L"]).mean()
print(avg_width)

# Winkler Score
width = df_results["U"] - df_results["L"]

winkler = np.where(
    df_results["actual"] < df_results["L"],
    width + (2/alpha) * (df_results["L"] - df_results["actual"]),
    np.where(
        df_results["actual"] > df_results["U"],
        width + (2/alpha) * (df_results["actual"] - df_results["U"]),
        width
    )
)

winkler_score = np.mean(winkler)
print(winkler_score)
