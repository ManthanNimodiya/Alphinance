import requests
import pandas as pd
import numpy as np
URL = "https://api.binance.com/api/v3/klines"

params = {
    "symbol":"BTCUSDT",
    "interval":"1h",
    "limit":720
}

response = requests.get(URL, params=params)
response.raise_for_status()
raw=response.json()

df = pd.DataFrame(raw)
df = df[[0,4]]
df.columns = ["open_time", "close"]
df["close"]= df["close"].astype(float)
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df["log_return"] = np.log(df["close"] / df["close"].shift(1))

df.to_csv("backend/data/btc_1h.csv", index=False)

print(df.head(10))
print(f"Total rows: {len(df)}")

