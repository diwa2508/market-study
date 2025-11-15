from nsetools import Nse
import pandas as pd
import requests
import datetime
import os

nse = Nse()

def get_delivery_data():
    today = datetime.date.today()
    date_str = today.strftime("%Y%m%d")
    url = f"https://www1.nseindia.com/archives/equities/mto/MTO_{date_str}.DAT"

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)

    lines = r.text.split("\n")

    rows = []
    for ln in lines[3:]:  # skip header lines
        parts = ln.split(",")
        if len(parts) >= 5:
            symbol = parts[1].strip()
            deliverable_qty = int(parts[2].strip())
            traded_qty = int(parts[3].strip())
            delivery_pct = float(parts[4].strip())
            rows.append([symbol, deliverable_qty, traded_qty, delivery_pct])

    df = pd.DataFrame(rows, columns=["symbol", "deliverable_qty", "traded_qty", "delivery_pct"])
    return df
def get_price_data(symbol):
    q = nse.get_quote(symbol)
    return {
        "symbol": symbol,
        "close": q["lastPrice"],
        "open": q["open"],
        "dayHigh": q["dayHigh"],
        "dayLow": q["dayLow"],
        "volume": q["totalTradedVolume"]
    }
def save_daily(symbol):
    # delivery data for all stocks
    del_df = get_delivery_data()
    
    # today's price for requested symbol
    price = get_price_data(symbol)
    price_df = pd.DataFrame([price])

    # merge both
    merged = price_df.merge(del_df, on="symbol", how="left")

    file = f"{symbol}_daily_data.csv"

    # append to CSV
    if os.path.exists(file):
        old = pd.read_csv(file)
        merged = pd.concat([old, merged], ignore_index=True)

    merged.to_csv(file, index=False)
    return merged
def add_ad_signal(df):
    df["price_change"] = df["close"].diff()
    df["ad_score"] = df["price_change"] * df["delivery_pct"]

    df["signal"] = df["ad_score"].apply(
        lambda x: "ACCUMULATION" if x > 1
        else "DISTRIBUTION" if x < -1
        else "NEUTRAL"
    )
    return df

df = save_daily("HDFCBANK")
df = add_ad_signal(df)
df.to_csv("HDFCBANK_daily_data.csv", index=False)
print(df.tail())
