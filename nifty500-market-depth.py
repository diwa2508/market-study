import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date
import requests
import pandas as pd
from bs4 import BeautifulSoup

#stock list can be downloaded here
#https://www.nseindia.com/products-services/indices-nifty500-index


# 1️⃣ Read Excel file
file_path = "ind_nifty500list.csv"   # <-- change this to your actual Excel filename
df = pd.read_csv(file_path)

# 2️⃣ Extract the 'Symbol' column
symbols = df['Symbol'].dropna().astype(str).tolist()

# 3️⃣ Optionally add ".NS" suffix for NSE tickers (for yfinance etc.)
symbols = [s + ".NS" for s in symbols]

# 4️⃣ Save to text file
with open("nse_tickers.txt", "w") as f:
    for sym in symbols:
        f.write(sym + "\n")

print(f"✅ Saved {len(symbols)} tickers to nse_tickers.txt")


# -------------------------------
# 1️⃣ Setup - Choose Universe
# -------------------------------

with open("nse_tickers.txt", "r") as f:
    tickers = [line.strip() for line in f if line.strip()]

remove_list = ['DUMMYTATAM.NS', 'DUMMYSKFIN.NS', 'DUMMYDBRLT.NS']

tickers = [t for t in tickers if t not in remove_list]
#tickers = [
#    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
#    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "LT.NS"
#]  # <- replace with your Nifty 100/200/500 list

nifty_symbol = "^NSEI"

# -------------------------------
# 2️⃣ Download last 6 months data
# -------------------------------
data = yf.download(tickers, period="6mo", interval="1d")['Close']
nifty = yf.download(nifty_symbol, period="6mo", interval="1d")['Close']

# -------------------------------
# 3️⃣ Compute Moving Averages
# -------------------------------
dma_5 = data.rolling(5).mean()
dma_10 = data.rolling(10).mean()
dma_20 = data.rolling(20).mean()
dma_40 = data.rolling(40).mean()
dma_50 = data.rolling(50).mean()

# -------------------------------
# 4️⃣ Daily % Change
# -------------------------------
pct_change = data.pct_change() * 100

# -------------------------------
# 5️⃣ Breadth Calculations
# -------------------------------
summary = pd.DataFrame(index=data.index)

summary["Advances"] = (pct_change > 0).sum(axis=1)
summary["Declines"] = (pct_change < 0).sum(axis=1)

summary["Up >4% (Daily)"] = (pct_change > 4).sum(axis=1)
summary["Down >4% (Daily)"] = (pct_change < -4).sum(axis=1)

# --- Monthly view: 21 trading days approx
monthly_change = data.pct_change(21) * 100
summary["Up >25% (Monthly)"] = (monthly_change > 25).sum(axis=1)
summary["Down >25% (Monthly)"] = (monthly_change < -25).sum(axis=1)
summary["Up >5% (Monthly)"] = (monthly_change > 5).sum(axis=1)
summary["Down >5% (Monthly)"] = (monthly_change < -5).sum(axis=1)

# -------------------------------
# 6️⃣ % Above Key DMAs
# -------------------------------
summary["% Above 10 DMA"] = (data > dma_10).sum(axis=1) / len(tickers) * 100
summary["% Above 20 DMA"] = (data > dma_20).sum(axis=1) / len(tickers) * 100
summary["% Above 40 DMA"] = (data > dma_40).sum(axis=1) / len(tickers) * 100
summary["% Above 50 DMA"] = (data > dma_50).sum(axis=1) / len(tickers) * 100

# -------------------------------
# 7️⃣ Add Nifty Data
# -------------------------------
summary["Nifty"] = nifty
summary["Nifty Chg %"] = nifty.pct_change() * 100

# -------------------------------
# 8️⃣ Optional: Add Weekday Column
# -------------------------------
summary["Day"] = summary.index.day_name()
summary.reset_index(inplace=True)
summary.rename(columns={'Date': 'Date'}, inplace=True)

# -------------------------------
# 🧭  NEW: 10 DMA > 40 DMA Breadth
# -------------------------------
summary["% 10DMA > 40DMA"] = ((dma_10 > dma_40).sum(axis=1) / len(tickers)) * 100

# Optional: Add 20>50 DMA too
summary["% 20DMA > 50DMA"] = ((dma_20 > dma_50).sum(axis=1) / len(tickers)) * 100

# -------------------------------
# 9️⃣ Reorder Columns
# -------------------------------
cols = [
    "Date", "Day", "Advances", "Declines",
    "Up >4% (Daily)", "Down >4% (Daily)",
    "Up >25% (Monthly)", "Down >25% (Monthly)",
    "Up >5% (Monthly)", "Down >5% (Monthly)",
    "% Above 10 DMA", "% Above 20 DMA",
    "% Above 40 DMA", "% Above 50 DMA",
    "% 10DMA > 40DMA", "% 20DMA > 50DMA",
    "Nifty", "Nifty Chg %"
]
summary = summary[cols]

# -------------------------------
# 🔟 Export to Excel
# -------------------------------
summary.to_excel("Market_Breadth_Monitor.xlsx", index=False)

print("✅ Market Breadth Monitor saved to Market_Breadth_Monitor.xlsx")
