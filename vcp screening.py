#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-28T16:50:13.348Z
"""

from nsepython import nse_eq_symbols
import yfinance as yf
import pandas as pd
import numpy as np
import nsepython as nse

file_path = "ind_nifty500list.csv"   # <-- change this to your actual Excel filename
df = pd.read_csv(file_path)

# 2️⃣ Extract the 'Symbol' column
symbols = df['Symbol'].dropna().astype(str).tolist()

# --- Step 1: Get NSE symbols ---
#symbols = nse_eq_symbols()[:50]  # limit to first 50 for speed; expand later
print(symbols)
# --- Step 2: Function to detect VCP on weekly timeframe ---
def detect_vcp(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period="5y", interval="1wk")
        if df.empty:
            return None

        df['volatility'] = df['High'] - df['Low']
        df['vol_ratio'] = df['volatility'] / df['Close']

        # Measure 3 contractions (approx)
        last_3 = df['vol_ratio'].tail(10).values

        if len(last_3) < 6:
            return None

        # Check if volatility contractions are reducing
        c1 = np.mean(last_3[-6:-4])
        c2 = np.mean(last_3[-4:-2])
        c3 = np.mean(last_3[-2:])

        # Volatility contraction condition
        if c1 > c2 > c3 and (c1 - c3) / c1 > 0.25:
            # Optional: price near resistance
            recent_close = df['Close'].iloc[-1]
            recent_high = df['High'].max()
            proximity = (recent_high - recent_close) / recent_high

            if proximity < 0.05:  # within 5% of high
                return {
                    'Symbol': symbol,
                    'Close': round(recent_close, 2),
                    'Volatility Drop': round((c1 - c3) / c1 * 100, 1)
                }
    except Exception:
        return None

# --- Step 3: Run screen ---
results = []
for sym in symbols:
    res = detect_vcp(sym)
    if res:
        results.append(res)

vcp_df = pd.DataFrame(results)
print(vcp_df)


# --- Inverse VCP Screener (Weekly) ---
from nsepython import nsefetch
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --------------------------
# Step 2: Define helper function
# --------------------------
def get_weekly_data(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period="5y", interval="1wk")
        return df if len(df) > 0 else None
    except:
        return None

# --------------------------
# Step 3: Define inverse VCP detector
# --------------------------
def is_inverse_vcp(df):
    if df is None or len(df) < 6:
        return False

    # Calculate ATR for volatility contraction
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=3).mean()

    # Moving averages
    df['20DMA'] = df['Close'].rolling(20).mean()
    df['50DMA'] = df['Close'].rolling(50).mean()

    # Volume contraction ratio
    vol_contraction = df['Volume'].iloc[-3:].mean() < df['Volume'].iloc[-6:-3].mean()

    # Lower highs
    highs = df['High'].iloc[-5:]
    lower_highs = all(x > y for x, y in zip(highs, highs[1:]))

    # ATR contraction
    atr_contraction = df['ATR'].iloc[-3:].mean() < df['ATR'].iloc[-6:-3].mean()

    # Trend weakness
    below_ma = df['Close'].iloc[-1] < df['20DMA'].iloc[-1] < df['50DMA'].iloc[-1]

    if lower_highs and atr_contraction and below_ma and vol_contraction:
        return True
    return False

# --------------------------
# Step 4: Scan all stocks
# --------------------------
inverse_vcp_list = []
for sym in symbols:
    df = get_weekly_data(sym)
    if is_inverse_vcp(df):
        inverse_vcp_list.append(sym)

# --------------------------
# Step 5: Output
# --------------------------
print("🧨 Inverse VCP Candidates (Weekly):")
print(inverse_vcp_list)