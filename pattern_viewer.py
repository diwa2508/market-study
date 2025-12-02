import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from collections import Counter
import plotly.graph_objects as go
import os

# ==========================================================
# 1. Load Nifty Data
# ==========================================================
@st.cache_data
def load_nifty():
    if os.path.exists("nifty_daily.csv"):
        df = pd.read_csv("nifty_daily.csv", index_col=0, parse_dates=True)
        print("Loaded local nifty_daily.csv")
    else:
        print("Fetching daily Nifty (10 years)...")
        df = yf.download("^NSEI", period="10y", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # Keep: Close, High, Low, Open, Volume
        df.to_csv("nifty_daily.csv")
        print("Saved nifty_daily.csv")
    df.index = pd.to_datetime(df.index)
    return df

df = load_nifty()


# ==========================================================
# 2. Candle Encoding (same as before)
# ==========================================================
def encode_candle(idx):
    o, h, l, c = df.iloc[idx][['Open','High','Low','Close']]
    prev_c = df.iloc[idx-1]['Close'] if idx > 0 else c
    
    ret = (c - o) / o  
    ret_bucket = int(ret * 100)

    direction = "B" if c > o else "R"

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body = abs(c - o)
    wick_ratio = "L" if upper_wick < body * 0.5 else "H"

    gap = (o - prev_c) / prev_c
    gap_dir = "G+" if gap > 0.004 else ("G-" if gap < -0.004 else "NG")

    date = df.index[idx]
    dow = date.weekday()
    wom = (date.day - 1) // 7
    expiry_week = 1 if (wom == 3 and dow >= 2) else 0

    return (direction, ret_bucket, wick_ratio, gap_dir, dow, date.month, wom, expiry_week)


# ==========================================================
# 3. Extract Patterns (min=2, max=6 candles)
# ==========================================================
@st.cache_data
def extract_patterns(min_len=2, max_len=6):
    encoded = [encode_candle(i) for i in range(len(df))]
    all_patterns = Counter()
    pattern_positions = {}

    for L in range(min_len, max_len+1):
        for i in range(L, len(encoded)):
            pattern = tuple(encoded[i-L:i])
            all_patterns[pattern] += 1
            pattern_positions.setdefault(pattern, []).append(i)

    return all_patterns, pattern_positions

patterns, pattern_positions = extract_patterns()


# ==========================================================
# 4. Streamlit UI
# ==========================================================
st.title("🔍 NIFTY 50 Candle Pattern Explorer (2–6 Candle Patterns)")
st.write("Select a pattern and visualize all occurrences on the chart.")

# Convert patterns to readable strings
pattern_list = [(str(pat), pat) for pat in patterns.keys()]
pattern_list_sorted = sorted(pattern_list, key=lambda x: patterns[x[1]], reverse=True)

pattern_label = st.selectbox(
    "Choose a Pattern (sorted by frequency):",
    pattern_list_sorted,
    format_func=lambda x: f"{x[0][:150]}...   | Occ: {patterns[x[1]]}"
)

selected_pattern = pattern_label[1]
positions = pattern_positions[selected_pattern]


# ==========================================================
# 5. Plot each pattern occurrence
# ==========================================================
st.subheader(f"Pattern Details")
st.write(f"**Occurrences:** {patterns[selected_pattern]}")
st.write(f"**Pattern Length:** {len(selected_pattern)} candles")

st.divider()

# Choose occurrence index
occ_idx = st.slider(
    "Select Occurrence to View:",
    0, len(positions)-1, 0
)

index = positions[occ_idx]

# Extract pattern window
window_start = index - len(selected_pattern)
window_end = index

pattern_df = df.iloc[window_start:window_end]

st.write(f"Showing occurrence **{occ_idx+1} / {len(positions)}**")
st.write(f"From {pattern_df.index[0].date()} to {pattern_df.index[-1].date()}")

# ==========================================================
# Candlestick Plot
# ==========================================================
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=pattern_df.index,
    open=pattern_df['Open'],
    high=pattern_df['High'],
    low=pattern_df['Low'],
    close=pattern_df['Close'],
    name="Pattern"
))

fig.update_layout(
    title="Pattern Candlestick View",
    xaxis_rangeslider_visible=False,
    height=500
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# Highlight ALL occurrences on main chart
# ==========================================================
st.subheader("🗺 All Occurrences on Full Chart")

full_fig = go.Figure()

full_fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="NIFTY"
))

# Mark occurrences
for pos in positions:
    s = df.index[pos - len(selected_pattern)]
    e = df.index[pos - 1]
    full_fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.25, line_width=0)

full_fig.update_layout(height=600, title="All Pattern Occurrences")
st.plotly_chart(full_fig, use_container_width=True)

 #py -m streamlit run pattern_viewer.p
