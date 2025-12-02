# lt_freefloat_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import sqlite3
from sqlalchemy import create_engine
import requests
from datetime import datetime
import plotly.graph_objects as go

# ---------------------------
# Config / Constants
# ---------------------------
DEFAULT_FREE_FLOAT_SHARES = 1.009e9  # 1.009 Billion (as provided)
DATA_DIR = "data"
DB_PATH = f"{DATA_DIR}/nse_data.db"

# Ensure data dir exists
import os
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------
# Helper functions
# ---------------------------
def smart_read_csv(file_like):
    """
    Read CSV robustly: handle BOM, stray quotes, and commas in numeric fields.
    Accepts file path, file-like or raw text.
    """
    # If bytes or str, wrap in StringIO
    if isinstance(file_like, (bytes, bytearray)):
        s = file_like.decode("utf-8", errors="ignore")
        bio = io.StringIO(s)
        df = pd.read_csv(bio, dtype=str)
    elif isinstance(file_like, str) and os.path.exists(file_like):
        # file path
        df = pd.read_csv(file_like, encoding="utf-8-sig", dtype=str)
    else:
        # assume file-like (uploaded)
        df = pd.read_csv(file_like, encoding="utf-8-sig", dtype=str)
    # Clean column names
    df.columns = df.columns.map(lambda c: str(c).replace('ï»¿','').replace('"','').strip())
    # Remove weird characters from column names
    df.columns = df.columns.str.replace('\n',' ').str.strip()
    # Remove commas inside numeric fields (thousands separators) across all cells
    df = df.replace({',': ''}, regex=True)
    return df

def normalize_nse_columns(df):
    """
    Map known NSE column names to canonical columns used by dashboard:
    Date, Symbol, Open, High, Low, Close, Volume, Deliverable, DeliveryPercent, Turnover, PrevClose
    """
    colmap = {}
    # Fix weird BOM or multibyte chars in column names
    df.columns = (
        df.columns
        .astype(str)
        .str.replace(r'ï»¿', '', regex=True)
        .str.replace('"', '', regex=True)
        .str.strip()
    )

    # Fix values with commas and currency symbols
    for c in df.columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("â¹", "", regex=False)
            .str.strip()
        )

    cols_lower = {c.lower():c for c in df.columns}
    def find(k):
        return cols_lower.get(k.lower())

    # heuristics
    possible_map = {
        "symbol": ["symbol","security code","scrip"],
        "date": ["date","timestamp","trade date"],
        "open": ["open price","open","open_price","openPrice"],
        "high": ["high price","high","high_price"],
        "low": ["low price","low","low_price"],
        "close": ["close price","close","close_price"],
        "p_close": ["Prev Close","close","close_price"],
        "last": ["last price","last"],
        "volume": ["total traded quantity","total traded qty","tottrdqty","totaltradedquantity","total traded quantity","total traded qty","total traded quantity"],
        "deliverable": ["deliverable qty","deliverable quantity","deliverable_qty","deliverableqty"],
        "deliverypercent": ["% dly qt to traded qty","% dly qt to traded qty","% dly qt to traded qty".lower(),"delivery %","deliverypercent"]
    }
    for target, names in possible_map.items():
        for n in names:
            if n in cols_lower:
                colmap[cols_lower[n]] = target.capitalize() if target!="deliverypercent" else "DeliveryPercent"
                break
    # fallback guesses by partial match
    for c in df.columns:
        cc = c.strip().lower()
        if "date" == cc:
            colmap[c] = "Date"
        elif "open price" in cc:
            colmap[c] = "Open"
        elif "high price" in cc:
            colmap[c] = "High"
        elif "low price" in cc:
            colmap[c] = "Low"
        elif "close price" in cc:
            colmap[c] = "Close"
        elif "last price" in cc:
            colmap[c] = "Last"
        elif "average price" in cc:
            colmap[c] = "Average"
        elif "total traded quantity" in cc or "tottrdqty" in cc:
            colmap[c] = "Volume"
        elif "turnover" in cc or "tottrdval" in cc:
            colmap[c] = "Turnover"
        elif "deliverable qty" in cc or "deliverable_qty" in cc:
            colmap[c] = "Deliverable"
        elif "% dly qt to traded qty" in cc or "delivery" in cc:
            colmap[c] = "DeliveryPercent"
        elif "prev close" in cc:
            colmap[c] = "PrevClose"
        elif "no. of trades" in cc or "tottrades" in cc:
            colmap[c] = "NoOfTrades"

    # rename
    df = df.rename(columns=colmap)
    df = df.replace({',': ''}, regex=True)

    # Ensure Date exists
    if "Date" not in df.columns:
        st.warning("No Date column detected — ensure your CSV contains a date column.")
        return df
    # Parse Date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    print(df.head())
    for c in df.columns:
        print(c, type(df[c]))
    # numeric conversions
    for c in ["Open","High","Low","Close","Volume","Deliverable","Turnover","DeliveryPercent"]:
        if c in df.columns:
            print(f" Converting {c} to numeric")
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If DeliveryPercent not present but both Deliverable & Volume present, compute it
    if "DeliveryPercent" not in df.columns and ("Deliverable" in df.columns and "Volume" in df.columns):
        df["DeliveryPercent"] = (df["Deliverable"] / df["Volume"]) * 100.0
    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def fetch_nse_securitywise(symbol, from_date, to_date):
    """
    Fetch using the security-wise API URL pattern you provided. Returns raw CSV text or raises.
    Dates must be dd-mm-YYYY.
    """
    base = "https://seindia.com/api/historicalOR/generateSecurityWiseHistoricalData"
    params = f"?from={from_date}&to={to_date}&symbol={symbol}&type=priceVolumeDeliverable&series=ALL&csv=true"
    url = base + params
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept":"text/csv, */*; q=0.01",
        "Referer":"https://www.nseindia.com/"
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text

def append_to_sqlite(df, table_name):
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql(table_name, engine, if_exists="append", index=False)
    return True

def load_nse_csv_text(csv_text):
    """Load NSE CSV returned by the endpoint into dataframe and normalize columns."""
    df = pd.read_csv(StringIO(csv_text))
    
    df.columns = (
    df.columns
    .str.replace('"', '', regex=False)
    .str.replace('ï»¿', '', regex=False)
    .str.strip()
    )
    print(df.columns)
    # df = df.rename(columns={
    # "Open Price": "Open",
    # "High Price": "High",
    # "Low Price": "Low",
    # "Close Price": "Close",
    # "Total Traded Quantity": "Volume"
    # })

    # try to normalize common columns
    col_map = {}
    # common names in the CSV are: Date, Prev Close, Open Price, High Price, Low Price, Last Price, Close Price, Average Price, Total Traded Quantity, Turnover ₹, No. of Trades, Deliverable Qty, % Dly Qt to Traded Qty
    for c in df.columns:
        cc = c.strip().lower()
        if "date" == cc:
            col_map[c] = "Date"
        elif "open price" in cc:
            col_map[c] = "Open"
        elif "high price" in cc:
            col_map[c] = "High"
        elif "low price" in cc:
            col_map[c] = "Low"
        elif "close price" in cc:
            col_map[c] = "Close"
        elif "last price" in cc:
            col_map[c] = "Last"
        elif "average price" in cc:
            col_map[c] = "Average"
        elif "total traded quantity" in cc or "tottrdqty" in cc:
            col_map[c] = "Volume"
        elif "turnover" in cc or "tottrdval" in cc:
            col_map[c] = "Turnover"
        elif "deliverable qty" in cc or "deliverable_qty" in cc:
            col_map[c] = "Deliverable"
        elif "% dly qt to traded qty" in cc or "delivery" in cc:
            col_map[c] = "DeliveryPercent"
        elif "prev close" in cc:
            col_map[c] = "PrevClose"
        elif "no. of trades" in cc or "tottrades" in cc:
            col_map[c] = "NoOfTrades"
        
    df = df.rename(columns=col_map)
    df = df.replace({',': ''}, regex=True)
    # Drop invalid rows
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    print("Normalized columns:", df.columns.tolist())
    print("Sample data:")
    print(df.head())
    # Parse date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
    # Convert numeric columns
    for col in ["Open","High","Low","Close","Last","Average","Volume","Deliverable","Turnover","PrevClose"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # compute DeliveryPercent if not present
    if "Deliverable" in df.columns and "Volume" in df.columns and "DeliveryPercent" not in df.columns:
        df["DeliveryPercent"] = (df["Deliverable"] / df["Volume"]) * 100.0
    # sort by date asc
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Free-Float Delivery Dashboard (LT)", layout="wide")
st.title("Free-Float Delivery Dashboard — LT")

# Sidebar controls
st.sidebar.header("Data input & settings")
input_mode = st.sidebar.radio("Data source", ("Upload CSV", "Download from NSE URL"))
symbol = st.sidebar.text_input("Symbol", value="LT")
from_date = st.sidebar.text_input("From (dd-mm-YYYY)", value="15-11-2024")
to_date = st.sidebar.text_input("To (dd-mm-YYYY)", value=datetime.today().strftime("%d-%m-%Y"))
free_float_shares = st.sidebar.number_input("Free Float Shares (defaults to 1.009B)", value=float(DEFAULT_FREE_FLOAT_SHARES), step=1.0)
save_to_db = st.sidebar.checkbox("Append cleaned data to SQLite DB", value=True)
save_csv = st.sidebar.checkbox("Save cleaned CSV locally", value=True)

uploaded_file = None
raw_df = None

if input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload security-wise CSV (NSE format)", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = smart_read_csv(uploaded_file)
            st.sidebar.success("CSV loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")
else:
    st.sidebar.markdown("Download from NSE security-wise endpoint")
    if st.sidebar.button("Download CSV from NSE"):
        try:
            csv_text = fetch_nse_securitywise(symbol, from_date, to_date)
            raw_df = smart_read_csv(csv_text)
            st.sidebar.success("Downloaded CSV")
        except Exception as e:
            st.sidebar.error(f"Download failed: {e}")

# If we have raw data, proceed
if raw_df is not None:
    st.subheader("Raw data preview (first 5 rows)")
    st.dataframe(raw_df.head())

    # Normalize
    df = normalize_nse_columns(raw_df)

    st.subheader("Normalized data preview (first 8 rows)")
    st.dataframe(df.head(8))

    # Basic stats and cleaning summary
    st.markdown("### Data cleaning summary")
    col_info = {c: str(df[c].dtype) for c in df.columns}
    st.json(col_info)

    # Save cleaned CSV
    if save_csv:
        cleaned_path = os.path.join(DATA_DIR, f"{symbol}_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(cleaned_path, index=False)
        st.success(f"Saved cleaned CSV: {cleaned_path}")

    # Append to DB
    if save_to_db:
        try:
            append_to_sqlite(df, f"{symbol}_price")
            st.success(f"Appended to DB: {DB_PATH} table: {symbol}_price")
        except Exception as e:
            st.error(f"Failed to append to DB: {e}")

    # Compute delivery metrics relative to free float
    st.subheader("Free-float based metrics")
    df = df.copy()
    # Ensure Deliverable and Volume exist
    if "Deliverable" not in df.columns or "Volume" not in df.columns:
        st.warning("Deliverable or Volume column missing — some metrics won't be available.")
    # Compute metrics
    if "Deliverable" in df.columns:
        df["Delivery_of_FreeFloat_%"] = (df["Deliverable"] / free_float_shares) * 100.0
        df["Cumulative_Delivery"] = df["Deliverable"].cumsum()
        df["Cumulative_Delivery_of_FreeFloat_%"] = (df["Cumulative_Delivery"] / free_float_shares) * 100.0
    if "Volume" in df.columns:
        df["Daily_Volume_of_FreeFloat_%"] = (df["Volume"] / free_float_shares) * 100.0
    # Short MAs of delivery% (5 and 20 days)
    if "Delivery_of_FreeFloat_%" in df.columns:
        df["DFF_MA5"] = df["Delivery_of_FreeFloat_%"].rolling(5, min_periods=1).mean()
        df["DFF_MA20"] = df["Delivery_of_FreeFloat_%"].rolling(20, min_periods=1).mean()

    # Show metrics summary
    st.metric("Free Float Shares (used)", f"{free_float_shares:,.0f}")
    if "Deliverable" in df.columns:
        last_row = df.iloc[-1]
        st.metric("Last day Deliverable Qty", f"{int(last_row['Deliverable']):,}" if not np.isnan(last_row['Deliverable']) else "N/A",
                  delta=f"{last_row['Delivery_of_FreeFloat_%']:.5f}% of free float" if "Delivery_of_FreeFloat_% " not in last_row else "")
    if "Volume" in df.columns:
        st.metric("Last day Volume", f"{int(last_row['Volume']):,}" if not np.isnan(last_row['Volume']) else "N/A")

    # ---------------------------
    # Interactive Charts (Plotly)
    # ---------------------------
    st.subheader("Interactive Charts")

    # Candlestick with volume + deliverable overlay
    if set(["Date","Open","High","Low","Close"]).issubset(df.columns):
        plot_df = df.set_index("Date").copy()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=plot_df.index,
                                     open=plot_df["Open"],
                                     high=plot_df["High"],
                                     low=plot_df["Low"],
                                     close=plot_df["Close"],
                                     name="Price"))
        # volume bars (secondary y as show)
        if "Volume" in plot_df.columns:
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], name="Volume", marker=dict(opacity=0.4), yaxis="y2"))
        # deliverable as line
        if "Deliverable" in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Deliverable"], name="Deliverable", mode="lines+markers", yaxis="y2"))

        # layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            yaxis_title="Price",
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Qty", position=0.97),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough OHLC data for candlestick plot.")

    # Delivery % of free float line + cumulative
    if "Delivery_of_FreeFloat_%" in df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df["Date"], y=df["Delivery_of_FreeFloat_%"], name="Daily Delivery % of Free Float"))
        fig2.add_trace(go.Scatter(x=df["Date"], y=df.get("DFF_MA5", np.nan), name="DFF MA5"))
        fig2.add_trace(go.Scatter(x=df["Date"], y=df.get("DFF_MA20", np.nan), name="DFF MA20"))
        fig2.update_layout(title="Delivery % of Free Float (daily) & moving averages", yaxis_title="% of free float")
        st.plotly_chart(fig2, use_container_width=True)

    if "Cumulative_Delivery_of_FreeFloat_%" in df.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df["Date"], y=df["Cumulative_Delivery_of_FreeFloat_%"], name="Cumulative % of Free Float", mode="lines+markers"))
        fig3.update_layout(title="Cumulative Delivery as % of Free Float", yaxis_title="% of free float")
        st.plotly_chart(fig3, use_container_width=True)

    # ---------------------------
    # Accumulation Scanner (table)
    # ---------------------------
    st.subheader("Accumulation / Distribution Scanner")
    scanner_df = df.loc[:, ["Date","Close","Volume","Deliverable","Delivery_of_FreeFloat_%"]].copy() if set(["Deliverable","Volume"]).issubset(df.columns) else df.loc[:, ["Date","Close","Volume"]].copy()
    if "Delivery_of_FreeFloat_%" in scanner_df.columns:
        scanner_df = scanner_df.sort_values("Delivery_of_FreeFloat_%", ascending=False)
        scanner_df["Delivery_of_FreeFloat_%"] = scanner_df["Delivery_of_FreeFloat_%"].map(lambda x: f"{x:.6f}%")
    st.dataframe(scanner_df.head(50))

    # CSV export
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download processed CSV", data=csv_out, file_name=f"{symbol}_processed_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

    st.markdown("---")
    st.info("Notes: \n- Free-float shares used are what you entered in sidebar. \n- Delivery % of free-float = Deliverable Qty / Free Float Shares × 100. \n- For meaningful signals combine Delivery-of-Free-Float with price direction (Close).")
else:
    st.info("Upload a CSV or download from NSE to begin.")


#py -m streamlit run lt_freefloat_dashboard.py