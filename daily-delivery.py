#!/usr/bin/env python3
"""
nse_delivery_terminal.py

Terminal tool to:
 - download security-wise price+volume+deliverable CSV from NSE
 - save CSV to data/
 - save/append to SQLite DB (data/nse_data.db)
 - compute Delivery %, optional free-float metrics (IWF / float-shares)
 - plot candlestick + volume + deliverable overlay and a multi-chart dashboard

Usage:
 python nse_delivery_terminal.py --symbol LT --from 15-11-2024 --to 15-11-2025

Optional flags:
 --no-csv        : don't save raw CSV file
 --no-db         : don't save to SQLite DB
 --iwf path.csv  : path to CSV containing IWF data (columns SYMBOL,DATE,IWF)
 --float path.csv: path to CSV containing FLOAT_SHARES data (columns SYMBOL,DATE,FLOAT_SHARES)
"""

import os
import argparse
from io import StringIO
from datetime import datetime
import requests
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine

# ---------------------------
# Configuration
# ---------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/csv,*/*;q=0.1",
    "Referer": "https://www.nseindia.com/"
}

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "nse_data.db")
RAW_CSV_TEMPLATE = os.path.join(DATA_DIR, "{symbol}_{from_date}_to_{to_date}.csv")

# ---------------------------
# Helpers
# ---------------------------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def build_nse_url(symbol, from_date, to_date):
    # Expecting from_date and to_date in dd-mm-YYYY
    base = "https://www.nseindia.com/api/historicalOR/generateSecurityWiseHistoricalData"
    params = (
        f"?from={from_date}&to={to_date}&symbol={symbol}"
        f"&type=priceVolumeDeliverable&series=ALL&csv=true"
    )
    return base + params

def download_csv(url):
    """Return CSV text (str). Raises on HTTP error."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} when fetching {url}. Body: {resp.text[:400]}")
    return resp.text

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

def save_csv(df, path):
    df.to_csv(path, index=False)
    print("Saved CSV to:", path)

def save_to_sqlite(df, table_name):
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"Appended data to SQLite DB at {DB_PATH}, table: {table_name}")

# ---------------------------
# Free-float helpers
# ---------------------------
def apply_iwf(df, iwf_df):
    """
    iwf_df expected columns: SYMBOL, DATE, IWF
    Will forward-fill IWF across dates and compute FREE_FLOAT_SHARES if ISSUED_SHARES found.
    """
    iwf_df = iwf_df.copy()
    iwf_df["DATE"] = pd.to_datetime(iwf_df["DATE"], dayfirst=True, errors='coerce')
    iwf_df = iwf_df.sort_values("DATE").set_index("DATE")
    working = df.copy()
    working = working.set_index("Date")
    # join
    merged = working.join(iwf_df["IWF"], how="left")
    merged["IWF"] = merged["IWF"].fillna(method="ffill").fillna(method="bfill")
    # If user provided ISSUED_SHARES column, compute FREE_FLOAT_SHARES and FREE_FLOAT_MKTCAP
    if "ISSUED_SHARES" in merged.columns:
        merged["FREE_FLOAT_SHARES"] = merged["ISSUED_SHARES"] * merged["IWF"]
        merged["FREE_FLOAT_MKTCAP"] = merged["FREE_FLOAT_SHARES"] * merged["Close"]
    merged = merged.reset_index().rename(columns={"index":"Date"})
    return merged

def apply_float_shares(df, float_df):
    """
    float_df expected columns: SYMBOL, DATE, FLOAT_SHARES
    """
    float_df = float_df.copy()
    float_df["DATE"] = pd.to_datetime(float_df["DATE"], dayfirst=True, errors='coerce')
    float_df = float_df.sort_values("DATE").set_index("DATE")
    working = df.copy().set_index("Date")
    merged = working.join(float_df["FLOAT_SHARES"], how="left")
    merged["FLOAT_SHARES"] = merged["FLOAT_SHARES"].fillna(method="ffill").fillna(method="bfill")
    if "Close" in merged.columns:
        merged["FREE_FLOAT_MKTCAP"] = merged["FLOAT_SHARES"] * merged["Close"]
    merged = merged.reset_index().rename(columns={"index":"Date"})
    return merged

# ---------------------------
# Plotting
# ---------------------------
def plot_candlestick_with_volume_deliverable(df, symbol):
    """
    Expects df indexed by Date with columns Open, High, Low, Close, Volume, Deliverable
    """
    print('# mplfinance expects columns: Open,High,Low,Close,Volume')
    df_plot = df.copy()
    print(df_plot.head())
    # mplfinance expects columns: Open,High,Low,Close,Volume
    ohlc_cols = ["Open","High","Low","Close"]
    if not all([c in df_plot.columns for c in ohlc_cols]):
        raise RuntimeError("OHLC columns missing in data for plotting candlestick.")
    df_plot = df_plot.set_index("Date")
    df_plot.index.name = "Date"
    if "Volume" in df_plot.columns:
        mpf.plot(df_plot, type='candle', volume=True, mav=(5,20), title=f"{symbol} - Candlestick + Volume", show_nontrading=False)
    else:
        mpf.plot(df_plot, type='candle', title=f"{symbol} - Candlestick (no volume)", show_nontrading=False)

    # Overlay deliverable on volume chart separately
    if "Deliverable" in df_plot.columns:
        plt.figure(figsize=(12,4))
        plt.plot(df_plot.index, df_plot["Volume"], label="Volume")
        plt.plot(df_plot.index, df_plot["Deliverable"], label="Deliverable", marker='o')
        plt.title(f"{symbol} - Volume vs Deliverable")
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_dashboard(df, symbol):
    """3-row dashboard: Close, Volume+Deliverable, Delivery %"""
    df2 = df.copy().set_index("Date")
    fig, axes = plt.subplots(3,1, figsize=(14,10), sharex=True)
    # Close
    axes[0].plot(df2.index, df2["Close"], marker='o')
    axes[0].set_title(f"{symbol} Close Price")
    axes[0].grid(True)
    # Volume + Deliverable
    axes[1].bar(df2.index, df2["Volume"], label="Volume")
    if "Deliverable" in df2.columns:
        axes[1].plot(df2.index, df2["Deliverable"], label="Deliverable", color='orange', marker='o')
    axes[1].set_title("Volume and Deliverable")
    axes[1].legend()
    axes[1].grid(True)
    # Delivery %
    if "DeliveryPercent" in df2.columns:
        axes[2].plot(df2.index, df2["DeliveryPercent"], marker='o', color='green')
        axes[2].set_title("Delivery %")
        axes[2].grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------
# CLI & Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="NSE security-wise price+volume+deliverable downloader and visualizer")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. LT")
    parser.add_argument("--from", dest="from_date", required=True, help="From date dd-mm-YYYY")
    parser.add_argument("--to", dest="to_date", required=True, help="To date dd-mm-YYYY")
    parser.add_argument("--no-csv", action="store_true", help="Do not save raw CSV")
    parser.add_argument("--no-db", action="store_true", help="Do not save to SQLite DB")
    parser.add_argument("--iwf", help="Path to CSV with IWF data (SYMBOL,DATE,IWF)")
    parser.add_argument("--float", dest="float_csv", help="Path to CSV with FLOAT_SHARES data (SYMBOL,DATE,FLOAT_SHARES)")
    args = parser.parse_args()

    ensure_data_dir()

    symbol = args.symbol.upper()
    from_date = args.from_date
    to_date = args.to_date

    url = build_nse_url(symbol, from_date, to_date)
    print("Fetching:", url)
    try:
        csv_text = download_csv(url)
    except Exception as e:
        print("ERROR downloading data:", e)
        return

    # Save raw csv
    csv_path = RAW_CSV_TEMPLATE.format(symbol=symbol, from_date=from_date.replace('-',''), to_date=to_date.replace('-',''))
    if not args.no_csv:
        save_csv(pd.read_csv(StringIO(csv_text)), csv_path)

    # Load & normalize
    df = load_nse_csv_text(csv_text)
    print("Data sample:", df.head())
    # Keep only this symbol rows if endpoint returned others (usually it's specific)
    # Some endpoints include SYMBOL column; else assume it's the requested symbol
    if "Symbol" in df.columns or "SYMBOL" in df.columns:
        # normalize case
        colnames = {c:c for c in df.columns}
        for c in df.columns:
            if c.lower() == "symbol":
                sym_col = c
                df = df[df[c].astype(str).str.upper() == symbol].copy()
                break

    # Ensure Date column exists
    if "Date" not in df.columns:
        print("No Date column in returned CSV. Aborting.")
        return

    # Save to DB
    if not args.no_db:
        # for DB, create a small "meta" table with symbol
        table_name = f"{symbol}_price"
        try:
            save_to_sqlite(df, table_name)
        except Exception as e:
            print("Warning: failed to save to DB:", e)

    # Free-float handling (optional)
    if args.iwf:
        try:
            iwf_df = pd.read_csv(args.iwf)
            df_ff = apply_iwf(df.rename(columns={c:c for c in df.columns}), iwf_df)
            print("Applied IWF. Sample:")
            print(df_ff.head())
        except Exception as e:
            print("Failed to apply IWF:", e)
    elif args.float_csv:
        try:
            float_df = pd.read_csv(args.float_csv)
            df_ff = apply_float_shares(df.rename(columns={c:c for c in df.columns}), float_df)
            print("Applied float shares. Sample:")
            print(df_ff.head())
        except Exception as e:
            print("Failed to apply float shares:", e)

    print("Plot candlestick + volume + deliverable")
    # Plot candlestick + volume + deliverable
    try:
        plot_candlestick_with_volume_deliverable(df, symbol)
    except Exception as e:
        print("Plotting candlestick failed:", e)

    print("Plot dashboard")
    # Plot dashboard
    try:
        plot_dashboard(df, symbol)
    except Exception as e:
        print("Plotting dashboard failed:", e)

if __name__ == "__main__":
    main()


#helper

#Basic (save CSV + DB):

#py .\daily-delivery.py --symbol LT --from 15-11-2024 --to 15-11-2025

#Do not save CSV, only show charts:

#py .\daily-delivery.py --symbol LT --from 15-11-2024 --to 15-11-2025 --no-csv

#Use an IWF CSV for free-float calculations:

#py .\daily-delivery.py --symbol LT --from 15-11-2024 --to 15-11-2025 --iwf path/to/iwf.csv

#iwf.csv should contain at least: SYMBOL,DATE,IWF (DATE in dd-mm-YYYY or ISO format).

#Provide FLOAT_SHARES CSV:

#py .\daily-delivery.py --symbol LT --from 15-11-2024 --to 15-11-2025 --float path/to/float.cs