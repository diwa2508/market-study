#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: vcp screening.ipynb
Conversion Date: 2025-10-28T16:50:29.160Z
"""

import pandas as pd
import numpy as np
import nsepython as nse
from scipy.stats import norm
from datetime import datetime, timedelta
from nsetools import Nse


#FNO Stocks
#FNO Stocks
symbols = [
    "360ONE", "ABB", "APLAPOLLO", "AUBANK", "ADANIENSOL", "ADANIENT",
    "ADANIGREEN", "ADANIPORTS", "ABCAPITAL", "ALKEM", "AMBER", "AMBUJACEM",
    "ANGELONE", "APOLLOHOSP", "ASHOKLEY", "ASIANPAINT", "ASTRAL", "AUROPHARMA",
    "DMART", "AXISBANK", "BSE", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BDL", "BEL", "BHARATFORG",
    "BHEL", "BPCL", "BHARTIARTL", "BIOCON", "BLUESTARCO", "BOSCHLTD",
    "BRITANNIA", "CGPOWER", "CANBK", "CDSL", "CHOLAFIN", "CIPLA",
    "COALINDIA", "COFORGE", "COLPAL", "CAMS", "CONCOR", "CROMPTON",
    "CUMMINSIND", "CYIENT", "DLF", "DABUR", "DALBHARAT", "DELHIVERY",
    "DIVISLAB", "DIXON", "DRREDDY", "ETERNAL", "EICHERMOT", "EXIDEIND",
    "NYKAA", "FORTIS", "GAIL", "GMRAIRPORT", "GLENMARK", "GODREJCP",
    "GODREJPROP", "GRASIM", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HFCL", "HAVELLS", "HEROMOTOCO", "HINDALCO", "HAL", "HINDPETRO",
    "HINDUNILVR", "HINDZINC", "POWERINDIA", "HUDCO", "ICICIBANK", "ICICIGI",
    "ICICIPRULI", "IDFCFIRSTB", "IIFL", "ITC", "INDIANB", "IEX",
    "IOC", "IRCTC", "IRFC", "IREDA", "IGL", "INDUSTOWER", "INDUSINDBK",
    "NAUKRI", "INFY", "INOXWIND", "INDIGO", "JINDALSTEL", "JSWENERGY",
    "JSWSTEEL", "JIOFIN", "JUBLFOOD", "KEI", "KPITTECH", "KALYANKJIL",
    "KAYNES", "KFINTECH", "KOTAKBANK", "LTF", "LICHSGFIN", "LTIM",
    "LT", "LAURUSLABS", "LICI", "LODHA", "LUPIN", "M&M", "MANAPPURAM",
    "MANKIND", "MARICO", "MARUTI", "MFSL", "MAXHEALTH", "MAZDOCK",
    "MPHASIS", "MCX", "MUTHOOTFIN", "NBCC", "NCC", "NHPC", "NMDC",
    "NTPC", "NATIONALUM", "NESTLEIND", "NUVAMA", "OBEROIRLTY", "ONGC",
    "OIL", "PAYTM", "OFSS", "POLICYBZR", "PGEL", "PIIND", "PNBHOUSING",
    "PAGEIND", "PATANJALI", "PERSISTENT", "PETRONET", "PIDILITIND", "PPLPHARMA",
    "POLYCAB", "PFC", "POWERGRID", "PRESTIGE", "PNB", "RBLBANK", "RECLTD",
    "RVNL", "RELIANCE", "SBICARD", "SBILIFE", "SHREECEM", "SRF", "SAMMAANCAP",
    "MOTHERSON", "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "SONACOMS", "SBIN",
    "SAIL", "SUNPHARMA", "SUPREMEIND", "SUZLON", "SYNGENE", "TATACONSUM",
    "TITAGARH", "TVSMOTOR", "TCS", "TATAELXSI", "TMPV", "TATAPOWER",
    "TATASTEEL", "TATATECH", "TECHM", "FEDERALBNK", "INDHOTEL", "PHOENIXLTD",
    "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TIINDIA", "UNOMINDA",
    "UPL", "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "VBL", "VEDL", "IDEA",
    "VOLTAS", "WIPRO", "YESBANK", "ZYDUSLIFE"
]

for symbol in symbols:
    print(f"\nAnalyzing {symbol}...")
    stock = nse.nse_quote(symbol)
    next_expiry = (stock["expiryDatesByInstrument"]["Stock Options"])[0]
    spot = stock["underlyingValue"]
    options_chain = nse.option_chain(symbol)
    options_chain_weekly = options_chain['filtered']['data']


    today = datetime.today()
    monthly_expiry = None

    # Find nearest weekly expiry after today
    # for d in expiry_dates:
    #     exp_date = datetime.strptime(d, "%d-%b-%Y")
    #     if exp_date >= today:
    #         monthly_expiry = d
    #         break

    # if monthly_expiry is None:
    #     raise Exception("No weekly expiry found!")

    print(f"Monthly Expiry: {next_expiry}")



    columns = ['CE_strikePrice', 'CE_expiryDate', 'CE_underlying', 'CE_identifier',
        'CE_openInterest', 'CE_changeinOpenInterest',
        'CE_pchangeinOpenInterest', 'CE_totalTradedVolume',
        'CE_impliedVolatility', 'CE_lastPrice', 'CE_change', 'CE_pChange',
        'CE_totalBuyQuantity', 'CE_totalSellQuantity', 'CE_bidQty',
        'CE_bidprice', 'CE_askQty', 'CE_askPrice', 'CE_underlyingValue',
        'strikePrice', 'expiryDate', 'PE_strikePrice', 'PE_expiryDate',
        'PE_underlying', 'PE_identifier', 'PE_openInterest',
        'PE_changeinOpenInterest', 'PE_pchangeinOpenInterest',
        'PE_totalTradedVolume', 'PE_impliedVolatility', 'PE_lastPrice',
        'PE_change', 'PE_pChange', 'PE_totalBuyQuantity',
        'PE_totalSellQuantity', 'PE_bidQty', 'PE_bidprice', 'PE_askQty',
        'PE_askPrice', 'PE_underlyingValue']

    option_df = pd.DataFrame(columns)
    rows= []
    for item in options_chain_weekly:

        strike = item.get('strikePrice')
        expiry = item.get('expiryDate')

        # Extract CE and PE data (handle missing keys safely)
        ce = item.get('CE', {})
        pe = item.get('PE', {})

        # Prefix keys for clarity and merge both sides
        ce_prefixed = {f'CE_{k}': v for k, v in ce.items()}
        pe_prefixed = {f'PE_{k}': v for k, v in pe.items()}

        # Merge both into one dictionary
        combined = {
            'strikePrice': strike,
            'expiryDate': expiry,
            **ce_prefixed,
            **pe_prefixed
        }
        # Append to list
        rows.append(combined)


    # Convert to DataFrame
    option_df = pd.DataFrame(rows)

    df = option_df

    # Optional: show first rows
    #print(df['CE_impliedVolatility'].head())


    # -----------------------------
    # 1. Determine ATM Strike
    # -----------------------------
    df['CE_strikePrice'] = pd.to_numeric(df['CE_strikePrice'], errors='coerce')

    # Compute the absolute difference from spot
    df['diff'] = (df['CE_strikePrice'] - spot).abs()

    # Find the strike with the minimum difference
    atm_strike = df.loc[df['diff'].idxmin(), 'CE_strikePrice']


    #print(f"Spot: {spot}, ATM Strike: {atm_strike}")

    # -----------------------------
    # 2. Calculate Expected Move (1 Std Dev)
    # -----------------------------
    # Use ATM CE IV (or PE IV if CE IV is 0)
    #✅ Option 1: Replace Zero IVs with Nearby Values (Interpolation)
    df['CE_impliedVolatility'] = df['CE_impliedVolatility'].replace(0, np.nan).interpolate()
    df['PE_impliedVolatility'] = df['PE_impliedVolatility'].replace(0, np.nan).interpolate()
    atm_row = df[df['CE_strikePrice'] == atm_strike].iloc[0]

    iv = atm_row['CE_impliedVolatility'] if atm_row['CE_impliedVolatility'] > 0 else atm_row['PE_impliedVolatility']
    iv = iv / 100  # Convert to decimal
    #print  (f"Interpolated ATM IV: {iv:.2f}")

    today = datetime.today()
    expiry = datetime.strptime(next_expiry, '%d-%b-%Y')
    DTE = (expiry - today).days


    expected_move = spot * iv * np.sqrt(DTE / 365)
    expected_move_2std = 2 * expected_move

    print(f"Expected Move (1σ): {expected_move:.2f} points")
    print(f"Expected Move (2σ): {expected_move_2std:.2f} points")

    # -----------------------------
    # 3. Determine Market Bias using OI
    # -----------------------------
    total_ce_oi = df['CE_openInterest'].sum()
    total_pe_oi = df['PE_openInterest'].sum()

    if total_ce_oi > total_pe_oi:
        direction = "Bearish (more CE OI)"
    else:
        direction = "Bullish (more PE OI)"
    print(f"Probable Direction: {direction}")

    # -----------------------------
    # 4. Select Strike to Capture ~50 Points
    # -----------------------------
    target_points = 50
    if direction.startswith("Bull"):
        # Select nearest strike above spot + 50
        filtered_ce_strike =  df['CE_strikePrice'].dropna()
        target_strike = filtered_ce_strike.iloc[(filtered_ce_strike - (spot + target_points)).abs().argsort()[:1]].values[0]
    else:
        # Ensure no NA values and reset index before sorting
        filtered_pe_strike = df['PE_strikePrice'].dropna()
        target_strike = filtered_pe_strike.iloc[(filtered_pe_strike - (spot - target_points)).abs().argsort()[:1]].values[0]

    print(f"Suggested Strike to capture ~50 points: {target_strike}")

    #print(df.iloc[(df['CE_strikePrice'] - spot).abs().argsort()[:1]])