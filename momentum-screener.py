#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: nifty50-selling.ipynb
Conversion Date: 2025-10-28T16:50:43.273Z
"""

%pip install nsepython
%pip install nsetools
%pip install pandas_ta

#FNO Stocks
symbols = [
    "360ONE.NS", "ABB.NS", "APLAPOLLO.NS", "AUBANK.NS", "ADANIENSOL.NS", "ADANIENT.NS",
    "ADANIGREEN.NS", "ADANIPORTS.NS", "ABCAPITAL.NS", "ALKEM.NS", "AMBER.NS", "AMBUJACEM.NS",
    "ANGELONE.NS", "APOLLOHOSP.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "AUROPHARMA.NS",
    "DMART.NS", "AXISBANK.NS", "BSE.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "BDL.NS", "BEL.NS", "BHARATFORG.NS",
    "BHEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BIOCON.NS", "BLUESTARCO.NS", "BOSCHLTD.NS",
    "BRITANNIA.NS", "CGPOWER.NS", "CANBK.NS", "CDSL.NS", "CHOLAFIN.NS", "CIPLA.NS",
    "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CAMS.NS", "CONCOR.NS", "CROMPTON.NS",
    "CUMMINSIND.NS", "CYIENT.NS", "DLF.NS", "DABUR.NS", "DALBHARAT.NS", "DELHIVERY.NS",
    "DIVISLAB.NS", "DIXON.NS", "DRREDDY.NS", "ETERNAL.NS", "EICHERMOT.NS", "EXIDEIND.NS",
    "NYKAA.NS", "FORTIS.NS", "GAIL.NS", "GMRAIRPORT.NS", "GLENMARK.NS", "GODREJCP.NS",
    "GODREJPROP.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HFCL.NS", "HAVELLS.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HAL.NS", "HINDPETRO.NS",
    "HINDUNILVR.NS", "HINDZINC.NS", "POWERINDIA.NS", "HUDCO.NS", "ICICIBANK.NS", "ICICIGI.NS",
    "ICICIPRULI.NS", "IDFCFIRSTB.NS", "IIFL.NS", "ITC.NS", "INDIANB.NS", "IEX.NS",
    "IOC.NS", "IRCTC.NS", "IRFC.NS", "IREDA.NS", "IGL.NS", "INDUSTOWER.NS", "INDUSINDBK.NS",
    "NAUKRI.NS", "INFY.NS", "INOXWIND.NS", "INDIGO.NS", "JINDALSTEL.NS", "JSWENERGY.NS",
    "JSWSTEEL.NS", "JIOFIN.NS", "JUBLFOOD.NS", "KEI.NS", "KPITTECH.NS", "KALYANKJIL.NS",
    "KAYNES.NS", "KFINTECH.NS", "KOTAKBANK.NS", "LTF.NS", "LICHSGFIN.NS", "LTIM.NS",
    "LT.NS", "LAURUSLABS.NS", "LICI.NS", "LODHA.NS", "LUPIN.NS", "M&M.NS", "MANAPPURAM.NS",
    "MANKIND.NS", "MARICO.NS", "MARUTI.NS", "MFSL.NS", "MAXHEALTH.NS", "MAZDOCK.NS",
    "MPHASIS.NS", "MCX.NS", "MUTHOOTFIN.NS", "NBCC.NS", "NCC.NS", "NHPC.NS", "NMDC.NS",
    "NTPC.NS", "NATIONALUM.NS", "NESTLEIND.NS", "NUVAMA.NS", "OBEROIRLTY.NS", "ONGC.NS",
    "OIL.NS", "PAYTM.NS", "OFSS.NS", "POLICYBZR.NS", "PGEL.NS", "PIIND.NS", "PNBHOUSING.NS",
    "PAGEIND.NS", "PATANJALI.NS", "PERSISTENT.NS", "PETRONET.NS", "PIDILITIND.NS", "PPLPHARMA.NS",
    "POLYCAB.NS", "PFC.NS", "POWERGRID.NS", "PRESTIGE.NS", "PNB.NS", "RBLBANK.NS", "RECLTD.NS",
    "RVNL.NS", "RELIANCE.NS", "SBICARD.NS", "SBILIFE.NS", "SHREECEM.NS", "SRF.NS", "SAMMAANCAP.NS",
    "MOTHERSON.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SOLARINDS.NS", "SONACOMS.NS", "SBIN.NS",
    "SAIL.NS", "SUNPHARMA.NS", "SUPREMEIND.NS", "SUZLON.NS", "SYNGENE.NS", "TATACONSUM.NS",
    "TITAGARH.NS", "TVSMOTOR.NS", "TCS.NS", "TATAELXSI.NS", "TMPV.NS", "TATAPOWER.NS",
    "TATASTEEL.NS", "TATATECH.NS", "TECHM.NS", "FEDERALBNK.NS", "INDHOTEL.NS", "PHOENIXLTD.NS",
    "TITAN.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TIINDIA.NS", "UNOMINDA.NS",
    "UPL.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "IDEA.NS",
    "VOLTAS.NS", "WIPRO.NS", "YESBANK.NS", "ZYDUSLIFE.NS"
]


#from nsepython import nse_eq,  nse_quote, nse_results
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from nsetools import Nse
from nsetools import nse

#results = nse_results('LT.NS',str = "Quaterly")


# Read Nifty FnO stock symbols (or your custom list)


momentum_candidates = []
cross_up_candidates = []
rsi_rising_candidates= []
divergence_candidates= []


for symbol in symbols:
    try:
        # Get last 100 days data
        df = yf.download(symbol , period="6mo", interval="1d", progress=False, auto_adjust=True)
        df.dropna(inplace=True)
        df.columns = df.columns.get_level_values(0)

        # Calculate 20 & 50 SMA
        df['SMA20'] = ta.sma(df['Close'], 20)
        df['SMA50'] = ta.sma(df['Close'], 50)

        # Calculate RSI
        df['RSI'] = ta.rsi(df['Close'], 14)

        # Condition 1: 20MA crossed above 50MA today
        cross_up = (df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1]) and (df['SMA20'].iloc[-2] <= df['SMA50'].iloc[-2])


        # Condition 2: RSI rising toward 70
        rsi_rising = df['RSI'].iloc[-1] > 60 and df['RSI'].iloc[-1] > df['RSI'].iloc[-2]



        # Condition 3: Positive divergence (Price lower low but RSI higher low)
        recent = df.tail(20)

        price_div = recent['Close'].iloc[-1] > recent['Close'].min()
        rsi_div = recent['RSI'].iloc[-1] > recent['RSI'].min()
        divergence = rsi_div and not price_div

        #print(f"Symbol - {symbol} | cross_up - {cross_up} | divergence - {divergence}")

        # Condition 4: EPS increasing (Fundamentals)
        #fundamentals = get_nse_fundamentals(symbol)
        # Assuming EPS is available and accessible like this based on common API structures
        #eps = fundamentals.get("EPS", {}).get("current", None)
        #prev_eps = fundamentals.get("EPS", {}).get("previous", None) # Adjust key based on actual API response

        #eps_increasing = eps and prev_eps and float(eps) > float(prev_eps)
        if cross_up:
            cross_up_candidates.append(symbol)

        if rsi_rising:
            rsi_rising_candidates.append(symbol)

        if divergence:
            divergence_candidates.append(symbol)

        # Combine filters
        if cross_up and rsi_rising and divergence :
            momentum_candidates.append(symbol)

    except Exception as e:
        print(e)
        continue

print("20 MA Cross up Stocks:", cross_up_candidates)
print("RSI Rising Stocks:", rsi_rising_candidates)
print("Positive Divergence Stocks:", divergence_candidates)
print("Momentum Stocks:", momentum_candidates)

# Support and resistance
import yfinance as yf
import pandas as pd
import numpy as np

def find_swing_points(series, window=3):
    """Return swing highs and lows indices."""
    highs = series[(series.shift(1) < series) & (series.shift(-1) < series)]
    lows  = series[(series.shift(1) > series) & (series.shift(-1) > series)]
    return highs, lows

def detect_near_zones(symbol, percent_threshold=2):
    df = yf.download(symbol, period="1y", interval="1wk", progress=False, auto_adjust=True)
    df.dropna(inplace=True)

    # Step 1: find swing highs/lows
    highs, lows = find_swing_points(df['High'])
    #print(f"Symbol {symbol} | High {highs} | Lows {lows}")
    swing_highs = df.loc[highs.index].tail(3)['High'].values
    swing_lows = df.loc[lows.index].tail(3)['Low'].values
    print(f"Symbol {symbol} | swing_highs {swing_highs} | swing_lows {swing_lows}")


    # Step 2: get latest price
    last_price = df['Close'].iloc[-1]
    #print(f"Symbol {symbol} | near_support {near_support} | near_resistance {near_resistance}")
    # Step 3: find nearest support/resistance
    near_support = any(abs(last_price - s) / s < percent_threshold/100 for s in swing_lows)
    near_resistance = any(abs(last_price - r) / r < percent_threshold/100 for r in swing_highs)

    # Step 4: classify
    if near_support:
        zone = "Near Support"
    elif near_resistance:
        zone = "Near Resistance"
    else:
        zone = None

    return zone, last_price, swing_highs, swing_lows


# Example: run on multiple stocks
tickers = symbols[:10]
results = []

for t in tickers:
    try:
        zone, price, resistances, supports = detect_near_zones(t, percent_threshold=2)
        if zone:
            results.append({
                "Symbol": t,
                "Price": round(price, 2),
                "Zone": zone,
                "Supports": supports,
                "Resistances": resistances
            })
    except Exception as e:
        continue

df_results = pd.DataFrame(results)
print(df_results)


import yfinance as yf
import pandas as pd
import numpy as np

def find_swing_points(series, window=3):
    """Return swing highs and lows indices."""
    highs = series[(series.shift(1) < series) & (series.shift(-1) < series)]
    lows  = series[(series.shift(1) > series) & (series.shift(-1) > series)]
    return highs, lows

def detect_near_zones(symbol, percent_threshold=2):
    df = yf.download(symbol, period="6mo", interval="1d", progress=False)
    df.dropna(inplace=True)

    # Step 1: find swing highs/lows
    highs, lows = find_swing_points(df['High'])
    swing_highs = df.loc[highs.index].tail(3)['High'].values
    swing_lows = df.loc[lows.index].tail(3)['Low'].values

    # Step 2: get latest price
    last_price = df['Close'].iloc[-1]

    # Step 3: find nearest support/resistance
    near_support = any(abs(last_price - s) / s < percent_threshold/100 for s in swing_lows)
    near_resistance = any(abs(last_price - r) / r < percent_threshold/100 for r in swing_highs)

    # Step 4: classify
    if near_support:
        zone = "Near Support"
    elif near_resistance:
        zone = "Near Resistance"
    else:
        zone = None

    return zone, last_price, swing_highs, swing_lows


# Example: run on multiple stocks
tickers = symbols
results = []

for t in tickers:
    try:
        zone, price, resistances, supports = detect_near_zones(t, percent_threshold=2)
        if zone:
            results.append({
                "Symbol": t,
                "Price": round(price, 2),
                "Zone": zone,
                "Supports": supports,
                "Resistances": resistances
            })
    except Exception as e:
        continue

df_results = pd.DataFrame(results)
print(df_results)