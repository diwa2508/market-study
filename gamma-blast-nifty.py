import requests
import time
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
SYMBOL = "NIFTY"       # or 'BANKNIFTY' etc.
LOT_SIZE = 50          # confirm your lot size (Nifty is usually 50)
R = 0.07               # annual risk-free rate approx (adjust)
THRESH_ABS_GAMMA = 200000.0    # absolute net gamma threshold (tunable)
THRESH_DAY_CHANGE = 50000.0    # day-over-day gamma change threshold (tunable)
# ----------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def fetch_option_chain(symbol):
    """Fetch NSE option chain JSON for the index (NIFTY)."""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    # initial request to get cookies (NSE blocks direct calls sometimes)
    s = requests.Session()
    s.headers.update(HEADERS)
    r0 = s.get("https://www.nseindia.com", timeout=5)
    time.sleep(0.5)
    r = s.get(url, timeout=10)
    r.raise_for_status()
    print(f"Fetched option chain for {symbol}, status {r.status_code}")
    return r.json()

def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma for European option (per underlying unit)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

def implied_vol_from_price(option_price, S, K, T, r, is_call=True):
    """Solve implied vol using brentq. Option_price must be >= intrinsic for solver to work."""
    # simple BS price functions
    def bs_price(sigma):
        if sigma <= 1e-12 or T <= 0:
            # return intrinsic
            if is_call:
                return max(0.0, S - K * math.exp(-r*T))
            else:
                return max(0.0, K * math.exp(-r*T) - S)
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        if is_call:
            return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
        else:
            return K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # bounds for sigma
    lower = 1e-6
    upper = 5.0
    try:
        # If option_price less than intrinsic, return small vol
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        if option_price <= intrinsic + 1e-8:
            return 1e-6
        # brentq requires function with different signs at bounds
        f_low = bs_price(lower) - option_price
        f_high = bs_price(upper) - option_price
        if f_low * f_high > 0:
            # expansion fallback
            return min(max(0.1, abs(np.log(option_price+1))), upper)
        iv = brentq(lambda sig: bs_price(sig) - option_price, lower, upper, maxiter=200)
        return iv
    except Exception:
        return np.nan

def parse_chain_and_compute_gamma(chain_json, symbol_spot=None):
    """Return DataFrame with strikes and gamma exposure."""
    records = []
    expiry_dates = set()
    # the JSON has 'records' with 'data' list containing CE & PE per strike
    rec = chain_json.get("records", {})
    expiry_dates = rec.get("expiryDates", [])
    underlying = rec.get("underlyingValue", None)
    data = rec.get("data", [])
    today_ts = pd.Timestamp.now(tz='Asia/Kolkata').normalize()
    for d in data:
        K = float(d.get("strikePrice"))
        # CE and PE may be present
        ce = d.get("CE", None)
        pe = d.get("PE", None)
        # expiry used from rec->expiryDates[0] or CE/PE expiryDate
        exp = None
        if ce:
            exp = ce.get("expiryDate", exp)
        if pe and not exp:
            exp = pe.get("expiryDate", exp)
        # parse expiry to days T
        try:
            exp_date = pd.to_datetime(exp, dayfirst=True)
            T = (exp_date - pd.Timestamp.now()).total_seconds() / (365.0*24*3600)
            if T < 0:
                continue
        except Exception:
            T = np.nan
        S = underlying if underlying is not None else symbol_spot
        # CE
        if ce:
            oi = int(ce.get("openInterest", 0))
            ltp = float(ce.get("lastPrice", 0.0))
            iv = ce.get("impliedVolatility")
            iv = float(iv)/100.0 if iv else np.nan
            if np.isnan(iv):
                iv = implied_vol_from_price(ltp, S, K, T, R, is_call=True)
            gamma = bs_gamma(S, K, T, R, iv)
            # gamma per contract (per underlying) * contracts * lot
            gamma_exposure = gamma * oi * LOT_SIZE
            records.append(dict(strike=K, side='CE', oi=oi, ltp=ltp, iv=iv, gamma=gamma, gamma_exp=gamma_exposure, T=T))
        # PE
        if pe:
            oi = int(pe.get("openInterest", 0))
            ltp = float(pe.get("lastPrice", 0.0))
            iv = pe.get("impliedVolatility")
            iv = float(iv)/100.0 if iv else np.nan
            if np.isnan(iv):
                iv = implied_vol_from_price(ltp, S, K, T, R, is_call=False)
            gamma = bs_gamma(S, K, T, R, iv)
            gamma_exposure = -gamma * oi * LOT_SIZE  # puts give negative gamma exposure for dealer short puts? Use sign convention below
            records.append(dict(strike=K, side='PE', oi=oi, ltp=ltp, iv=iv, gamma=gamma, gamma_exp=gamma_exposure, T=T))
    df = pd.DataFrame(records)
    # Group by strike to get net gamma (CE positive, PE negative here by construction)
    grouped = df.groupby('strike').agg({'oi':'sum','ltp':'mean','iv':'mean','gamma':'mean','gamma_exp':'sum'}).reset_index()
    return grouped, underlying

def detect_gamma_blast(grouped_df, spot):
    """Basic detectors and flags."""
    # net gamma total
    net_gamma = grouped_df['gamma_exp'].sum()
    # gamma density near spot +/- window
    window = 200  # points
    near = grouped_df[(grouped_df['strike'] >= spot-window) & (grouped_df['strike'] <= spot+window)]
    near_gamma = near['gamma_exp'].sum()
    # maximum single-strike gamma exposure
    max_strike_gamma = grouped_df.iloc[grouped_df['gamma_exp'].abs().argmax()]
    results = {
        "net_gamma": net_gamma,
        "near_gamma": near_gamma,
        "max_strike": max_strike_gamma['strike'],
        "max_strike_gamma": max_strike_gamma['gamma_exp'],
    }
    return results

def plot_gamma_map(grouped_df, spot):
    plt.figure(figsize=(12,5))
    plt.bar(grouped_df['strike'], grouped_df['gamma_exp'])
    plt.axvline(spot, color='red', linestyle='--', label=f"Spot {spot:.2f}")
    plt.xlabel("Strike")
    plt.ylabel("Gamma Exposure (signed)")
    plt.title("Gamma Exposure by Strike")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("Fetching option chain...")
    j = fetch_option_chain(SYMBOL)
    grouped, spot = parse_chain_and_compute_gamma(j)
    print("Spot from chain:", spot)
    res = detect_gamma_blast(grouped, spot)
    print("Net Gamma (signed):", res['net_gamma'])
    print("Gamma within +/-200 points of spot:", res['near_gamma'])
    print("Max strike gamma:", res['max_strike'], "gamma:", res['max_strike_gamma'])
    plot_gamma_map(grouped, spot)
    # Simple threshold test
    if abs(res['net_gamma']) > THRESH_ABS_GAMMA:
        print("ALERT: Net gamma large -> potential for directional hedging flows (gamma blast risk).")
    if abs(res['near_gamma']) > THRESH_ABS_GAMMA/2:
        print("ALERT: Gamma concentrated near spot -> local gamma squeeze risk.")
    # optional: save CSV
    grouped.to_csv("nifty_gamma_by_strike.csv", index=False)
    print("Saved nifty_gamma_by_strike.csv")

if __name__ == "__main__":
    main()
