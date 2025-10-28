import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date
import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_nifty500_tickers():
    # URL that offers CSV or table of Nifty 500 constituents
    url = "https://www.nseindia.com/products-services/indices-nifty50-index"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    # get page
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    html = resp.text
    print(html)

    # parse HTML to find CSV link or the table
    soup = BeautifulSoup(html, "html.parser")
    # look for a link to CSV (analysis might need adaptation because site may require authentication or dynamic JS)
    csv_link = None
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.lower().endswith(".csv"):
            csv_link = href
            break
    tickers = []
    if csv_link:
        # absolute url if relative
        if not csv_link.startswith("http"):
            base = "https://www.nseindia.com"
            csv_link = base + csv_link
        df = pd.read_csv(csv_link)
        # assume there is a column like "Symbol" or "Ticker"
        if "Symbol" in df.columns:
            tickers = df["Symbol"].tolist()
        elif "Ticker" in df.columns:
            tickers = df["Ticker"].tolist()
    else:
        # fallback: parse a table of constituents
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            for row in rows[1:]:
                cols = row.find_all("td")
                if cols:
                    ticker = cols[0].get_text(strip=True)
                    tickers.append(ticker)
    return tickers

tickers = get_nifty500_tickers()
print("Number of tickers:", len(tickers))
print(tickers[:20])  # first 20 tickers

with open("nse_tickers.txt", "w") as f:
    for t in tickers:
        f.write(t + "\n")

print("✅ Saved", len(tickers), "tickers to nse_tickers.txt")
