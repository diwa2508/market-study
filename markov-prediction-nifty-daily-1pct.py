"""
Daily Markov prediction: detect >=1% moves in Nifty for the NEXT DAY.
"""

from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

# ---------------- Data load helpers ----------------

def ensure_date_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

# -------------------- State definition --------------------

def states_nextday_1pct(df: pd.DataFrame, thresh: float = 0.01) -> pd.Series:
    """
    Define state using NEXT-DAY return:
      UP_BIG if next day return >= +1%
      DN_BIG if next day return <= -1%
      NEUTRAL otherwise
    """
    c = df['Close']
    next_ret = c.shift(-1) / c - 1

    labels = pd.Series(index=df.index, dtype=object)
    labels[next_ret >= thresh] = 'UP_BIG'
    labels[next_ret <= -thresh] = 'DN_BIG'
    labels[(next_ret > -thresh) & (next_ret < thresh)] = 'NEUTRAL'
    return labels

# -------------------- Markov Chain --------------------

class MarkovChain:
    def __init__(self, order=2):
        self.order = order
        self.transition = {}
        self.states = None

    def fit(self, state_series: pd.Series):
        s = state_series.dropna()
        seq = []
        for i in range(self.order, len(s)):
            prev = tuple(s.iloc[i-self.order:i])
            cur = s.iloc[i]
            seq.append((prev, cur))

        counts = {}
        for prev, cur in seq:
            counts.setdefault(prev, {})
            counts[prev][cur] = counts[prev].get(cur, 0) + 1

        probs = {
            prev: {k: v / sum(cur.values()) for k, v in cur.items()}
            for prev, cur in counts.items()
        }
        self.transition = probs

    def predict(self, prev_state_tuple: Tuple) -> Dict:
        if prev_state_tuple in self.transition:
            return self.transition[prev_state_tuple]

        # fallback: marginal distribution
        fallback = {}
        for d in self.transition.values():
            for k, v in d.items():
                fallback[k] = fallback.get(k, 0) + v

        s = sum(fallback.values()) if fallback else 1
        return {k: v / s for k, v in fallback.items()}

# -------------------- Backtest --------------------

def backtest_daily(df: pd.DataFrame,
                   states: pd.Series,
                   order=2,
                   train_window=252,
                   prob_threshold=0.55):
    """
    Walk-forward daily prediction.
    Predict NEXT-DAY >=1% move.
    """
    dates = []
    pred = []
    prob = []
    action = []
    pct_return = []

    for i in range(train_window, len(states)-1):
        train = states.iloc[i-train_window:i]
        mc = MarkovChain(order=order)
        mc.fit(train)

        prev_state = tuple(states.iloc[i-order:i])
        p = mc.predict(prev_state)

        p_up = p.get('UP_BIG', 0)
        p_dn = p.get('DN_BIG', 0)

        chosen = 'NEUTRAL'
        act = 0
        strength = 0

        if p_up >= prob_threshold and p_up >= p_dn:
            chosen = 'UP_BIG'
            act = 1
            strength = p_up
        elif p_dn >= prob_threshold:
            chosen = 'DN_BIG'
            act = -1
            strength = p_dn

        # Realized next-day return (close-to-close)
        next_ret = df['Close'].iloc[i+1] / df['Close'].iloc[i] - 1
        ret = act * next_ret

        dates.append(df.index[i+1])
        pred.append(chosen)
        prob.append(strength)
        action.append(act)
        pct_return.append(ret)

    res = pd.DataFrame({
        'pred': pred,
        'prob': prob,
        'action': action,
        'pct_return': pct_return
    }, index=dates)

    res['cum_strategy'] = (1 + res['pct_return']).cumprod() - 1
    res['next_day_ret'] = df['Close'].pct_change().shift(-1).reindex(res.index)
    res['cum_hold'] = (1 + res['next_day_ret']).cumprod() - 1
    return res

# -------------------- Evaluation --------------------

def evaluate(res: pd.DataFrame):
    trades = res[res.action != 0]
    hits = 0
    for i, row in trades.iterrows():
        if row.pred == 'UP_BIG' and row.next_day_ret >= 0.01:
            hits += 1
        if row.pred == 'DN_BIG' and row.next_day_ret <= -0.01:
            hits += 1

    hit_rate = hits / len(trades) if len(trades) else 0
    total = res['cum_strategy'].iloc[-1]

    return {
        'n_predictions': len(res),
        'n_trades': len(trades),
        'hit_rate': hit_rate,
        'total_return': total
    }

def download_nifty_daily(period="10y", save_as="nifty_daily.csv"):
    """
    Downloads Nifty 50 (^NSEI) daily OHLCV data using yfinance.
    Handles missing data and returns a clean DataFrame.
    """

    print(f"Downloading Nifty daily data for last {period}...")
    df = yf.download("^NSEI", period=period, interval="1d", auto_adjust=False)

    if df.empty:
        raise ValueError("Failed to download Nifty data. Check internet or yfinance version.")

    # Clean up
    df = df.dropna(subset=["Close"])
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Save locally
    df.to_csv(save_as)
    print(f"Saved daily Nifty data to {save_as}")

    return df

# -------------------- Run --------------------

def main():
    # load daily data
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
    print(df.head())
    df = ensure_date_index(df)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    states = states_nextday_1pct(df)

    res = backtest_daily(
        df,
        states,
        order=2,
        train_window=252,
        prob_threshold=0.55
    )

    metrics = evaluate(res)
    print(metrics)

    res.to_csv("markov_daily_1pct_results.csv")
    print("Saved markov_daily_1pct_results.csv")

    res[['cum_strategy', 'cum_hold']].plot(figsize=(12,5), title="Daily >=1% move prediction")
    plt.show()


if __name__ == "__main__":
    main()
