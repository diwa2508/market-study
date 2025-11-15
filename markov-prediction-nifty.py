"""
Markov-chain Nifty-50 next-candle predictor + backtester

Usage:
 - install dependencies: pip install pandas numpy matplotlib yfinance scipy
 - run the script or import the functions in a notebook

Features:
 - fetches Nifty 50 history from Yahoo (ticker: ^NSEI) or accepts a CSV
 - creates discrete states (simple: Up/Down/Neutral) or binned returns (quantiles)
 - builds an n-order Markov transition matrix from training window
 - predicts next candle probability distribution
 - backtests predictions walk-forward and reports accuracy, directional hit-rate, cumulative returns and a simple strategy P/L
 - plotting helpers

Note: This is a simple educational implementation. Use with care for live trading.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import zscore


def fetch_nifty(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical NIFTY 50 data from Yahoo Finance (ticker ^NSEI).
    Returns a DataFrame with Date index and columns O H L C Adj Close Volume
    """
    ticker = "^NSEI"
    df = yf.download(ticker, period=period, interval=interval, progress=False,auto_adjust=True)
    df = df.dropna()
    return df


def load_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def candle_states(df: pd.DataFrame, method: str = "sign", neutral_thresh: float = 0.001, bins: int = 5) -> pd.Series:
    """Create discrete states from OHLC data.

    method:
      - 'sign': simple Up/Down/Neutral based on close-open and neutral_thresh (fraction of open)
      - 'quantile': discretize returns into `bins` quantile buckets labeled 0..bins-1

    Returns a pandas Series of states (strings for 'sign', ints for 'quantile').
    """
    o = df['Open']
    c = df['Close']
    ret = (c - o) / o

    if method == 'sign':
        states = pd.Series(index=df.index, dtype=object)
        states.loc[ret > neutral_thresh] = 'U'  # Up
        states.loc[ret < -neutral_thresh] = 'D'  # Down
        states.loc[ret.abs() <= neutral_thresh] = 'N'  # Neutral / Doji
        return states

    elif method == 'quantile':
        # use daily returns (close-to-close)
        r = df['Close'].pct_change().fillna(0)
        # cut into bins by quantile
        labels = list(range(bins))
        states = pd.qcut(r, q=bins, labels=labels, duplicates='drop')
        return states.astype(int)

    else:
        raise ValueError("Unknown method for candle_states")


class MarkovChainPredictor:
    def __init__(self, order: int = 1):
        self.order = order
        self.transition_counts = None
        self.transition_probs = None
        self.states = None

    def fit(self, state_series: pd.Series) -> None:
        """Fit transition counts/probabilities from a pandas Series of states.
        Supports n-th order Markov by treating history tuples.
        """
        s = state_series.dropna().astype(object)
        # generate observed state tuples
        sequences = []
        for i in range(self.order, len(s)):
            prev = tuple(s.iloc[i - self.order:i].tolist())
            cur = s.iloc[i]
            sequences.append((prev, cur))

        # unique states set
        unique_states = sorted(s.unique(), key=lambda x: str(x))
        self.states = unique_states

        # build counts dict
        counts = {}
        for prev, cur in sequences:
            counts.setdefault(prev, {})
            counts[prev][cur] = counts[prev].get(cur, 0) + 1

        # convert to matrices (dicts)
        probs = {}
        for prev, d in counts.items():
            total = sum(d.values())
            probs[prev] = {k: v / total for k, v in d.items()}

        self.transition_counts = counts
        self.transition_probs = probs

    def predict_next_proba(self, prev_state: Tuple) -> dict:
        """Return a dict of probabilities for the next state given prev_state tuple."""
        if self.transition_probs is None:
            raise RuntimeError('Model not fitted')
        # if unseen prev_state, fall back to global marginal distribution
        if prev_state in self.transition_probs:
            return self.transition_probs[prev_state]
        else:
            # aggregate all next-state probabilities
            agg = {}
            for p, d in self.transition_probs.items():
                for k, v in d.items():
                    agg[k] = agg.get(k, 0) + v
            # normalize
            s = sum(agg.values())
            return {k: v / s for k, v in agg.items()}

    def most_likely(self, prev_state: Tuple):
        proba = self.predict_next_proba(prev_state)
        # return state with highest probability
        return max(proba.items(), key=lambda x: x[1])


def backtest_markov(df: pd.DataFrame,
                     state_series: pd.Series,
                     order: int = 1,
                     train_window: int = 252,
                     predict_on: str = 'close',
                     method_for_trade: str = 'direction') -> pd.DataFrame:
    """Walk-forward backtest that at each step fits a Markov chain on the previous train_window days
    and predicts the next day's state. Returns a DataFrame with predictions and PnL for a simple strategy.

    predict_on: 'close' means we evaluate prediction vs next day's close/open sign-based outcome
    method_for_trade: 'direction' -> go long when predict 'U', short when predict 'D', flat on 'N'
    """
    preds = []
    actions = []
    returns = []
    index_dates = []

    for i in range(train_window, len(state_series) - 1):
        train_states = state_series.iloc[i - train_window:i]
        mc = MarkovChainPredictor(order=order)

        mc.fit(train_states)
        prev_state = tuple(state_series.iloc[i - order:i].tolist())
        proba = mc.predict_next_proba(prev_state)
        mc.fit(states)
        pd.DataFrame(mc.transition_probs).fillna(0)
        # choose most likely
        pred_state, pred_p = max(proba.items(), key=lambda x: x[1])

        # next day's actual state (we predicted for i -> i+1)
        actual = state_series.iloc[i + 1]

        # define simple action
        if method_for_trade == 'direction' and isinstance(pred_state, str):
            if pred_state == 'U':
                action = 1
            elif pred_state == 'D':
                action = -1
            else:
                action = 0
            # return based on open->close of next day
            next_open = df['Open'].iloc[i + 1]
            next_close = df['Close'].iloc[i + 1]
            ret = (next_close - next_open) / next_open * action
        else:
            # fallback: no action
            action = 0
            ret = 0

        preds.append(pred_state)
        actions.append(action)
        returns.append(ret)
        index_dates.append(df.index[i + 1])

    res = pd.DataFrame({
        'date': index_dates,
        'pred': preds,
        'action': actions,
        'pct_return': returns
    }).set_index('date')

    res['cum_return_strategy'] = (1 + res['pct_return']).cumprod() - 1
    res['cum_return_hold'] = (1 + df['Close'].pct_change().shift(-1).reindex(res.index).fillna(0)).cumprod() - 1
    return res


def evaluate_backtest(res: pd.DataFrame) -> dict:
    """Simple metrics: accuracy of predicted state (if categorical), directional hit-rate, final returns, annualized return and Sharpe (assumes daily returns)
    """
    # prediction accuracy if pred equals actual (but we didn't save actual in backtest above) - skip unless available
    # compute hit rate: when action non-zero, check sign of pct_return
    trades = res[res['action'] != 0]
    if len(trades) == 0:
        hit_rate = None
    else:
        wins = (trades['pct_return'] > 0).sum()
        hit_rate = wins / len(trades)

    total_ret = res['cum_return_strategy'].iloc[-1] if len(res) > 0 else 0
    # annualized return (approx)
    days = len(res)
    ann_ret = (1 + total_ret) ** (252 / days) - 1 if days > 0 else 0
    # daily returns series for sharpe
    dr = res['pct_return'].fillna(0)
    if dr.std() == 0:
        sharpe = None
    else:
        sharpe = np.sqrt(252) * dr.mean() / dr.std()

    return {
        'n_predictions': len(res),
        'n_trades': len(trades),
        'hit_rate': hit_rate,
        'total_return': total_ret,
        'annualized_return': ann_ret,
        'sharpe': sharpe
    }


def plot_backtest(res: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res.index, res['cum_return_strategy'], label='Strategy cumulative')
    ax.plot(res.index, res['cum_return_hold'], label='Next-day hold cum')
    ax.set_title('Backtest cumulative returns')
    ax.legend()
    plt.tight_layout()
    plt.scatter(res.index, res['pct_return'].cumsum(), c=res['action'], cmap='coolwarm')
    plt.show()

# Example of use as script
if __name__ == '__main__':
    # 1) fetch data
    df = pd.DataFrame()
    try:
        df = pd.read_csv("nifty_data.csv")
        if 'Date' not in df.columns:
            raise ValueError("Missing 'Date' column in nifty_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        print("Loaded Nifty data from nifty_data.csv")
        print(df.head())
    except Exception as e:
        print(f"Error loading nifty_data.csv: {e}")
        df = fetch_nifty(period='10y', interval='1wk')
        df = df.rename_axis(None, axis=0)  # Ensure axis renaming is correct
        df.to_csv("nifty_data.csv")  # save copy
        print("Fetched Nifty data and saved to nifty_data.csv")
        print(df.head())
    
    df.columns = df.columns.get_level_values(0)
    df.columns = [col.strip() for col in df.columns]

        # If the first column is the Date, convert and set it as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    print(df.head())
    # 2) create states (try 'sign' or 'quantile')
    #states = candle_states(df, method='quantile', bins=5)
    states = candle_states(df, method='sign', neutral_thresh=0.0005)

    # 3) run backtest with 1st order Markov and 252-day training window
    #res = backtest_markov(df, states, order=1, train_window=252)
    #res = backtest_markov(df, states, order=2, train_window=252)
    #res = backtest_markov(df, states, order=3, train_window=252)
    res = backtest_markov(df, states, order=3, train_window=104)


    # 4) evaluate
    metrics = evaluate_backtest(res)
    
    print('Backtest metrics:')
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 5) plot
    plot_backtest(res)
