"""
Upgraded Markov-chain Nifty-50 weekly predictor + backtester + option sell recommendation

New additions in this version:
 - Volatility-weighted position sizing (uses rolling weekly returns volatility)
 - Dynamic selection of weekly OTM strike based on 1σ expected weekly move
 - Logs an option recommendation for every backtest week (side, strike, otm_steps, note)
 - Saves per-trade CSV with the option suggestion column included

Usage: run the script or import functions in a notebook. If CSV `nifty_weekly.csv` exists it will be used; otherwise the script fetches daily data from Yahoo and resamples to weekly.

Note: Option recommendations only pick strikes by heuristic. To execute option trades you must query your broker for live option chains and premium, and verify margin and risk.
"""

from typing import Tuple, List, Optional, Dict
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# ------------------------- Data helpers -------------------------

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        lvl1 = list(df.columns.get_level_values(1))
        if any(x and 'Close' in str(x) for x in lvl0) or any(x and 'Open' in str(x) for x in lvl0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_date_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError('Could not convert index to datetime. Provide a Date column or valid datetime index.')
    df = df.sort_index()
    return df


def resample_weekly_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    inferred = pd.infer_freq(df.index[:10]) if len(df.index) >= 3 else None
    if inferred and ('W' in inferred or 'W-' in str(inferred)):
        return df
    o = df['Open'].resample('W-FRI').first()
    h = df['High'].resample('W-FRI').max()
    l = df['Low'].resample('W-FRI').min()
    c = df['Close'].resample('W-FRI').last()
    v = df['Volume'].resample('W-FRI').sum()
    weekly = pd.concat([o, h, l, c, v], axis=1)
    weekly.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    weekly = weekly.dropna()
    return weekly


# ------------------------- State creation -------------------------

def candle_states(df: pd.DataFrame, method: str = 'sign', neutral_thresh: float = 0.002, bins: int = 5) -> pd.Series:
    o = df['Open']
    c = df['Close']
    ret_open = (c - o) / o
    if method == 'sign':
        states = pd.Series(index=df.index, dtype=object)
        states.loc[ret_open > neutral_thresh] = 'U'
        states.loc[ret_open < -neutral_thresh] = 'D'
        states.loc[ret_open.abs() <= neutral_thresh] = 'N'
        return states
    elif method == 'quantile':
        r = df['Close'].pct_change().fillna(0)
        labels = list(range(bins))
        states = pd.qcut(r, q=bins, labels=labels, duplicates='drop')
        return states.astype(int)
    else:
        raise ValueError('Unknown method for candle_states')


# ------------------------- Markov chain model -------------------------

class MarkovChainPredictor:
    def __init__(self, order: int = 1):
        self.order = order
        self.transition_counts = {}
        self.transition_probs = {}
        self.states = None

    def fit(self, state_series: pd.Series) -> None:
        s = state_series.dropna().astype(object)
        sequences = []
        for i in range(self.order, len(s)):
            prev = tuple(s.iloc[i - self.order:i].tolist())
            cur = s.iloc[i]
            sequences.append((prev, cur))
        unique_states = sorted(s.unique(), key=lambda x: str(x))
        self.states = unique_states
        counts = {}
        for prev, cur in sequences:
            counts.setdefault(prev, {})
            counts[prev][cur] = counts[prev].get(cur, 0) + 1
        probs = {}
        for prev, d in counts.items():
            total = sum(d.values())
            probs[prev] = {k: v / total for k, v in d.items()}
        self.transition_counts = counts
        self.transition_probs = probs

    def predict_next_proba(self, prev_state: Tuple) -> Dict:
        if self.transition_probs is None:
            raise RuntimeError('Model not fitted')
        if prev_state in self.transition_probs:
            return self.transition_probs[prev_state]
        else:
            agg = {}
            for p, d in self.transition_probs.items():
                for k, v in d.items():
                    agg[k] = agg.get(k, 0) + v
            s = sum(agg.values())
            return {k: v / s for k, v in agg.items()}

    def most_likely(self, prev_state: Tuple) -> Tuple:
        proba = self.predict_next_proba(prev_state)
        return max(proba.items(), key=lambda x: x[1])


# ------------------------- Option helper -------------------------

def recommend_weekly_option_sell_dynamic(close_price: float,
                                          vol_sigma: float,
                                          direction: int,
                                          strike_step: int = 50,
                                          min_steps: int = 1) -> Dict:
    """
    Choose an OTM strike based on expected 1σ weekly move (vol_sigma is weekly std of returns)
    - expected_move = close * vol_sigma
    - otm_steps = max(min_steps, int(expected_move // strike_step))
    direction: 1 (bullish) -> sell PE OTM below spot
               -1 (bearish) -> sell CE OTM above spot
    """
    if direction == 0:
        return {'side': None, 'strike': None, 'otm_steps': None, 'note': 'No trade (neutral)'}

    expected_move = close_price * vol_sigma
    otm_steps = max(min_steps, int(max(1, np.round(expected_move / strike_step))))
    nearest = int(round(close_price / strike_step) * strike_step)
    if direction == 1:
        strike = nearest - otm_steps * strike_step
        side = 'PE'
        note = f'Sell weekly {side} strike {strike} (OTM {otm_steps * strike_step} pts), exp_move≈{expected_move:.1f}'
    else:
        strike = nearest + otm_steps * strike_step
        side = 'CE'
        note = f'Sell weekly {side} strike {strike} (OTM {otm_steps * strike_step} pts), exp_move≈{expected_move:.1f}'

    return {'side': side, 'strike': strike, 'otm_steps': otm_steps, 'note': note}


# ------------------------- Backtester with vol-weighted sizing -------------------------

def backtest_markov(df: pd.DataFrame,
                     state_series: pd.Series,
                     order: int = 3,
                     train_window: int = 104,
                     prob_threshold: float = 0.6,
                     use_trend_filter: bool = True,
                     sma_short: int = 5,
                     sma_long: int = 20,
                     position_scale: bool = True,
                     vol_window: int = 10,
                     strike_step: int = 50) -> pd.DataFrame:
    """Walk-forward weekly backtest with volatility-weighted sizing and option recommendation logging."""
    preds = []
    probs = []
    prev_states = []
    actions = []
    returns = []
    sizes = []
    dates = []
    option_recs = []

    # precompute SMAs and volatility
    if use_trend_filter:
        sma_s = df['Close'].rolling(sma_short).mean()
        sma_l = df['Close'].rolling(sma_long).mean()
    vol = df['Close'].pct_change().rolling(vol_window).std()  # weekly sigma
    vol_median = vol.median() if not np.isnan(vol.median()) else 0.0

    for i in range(train_window, len(state_series) - 1):
        train_states = state_series.iloc[i - train_window:i]
        mc = MarkovChainPredictor(order=order)
        mc.fit(train_states)
        prev_state = tuple(state_series.iloc[i - order:i].tolist())
        proba_dict = mc.predict_next_proba(prev_state)
        pred_state, pred_p = max(proba_dict.items(), key=lambda x: x[1])

        action = 0
        size = 0.0

        trend_ok = True
        if use_trend_filter and not (np.isnan(sma_s.iloc[i]) or np.isnan(sma_l.iloc[i])):
            trend_ok = sma_s.iloc[i] > sma_l.iloc[i]

        if pred_p >= prob_threshold and trend_ok:
            if isinstance(pred_state, str):
                if pred_state == 'U':
                    action = 1
                elif pred_state == 'D':
                    action = -1
                else:
                    action = 0
            else:
                median_label = np.median(list(mc.states)) if mc.states else 0
                action = 1 if pred_state > median_label else -1

            # volatility-weighted size: lean on confidence, but reduce size in high vol
            if position_scale:
                sigma_i = vol.iloc[i] if not np.isnan(vol.iloc[i]) else vol_median if vol_median > 0 else 1.0
                # avoid division by zero; size = pred_p * (vol_median / sigma_i)
                if sigma_i <= 0 or vol_median <= 0:
                    size = pred_p
                else:
                    size = pred_p * (vol_median / sigma_i)
                # cap size to [0, 1.5] for safety
                size = float(np.clip(size, 0.0, 1.5))
            else:
                size = 1.0

        next_open = df['Open'].iloc[i + 1]
        next_close = df['Close'].iloc[i + 1]
        ret = ((next_close - next_open) / next_open) * action * size

        # prepare option recommendation for this week (based on close at i)
        last_close = df['Close'].iloc[i]
        sigma_for_option = vol.iloc[i] if not np.isnan(vol.iloc[i]) else vol_median if vol_median > 0 else 0.01
        direction = int(np.sign(action))
        opt = recommend_weekly_option_sell_dynamic(last_close, sigma_for_option, direction, strike_step=strike_step, min_steps=1)

        preds.append(pred_state)
        probs.append(pred_p)
        prev_states.append(prev_state)
        actions.append(action)
        returns.append(ret)
        sizes.append(size)
        dates.append(df.index[i + 1])
        option_recs.append(opt)

    res = pd.DataFrame({
        'date': dates,
        'prev_state': prev_states,
        'pred': preds,
        'prob': probs,
        'action': actions,
        'size': sizes,
        'pct_return': returns,
        'option_rec': option_recs
    }).set_index('date')

    res['cum_return_strategy'] = (1 + res['pct_return']).cumprod() - 1
    res['cum_return_hold'] = (1 + df['Close'].pct_change().shift(-1).reindex(res.index).fillna(0)).cumprod() - 1
    return res


def evaluate_backtest(res: pd.DataFrame) -> dict:
    trades = res[res['action'] != 0]
    if len(trades) == 0:
        hit_rate = None
    else:
        wins = (trades['pct_return'] > 0).sum()
        hit_rate = wins / len(trades)
    total_ret = res['cum_return_strategy'].iloc[-1] if len(res) > 0 else 0
    periods = len(res)
    ann_ret = (1 + total_ret) ** (52 / periods) - 1 if periods > 0 else 0
    dr = res['pct_return'].fillna(0)
    sharpe = (np.sqrt(52) * dr.mean() / dr.std()) if dr.std() != 0 else None
    return {
        'n_predictions': len(res),
        'n_trades': len(trades),
        'hit_rate': hit_rate,
        'total_return': total_ret,
        'annualized_return': ann_ret,
        'sharpe': sharpe
    }


def plot_backtest(res: pd.DataFrame, title: str = 'Backtest cumulative returns'):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(res.index, res['cum_return_strategy'], label='Strategy cumulative')
    ax.plot(res.index, res['cum_return_hold'], label='Next-week hold cum')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

# --- Upcoming Week Recommendation ---

def recommend_next_week(df, markov_model, order, prob_threshold, lot_step=50):
    """Generate next-week directional and option recommendation."""
    last_seq = df['state'].iloc[-order:].tolist()
    state_tuple = tuple(last_seq)
    if state_tuple not in markov_model:
        print("No known transition for current state, skipping.")
        return None

    proba = markov_model[state_tuple]
    pred_state, pred_p = max(proba.items(), key=lambda x: x[1])
    direction = 'Bullish' if pred_state == 'Up' else 'Bearish'

    # expected 1σ move
    sigma = df['Close'].pct_change().rolling(20).std().iloc[-1]
    expected_move = df['Close'].iloc[-1] * sigma
    base_price = df['Close'].iloc[-1]

    if direction == 'Bullish':
        strike = base_price - (expected_move // lot_step + 1) * lot_step
        rec = f"Sell {int(strike)} PE"
    else:
        strike = base_price + (expected_move // lot_step + 1) * lot_step
        rec = f"Sell {int(strike)} CE"

    print(f"\nNext week prediction (ending {df.index[-1] + pd.Timedelta(days=7):%Y-%m-%d}):")
    print(f"  ➤ Signal: {direction}")
    print(f"  ➤ Expected move: ±{expected_move:.0f} pts")
    print(f"  ➤ Recommendation: {rec}")
    print(f"  ➤ Confidence: {pred_p:.2f}")

    return {
        "week": df.index[-1],
        "direction": direction,
        "expected_move": expected_move,
        "recommendation": rec,
        "confidence": pred_p
    }


# ------------------------- Main execution -------------------------
if __name__ == '__main__':
    try:
        df = pd.read_csv('nifty_weekly.csv', index_col=0, parse_dates=True)
    except Exception:
        print('Could not load CSV; fetching from Yahoo and resampling to weekly...')
        df = yf.download('^NSEI', period='10y', interval='1d', progress=False)

    df = flatten_columns(df)
    df = ensure_date_index(df)
    df = resample_weekly_if_needed(df)

    states = candle_states(df, method='sign', neutral_thresh=0.002)

    res = backtest_markov(df, states, order=3, train_window=104, prob_threshold=0.6,
                          use_trend_filter=True, sma_short=5, sma_long=20, position_scale=True,
                          vol_window=10, strike_step=50)

    metrics = evaluate_backtest(res)
    print('Backtest metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')

    print('Sample trades:')
    # expand option_rec dict columns for readability in head
    sample = res.head(20).copy()
    sample['option_side'] = sample['option_rec'].apply(lambda x: x.get('side') if isinstance(x, dict) else None)
    sample['option_strike'] = sample['option_rec'].apply(lambda x: x.get('strike') if isinstance(x, dict) else None)
    sample['option_note'] = sample['option_rec'].apply(lambda x: x.get('note') if isinstance(x, dict) else None)
    print(sample.head(20))

    plot_backtest(res)

    # save results including option recommendation
    res.to_csv('markov_weekly_backtest_results_with_options.csv')
    print('Saved backtest results to markov_weekly_backtest_results_with_options.csv')
    
    
    # Call this at the end of your script
    # --- Rebuild model on full weekly data ---
    # final_model = MarkovChainPredictor(order=3)
    # final_model.fit(df['state'])
    # next_week_rec = recommend_next_week(df, final_model, order=3, prob_threshold=0.6)
