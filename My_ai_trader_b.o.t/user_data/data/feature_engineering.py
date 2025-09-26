# user_data/data/feature_engineering.py
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from user_data.data.news_predictor import get_news_prediction, MONITORED
from user_data.data.fetch_market_data import load_price_history

def price_features(df: pd.DataFrame):
    df = df.copy()
    df['close_ret_1d'] = df['Close'].pct_change(1)
    df['close_ret_3d'] = df['Close'].pct_change(3)
    df['vol_rolling_7d'] = df['Volume'].rolling(7).mean().fillna(0)
    df['sma_7'] = df['Close'].rolling(7).mean()
    df['sma_21'] = df['Close'].rolling(21).mean()
    df['sma_diff'] = df['sma_7'] - df['sma_21']
    return df

def get_news_features_for_date(symbol_key, date):
    """
    Use your news_predictor to get score & reason. Returns numeric features.
    date param not used for now (predictor uses recent news, cached).
    """
    pred = get_news_prediction(symbol_key)
    score = pred.get("score", 0.0)
    pred_dir = pred.get("prediction", 0)
    reasons = " ".join(pred.get("reasons", []))
    # simple keyword counts
    kw_counts = {
        "war": reasons.lower().count("war"),
        "sanction": reasons.lower().count("sanction"),
        "opec": reasons.lower().count("opec"),
        "contract": reasons.lower().count("contract"),
        "earnings": reasons.lower().count("q")  # crude, naive
    }
    # return dict for merging
    out = {"news_score": score, "news_pred": pred_dir}
    out.update(kw_counts)
    return out

def build_feature_dataset(ticker_map: dict, period_days=365):
    """
    ticker_map: dict of monitored_key -> yfinance ticker (e.g., {"RELIANCE":"RELIANCE.NS","BTC":"BTC-USD","CRUDE":"CL=F"})
    Returns DataFrame with features and label (next-day direction).
    """
    rows = []
    for key, ticker in ticker_map.items():
        df = load_price_history(ticker)
        if df.empty: 
            continue
        df = price_features(df)
        # create date index
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

        # compute next-day direction (label)
        df['next_close'] = df['Close'].shift(-1)
        df['next_ret'] = (df['next_close'] - df['Close']) / df['Close']
        df['label'] = (df['next_ret'] > 0).astype(int)  # 1 = up next day, 0 = down/flat

        # iterate rows and attach news features (use today's news to predict next-day)
        for _, r in df[:-1].iterrows():  # skip last because no next_close
            news_feats = get_news_features_for_date(key, r['date'])
            feat = {
                "symbol": key,
                "date": r['date'],
                "close": r['Close'],
                "volume": r['Volume'],
                "close_ret_1d": r.get('close_ret_1d', 0.0),
                "close_ret_3d": r.get('close_ret_3d', 0.0),
                "sma_diff": r.get('sma_diff', 0.0),
                **news_feats,
                "label": int(r['label'])
            }
            rows.append(feat)
    return pd.DataFrame(rows)
