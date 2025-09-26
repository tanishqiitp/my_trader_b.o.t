# user_data/data/fetch_market_data.py
import os
import requests
import yfinance as yf
import pandas as pd

# Directory to save CSV files
OUT_DIR = os.path.join(os.path.dirname(__file__), "market_data")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "SAMZ9513W2L4CMVA")  # fallback key

def fetch_price_history_yf(ticker: str, period="1y", interval="1d"):
    """
    Fetch price data using Yahoo Finance.
    Returns DataFrame with columns: date, Open, High, Low, Close, Volume
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} from Yahoo Finance.")
    df = df.reset_index().rename(columns={"Date": "date"})
    save_path = os.path.join(OUT_DIR, f"{ticker.replace('/', '_')}.csv")
    df.to_csv(save_path, index=False)
    return df

def fetch_price_history_alpha(symbol: str):
    """
    Fetch daily OHLC data using Alpha Vantage (for Indian stocks: symbol.NSE or symbol.BSE).
    Falls back to Yahoo Finance if Alpha Vantage returns no data.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}&outputsize=compact"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"⚠ Alpha Vantage request failed for {symbol}: {e}")
        yahoo_symbol = symbol.replace(".NSE", ".NS").replace(".BSE", ".BO")
        return fetch_price_history_yf(yahoo_symbol)

    if "Time Series (Daily)" not in data:
        print(f"⚠ Alpha Vantage returned no data for {symbol}, falling back to Yahoo Finance")
        yahoo_symbol = symbol.replace(".NSE", ".NS").replace(".BSE", ".BO")
        return fetch_price_history_yf(yahoo_symbol)

    time_series = data["Time Series (Daily)"]
    rows = []
    for date_str, values in time_series.items():
        rows.append({
            "date": pd.to_datetime(date_str),
            "Open": float(values["1. open"]),
            "High": float(values["2. high"]),
            "Low": float(values["3. low"]),
            "Close": float(values["4. close"]),
            "Volume": int(float(values["5. volume"]))
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    save_path = os.path.join(OUT_DIR, f"{symbol.replace('.', '_')}.csv")
    df.to_csv(save_path, index=False)
    return df

def load_price_history(ticker: str):
    """
    Load saved data if present, otherwise fetch from Yahoo Finance.
    """
    path = os.path.join(OUT_DIR, f"{ticker.replace('/', '_')}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return fetch_price_history_yf(ticker)
