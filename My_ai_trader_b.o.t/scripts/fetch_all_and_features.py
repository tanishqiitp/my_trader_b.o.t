import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from user_data.data.fetch_market_data import fetch_price_history_alpha, fetch_price_history_yf

ticker_map = {
    "BTC": "BTC-USD",       # Yahoo Finance
    "GOLD": "GC=F",         # Yahoo Finance
    "CRUDE": "CL=F",        # Yahoo Finance
    "TCS": "TCS.BSE",       # Alpha Vantage
    "RELIANCE": "RELIANCE.NSE"
}

# Fetch data
for key, ticker in ticker_map.items():
    if ticker.endswith(".BSE") or ticker.endswith(".NSE"):
        df = fetch_price_history_alpha(ticker)
    else:
        df = fetch_price_history_yf(ticker)
    print(f"Fetched {key}: {len(df)} rows")
