# AI-Powered Market Prediction Bot

This project predicts market trends for crypto, commodities, and Indian stocks (Nifty 50) by combining news sentiment analysis and recent price action.

## Features
- ✅ Predicts BTC, GOLD, CRUDE, and Nifty 50 stocks (e.g., TCS, Reliance)
- ✅ Fetches latest news from APIs and performs sentiment scoring
- ✅ Lightweight ML model for up/down prediction
- ✅ Dashboard to visualize predictions (Streamlit)
- ✅ Easy to expand — just edit `monitored_assets.json`

## Getting Started
```bash
git clone https://github.com/YOUR_USERNAME/my_trading_ai.git
cd my_trading_ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn dashboard.api.main:app --reload --port 8000
streamlit run dashboard/app.py
