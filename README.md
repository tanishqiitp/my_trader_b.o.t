# NewsTrader

Main app folder: `My_ai_trader_b.o.t`

This bot analyzes manually entered stocks, or broker-exported watchlists/holdings, using current market news. It now supports rules, trained ML, optional FinBERT DL sentiment, and a hybrid mode. The dashboard groups stocks into three columns:

- `TO BUY`
- `HOLD`
- `SELL`

Run from `My_ai_trader_b.o.t`:

```bash
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
streamlit run dashboard/app.py
```
