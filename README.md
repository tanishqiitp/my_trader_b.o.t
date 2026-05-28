# NewsTrader

NewsTrader analyzes your stock watchlist from current market news and groups every stock into:

- `TO BUY`
- `HOLD`
- `SELL`

The current version is news-driven. It can run with transparent rules only, use a trained ML classifier when you provide labeled news examples, or use an optional FinBERT deep-learning sentiment backend.

## User Flow

1. Start the API.
2. Start the dashboard.
3. Paste stocks manually, for example `TCS, RELIANCE, INFY, HDFCBANK`.
4. Click `Analyze Manual List` or `Save As Watchlist`.
5. Review the three dashboard columns: `TO BUY`, `HOLD`, `SELL`.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```

In another terminal:

```bash
streamlit run dashboard/app.py
```

## ML/DL Modes

The dashboard has a `Model mode` selector:

- `rules`: deterministic sentiment + event scoring.
- `ml`: trained text classifier only.
- `dl`: FinBERT sentiment, if installed.
- `hybrid`: rules plus trained ML, and FinBERT when enabled.

Rules are the fallback because they work without training data. ML/DL can improve the signal only after you give the model good examples or install a real finance NLP model.

### Train The ML Classifier

Create a CSV with at least 15 rows:

```csv
text,label
"TCS wins major AI contract from global manufacturer",TO_BUY
"Infosys reports results in line with expectations",HOLD
"Reliance faces regulatory penalty and weak demand",SELL
```

Allowed labels are `TO_BUY`, `HOLD`, and `SELL`.

Then run:

```bash
pip install -r requirements-ml.txt
python scripts/train_news_model.py C:\path\to\your_news_labels.csv
```

The model is saved to `models/news_action_classifier.joblib` and the API/dashboard will use it in `ml` or `hybrid` mode.

### Enable The DL Backend

FinBERT is optional because it downloads a large model the first time it runs:

```bash
pip install -r requirements-dl.txt
set NEWS_ENABLE_FINBERT=1
```

Then choose `dl`, or choose `hybrid` and tick `Use FinBERT in hybrid`.

## Manual Watchlist Format

Use comma-separated or newline-separated symbols:

```text
TCS, RELIANCE, INFY, HDFCBANK, SBIN
```

You can also give a display name:

```text
Tata Motors:TATAMOTORS.NS
ICICI Bank:ICICIBANK.NS
```

## Broker Export Bridge

Live Zerodha/Groww integration needs account credentials and broker approval. For now, the safe bridge is to export holdings/watchlist as CSV or JSON and point the app to it:

```bash
set ZERODHA_HOLDINGS_FILE=C:\path\to\zerodha_holdings.csv
set GROWW_HOLDINGS_FILE=C:\path\to\groww_holdings.csv
```

Then use `Import Broker Export` in the dashboard.

## News Sources

If `NEWSAPI_KEY` is set, the app uses NewsAPI first. Otherwise it falls back to Google News RSS.

```bash
set NEWSAPI_KEY=your_key_here
```

## API

- `GET /watchlist`
- `POST /watchlist/manual`
- `POST /watchlist/replace`
- `POST /watchlist/broker/{zerodha|groww}`
- `GET /predict?asset=TCS`
- `GET /signals`
- `GET /model/status`
- `POST /model/train`

This is an analysis tool, not financial advice or an auto-trading system.
