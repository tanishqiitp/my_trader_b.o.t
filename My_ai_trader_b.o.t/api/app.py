# api/app.py
from fastapi import FastAPI, HTTPException, Query
import uvicorn
from models.model_utils import load_model, predict_single
from user_data.data.feature_engineering import price_features
from user_data.data.fetch_market_data import load_price_history
import pandas as pd

app = FastAPI(title="NewsTrader API")

model = None
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get("/predict")
def predict(asset: str = Query(..., description="MONITORED key, e.g., BTC, RELIANCE")):
    # map asset to ticker; maintain same ticker_map used in training
    ticker_map = {
        "BTC":"BTC-USD",
        "GOLD":"GC=F",
        "CRUDE":"CL=F",
        "RELIANCE":"RELIANCE.NS"
    }
    if asset not in ticker_map:
        raise HTTPException(status_code=400, detail="Unknown asset")
    ticker = ticker_map[asset]
    # load latest price row and compute price features for the last day
    df = load_price_history(ticker)
    df = price_features(df)
    last = df.iloc[-1].to_dict()
    # get news features
    from user_data.data.news_predictor import get_news_prediction
    news = get_news_prediction(asset)
    # build feat dict
    feat = {
        "close_ret_1d": last.get("close_ret_1d",0.0),
        "close_ret_3d": last.get("close_ret_3d",0.0),
        "sma_diff": last.get("sma_diff",0.0),
        "news_score": news.get("score", 0.0),
        "news_pred": news.get("prediction", 0),
        "war": int("war" in " ".join(news.get("reasons",[])).lower()),
        "sanction": int("sanction" in " ".join(news.get("reasons",[])).lower()),
        "opec": int("opec" in " ".join(news.get("reasons",[])).lower()),
        "contract": int("contract" in " ".join(news.get("reasons",[])).lower()),
        "earnings": int("earnings" in " ".join(news.get("reasons",[])).lower()),
        "volume": last.get("Volume", 0)
    }
    out = predict_single(model, feat)
    return {"asset": asset, "features": feat, "prediction": out, "news": news}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
