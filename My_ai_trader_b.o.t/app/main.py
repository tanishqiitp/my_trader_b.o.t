from fastapi import FastAPI
from user_data.data.news_predictor import get_news_prediction
from user_data.data.fetch_market_data import load_price_history

app = FastAPI()

@app.get("/predict")
def predict(asset: str):
    try:
        result = get_news_prediction(asset.upper())
        return {"asset": asset, "result": result}
    except Exception as e:
        return {"error": str(e)}

@app.get("/price-history")
def price_history(asset: str):
    try:
        df = load_price_history(asset.upper())
        return df.tail(10).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
