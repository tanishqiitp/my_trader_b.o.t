# dashboard/api/main.py
from fastapi import FastAPI
from user_data.data.news_predictor import get_news_prediction

app = FastAPI()

@app.get("/")
def root():
    return {"message": "✅ API is running"}

@app.get("/predict")
def predict(asset: str):
    try:
        result = get_news_prediction(asset)
        return {"status": "success", "asset": asset, "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

