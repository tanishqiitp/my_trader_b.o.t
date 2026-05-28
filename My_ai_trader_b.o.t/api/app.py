from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from user_data.data.ml_signal import model_status, train_news_classifier
from user_data.data.news_predictor import (
    analyze_watchlist,
    get_news_prediction,
    refresh_monitored_assets,
)
from user_data.data.watchlist import (
    load_broker_export,
    load_watchlist,
    replace_with_manual_entries,
    upsert_manual_entries,
)

app = FastAPI(title="NewsTrader API", version="2.0.0")


class ManualWatchlistRequest(BaseModel):
    stocks: str
    persist: bool = True


class TrainModelRequest(BaseModel):
    csv_path: Optional[str] = None


@app.get("/")
def root():
    return {
        "status": "running",
        "mode": "hybrid_news",
        "message": "Submit stocks manually or import broker exports, then read grouped TO_BUY/HOLD/SELL signals.",
    }


@app.get("/watchlist")
def watchlist():
    return {"watchlist": load_watchlist()}


@app.post("/watchlist/manual")
def add_manual_watchlist(payload: ManualWatchlistRequest):
    if not payload.stocks.strip():
        raise HTTPException(status_code=400, detail="Provide at least one stock symbol.")
    updated = upsert_manual_entries(payload.stocks, persist=payload.persist)
    refresh_monitored_assets()
    return {"status": "updated", "count": len(updated), "watchlist": updated}


@app.post("/watchlist/replace")
def replace_watchlist(payload: ManualWatchlistRequest):
    if not payload.stocks.strip():
        raise HTTPException(status_code=400, detail="Provide at least one stock symbol.")
    updated = replace_with_manual_entries(payload.stocks, persist=payload.persist)
    refresh_monitored_assets()
    return {"status": "replaced", "count": len(updated), "watchlist": updated}


@app.post("/watchlist/broker/{provider}")
def import_broker_watchlist(provider: str):
    if provider.lower() not in {"zerodha", "kite", "groww", "grow"}:
        raise HTTPException(status_code=400, detail="Supported broker imports: zerodha, groww.")
    result = load_broker_export(provider)
    refresh_monitored_assets()
    return result


@app.get("/predict")
def predict(
    asset: str = Query(..., description="Watchlist key, e.g. TCS, RELIANCE, HDFCBANK"),
    max_headlines: int = Query(25, ge=5, le=50),
    use_newsapi: bool = True,
    model_mode: str = Query("hybrid", description="rules, ml, dl, or hybrid"),
    enable_finbert: Optional[bool] = Query(None, description="Force FinBERT on/off for dl/hybrid mode."),
):
    try:
        return get_news_prediction(
            asset,
            use_newsapi=use_newsapi,
            max_headlines=max_headlines,
            assets=load_watchlist(),
            model_mode=model_mode,
            enable_finbert=enable_finbert,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/signals")
def signals(
    max_headlines: int = Query(25, ge=5, le=50),
    use_newsapi: bool = True,
    model_mode: str = Query("hybrid", description="rules, ml, dl, or hybrid"),
    enable_finbert: Optional[bool] = Query(None, description="Force FinBERT on/off for dl/hybrid mode."),
    manual: Optional[str] = Query(
        None,
        description="Optional comma-separated stocks to analyze without saving, e.g. TCS,RELIANCE,HDFCBANK",
    ),
):
    if manual:
        assets = replace_with_manual_entries(manual, persist=False)
    else:
        assets = load_watchlist()
    return analyze_watchlist(
        assets=assets,
        use_newsapi=use_newsapi,
        max_headlines=max_headlines,
        model_mode=model_mode,
        enable_finbert=enable_finbert,
    )


@app.get("/model/status")
def get_model_status():
    return model_status()


@app.post("/model/train")
def train_model(payload: TrainModelRequest):
    try:
        return {"status": "trained", "metrics": train_news_classifier(payload.csv_path)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
