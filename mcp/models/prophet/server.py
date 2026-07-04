import logging
import hashlib
import json
from fastmcp import FastMCP
from fastapi import FastAPI
from prophet import Prophet as ProphetModel
from pydantic import BaseModel
from typing import List
import pandas as pd
import uvicorn
from dotenv import load_dotenv

load_dotenv()

logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

mcp = FastMCP("prophet-service")
app = FastAPI(title="Prophet Forecasting Service")
app.mount("/mcp", mcp.http_app(path="/sse", transport="sse"))

class ForecastRequest(BaseModel):
    dates:  List[str]
    values: List[float]
    n:      int

# ── Model cache ───────────────────────────────────────────
# Stores: { data_hash: trained_model }
# Key is a hash of the input data so different datasets
# get their own cached model
_model_cache = {}

def get_cache_key(dates: List[str], values: List[float]) -> str:
    """
    Create a unique key from the input data.
    Same data = same key = cache hit.
    Different data = different key = retrain.
    """
    payload = json.dumps({"dates": dates, "values": values}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()

def get_or_train_model(dates: List[str], values: List[float]) -> ProphetModel:
    """
    Return cached model if data matches.
    Train a new one and cache it if not.
    """
    key = get_cache_key(dates, values)

    if key in _model_cache:
        # Cache hit — return existing trained model
        logging.info("Prophet: cache hit, skipping training")
        return _model_cache[key]

    # Cache miss — train from scratch
    logging.info("Prophet: cache miss, training model")

    df = pd.DataFrame({
        'ds': pd.to_datetime(dates),
        'y':  values
    })

    model = ProphetModel(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df)

    # Save to cache for next time
    _model_cache[key] = model
    logging.info("Prophet: model cached")

    return model

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "prophet",
        "cached_models": len(_model_cache)  # shows how many models are cached
    }

@mcp.tool
def forecast(request: ForecastRequest) -> dict:
    """
    Forecast next n values using Prophet.
    Best for data with seasonal patterns.
    Uses model caching to avoid retraining on repeated datasets.

    Args:
        dates: list of date strings in YYYY-MM-DD format
        values: list of historical float values
        n: number of future steps to predict
    """
    if len(request.values) < 5:
        return {"model": "prophet", "error": "Need at least 5 data points", "forecast": []}

    try:
        # Get cached model or train new one
        model = get_or_train_model(request.dates, request.values)

        future = model.make_future_dataframe(
            periods=request.n,
            freq='D',
            include_history=False
        )
        forecast_df = model.predict(future)

        return {
            "model": "prophet",
            "forecast": [round(float(v), 4) for v in forecast_df['yhat'].values],
            "steps": request.n,
            "cache_hit": get_cache_key(request.dates, request.values) in _model_cache
        }

    except Exception as e:
        return {"model": "prophet", "error": str(e), "forecast": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
