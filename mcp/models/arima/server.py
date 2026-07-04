import warnings
from fastmcp import FastMCP
from fastapi import FastAPI
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
from pydantic import BaseModel
from typing import List
import uvicorn
from dotenv import load_dotenv

load_dotenv()

class ForecastRequest(BaseModel):
    dates: List[str]
    values: List[float]
    n: int

mcp = FastMCP("arima-service")
app = FastAPI(title="ARIMA Forecasting Service")

app.mount("/mcp",mcp.http_app(path="/sse",transport="sse"))

@app.get("/health")
def health():
    return{"status":"ok", "service":"arima"}

@mcp.tool
def forecast(request : ForecastRequest)->dict:
    """
    Forecast next n values using ARIMA model.
    Best for data with consistent trends (steadily increasing or decreasing).
    Requires minimum 10 data points.

    Args:
        dates: list of date strings in YYYY-MM-DD format
        values: list of historical float values
        n: number of future steps to predict
    """
    
    if len(request.values) < 10:
        return{"model":"arima","error":"Need at least 10 data points","forecast":[]}
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = StatsARIMA(request.values,order=(2,1,2))
            fitted = model.fit()
            result = fitted.forecast(steps = request.n)
        return{
            "model" : "arima",
            "forecast":[round(float(v),4) for v in result],
            "steps" : request.n
        }
    except Exception as e:
        return{"model":"arima","error":str(e),"forecast":[]}

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8001)
