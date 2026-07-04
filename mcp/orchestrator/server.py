#PART 1 : Import and Setup 
import asyncio
import logging
from fastmcp import FastMCP, Client
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph
from typing import TypedDict,List
import uvicorn
from registry import get_url, list_models
from dotenv import load_dotenv

load_dotenv()

import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("orchestrator")
app = FastAPI(title="Forecasting Orchestrator")
app.mount("/mcp",mcp.http_app(path="/sse",transport="sse"))


#PART 2: Shared request model

class ForecastRequest(BaseModel):
    dates: List[str]
    values: List[float]
    n: int

class ForecastState(TypedDict):

    #input
    dates: List[str]
    values: List[float]
    n: int

    #after split node
    train_dates: List[str]
    train_values: List[float]
    test_dates: List[str]
    test_values: List[float]

    #after validation node
    arima_val: List[float]
    prophet_val: List[float]
    arima_mae: float
    prophet_mae: float
    winner: str

    #after forecast node:
    final_forecast: List[float]
    

#PART 3: Helper Functions

def calculate_mae(actual: List[float], predicted:List[float]) -> float:
    if not predicted or not actual:  # added this check
        return float('inf') 
    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
    return sum(abs(a-p) for a,p in zip(actual,predicted))/len(actual)

async def call_model_service(model_name:str, dates:List[str], values:List[float], n:int) -> List[float]:
    url = get_url(model_name)
    try:
        async with Client(url) as client:
            result = await client.call_tool(
                "forecast",
                {
                    "request":{
                        "dates": dates,
                        "values": values,
                        "n": n
                    }
                }
            )
            return result.data.get("forecast",[])
    except Exception as e:
        logger.error(f"{model_name} service failed:{e}")
        return []
    
#PART 4: LangGraph Nodes

#Node 1: Split Data
def split_data(state: ForecastState) -> ForecastState:
    dates = state["dates"]
    values = state["values"]

    test_size = max(3, int(len(values)* 0.2))
    train_size = len(values) - test_size

    logger.info(f"Split: {train_size} train, {test_size} test")

    return{
        **state,
        "train_dates":dates[:train_size],
        "train_values":values[:train_size],
        "test_dates":dates[train_size:],
        "test_values":values[train_size:],
    }

#Node 2: Validate both models async

async def validate_models(state: ForecastState) -> ForecastState:
    
    test_size = len(state["test_values"])

    arima_pred, prophet_pred = await asyncio.gather(
        call_model_service(
            "arima",
            state["train_dates"],
            state["train_values"],
            test_size
        ),
        call_model_service(
            "prophet",
            state["train_dates"],
            state["train_values"],
            test_size
        )
    )
    if not arima_pred:
        logger.error("ARIMA returned empty — service may be down")
    if not prophet_pred:
        logger.error("Prophet returned empty — service may be down")

    arima_mae   = calculate_mae(state["test_values"], arima_pred)
    prophet_mae = calculate_mae(state["test_values"], prophet_pred)

    winner = "arima" if arima_mae <= prophet_mae else "prophet"

    logger.info(f"ARIMA MAE: {arima_mae}")
    logger.info(f"Prophet MAE: {prophet_mae}")
    logger.info(f"Winner: {winner}")

    return {
        **state,
        "arima_val":   arima_pred,
        "prophet_val": prophet_pred,
        "arima_mae":   arima_mae if arima_mae == float('inf') else round(arima_mae, 4),
        "prophet_mae": prophet_mae if prophet_mae == float('inf') else round(prophet_mae, 4),
        "winner":      winner,
    }

#Node 3: Final Forecast

async def final_forecast(state: ForecastState) -> ForecastState:

    logger.info(f"Running final forecast with {state['winner']}")

    forecast = await call_model_service(
        state["winner"],
        state["dates"],
        state["values"],
        state["n"]
    )

    return {
        **state,
        "final_forecast": forecast
    }

#PART 5: Build LangGraph

def build_graph():
    graph = StateGraph(ForecastState)

    graph.add_node("split_data", split_data)
    graph.add_node("validate_models", validate_models)
    graph.add_node("final_forecast", final_forecast)

    graph.set_entry_point("split_data")
    graph.add_edge("split_data","validate_models")
    graph.add_edge("validate_models","final_forecast")
    graph.set_finish_point("final_forecast")

    return graph.compile()

forecast_graph = build_graph()

@mcp.tool
async def getForecast(request: ForecastRequest) -> dict:
    """
    Forecast future values of a time series using backtested model selection.
    Compares ARIMA and Prophet on a validation split, picks the more accurate 
    model, and returns a forecast.

    Args:
        dates: list of date strings in YYYY-MM-DD format, one per data point.
               REQUIRED — if the user has not provided dates, ask them:
               "What date does the first value correspond to, and how frequently 
               is the data recorded? (daily, weekly, monthly)"
               Then construct dates accordingly before calling this tool.
               
        values: list of float values, one per date.
                Accept data in any format the user provides:
                - comma separated: "10, 13, 16, 19"
                - list format: [10, 13, 16, 19]
                - table or column format
                - natural language: "starts at 10, increases by 3 each day"
                Always convert to a clean list of floats before calling.

        n: number of future values to predict.
           If user says "next week" and data is daily, n=7.
           If user says "next month" and data is weekly, n=4.
           If user doesn't specify, ask them.
           Any positive integer is valid — do not reduce n without asking.

    Requires minimum 15 data points.
    Returns winner model, forecast, MAE comparison, and validation results.
    """
    if len(request.values) < 15:
        return{
            "error" : "Need at least 15 data points for reliable backtesting"
        }
    
    result = await forecast_graph.ainvoke({
        "dates": request.dates,
        "values": request.values,
        "n": request.n,
        "train_dates": [],
        "train_values": [],
        "test_dates": [],
        "test_values": [],
        "arima_val": [],
        "prophet_val": [],
        "arima_mae": 0.0,
        "prophet_mae": 0.0,
        "winner": "",
        "final_forecast": []
    })
    return {
        "winner":   result["winner"],
        "forecast": result["final_forecast"],
        "accuracy": {
            "arima_mae":   result["arima_mae"],
            "prophet_mae": result["prophet_mae"],
        },
        "validation": {
            "actual":          result["test_values"],
            "arima_predicted": result["arima_val"],
            "prophet_predicted": result["prophet_val"],
        }
    }



# Add to orchestrator/server.py temporarily for benchmarking

@mcp.tool
async def getForecast_sequential(dates: List[str], values: List[float], n: int) -> dict:
    """Same as getForecast but calls models sequentially instead of in parallel"""
    
    # split data same way
    test_size  = max(3, int(len(values) * 0.2))
    train_size = len(values) - test_size
    train_dates  = dates[:train_size]
    train_values = values[:train_size]

    start = time.time()

    # Sequential — one after the other
    arima_pred   = await call_model_service("arima",   train_dates, train_values, test_size)
    prophet_pred = await call_model_service("prophet", train_dates, train_values, test_size)

    sequential_time = time.time() - start

    return {
        "method": "sequential",
        "time_seconds": round(sequential_time, 4),
        "arima_pred": arima_pred,
        "prophet_pred": prophet_pred
    }

@mcp.tool
async def getForecast_parallel(dates: List[str], values: List[float], n: int) -> dict:
    """Same as getForecast but explicitly timed for benchmarking"""

    test_size  = max(3, int(len(values) * 0.2))
    train_size = len(values) - test_size
    train_dates  = dates[:train_size]
    train_values = values[:train_size]

    start = time.time()

    # Parallel — both at same time
    arima_pred, prophet_pred = await asyncio.gather(
        call_model_service("arima",   train_dates, train_values, test_size),
        call_model_service("prophet", train_dates, train_values, test_size)
    )

    parallel_time = time.time() - start

    return {
        "method": "parallel",
        "time_seconds": round(parallel_time, 4),
        "arima_pred": arima_pred,
        "prophet_pred": prophet_pred
    }



@app.get("/health")
async def health():
    service_health = {}

    for model_name in list_models():  
        url = get_url(model_name)
        try:
            async with Client(url) as client:
                await client.ping()
            service_health[model_name] = "ok"
        except:
            service_health[model_name] = "unreachable"

    return {
        "orchestrator": "ok",
        "services": service_health
    }

if __name__ == "__main__":
    uvicorn.run(app,host = "0.0.0.0", port = 8000)
