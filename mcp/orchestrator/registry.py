import os

MODEL_REGISTRY = {
    "arima":os.getenv("ARIMA_URL","http://localhost:8001/mcp/sse"),
    "prophet":os.getenv("PROPHET_URL","http://localhost:8002/mcp/sse"),
}

def get_url(model_name:str) -> str:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]

def list_models():
    return list(MODEL_REGISTRY.keys())
