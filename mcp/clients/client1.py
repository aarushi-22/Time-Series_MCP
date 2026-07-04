import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport
import os
import logging
logging.getLogger("mcp.client.sse").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)


url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000/mcp/sse")
transport = SSETransport(url)

dataset_1 = {
    "dates": [
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10",
        "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15",
        "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-20"
    ],
    "values": [
        10, 13, 16, 19, 22, 25, 28, 31, 34, 37,
        40, 43, 46, 49, 52, 55, 58, 61, 64, 67
    ],
    "n": 5
}

dataset_2 = {
    "dates": [
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10",
        "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15",
        "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-20"
    ],
    "values": [
        10, 18, 11, 17, 12, 19, 11, 18, 12, 17,
        10, 19, 11, 18, 12, 17, 11, 19, 10, 18
    ],
    "n": 5
}

def print_result(label: str, result):
    data = result.data   

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    if "error" in data:
        print(f"  ERROR: {data['error']}")
        return

    print(f"  Winner   : {data['winner'].upper()}")
    print(f"  Forecast : {data['forecast']}")

    print(f"\n  Accuracy (MAE — lower = more accurate):")
    print(f"    ARIMA   : {data['accuracy']['arima_mae']}")
    print(f"    Prophet : {data['accuracy']['prophet_mae']}")

    print(f"\n  Validation (how well each model predicted known data):")
    print(f"    Actual values      : {data['validation']['actual']}")
    print(f"    ARIMA predicted    : {data['validation']['arima_predicted']}")
    print(f"    Prophet predicted  : {data['validation']['prophet_predicted']}")
    print(f"{'='*55}")

# ── Main ──────────────────────────────────────────────────
async def main():
    try:
        async with Client(transport) as client:

            print("Connected to orchestrator\n")

            print("Sending dataset 1 (trending)")
            result = await client.call_tool(
                "getForecast",       
                {
                    "request": {     
                        "dates":  dataset_1["dates"],
                        "values": dataset_1["values"],
                        "n":      dataset_1["n"]
                    }
                }
            )

            print_result("TEST 1 — Trending Data", result)

            print("\nSending dataset 2 (seasonal)")
            result = await client.call_tool(
                "getForecast",
                {
                    "request": {
                        "dates":  dataset_2["dates"],
                        "values": dataset_2["values"],
                        "n":      dataset_2["n"]
                    }
                }
            )
            print_result("TEST 2 — Seasonal Data", result)
    except Exception as e:
        print(f"\n Connection error : Orchestrator may be down.")
        print(f" Details: {e}")


asyncio.run(main())
