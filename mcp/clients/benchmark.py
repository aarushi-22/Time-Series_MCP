# clients/benchmark.py
import asyncio
import time
from fastmcp import Client
from fastmcp.client.transports import SSETransport

transport = SSETransport("http://localhost:8000/mcp/sse")

# Test data
dates = [
    "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10",
    "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15",
    "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-20"
]
values = [10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67]

async def main():
    RUNS = 50  # reduce to 10, enough for a reliable average

    sequential_times = []
    parallel_times   = []

    print("Running benchmark...")
    print(f"Each method tested {RUNS} times\n")

    for i in range(RUNS):
        # Fresh connection each run — avoids timeout
        async with Client(transport) as client:
            
            # Sequential
            result = await client.call_tool(
                "getForecast_sequential",
                {"dates": dates, "values": values, "n": 5}
            )
            sequential_times.append(result.data["time_seconds"])
            print(f"Run {i+1} sequential: {result.data['time_seconds']}s")

            # Parallel
            result = await client.call_tool(
                "getForecast_parallel",
                {"dates": dates, "values": values, "n": 5}
            )
            parallel_times.append(result.data["time_seconds"])
            print(f"Run {i+1} parallel:   {result.data['time_seconds']}s")

   # Skip first run — cold start noise
    avg_sequential = sum(sequential_times[1:]) / len(sequential_times[1:])
    avg_parallel   = sum(parallel_times[1:])   / len(parallel_times[1:])
    reduction_pct  = ((avg_sequential - avg_parallel) / avg_sequential) * 100

    print(f"\n{'='*50}")
    print(f"  Sequential avg    : {avg_sequential:.4f}s")
    print(f"  Parallel avg      : {avg_parallel:.4f}s")
    print(f"  Latency reduction : {reduction_pct:.1f}%")
    print(f"{'='*50}")

asyncio.run(main())
