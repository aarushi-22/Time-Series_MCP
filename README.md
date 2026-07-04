# MCP-Native Time Series Forecasting Platform

A microservices-based forecasting system that exposes ARIMA and Prophet models as MCP tools, enabling LLM clients to discover and invoke forecasting capabilities via the Model Context Protocol.

---

## Architecture

```
LLM Client (Continue.dev / Groq LLaMA 3.3)
        │
        ▼
┌─────────────────────┐
│  Orchestrator       │  FastAPI + FastMCP  (port 8000)
└──────────┬──────────┘
           │  parallel dispatch
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌─────────┐
│  ARIMA  │  │ Prophet │
│ Service │  │ Service │
│ :8001   │  │ :8002   │
└─────────┘  └─────────┘
```

Three independent FastAPI services, each exposed as an MCP server over SSE transport:

- **Orchestrator** — unified MCP interface for LLM clients, coordinates both forecasting services
- **ARIMA service** — classical statistical time series forecasting
- **Prophet service** — Meta's Prophet model with input-hash-based response caching

---

## Tech Stack

| Layer | Technology |
|---|---|
| Services | FastAPI |
| MCP protocol | FastMCP 3.3.1 (SSE transport) |
| Orchestration | LangGraph |
| Forecasting | ARIMA, Prophet |
| Containerization | Docker |

---

## Notes

- Both models run in parallel rather than sequentially, reducing end-to-end latency by 17%
- Prophet responses are cached using an MD5 hash of input parameters to avoid redundant computation on repeated requests
- Validated end-to-end with Continue.dev + Groq LLaMA 3.3 70B — MCP tool discovery and invocation confirmed across all three services
