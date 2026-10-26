# RepoContext AI

[![CI](https://github.com/pavansde/repo-context-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/pavansde/repo-context-ai/actions)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-orange.svg)

A production-oriented Python application that answers questions about code repositories using Retrieval-Augmented Generation (RAG) and a stateful LangGraph agent.

---

## Features

| Feature | Details |
|---|---|
| **Repository-aware RAG** | Code-aware chunking, SHA-256 chunk IDs, incremental re-indexing, and metadata-filtered retrieval |
| **LangGraph agent** | Stateful, tool-calling agent with `search_repository` and `inspect_file` tools |
| **Persistent memory** | SQLite-backed conversation checkpoints (per thread, per repository) |
| **Multi-provider LLMs** | Ollama (default) or OpenAI, with optional primary→fallback chain |
| **Streaming** | Server-Sent Events (SSE) streaming endpoint |
| **Security** | Bearer-token API authentication, per-repository authorization, path-traversal guard, prompt injection resistance |
| **Observability** | Structured logging, LangSmith tracing integration |
| **Retrieval evaluation** | Precision@K / Recall@K evaluation with JSON reports |
| **Containerization** | Multi-stage Dockerfile and Docker Compose |

---

## Architecture

```
                          ┌──────────────────────────────────────┐
                          │              FastAPI App              │
                          │                                      │
  Client ──────────────► │  POST /api/v1/chat                   │
  (Bearer Token)          │  POST /api/v1/chat/stream (SSE)      │
                          │  GET  /health                        │
                          │  GET  /health/ready                  │
                          └──────────┬────────────────────────────┘
                                     │  Auth → Registry → AppState
                                     ▼
                          ┌──────────────────────────────────────┐
                          │          LangGraph Agent             │
                          │                                      │
                          │  MessagesState graph:                │
                          │   START → agent ──► tools ──► agent │
                          │                 └──────────► END     │
                          │                                      │
                          │  Tools:                              │
                          │   • search_repository (RAG lookup)  │
                          │   • inspect_file (path-safe read)   │
                          └──────────┬───────────────────────────┘
                                     │
                     ┌───────────────┴──────────────┐
                     ▼                              ▼
           ┌─────────────────┐          ┌──────────────────────┐
           │  Chroma Vector  │          │  Ollama / OpenAI LLM │
           │  Store          │          │  (configurable)       │
           │  (filtered by   │          │                       │
           │   repository_id)│          │  Optional fallback    │
           └─────────────────┘          └──────────────────────┘
                     ▲
                     │ embed + store
           ┌─────────────────┐
           │  RAG Pipeline   │
           │  (rag.py)       │
           │                 │
           │  load → split   │
           │  → embed → upsert│
           │  → prune stale  │
           └─────────────────┘
```

The conversation state is persisted in SQLite using LangGraph's `SqliteSaver` checkpointer. Each `thread_id` maintains independent history, enabling multi-turn Q&A.

---

## Technology Stack

- **Python 3.13**
- **FastAPI** — REST API and SSE streaming
- **LangChain / LCEL** — chains, prompts, output parsers
- **LangGraph** — stateful agent with tool-calling and checkpointing
- **Chroma** — local vector store with metadata filtering
- **Ollama** — local LLM and embedding model server
- **OpenAI** — optional cloud LLM / embedding provider
- **SQLite** — conversation memory (via `langgraph-checkpoint-sqlite`)
- **Pydantic / pydantic-settings** — validated configuration

---

## Prerequisites

1. **Python 3.13** (or 3.11+)
2. **Ollama** running locally or on a network host with:
   - A chat model: `ollama pull qwen2.5-coder:7b`
   - An embedding model: `ollama pull nomic-embed-text`
3. (Optional) An **OpenAI API key** if using OpenAI as provider

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/pavansde/repo-context-ai.git
cd repo-context-ai

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   OLLAMA_HOST       (hostname or IP, not a URL)
#   API_KEY           (any non-empty secret)
#   REPOSITORIES      (comma-separated id:path entries)
```

### 3. Index a repository

```python
# index_repo.py (run once, or on each update)
from rag import index_repository
from core.config import get_settings

settings = get_settings()
chunks = index_repository(
    repository_path="/path/to/your/repo",
    repository_id="my-repo",   # must match REPOSITORIES setting
    settings=settings,
)
print(f"Indexed {chunks} chunks.")
```

```bash
python index_repo.py
```

### 4. Start the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Query

```bash
# Synchronous
curl -s http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "repository_id": "my-repo",
    "message": "How does authentication work?",
    "thread_id": "session-1"
  }'

# Streaming (SSE)
curl -N http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "repository_id": "my-repo",
    "message": "Find the main entry point.",
    "thread_id": "session-1"
  }'
```

---

## Docker

```bash
cp .env.example .env   # configure OLLAMA_HOST, API_KEY, REPOSITORIES

docker compose up --build
```

The container mounts `./data/` for persistent Chroma and SQLite storage.

---

## Configuration Reference

All settings are loaded from environment variables or `.env`. See [`.env.example`](.env.example) for the full list with comments.

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development`, `testing`, or `production` |
| `API_KEY` | _(empty)_ | Bearer token for all protected endpoints. Required in `production`. |
| `LLM_PROVIDER` | `ollama` | `openai` or `ollama` |
| `LLM_MODEL` | `gpt-4.1-mini` | Used when `LLM_PROVIDER=openai` |
| `OLLAMA_CHAT_MODEL` | `qwen2.5-coder:7b` | Chat model for Ollama |
| `OLLAMA_HOST` | `localhost` | Hostname or IP (not a URL) |
| `OLLAMA_TIMEOUT` | `120` | HTTP timeout in seconds |
| `LLM_FALLBACK_PROVIDER` | _(disabled)_ | Optional fallback provider on primary failure |
| `LLM_FALLBACK_MODEL` | _(disabled)_ | Model name for fallback |
| `EMBEDDING_PROVIDER` | `ollama` | `openai` or `ollama` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `REPOSITORIES` | _(empty)_ | `id:/path,id2:/path2` — registered trusted repos |
| `RETRIEVER_TOP_K` | `5` | Chunks returned per retrieval |
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `VECTOR_STORE_PATH` | `./data/chroma` | Chroma persistence directory |
| `CHECKPOINT_DB_PATH` | `./data/checkpoints.db` | SQLite file for conversation memory |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |

---

## API Reference

### Authentication

All `/api/v1/*` endpoints require:

```
Authorization: Bearer <API_KEY>
```

### Endpoints

#### `POST /api/v1/chat`

Synchronous question → answer.

**Request:**
```json
{
  "repository_id": "my-repo",
  "message": "How does authentication work?",
  "thread_id": "any-unique-session-id"
}
```

**Response (200):**
```json
{
  "thread_id": "any-unique-session-id",
  "answer": "Authentication is implemented in api/auth.py ..."
}
```

**Error responses:** `401` (missing/invalid API key), `403` (unauthorized repository), `404` (repository path not found), `500` (agent execution error).

---

#### `POST /api/v1/chat/stream`

Same request format as `/chat`. Response is a Server-Sent Events stream:

```
data: {"type": "token", "content": "Auth"}

data: {"type": "token", "content": "entication is..."}

data: {"type": "done", "request_id": "uuid"}
```

On error: `data: {"type": "error", "error": "Agent execution failed", "request_id": "uuid"}`

---

#### `GET /health`

Liveness probe. Always returns `200` once the process is running.

```json
{"status": "ok", "service": "repo-context-ai"}
```

---

#### `GET /health/ready`

Readiness probe. Returns `503` until the SQLite checkpointer is initialized.

```json
{"status": "ready", "service": "repo-context-ai"}
```

---

## Testing

The test suite is split into two groups:

### Fast tests (no LLM required)

```bash
# Run all 44 offline unit tests (configured via pytest.ini)
pytest

# Or run explicitly
python -m pytest -v
# Expected: 44 passed
```

### LLM-dependent tests (require live Ollama)

```bash
python -m pytest tests/test_agent.py \
                 tests/test_agent_memory.py \
                 tests/test_repository_isolation.py \
                 tests/test_api.py::test_chat_accepts_valid_api_key \
                 tests/test_api.py::test_chat_endpoint \
                 tests/test_api.py::test_chat_stream_endpoint \
                 tests/test_api.py::test_chat_returns_request_id \
                 tests/test_ollama_tools.py \
                 -v
```

### Manual scripts (run directly, not via pytest)

```bash
python tests/test_rag.py          # Index and query a test repository
python tests/test_reindexing.py   # Verify incremental re-indexing
python tests/test_models.py       # Verify model initialization and a live call
```

---

## Retrieval Evaluation

The evaluation framework measures Precision@K and Recall@K for the 8-question dataset in `evaluation/dataset.py`.

```bash
# Baseline evaluation (index the repo first)
python -m evaluation.runner \
  --repository-id my-repo \
  --k 3 \
  --output evaluation/results.json

# With experimental LLM reranker
python -m evaluation.runner \
  --repository-id my-repo \
  --k 3 \
  --rerank \
  --output evaluation/results_reranked.json

# With debug diagnostics
python -m evaluation.runner \
  --repository-id my-repo \
  --k 3 \
  --debug
```

**Documented baseline (this repository as the target):**

| Metric | Baseline (K=3) |
|---|---|
| Mean Precision@3 | 0.3333 |
| Mean Recall@3 | 1.0000 |

The experimental path-only reranker produced identical metrics; it is preserved for experimentation but not recommended as a production enhancement.

---

## Project Structure

```
repo-context-ai/
├── app.py                  # FastAPI application + lifespan + health endpoints
├── rag.py                  # Full RAG pipeline: load → split → embed → retrieve
├── agent/
│   ├── graph.py            # LangGraph agent (MessagesState + tool-calling loop)
│   └── tools.py            # search_repository and inspect_file tools
├── api/
│   ├── auth.py             # Bearer token authentication
│   ├── authorization.py    # Repository access guard
│   ├── routes.py           # /chat and /chat/stream endpoints
│   └── schemas.py          # Pydantic request/response models
├── core/
│   ├── app_state.py        # AppState (SQLite connection, checkpointer)
│   ├── config.py           # Validated settings via pydantic-settings
│   ├── exceptions.py       # Application-specific exception hierarchy
│   ├── logging_config.py   # Structured stdout logging
│   ├── model_factory.py    # Provider-independent model construction
│   └── repository_registry.py  # Trusted repository ID → path mapping
├── evaluation/
│   ├── dataset.py          # 8-question ground-truth dataset
│   ├── metrics.py          # precision_at_k, recall_at_k
│   ├── reranker.py         # Experimental LLM-based reranker
│   └── runner.py           # Evaluation CLI with JSON report output
├── tests/
│   ├── test_agent.py           # LLM: agent invocation
│   ├── test_agent_memory.py    # LLM: SQLite memory persistence
│   ├── test_api.py             # FastAPI endpoint tests
│   ├── test_app_state.py       # AppState startup/shutdown
│   ├── test_evaluation_dataset.py  # Dataset structure validation
│   ├── test_evaluation_metrics.py  # Metric function unit tests
│   ├── test_ollama_tools.py    # LLM: tool calling integration
│   ├── test_rag.py             # Manual: index + query script
│   ├── test_reindexing.py      # Manual: incremental re-indexing script
│   ├── test_reliability.py     # Unit: config, security, failure scenarios
│   └── test_repository_isolation.py  # LLM: cross-repo isolation
├── data/                   # Runtime data (gitignored)
│   ├── chroma/             # Chroma vector store
│   └── checkpoints.db      # SQLite conversation memory
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Limitations and Known Issues

- **No repository ingestion API**: Repositories must be indexed manually (via `index_repository()`) before the chat endpoint can answer questions about them.
- **Single Chroma collection**: All repositories share one Chroma collection, isolated by metadata filter. For very large deployments, separate collections per repo would be more scalable.
- **LLM reranker is experimental**: The `--rerank` flag did not improve evaluation metrics; it is preserved for experimentation.
- **LLM-dependent tests require live Ollama**: Tests that invoke the agent cannot run in a CI environment without a running Ollama server.
- **No rate limiting**: The API does not currently enforce per-client rate limits. For public deployments, a reverse proxy (nginx, Caddy) or API gateway should be used.
- **SQLite concurrency**: SQLite is suitable for development and single-server deployments. High-concurrency production deployments may require a different checkpointer backend.

---

## Portfolio Description

**RepoContext AI** is a production-oriented GenAI application that enables engineers to ask natural-language questions about any software repository and receive accurate, source-cited answers.

It demonstrates:
- **LangGraph agent design**: stateful tool-calling loop with memory, where the LLM decides when to search the codebase vs. respond directly
- **RAG pipeline engineering**: code-aware chunking, deterministic chunk IDs, incremental indexing with stale-chunk pruning, and repository-scoped retrieval
- **Production API patterns**: Bearer authentication, repository authorization, structured logging, request tracing, SSE streaming, readiness vs. liveness endpoints
- **Testable architecture**: 44 unit/integration tests covering security constraints, failure modes, and configuration validation — all running without a live LLM

---

## Interview Discussion Points

### Architecture decisions

**Why LangGraph over a plain LangChain agent?**
LangGraph provides explicit control over the agent loop, enabling stateful multi-turn conversations via SQLite checkpoints and making tool execution deterministic and debuggable. The MessagesState graph makes conversation history management explicit.

**Why Chroma with metadata filtering instead of separate collections per repository?**
A single collection with `repository_id` metadata filtering simplifies operational management (one Chroma instance, one set of embeddings) while enforcing isolation at query time. The tradeoff is that the collection grows with more repositories, which could impact query latency at very large scale.

**Why deterministic SHA-256 chunk IDs?**
Chroma's `add_documents` with explicit IDs is idempotent — re-indexing a file that hasn't changed is a no-op (upsert). Stale IDs are deleted after each indexing run, keeping the store consistent with the filesystem.

**Why SQLite for conversation memory?**
It requires zero infrastructure — no Redis, no PostgreSQL — and the SQLite WAL mode is sufficient for a single-server deployment. LangGraph's `SqliteSaver` handles schema and serialization transparently.

**Why not expose internal exceptions in API responses?**
The API catches all agent exceptions and returns a generic `"Agent execution failed"` message. Full exception details are logged server-side. This prevents leaking model prompts, stack traces, or file paths to API clients.

### Trade-offs

| Decision | Trade-off |
|---|---|
| Ollama default | Free, private, no API costs — but latency and quality depend on hardware and model choice |
| Code-aware chunking | Better semantic boundaries than character splits — but splits by language keywords, not AST, so cross-function chunks are possible |
| Evaluation dataset is self-referential | Questions are about this repository's own code — makes ground truth verifiable but limits generalizability claims |
| No rate limiting in API layer | Simpler implementation — delegated to reverse proxy in production |

---

## License

MIT