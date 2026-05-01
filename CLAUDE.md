# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Install dependencies
uv sync

# Pull local LLM model (required for /weather/ask)
ollama pull gemma4:e2b

# Start server
uv run uvicorn main:app --port 8000

# API docs: http://localhost:8000/docs
# Chat UI: http://localhost:8000/chat/
```

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) system** for weather-related documents. The architecture follows a clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│  INGESTION → STORAGE → RETRIEVAL → GENERATION               │
│  (controller)  (vector_store)   (query)    (llm.py)         │
└─────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `main.py` | FastAPI app entry point, mounts static files (`/chat/`) and routers |
| `routes/controller.py` | API endpoints, text extraction, chunking, request validation |
| `vector_store.py` | ChromaDB client, embedding model (`all-MiniLM-L6-v2`), CRUD operations |
| `llm.py` | LLM abstraction layer supporting Ollama (default: `gemma4:e2b`) and Anthropic |
| `static/` | Vanilla JS chat interface with streaming responses |

### Key Design Decisions

1. **Local-first LLM**: Defaults to Ollama with `gemma4:e2b` for private, cost-free inference. Falls back to Anthropic if `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` is set.

2. **Per-upload isolation**: Each document upload gets a unique `upload_id` to prevent same-filename collisions. Chunk IDs are derived from `upload_id + chunk_index`, not filename.

3. **In-memory conversation history**: Per-user sequential memory (last 5 turns) enables follow-up questions. Stored in `defaultdict` keyed by `user_id` (default: `"anonymous"`).

4. **Persistent vector storage**: ChromaDB stores embeddings to `./chroma_db/` - data survives restarts.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/weather/upload-document` | POST | Upload PDF/DOCX, extract text, chunk, embed, store |
| `/weather/query` | POST | Semantic search only (no LLM) |
| `/weather/ask` | POST | Retrieve + generate answer (with conversation memory) |
| `/weather/ask-stream` | POST | Streaming version of `/ask` |
| `/weather/stats` | GET | Vector store statistics |
| `/weather/documents` | DELETE | Remove chunks by filename |
| `/weather/source-doc/{upload_id}/{filename}` | GET | Browse/download uploaded source |

## Common Operations

### Run a specific test (if tests exist)
```bash
uv run pytest -k test_name
```

### Check LLM availability
```bash
curl http://localhost:8000/weather/health  # If health endpoint exists
```

### Clear conversation memory
Memory is in-process only - restart the server to clear.

### Inspect vector store
```bash
curl http://localhost:8000/weather/stats
```

## Configuration

Environment variables (all optional, defaults in `llm.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `anthropic` |
| `OLLAMA_MODEL` | `gemma4:e2b` | Local model name |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Cloud model name |
| `ANTHROPIC_API_KEY` | (empty) | Required for Anthropic provider |

## Important Behaviors

- **Conversation memory**: Tied to `user_id`. Requests without `user_id` share the `"anonymous"` bucket.
- **Memory persistence**: In-memory only - lost on server restart.
- **Embedding model**: Lazy-loaded on first use (~2s delay), then cached globally.
- **Chunking**: Word-based (not token-based) with configurable overlap. Default: 100 words/chunk, 20-word overlap.

## Files of Note

- `chat-client.md`: Design doc for the chat interface
- `uploaded_docs/`: Stored source documents (for linking back in responses)
- `chroma_db/`: Persistent vector database files
