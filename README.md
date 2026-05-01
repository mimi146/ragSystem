# 🧠 Weather RAG API: Document-Grounded Intelligence

FastAPI-based Retrieval-Augmented Generation (RAG) service for document-grounded Q&A.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-000000?style=flat)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-blue)](https://ollama.ai/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-8e44ad)](https://www.anthropic.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-111111)](https://openai.com/)

## 🚀 Overview

This service transforms your documents into an interactive knowledge base. It allows you to:
- Upload PDF/DOCX documents
- Chunk and embed them into ChromaDB
- Retrieve top-k relevant chunks for a query
- Generate grounded answers with Ollama, Anthropic, or OpenAI GPT-4o
- Stream responses token-by-token
- Persist short conversation history per `user_id`
- Choose model from chat UI (`gemma4:e2b` or `deepseek-r1:1.5b`)

The backend is now **RAG-only** (no legacy weather CRUD endpoints).

---

## 🏗️ Architecture

For a detailed breakdown of the system modules, see [AGENTS.md](./AGENTS.md).

### Runtime Modules

- **`main.py`**: Creates FastAPI app, preloads embedding model on startup, and mounts the static chat UI.
- **`routes/controller.py`**: API layer handling request/response and delegating logic to the workflow.
- **`rag_workflow.py`**: Explicit orchestration stages (Ingestion, Retriever, Reader).
- **`vector_store.py`**: Embedding helpers and ChromaDB operations (storage, search, history).
- **`llm.py`**: Provider abstraction for Ollama, Anthropic, and OpenAI generation.

### High-Level Flow

```text
Upload document
  -> extract text
  -> chunk text
  -> embed chunks
  -> store vectors + metadata in ChromaDB

User query
  -> embed query
  -> retrieve top-k similar chunks
  -> build context + prompt
  -> generate answer (or stream answer)
  -> persist conversation turn
```

---

## 📂 Project Structure

```text
weather/
├── main.py              # Application entry point
├── rag_workflow.py      # Core RAG orchestration
├── vector_store.py      # Vector DB management
├── llm.py               # LLM provider interface
├── routes/
│   └── controller.py    # API Endpoints
├── static/              # Chat UI (HTML/CSS/JS)
├── chroma_db/           # Persistent storage
└── uploaded_docs/       # Source file storage
```

---

## 📡 API Endpoints

Base router prefix: `/weather`

| Endpoint | Method | Purpose |
|---|---|---|
| `/weather/` | GET | Router health check |
| `/weather/upload-document` | POST | Ingest file into vector DB |
| `/weather/query` | POST | Retrieve similar chunks |
| `/weather/ask` | POST | Retrieve + generate answer |
| `/weather/ask-stream` | POST | Retrieve + stream answer |
| `/weather/source-doc/{upload_id}/{filename}` | GET | Open/download original file |
| `/weather/stats` | GET | Collection stats |
| `/weather/documents` | DELETE | Delete chunks by filename |
| `/weather/memory/clear` | POST | Clear conversation history for user |

### 1) Upload Document
```bash
curl -X POST http://localhost:8000/weather/upload-document \
  -F "file=@document.pdf" \
  -F "tokens_per_chunk=100" \
  -F "overlap=20"
```

### 2) Query (Retriever only)
```bash
curl -X POST http://localhost:8000/weather/query \
  -H "Content-Type: application/json" \
  -d '{"query":"what is this document about?","n_results":3}'
```

### 3) Ask (Retriever + Reader)
```bash
curl -X POST http://localhost:8000/weather/ask \
  -H "Content-Type: application/json" \
  -d '{"user_id":"session-1","query":"summarize key points","n_results":3,"model":"deepseek-r1:1.5b"}'
```

### 4) Ask Stream
```bash
curl -N -X POST http://localhost:8000/weather/ask-stream \
  -H "Content-Type: application/json" \
  -d '{"user_id":"session-1","query":"give me a concise summary","n_results":3,"model":"deepseek-r1:1.5b"}'
```
*Note: Source metadata is sent in the `X-Weather-Sources` response header.*

### 5) Clear Memory
```bash
curl -X POST http://localhost:8000/weather/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"user_id":"session-1"}'
```

---

## 💾 Conversation Memory

Conversation turns are persisted in ChromaDB collection `user-conversations`.
- Keyed by `user_id`
- Sliding window applied (`MAX_MEMORY_TURNS`, currently 5)
- Persistent across process restarts

---

## ⚙️ Configuration

Environment variables used by `llm.py`:
- `LLM_PROVIDER`: `ollama` by default, or `anthropic` / `openai` to override
- `OLLAMA_MODEL`: Default: `gemma4:e2b`
- `ANTHROPIC_MODEL`: Default: `claude-sonnet-4-20250514`
- `ANTHROPIC_API_KEY`: Required for Anthropic provider
- `OPENAI_MODEL`: Default: `gpt-4o`
- `OPENAI_API_KEY`: Required for OpenAI provider
- `OPENAI_BASE_URL`: Optional custom OpenAI-compatible API endpoint
- `AZURE_OPENAI_API_KEY`: Required for Azure OpenAI
- `AZURE_OPENAI_ENDPOINT`: Azure endpoint base URL, for example `https://<resource>.openai.azure.com`
- `AZURE_OPENAI_API_VERSION`: Azure API version, default `2024-10-21`
- `AZURE_OPENAI_MODEL`: Azure deployment name or model alias

If `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` is set and `LLM_PROVIDER` is not explicitly set, the app uses OpenAI by default.

---

## 🛠️ Local Setup

### Prerequisites
- Python 3.11+
- `uv` (recommended)
- Ollama installed locally

### Install and Run
```bash
uv sync
ollama pull gemma4:e2b
ollama pull deepseek-r1:1.5b
uv run uvicorn main:app --port 8000
```

Open:
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Chat UI**: [http://localhost:8000/chat](http://localhost:8000/chat)

---

## 📝 Notes for Development
- Chroma data is stored in `./chroma_db`.
- Uploaded source files are stored in `./uploaded_docs` and linked in responses.
- Same filename can be uploaded multiple times safely; chunk IDs are upload-scoped.
