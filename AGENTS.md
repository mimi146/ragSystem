# System Agents & Modules

This document describes the primary "agents" (modules) that make up the Weather RAG (Retrieval-Augmented Generation) system. Each module has a specific responsibility in the data processing and retrieval pipeline.

## 1. API Orchestrator (`routes/controller.py`)
The central coordinator for all incoming requests.
- **Responsibilities:**
    - Orchestrates the RAG flow (`upload` -> `chunk` -> `embed` -> `store`).
    - Handles file extraction (PDF, DOCX) using `pypdf` and `python-docx`.
    - Manages text chunking logic with configurable overlap.
    - Maintains in-memory conversation history for context-aware queries.
- **Endpoints:** `/weather/upload-document`, `/weather/query`, `/weather/ask`.

## 2. Vector Librarian (`vector_store.py`)
Manages the "memory" of the system using semantic embeddings.
- **Responsibilities:**
    - Converts text chunks into 384-dimensional vectors using `all-MiniLM-L6-v2`.
    - Interface with **ChromaDB** for persistent vector storage and retrieval.
    - Performs similarity searches to find relevant context for user queries.

## 3. Knowledge Synthesizer (`llm.py`)
The reasoning engine that generates natural language responses.
- **Responsibilities:**
    - Provides a unified interface for multiple LLM providers (**Ollama** and **Anthropic**).
    - Defaults to local inference using **Gemma** (via Ollama).
    - Formats system and user prompts to ensure the LLM stays within the provided context.
    - Supports both full-text generation and streaming responses.

## 4. System Core (`main.py`)
The application lifecycle manager.
- **Responsibilities:**
    - Initializes the FastAPI application.
    - Mounts sub-routers and handles global health checks.

---

## Data Flow Diagram
```text
[User] -> [API Orchestrator] -> [Vector Librarian] (Search)
              |                      |
              v                      v
[LLM Synthesizer] <- (Context) <- [ChromaDB]
      |
      v
[Natural Language Response]
```
