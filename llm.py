"""
LLM Module for RAG Generation

This module provides the "Generation" part of RAG. After retrieving relevant
document chunks, we pass them to an LLM along with the user's query to generate
a natural language answer.

Two providers are supported:
1. Ollama (local, free, private) - Default
2. Anthropic Claude (cloud, highest quality)

The module uses a simple provider pattern - switch between them via config.
"""

import logging
import os
import importlib
from typing import Generator

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Choose your LLM provider
# =============================================================================

# Set to "ollama" for local inference or "anthropic" for cloud API
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Model names for each provider
# Default to the locally hosted Gemma model requested for this project.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# API key for Anthropic (only needed if using Anthropic)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# =============================================================================
# PROVIDER INITIALIZATION
# =============================================================================

def get_llm_client():
    """
    Get the appropriate LLM client based on configured provider.

    Returns:
        tuple: (client, provider_name) - client is either an Ollama instance
               or Anthropic instance, depending on configuration.
    """
    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set, falling back to Ollama")
            return _get_ollama_client(), "ollama"
        return _get_anthropic_client(), "anthropic"
    else:
        return _get_ollama_client(), "ollama"


def _get_ollama_client():
    """Initialize Ollama client (local LLM runner)."""
    try:
        ollama = importlib.import_module("ollama")
        logger.info(f"Ollama client initialized (model: {OLLAMA_MODEL})")
        return ollama
    except ImportError:
        logger.error("ollama package not installed. Run: pip install ollama")
        raise


def _get_anthropic_client():
    """Initialize Anthropic client (cloud API)."""
    try:
        anthropic = importlib.import_module("anthropic")
        Anthropic = getattr(anthropic, "Anthropic")
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info(f"Anthropic client initialized (model: {ANTHROPIC_MODEL})")
        return client
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        raise


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context.
Follow these rules:
1. Answer ONLY using the provided context - do not use outside knowledge
2. If the answer is not in the context, say "I don't have enough information to answer that"
3. Cite your sources by mentioning which document the information came from
4. Be concise but complete
5. If different sources contradict each other, mention the discrepancy"""

RAG_USER_PROMPT = """Based on the following context from our documents, please answer this question:

Conversation history (most recent turns):
{history}

Question: {query}

Context:
{context}

Answer:"""


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_answer(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    model: str | None = None,
) -> str:
    """
    Generate an answer using the configured LLM provider.

    This is the main entry point for RAG generation. It takes the retrieved
    chunks and generates a natural language answer.

    Args:
        query: The user's question
        chunks: List of retrieved document chunks from query_similar()

    Returns:
        Generated answer as a string

    Example:
        results = query_similar("what causes solar flares?")
        answer = generate_answer("what causes solar flares?", results)
    """
    client, provider = get_llm_client()
    logger.info(f"Generating answer using {provider}...")

    # Build context from retrieved chunks
    context = build_context(chunks)

    if provider == "ollama":
        return _generate_ollama(client, query, context, history, model=model)
    else:
        return _generate_anthropic(client, query, context, history, model=model)


def generate_answer_stream(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    model: str | None = None,
) -> Generator[str, None, None]:
    """
    Generate an answer with streaming output.

    Same as generate_answer() but yields tokens as they're generated.
    Useful for real-time UI updates.

    Args:
        query: The user's question
        chunks: List of retrieved document chunks

    Yields:
        Text tokens as they're generated by the LLM
    """
    client, provider = get_llm_client()
    logger.info(f"Streaming answer using {provider}...")

    context = build_context(chunks)

    if provider == "ollama":
        yield from _stream_ollama(client, query, context, history, model=model)
    else:
        yield from _stream_anthropic(client, query, context, history, model=model)


def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the prompt.

    Includes source metadata so the LLM can cite documents.

    Args:
        chunks: List of dicts with 'content' and 'metadata' keys

    Returns:
        Formatted context string with source citations
    """
    context_parts: list[str] = []

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        filename = metadata.get("filename", "unknown")
        chunk_idx = metadata.get("chunk_index", i)

        # Format each chunk with source info
        context_parts.append(
            f"[Source: {filename}, chunk {chunk_idx}]\n{content}\n"
        )

    context = "\n---\n".join(context_parts)
    logger.info(f"Built context from {len(chunks)} chunks ({len(context)} chars)")

    return context


def build_user_prompt(query: str, context: str, history: list[dict] | None = None) -> str:
    """
    Build the reader prompt from user query + retrieved context + short history.
    """
    return RAG_USER_PROMPT.format(
        query=query,
        context=context,
        history=_format_history(history),
    )


def _format_history(history: list[dict] | None) -> str:
    """Format recent conversation turns for the prompt."""
    if not history:
        return "None"

    parts = []
    for turn in history:
        user = turn.get("query", "")
        assistant = turn.get("answer", "")
        parts.append(f"User: {user}\nAssistant: {assistant}")

    return "\n---\n".join(parts)

# =============================================================================
# OLLAMA IMPLEMENTATION
# =============================================================================

def _generate_ollama(
    client,
    query: str,
    context: str,
    history: list[dict] | None = None,
    model: str | None = None,
) -> str:
    """Generate answer using Ollama (local)."""
    user_prompt = build_user_prompt(query=query, context=context, history=history)
    selected_model = model or OLLAMA_MODEL

    logger.debug(f"Sending request to Ollama (model: {selected_model})")

    response = client.generate(
        model=selected_model,
        system=RAG_SYSTEM_PROMPT,
        prompt=user_prompt
    )

    answer = response.get("response", "")
    logger.info(f"Generated answer ({len(answer)} chars)")

    return answer


def _stream_ollama(
    client,
    query: str,
    context: str,
    history: list[dict] | None = None,
    model: str | None = None,
) -> Generator[str, None, None]:
    """Stream answer from Ollama."""
    user_prompt = build_user_prompt(query=query, context=context, history=history)
    selected_model = model or OLLAMA_MODEL

    response = client.generate(
        model=selected_model,
        system=RAG_SYSTEM_PROMPT,
        prompt=user_prompt,
        stream=True
    )

    # Ollama streaming returns a generator of dicts with 'response' key
    for chunk in response:
        if "response" in chunk:
            yield chunk["response"]


# =============================================================================
# ANTHROPIC IMPLEMENTATION
# =============================================================================

def _generate_anthropic(
    client,
    query: str,
    context: str,
    history: list[dict] | None = None,
    model: str | None = None,
) -> str:
    """Generate answer using Anthropic Claude (cloud)."""
    user_prompt = build_user_prompt(query=query, context=context, history=history)
    selected_model = model or ANTHROPIC_MODEL

    logger.debug(f"Sending request to Anthropic (model: {selected_model})")

    response = client.messages.create(
        model=selected_model,
        max_tokens=1024,
        system=RAG_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = response.content[0].text
    logger.info(f"Generated answer ({len(answer)} chars)")

    return answer


def _stream_anthropic(
    client,
    query: str,
    context: str,
    history: list[dict] | None = None,
    model: str | None = None,
) -> Generator[str, None, None]:
    """Stream answer from Anthropic."""
    user_prompt = build_user_prompt(query=query, context=context, history=history)
    selected_model = model or ANTHROPIC_MODEL

    with client.messages.stream(
        model=selected_model,
        max_tokens=1024,
        system=RAG_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    ) as stream:
        for text in stream.text_stream:
            yield text


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_llm_available() -> dict:
    """
    Check if the configured LLM provider is available and working.

    Returns:
        Dict with provider status and configuration details.
    """
    result = {
        "provider": LLM_PROVIDER,
        "available": False,
        "model": None,
        "error": None
    }

    try:
        if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
            result["model"] = ANTHROPIC_MODEL
            # Try a minimal request to verify API key
            import importlib
            anthropic_module = importlib.import_module("anthropic")
            client = anthropic_module.Anthropic(api_key=ANTHROPIC_API_KEY)
            client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            result["available"] = True
        else:
            result["model"] = OLLAMA_MODEL
            # Try to connect to Ollama and ensure the configured model exists locally
            import importlib
            ollama_module = importlib.import_module("ollama")
            model_list = ollama_module.list()
            # Ollama may return model name under either "model" or "name".
            available_names = {
                m.get("model") or m.get("name")
                for m in model_list.get("models", [])
                if isinstance(m, dict)
            }

            # Fail fast with a clear message when the default model is missing.
            if OLLAMA_MODEL not in available_names:
                raise RuntimeError(
                    f"Ollama model '{OLLAMA_MODEL}' not found locally. "
                    f"Available models: {sorted(n for n in available_names if n)}"
                )
            result["available"] = True
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"LLM provider check failed: {e}")

    return result
