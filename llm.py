"""
LLM Module for RAG Generation

This module provides the "Generation" part of RAG. After retrieving relevant
document chunks, we pass them to an LLM along with the user's query to generate
a natural language answer.

Three providers are supported:
1. OpenAI GPT-4o (cloud, state-of-the-art) - Primary
2. Anthropic Claude (cloud, high quality)
3. Ollama (local, fallback)

The module defaults to OpenAI GPT-4o via API.
"""

import logging
import os
import importlib
from urllib.parse import urlsplit, urlunsplit
from typing import Generator

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Choose your LLM provider
# =============================================================================

# Model names for each provider
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")

# API keys and endpoints for cloud providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", OPENAI_MODEL)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Default to OpenAI/Azure as requested.
DEFAULT_LLM_PROVIDER = "openai"

# Set to "openai" for cloud inference (GPT-4o), "anthropic" for Claude, or "ollama" for local fallback.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()

# =============================================================================
# PROVIDER INITIALIZATION
# =============================================================================

def get_llm_client():
    """
    Get the appropriate LLM client based on configured provider.

    Returns:
         tuple: (client, provider_name)
    """
    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set, falling back to OpenAI")
            return _get_openai_client(), "openai"
        return _get_anthropic_client(), "anthropic"
    if LLM_PROVIDER == "openai":
        if not (OPENAI_API_KEY or (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)):
            logger.warning("OpenAI/Azure OpenAI credentials not set, falling back to Ollama")
            return _get_ollama_client(), "ollama"
        return _get_openai_client(), "openai"
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


def _get_openai_client():
    """Initialize OpenAI or Azure OpenAI client."""
    try:
        openai = importlib.import_module("openai")
        if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
            OpenAI = getattr(openai, "OpenAI")
            azure_base_url = _normalize_azure_openai_base_url(AZURE_OPENAI_ENDPOINT)
            client = OpenAI(
                api_key="", # Azure uses api-key header
                base_url=azure_base_url,
                default_headers={"api-key": AZURE_OPENAI_API_KEY},
            )
            logger.info(f"Azure OpenAI client initialized (model: {AZURE_OPENAI_MODEL})")
            return client

        OpenAI = getattr(openai, "OpenAI")
        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**client_kwargs)
        logger.info(f"OpenAI client initialized (model: {OPENAI_MODEL})")
        return client
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        raise


def _normalize_azure_openai_base_url(endpoint: str) -> str:
    """Normalize Azure endpoint to origin + /openai/v1."""
    parsed = urlsplit(endpoint)
    if parsed.scheme and parsed.netloc:
        origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    else:
        origin = endpoint.split("/")[0]
    return origin.rstrip("/") + "/openai/v1"


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
    elif provider == "openai":
        return _generate_openai(client, query, context, history, model=model)
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
    elif provider == "openai":
        yield from _stream_openai(client, query, context, history, model=model)
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

def _generate_openai(
    client,
    query: str,
    context: str,
    history: list[dict] | None = None,
    model: str | None = None,
) -> str:
    """Generate answer using OpenAI or Azure OpenAI chat.completions API."""
    user_prompt = build_user_prompt(query=query, context=context, history=history)
    selected_model = model or _resolve_openai_model()

    logger.debug(f"Sending request to OpenAI/Azure (model: {selected_model})")

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    logger.info(f"Generated answer ({len(answer)} chars)")

    return answer


def _stream_openai(
    client,
    query: str,
    context: str,
    history: list[dict] | None = None,
    model: str | None = None,
) -> Generator[str, None, None]:
    """Stream answer from OpenAI or Azure OpenAI chat.completions API."""
    user_prompt = build_user_prompt(query=query, context=context, history=history)
    selected_model = model or _resolve_openai_model()

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024,
        stream=True
    )
    
    for chunk in response:
        try:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except (AttributeError, IndexError, TypeError) as e:
            logger.error(f"Error parsing chunk: {e}")
            pass
    logger.info("OpenAI/Azure stream finished")


def _resolve_openai_model() -> str:
    """Return the correct model name for OpenAI or Azure."""
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        return AZURE_OPENAI_MODEL
    return OPENAI_MODEL

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
        elif LLM_PROVIDER == "openai" and (OPENAI_API_KEY or (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)):
            result["model"] = _resolve_openai_model()
            client, _ = get_llm_client()
            client.chat.completions.create(
                model=_resolve_openai_model(),
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
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
