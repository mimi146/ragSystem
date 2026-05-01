"""
Vector Store Module for RAG Pipeline

This module handles the "storage" and "retrieval" parts of RAG:
1. Generate embeddings for document chunks using sentence-transformers
2. Store embeddings + metadata in ChromaDB vector database
3. Query for semantically similar chunks

The embedding model (all-MiniLM-L6-v2) converts text to 384-dimensional vectors
where semantically similar texts are close together in vector space.
"""

import logging
import hashlib
import uuid
import json
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# Configure logging - inherits root logger settings from controller.py
logger = logging.getLogger(__name__)

# =============================================================================
# EMBEDDING MODEL - Converts text to vectors for semantic comparison
# =============================================================================

# Global variable to cache the model (avoids reloading on every request)
# The model is ~80MB and takes ~1-2 seconds to load on first use
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or create the sentence-transformers embedding model.

    Uses lazy initialization - the model is only loaded when first requested.
    This avoids the ~2 second load time if no embeddings are needed.

    Model: all-MiniLM-L6-v2
    - 384-dimensional embeddings
    - Fast inference (~50ms per query)
    - Good balance of speed and quality for semantic search
    - Trained on 1B+ sentence pairs

    Returns:
        SentenceTransformer model instance

    Note: For production, consider:
    - Larger models (all-mpnet-base-v2) for better quality
    - GPU acceleration for batch embedding
    - Caching embeddings to avoid recomputation
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info("Loading sentence-transformers model (first time - may take a few seconds)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")

    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert texts into embeddings using the configured sentence transformer.

    This maps directly to the "Embed documents" workflow stage.
    """
    if not texts:
        return []

    model = get_embedding_model()
    return model.encode(texts).tolist()


def embed_query(query: str) -> list[float]:
    """
    Convert a single user query into an embedding vector.

    This maps directly to the retriever stage "Embed user query".
    """
    embeddings = embed_texts([query])
    return embeddings[0] if embeddings else []


# =============================================================================
# CHROMADB CLIENT - Persistent vector database
# =============================================================================

# PersistentClient stores data to disk (./chroma_db directory)
# Without this, data would be lost when the process exits
client = chromadb.PersistentClient(path="./chroma_db")
logger.info("ChromaDB client initialized (data persists to ./chroma_db)")

# Collection is like a "table" in the vector database
# Each collection holds documents with their embeddings and metadata
collection = client.get_or_create_collection(
    name="weather-documents",
    metadata={"description": "Weather document chunks for RAG"}
)
logger.info(f"Using collection: {collection.name}")

# =============================================================================
# CONVERSATION MEMORY - Persistent storage for user conversation history
# =============================================================================

# Separate collection for conversation turns (not used for semantic search)
# Stores conversation history so it survives server restarts
conversation_collection = client.get_or_create_collection(
    name="user-conversations",
    metadata={"description": "User conversation history for RAG context"}
)
logger.info(f"Using conversation collection: {conversation_collection.name}")


# =============================================================================
# VECTOR STORE OPERATIONS - Add, Query, Delete
# =============================================================================

def store_document_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
    upload_id: Optional[str] = None,
) -> list[str]:
    """
    Store chunk text + embedding vectors in ChromaDB.

    This maps directly to the "Knowledge base as a vector database" stage.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    logger.info(
        "Storing %d chunks from '%s' in vector store",
        len(chunks),
        filename,
    )

    if upload_id is None:
        upload_id = str(uuid.uuid4())

    ids = []
    for i, _ in enumerate(chunks):
        chunk_hash = hashlib.md5(f"{upload_id}-{i}".encode()).hexdigest()[:12]
        ids.append(f"chunk-{chunk_hash}")

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {"filename": filename, "chunk_index": i, "upload_id": upload_id}
            for i in range(len(chunks))
        ],
    )

    logger.info("Successfully added %d chunks to vector store", len(ids))
    return ids


def add_documents(chunks: list[str], filename: str, upload_id: Optional[str] = None) -> list[str]:
    """
    Add document chunks to the vector store with embeddings.

    This is the "indexing" step of RAG. Each chunk is:
    1. Converted to an embedding vector (384 dimensions)
    2. Assigned a unique ID (hash of filename + index)
    3. Stored with metadata for filtering later

    Args:
        chunks: List of text chunks from document splitting
        filename: Name of the source document (stored as metadata)

    Returns:
        List of chunk IDs that were added to the store

    Note: Embeddings are the expensive part - encoding N chunks takes
    roughly N * 50ms on CPU. For large documents, consider batching.
    """
    logger.info("Adding %d chunks from '%s' to vector store", len(chunks), filename)
    embeddings = embed_texts(chunks)
    return store_document_chunks(chunks, embeddings, filename, upload_id=upload_id)


def search_by_query_embedding(
    query_embedding: list[float],
    n_results: int = 3,
    filename: Optional[str] = None
) -> list[dict]:
    """
    Find nearest document chunks for a pre-embedded query vector.
    """
    where_clause = None
    if filename:
        where_clause = {"filename": filename}
        logger.debug("Filtering by filename: %s", filename)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_clause,  # type: ignore
        include=["documents", "distances", "metadatas"],
    )

    matches = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            match = {
                "content": doc,
                "distance": results["distances"][0][i] if results["distances"] else None,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
            }
            matches.append(match)
            distance = match.get("distance")
            if isinstance(distance, (int, float)):
                logger.debug("  Match %d: distance=%.4f", i + 1, float(distance))

    logger.info("Query returned %d results", len(matches))
    return matches


def query_similar(
    query: str,
    n_results: int = 3,
    filename: Optional[str] = None
) -> list[dict]:
    """
    Find document chunks most similar to a query using semantic search.

    How it works:
    1. Convert query text to embedding vector (same model as documents)
    2. Find nearest neighbors in vector space using cosine similarity
    3. Return matching chunks with similarity scores

    The "distance" in results is cosine distance (0 = identical, 2 = opposite).
    Lower distance = more semantically similar.

    Args:
        query: Search question or text to find similar content for
        n_results: Number of top matches to return (default: 3)
        filename: Optional filter - only search chunks from this file

    Returns:
        List of match dicts with:
        - content: The chunk text
        - distance: Cosine distance (lower = more similar)
        - metadata: Dict with filename and chunk_index

    Example:
        results = query_similar("solar flares", n_results=3)
        # Returns: [{"content": "...", "distance": 0.32, "metadata": {...}}, ...]
    """
    logger.debug(f"Querying for: '{query[:50]}...' (top {n_results})")

    query_embedding = embed_query(query)
    return search_by_query_embedding(
        query_embedding=query_embedding,
        n_results=n_results,
        filename=filename,
    )


def get_collection_stats() -> dict:
    """
    Get statistics about the vector store.

    Returns:
        Dict with:
        - total_chunks: Number of document chunks indexed
        - collection_name: Name of the ChromaDB collection

    Use this to monitor index size or verify uploads succeeded.
    """
    count = collection.count()
    logger.debug(f"Collection stats: {count} chunks")
    return {
        "total_chunks": count,
        "collection_name": collection.name
    }


def delete_by_filename(filename: str) -> int:
    """
    Delete all chunks from a specific file.

    This is useful for:
    - Removing outdated documents
    - Letting users delete their uploads
    - Cleaning up test data

    Args:
        filename: Exact filename to delete (must match upload name)

    Returns:
        Number of chunks deleted (0 if file not found)

    Note: Deletion is permanent. Consider soft-delete in production.
    """
    logger.info(f"Deleting chunks for filename: {filename}")

    # First, find all chunk IDs for this file
    results = collection.get(
        where={"filename": filename},
        include=["metadatas"]
    )

    if results["ids"]:
        # Delete all matching chunks
        collection.delete(ids=results["ids"])
        logger.info(f"Deleted {len(results['ids'])} chunks")
        return len(results["ids"])

    logger.warning(f"No chunks found for filename: {filename}")
    return 0


# =============================================================================
# CONVERSATION MEMORY FUNCTIONS - Persistent conversation history
# =============================================================================

def add_conversation_turn(user_id: str, query: str, answer: str, max_turns: int = 5) -> int:
    """
    Add a conversation turn to the user's history in ChromaDB.

    Automatically maintains a sliding window of max_turns by deleting
    the oldest turn when the limit is exceeded.

    Args:
        user_id: Unique identifier for the user/conversation
        query: User's question
        answer: LLM's response
        max_turns: Maximum number of turns to keep (default: 5)

    Returns:
        Total number of turns now stored for this user
    """
    import time

    logger.debug(f"Adding conversation turn for user {user_id[:8]}...")

    # Generate unique ID with timestamp for ordering
    timestamp = time.time()
    turn_id = f"{user_id}-{timestamp}"

    # Store the conversation turn as a JSON document
    turn_data = json.dumps({"query": query, "answer": answer})

    # Add to ChromaDB with zero embedding (not used for search)
    # Using a fixed 384-dim zero vector to satisfy ChromaDB's schema
    zero_embedding = [0.0] * 384

    conversation_collection.add(
        ids=[turn_id],
        embeddings=[zero_embedding],
        documents=[turn_data],
        metadatas=[{"user_id": user_id, "timestamp": timestamp}]
    )

    # Get current count of turns for this user
    results = conversation_collection.get(
        where={"user_id": user_id},
        include=["metadatas"]
    )
    current_count = len(results["ids"])

    # If over the limit, delete the oldest turn(s)
    turns_to_delete = current_count - max_turns
    metadatas = results.get("metadatas")
    
    if turns_to_delete > 0 and metadatas is not None:
        # Sort by timestamp to find oldest
        turns_with_meta = list(zip(results["ids"], metadatas))
        turns_with_meta.sort(key=lambda x: float(str(x[1].get("timestamp", 0))))

        # Delete oldest turns
        ids_to_delete = [turn_id for turn_id, _ in turns_with_meta[:turns_to_delete]]
        conversation_collection.delete(ids=ids_to_delete)
        logger.debug(f"Deleted {turns_to_delete} oldest turns for user {user_id[:8]}")

    return max(0, current_count - turns_to_delete)


def get_conversation_history(user_id: str, max_turns: int = 5) -> list[dict]:
    """
    Retrieve a user's conversation history from ChromaDB.

    Returns the most recent max_turns turns, ordered chronologically.

    Args:
        user_id: Unique identifier for the user/conversation
        max_turns: Maximum number of turns to return (default: 5)

    Returns:
        List of dicts with 'query' and 'answer' keys, in chronological order
    """
    logger.debug(f"Fetching conversation history for user {user_id[:8]}...")

    # Get all turns for this user
    results = conversation_collection.get(
        where={"user_id": user_id},
        include=["documents", "metadatas"]
    )

    if not results["ids"]:
        logger.debug(f"No conversation history found for user {user_id[:8]}")
        return []

    # Parse and sort by timestamp
    turns = []
    docs = results.get("documents")
    metadatas = results.get("metadatas")
    
    if not docs or not metadatas:
        return []

    for doc, meta in zip(docs, metadatas):
        try:
            turn_data = json.loads(doc)
            turns.append({
                "query": turn_data["query"],
                "answer": turn_data["answer"],
                "timestamp": meta["timestamp"]
            })
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid conversation turn data: {e}")

    # Sort by timestamp and return most recent max_turns
    turns.sort(key=lambda x: x["timestamp"])
    recent_turns = turns[-max_turns:]

    # Remove timestamp from output (not needed by caller)
    return [{"query": t["query"], "answer": t["answer"]} for t in recent_turns]


def clear_conversation(user_id: str) -> int:
    """
    Delete all conversation turns for a specific user.

    Args:
        user_id: Unique identifier for the user/conversation

    Returns:
        Number of turns deleted
    """
    logger.info(f"Clearing conversation history for user {user_id[:8]}...")

    # Find all turns for this user
    results = conversation_collection.get(
        where={"user_id": user_id},
        include=["metadatas"]
    )

    if results["ids"]:
        conversation_collection.delete(ids=results["ids"])
        logger.info(f"Cleared {len(results['ids'])} turns for user {user_id[:8]}")
        return len(results["ids"])

    logger.debug(f"No conversation history found for user {user_id[:8]}")
    return 0
