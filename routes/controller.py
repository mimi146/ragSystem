import logging
import mimetypes
import json
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from vector_store import (
    get_collection_stats,
    delete_by_filename,
    clear_conversation,
)
from rag_workflow import (
    WorkflowInputError,
    answer_query,
    doc_storage_path,
    ingest_document,
    normalize_filename,
    retrieve_top_k,
    stream_answer_query,
)

# Configure logging - shows output in console with timestamp and log level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

UPLOADED_DOCS_DIR = Path("./uploaded_docs")
UPLOADED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Conversation memory configuration
# Turns are now persisted to ChromaDB via vector_store functions
MAX_MEMORY_TURNS = 5

# APIRouter creates a modular set of routes
# prefix="/weather" means all routes here start with /weather
# tags=["weather"] groups these endpoints together in Swagger UI
router = APIRouter(prefix="/weather", tags=["weather"])

# =============================================================================
# PYDANTIC MODELS - Define the shape of data for request/response validation
# =============================================================================
# Pydantic models automatically validate incoming JSON data and convert types.
# They also generate OpenAPI schema for Swagger UI documentation.

# =============================================================================
# RAG MODELS - For document upload, query, and management
# =============================================================================

class QueryRequest(BaseModel):
    """
    Request model for semantic search queries.

    Attributes:
        query: The search question or text to find similar content for
        n_results: Number of most similar chunks to return (default: 3)
        model: Optional model override for generation endpoints
    """
    query: str
    n_results: int = 3
    # Client-provided conversation key used to fetch/store short-term memory.
    user_id: str = "anonymous"
    # Optional per-request model override (for example: deepseek-r1:1.5b).
    model: str | None = None

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model_override(cls, value):
        """Treat empty/placeholder model values as no override.

        Swagger UI often pre-fills optional strings with "string". Passing that
        through to Azure/OpenAI causes deployment/model lookup failures.
        """
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            # Swagger UI often sends "string" for optional fields; ignore it so
            # provider defaults are used instead of a non-existent deployment.
            if not normalized or normalized.lower() == "string":
                return None
            return normalized
        return value

class QueryResponse(BaseModel):
    """
    Response model for semantic search results.

    Attributes:
        query: Echo of the original query for reference
        results: List of matching document chunks with similarity scores
    """
    query: str
    results: list[dict]

class DeleteRequest(BaseModel):
    """
    Request model for deleting documents from the vector store.

    Attributes:
        filename: Name of the file to delete chunks for
    """
    filename: str


@router.get("/")
def hello():
    """
    Health check endpoint for the weather module.

    Useful for verifying the API is running and the router is mounted correctly.
    """
    logger.debug("Health check requested")
    return {"message": "Hello World"}


@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    tokens_per_chunk: int = 100,
    overlap: int = 20
):
    """
    RAG INGESTION ENDPOINT - Upload and process a document for semantic search.

    This is the first step in the RAG pipeline. The endpoint:
    1. Validates file type (PDF or Word document)
    2. Extracts raw text from the file
    3. Splits text into overlapping chunks
    4. Generates embeddings for each chunk using sentence-transformers
    5. Stores chunks + embeddings in ChromaDB vector store

    The embeddings enable semantic search later - you can query with natural
    language and find relevant chunks even without keyword matches.

    Args:
        file: Uploaded file (multipart/form-data). Must be PDF or DOC/DOCX.
        tokens_per_chunk: Words per chunk. Smaller = more precise retrieval.
        overlap: Overlapping words between chunks. Prevents context loss.

    Returns:
        Processing summary including chunk count and preview.

    Raises:
        HTTPException 400: If file type is unsupported or text extraction fails.

    Example curl:
        curl -X POST http://localhost:8000/weather/upload-document \\
             -F "file=@document.pdf" -F "tokens_per_chunk=100" -F "overlap=20"
    """
    logger.info("Upload request: %s, type=%s", normalize_filename(file.filename), file.content_type)

    content = await file.read()
    logger.debug("Read %d bytes from file", len(content))

    try:
        ingest_result = ingest_document(
            file_content=content,
            content_type=file.content_type or "",
            filename=file.filename,
            storage_dir=UPLOADED_DOCS_DIR,
            tokens_per_chunk=tokens_per_chunk,
            overlap=overlap,
        )
    except WorkflowInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    response = ingest_result.to_dict()
    response["message"] = "File processed, embedded, and stored in vector DB!"
    return response


@router.post("/query")
def query_documents(payload: QueryRequest):
    """
    RAG RETRIEVAL ENDPOINT - Find document chunks similar to your query.

    This uses semantic search (not keyword matching). The query is converted
    to an embedding vector, then compared to stored document embeddings using
    cosine similarity. This means:
    - "hot star" can find documents about "the sun"
    - Spelling variations and synonyms work naturally
    - Results ranked by semantic relevance, not word frequency

    Args:
        payload: QueryRequest with:
            - query: Natural language question or search text
            - n_results: How many matching chunks to return (default: 3)

    Returns:
        QueryResponse with original query and matching chunks with distances.
        Lower distance = more similar (0 = identical embeddings).

    Example curl:
        curl -X POST http://localhost:8000/weather/query \\
             -H "Content-Type: application/json" \\
             -d '{"query": "what causes solar flares?", "n_results": 3}'
    """
    logger.info(f"Query request: '{payload.query}' (asking for {payload.n_results} results)")

    # Retriever stage: embed query + similarity search in vector DB
    results = retrieve_top_k(payload.query, n_results=payload.n_results)

    if results:
        best_distance = results[0].get("distance")
        logger.info(
            "Found %d matching chunks, best distance: %s",
            len(results),
            f"{float(best_distance):.4f}" if isinstance(best_distance, (int, float)) else "n/a",
        )
    else:
        logger.warning("No matching documents found")

    return QueryResponse(query=payload.query, results=results)


@router.post("/ask")
def ask_question(payload: QueryRequest):
    """
    RAG GENERATION ENDPOINT - Retrieves documents and generates an LLM answer.

    Like /query, this searches the vector database for relevant chunks,
    but it also passes those chunks to the configured LLM to synthesize
    a short, plain-English answer to the user's question.

    Conversation history is persisted to ChromaDB for continuity across restarts.
    """
    logger.info(f"Ask request: '{payload.query}'")

    try:
        return answer_query(
            query=payload.query,
            n_results=payload.n_results,
            user_id=payload.user_id,
            max_memory_turns=MAX_MEMORY_TURNS,
            model=payload.model,
        )
    except Exception as e:
        logger.error(f"Failed to generate LLM answer: {e}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")


@router.post("/ask-stream")
async def ask_question_stream(payload: QueryRequest):
    """
    RAG GENERATION ENDPOINT (STREAMING) - Streams the LLM answer token-by-token.

    This is the "fast" version of /ask. It starts sending text immediately
    as it's generated by the local LLM, greatly improving perceived speed.

    Conversation history is persisted to ChromaDB for continuity across restarts.
    """
    logger.info(f"Ask-stream request: '{payload.query}'")

    stream, sources = stream_answer_query(
        query=payload.query,
        n_results=payload.n_results,
        user_id=payload.user_id,
        max_memory_turns=MAX_MEMORY_TURNS,
        model=payload.model,
    )
  

    if stream is None:
        return {"query": payload.query, "answer": "No relevant context found.", "sources": []}

    # Return a StreamingResponse so the client starts receiving data immediately
    return StreamingResponse(
        stream,
        media_type="text/plain",
        # Surface retrieval sources without changing token stream payload format.
        headers={"X-Weather-Sources": json.dumps(sources, separators=(",", ":"))},
    )


@router.get("/source-doc/{upload_id}/{filename:path}")
def browse_source_document(upload_id: str, filename: str, download: bool = False):
    """
    Serve the original uploaded source document for browser preview/download.
    """
    safe_filename = normalize_filename(filename)
    doc_path = doc_storage_path(UPLOADED_DOCS_DIR, upload_id, safe_filename)

    if not doc_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source document not found."
        )

    media_type = mimetypes.guess_type(safe_filename)[0] or "application/octet-stream"
    content_disposition_type = "attachment" if download else "inline"
    return FileResponse(
        path=doc_path,
        media_type=media_type,
        filename=safe_filename,
        content_disposition_type=content_disposition_type,
    )


@router.get("/stats")
def get_stats():
    """
    Get vector store statistics.

    Useful for monitoring how many document chunks are indexed.
    In production, you might also expose memory usage, collection count, etc.

    Returns:
        Dict with total_chunks and collection_name.
    """
    stats = get_collection_stats()
    logger.debug(f"Stats requested: {stats['total_chunks']} chunks in vector DB")
    return stats


@router.delete("/documents")
def delete_documents(payload: DeleteRequest):
    """
    Delete all chunks from a specific uploaded file.

    Use this to remove documents from the vector store when they're outdated
    or were uploaded by mistake. The filename must match exactly what was
    provided during upload.

    Note: In production, consider using document IDs or user IDs for more
    flexible deletion (e.g., delete all docs uploaded by a specific user).

    Args:
        payload: DeleteRequest with filename to remove.

    Returns:
        Confirmation with count of deleted chunks.

    Raises:
        HTTPException 404: If no chunks exist for the given filename.
    """
    logger.info(f"Delete request for filename: {payload.filename}")

    deleted_count = delete_by_filename(payload.filename)

    if deleted_count == 0:
        logger.warning(f"Delete found no chunks for: {payload.filename}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for filename: {payload.filename}"
        )

    logger.info(f"Successfully deleted {deleted_count} chunks")
    return {"message": f"Deleted {deleted_count} chunks", "filename": payload.filename}


class MemoryClearRequest(BaseModel):
    """
    Request model for clearing conversation memory.

    Attributes:
        user_id: ID of the user whose conversation history to clear.
                 Defaults to "anonymous" if not provided.
    """
    user_id: str = "anonymous"


@router.post("/memory/clear")
def clear_memory(payload: MemoryClearRequest):
    """
    Clear conversation history for a specific user.

    This endpoint allows users to delete their conversation history
    from the persistent storage. Useful for privacy or starting fresh.

    Args:
        payload: MemoryClearRequest with user_id (default: "anonymous")

    Returns:
        Confirmation with count of turns deleted.
    """
    logger.info(f"Memory clear request for user: {payload.user_id[:8]}...")

    deleted_count = clear_conversation(payload.user_id)

    return {
        "message": f"Cleared {deleted_count} conversation turns",
        "user_id": payload.user_id,
        "turns_deleted": deleted_count
    }
