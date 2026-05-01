from contextlib import asynccontextmanager
from fastapi import FastAPI  # pyright: ignore[reportMissingImports]
from fastapi.staticfiles import StaticFiles # NEW: Needed to serve HTML/CSS/JS

from routes.controller import router as weather_router
from vector_store import get_embedding_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Pre-load the embedding model into memory
    # This avoids the ~2 second delay when the first user uploads a file or queries
    print("Pre-loading embedding model...")
    get_embedding_model()
    print("Embedding model ready.")
    yield
    # SHUTDOWN: Clean up resources here if needed
    pass

app = FastAPI(
    title="Weather API",
    lifespan=lifespan
)

# NEW: Mount the static directory to serve the chat interface
# Access it at: http://localhost:8000/chat/index.html (or just /chat/)
app.mount("/chat", StaticFiles(directory="static", html=True), name="chat")

app.include_router(weather_router)


@app.get("/", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
