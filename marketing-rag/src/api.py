"""
api.py
FastAPI application for the Marketing Insights RAG system.

Endpoints:
  POST /api/ask   — answer a marketing question from the knowledge base
  GET  /api/health — health check

The Retriever is instantiated once at startup (singleton) to avoid rebuilding
the BM25 index on every request.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .models import AskRequest, AskResponse
from .retriever import Retriever
from .rag_engine import generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Singleton retriever (initialized at startup) ───────────────────────────────
_retriever: Retriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    logger.info("Initializing retriever (building BM25 index from Qdrant)...")
    try:
        _retriever = Retriever()
        logger.info("✅ Retriever ready")
    except RuntimeError as e:
        logger.error(f"❌ Retriever init failed: {e}")
        logger.error("Run 'python scripts/ingest.py' first to build the knowledge base.")
        # Don't crash the server — health endpoint will surface the issue
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Marketing Insights RAG",
    description="Ask questions about campaigns, customer research, and brand strategy.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok" if _retriever is not None else "degraded",
        "retriever_ready": _retriever is not None,
        "message": "Retriever not initialized. Run ingest.py." if _retriever is None else "Ready",
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Answer a marketing knowledge question using hybrid RAG.
    """
    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not initialized. Run scripts/ingest.py first.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"Query: {query[:100]}")

    # Retrieval
    t0 = time.time()
    try:
        chunks = _retriever.retrieve(query, top_n_rerank=5)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")
    retrieval_ms = int((time.time() - t0) * 1000)

    # Generation
    t1 = time.time()
    try:
        answer, p_tokens, c_tokens = await generate_answer(query, chunks)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")
    generation_ms = int((time.time() - t1) * 1000)
    total_ms = int((time.time() - t0) * 1000)

    logger.info(
        f"Done — retrieval: {retrieval_ms}ms, generation: {generation_ms}ms, "
        f"chunks: {len(chunks)}, thin: {answer.thin_context}, "
        f"tokens: {p_tokens+c_tokens}, confidence: {answer.confidence}"
    )

    return AskResponse(
        answer=answer,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        chunks_retrieved=len(chunks),
    )


# Mount frontend as last route (catches all unmatched paths)
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8001, reload=False)
