"""
Search API routes for CiteQuest-RAG.

POST /search — keyword / vector / hybrid paper search.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.schemas import SearchRequest, SearchResponse
from app.retrieval.lexical import search_lexical
from app.retrieval.vector_store import search_vector
from app.retrieval.hybrid import search_hybrid

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_DB_PATH = Path("data/indexes/metadata.sqlite")
DEFAULT_INDEX_DIR = Path("data/indexes/faiss")

SUPPORTED_MODES = {"lexical", "vector", "hybrid"}


@router.post("/search", response_model=SearchResponse)
def search_papers(request: SearchRequest) -> SearchResponse:
    """Search papers by keyword, semantic meaning, or both.

    Modes:
    - lexical: SQLite FTS5 BM25 keyword search
    - vector:  FAISS cosine similarity over BGE-small-en embeddings
    - hybrid:  weighted combination (alpha controls lexical vs vector weight)
    """
    start = time.perf_counter()

    # — mode guard ——————————————————————————————————————————————————————
    if request.mode not in SUPPORTED_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{request.mode}'. Supported: {', '.join(sorted(SUPPORTED_MODES))}.",
        )

    # — search ——————————————————————————————————————————————————————————
    if request.mode == "lexical":
        results = search_lexical(
            query=request.query,
            top_k=request.top_k,
            db_path=DEFAULT_DB_PATH,
        )
    elif request.mode == "vector":
        results = search_vector(
            query=request.query,
            top_k=request.top_k,
            db_path=DEFAULT_DB_PATH,
            index_dir=DEFAULT_INDEX_DIR,
        )
    else:  # hybrid
        results = search_hybrid(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha,
            db_path=DEFAULT_DB_PATH,
            index_dir=DEFAULT_INDEX_DIR,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "search mode=%s alpha=%.2f query=%r top_k=%d → %d results in %.1f ms",
        request.mode,
        request.alpha,
        request.query,
        request.top_k,
        len(results),
        elapsed_ms,
    )

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        total_results=len(results),
        results=results,
        latency_ms=round(elapsed_ms, 2),
    )
