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
from app.rag.router import route_query
from app.rag.answer import answer_question
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

    When include_overview=True, the router decides whether to generate
    an AI Overview (RAG answer) alongside the search results.
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

    # — AI Overview (optional) ——————————————————————————————————————————
    ai_overview = None
    if request.include_overview:
        router_result = route_query(request.query)
        if router_result.should_rag:
            logger.info("AI Overview triggered: %s", router_result.reason)
            ai_overview = answer_question(
                question=request.query,
                top_k=min(request.top_k, 8),
                retrieval_mode=request.mode,
                alpha=request.alpha,
                use_rewrite=router_result.needs_rewrite,
                db_path=DEFAULT_DB_PATH,
                index_dir=DEFAULT_INDEX_DIR,
            )
        else:
            logger.info("AI Overview skipped: %s", router_result.reason)

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "search mode=%s alpha=%.2f overview=%s query=%r → %d results in %.1f ms",
        request.mode,
        request.alpha,
        ai_overview is not None,
        request.query,
        len(results),
        elapsed_ms,
    )

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        total_results=len(results),
        results=results,
        ai_overview=ai_overview,
        latency_ms=round(elapsed_ms, 2),
    )
