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
from app.rag.rewriter import rewrite_query, detect_language
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

    # — Query enrichment: Chinese → extract English keywords → extra lexical search —
    if detect_language(request.query) == "zh":
        keywords = rewrite_query(request.query)
        if keywords and keywords != request.query:
            kw_results = search_lexical(
                keywords, top_k=request.top_k, db_path=DEFAULT_DB_PATH,
            )
            seen = {r.chunk_id for r in results}
            for r in kw_results:
                if r.chunk_id not in seen:
                    results.append(r)
                    seen.add(r.chunk_id)
            logger.info("Query enriched: '%s' → '%s', +%d extra results",
                        request.query[:60], keywords[:60], len(kw_results))

    # — Router (always run, even when include_overview=False) ———————————
    router_result = route_query(request.query)

    # — AI Overview (optional, reuses search results — no separate retrieval) —
    ai_overview = None
    if request.include_overview and router_result.should_rag:
        logger.info("AI Overview triggered: %s", router_result.reason)
        n_chunks = min(request.top_k, 8)
        ai_overview = answer_question(
            question=request.query,
            pre_retrieved=results[:n_chunks],
            top_k=n_chunks,
            retrieval_mode=request.mode,
            alpha=request.alpha,
            db_path=DEFAULT_DB_PATH,
            index_dir=DEFAULT_INDEX_DIR,
        )
    elif request.include_overview:
        logger.info("AI Overview skipped: %s", router_result.reason)

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "search mode=%s alpha=%.2f overview=%s should_rag=%s query=%r → %d results in %.1f ms",
        request.mode,
        request.alpha,
        ai_overview is not None,
        router_result.should_rag,
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
        should_rag=router_result.should_rag,
        rag_reason=router_result.reason,
    )
