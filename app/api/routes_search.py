"""
Search API routes for CiteQuest-RAG.

POST /search — keyword / vector / hybrid paper search.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import get_db_path, get_faiss_dir
from app.core.schemas import SearchRequest, SearchResponse, SearchResult
from app.rag.router import route_query
from app.rag.answer import answer_question
from app.rag.rewriter import rewrite_query, detect_language
from app.retrieval.lexical import search_lexical
from app.retrieval.vector_store import search_vector
from app.retrieval.hybrid import search_hybrid

logger = logging.getLogger(__name__)

router = APIRouter()

SUPPORTED_MODES = {"lexical", "vector", "hybrid"}
FAISS_REQUIRED_MODES = {"vector", "hybrid"}


def _index_not_ready_detail(index_dir: Path) -> dict[str, str]:
    return {
        "error_code": "INDEX_NOT_READY",
        "message": (
            "FAISS index is not ready. Run `python scripts/build_faiss.py` "
            f"to create {index_dir / 'index.faiss'} and {index_dir / 'id_map.json'}."
        ),
    }


def _faiss_ready(index_dir: Path) -> bool:
    """Return True only when both FAISS files required for lookup exist."""
    return (index_dir / "index.faiss").exists() and (index_dir / "id_map.json").exists()


def _require_faiss_for_mode(mode: str, index_dir: Path) -> None:
    """Reject vector/hybrid search when the local FAISS artifacts are missing."""
    if mode in FAISS_REQUIRED_MODES and not _faiss_ready(index_dir):
        raise HTTPException(status_code=503, detail=_index_not_ready_detail(index_dir))


@router.post("/search", response_model=SearchResponse)
def search_papers(request: SearchRequest) -> SearchResponse:
    """Search papers by keyword, semantic meaning, or both.

    Modes:
    - lexical: SQLite FTS5 BM25 keyword search
    - vector:  FAISS cosine similarity over BGE-M3 embeddings
    - hybrid:  weighted combination (alpha controls lexical vs vector weight)

    When include_overview=True, the router decides whether to generate
    an AI Overview (RAG answer) alongside the search results.
    """
    start = time.perf_counter()
    db_path = get_db_path()
    index_dir = get_faiss_dir()

    # — mode guard ——————————————————————————————————————————————————————
    if request.mode not in SUPPORTED_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{request.mode}'. Supported: {', '.join(sorted(SUPPORTED_MODES))}.",
        )
    _require_faiss_for_mode(request.mode, index_dir)

    # — search ——————————————————————————————————————————————————————————
    if request.mode == "lexical":
        results = search_lexical(
            query=request.query,
            top_k=request.top_k,
            db_path=db_path,
        )
    elif request.mode == "vector":
        results = search_vector(
            query=request.query,
            top_k=request.top_k,
            db_path=db_path,
            index_dir=index_dir,
        )
    else:  # hybrid
        results = search_hybrid(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha,
            db_path=db_path,
            index_dir=index_dir,
        )

    # — Query enrichment: Chinese → extract English keywords → extra lexical search —
    rewrite_keywords = ""
    if detect_language(request.query) == "zh":
        keywords = rewrite_query(request.query)
        if keywords and keywords != request.query:
            rewrite_keywords = keywords
            kw_results = search_lexical(
                keywords, top_k=request.top_k, db_path=db_path,
            )
            seen = {r.chunk_id for r in results}
            for r in kw_results:
                if r.chunk_id not in seen:
                    results.append(r)
                    seen.add(r.chunk_id)
            logger.info("Query enriched: '%s' → '%s', +%d extra results",
                        request.query[:60], keywords[:60], len(kw_results))

    # — Save chunk-level results for RAG (before paper-level dedup) ——
    rag_candidates = list(results)

    # — Dedup: preserve the highest-ranked chunk for each paper ———————
    seen_paper_ids: set[str] = set()
    deduped: list[SearchResult] = []
    for r in results:
        if r.paper_id in seen_paper_ids:
            continue
        seen_paper_ids.add(r.paper_id)
        deduped.append(r)
    results = deduped

    # — Snippet: always use abstract preview ——————————————————————————
    for r in results:
        r.snippet = r.abstract[:300] if r.abstract else ""

    # — Router (always run) ———————————————————————————————————————————
    router_result = route_query(request.query)

    # — AI Overview (uses chunk-level results — NOT deduped — for richer evidence) —
    ai_overview = None
    if request.include_overview and router_result.should_rag:
        logger.info("AI Overview triggered: %s", router_result.reason)
        n_chunks = min(request.top_k, 8)
        # Use pre-dedup results so LLM gets multiple chunks from the same paper
        ai_overview = answer_question(
            question=request.query,
            pre_retrieved=rag_candidates[:n_chunks],
            top_k=n_chunks,
            retrieval_mode=request.mode,
            alpha=request.alpha,
            db_path=db_path,
            index_dir=index_dir,
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
        rewrite_keywords=rewrite_keywords,
    )
