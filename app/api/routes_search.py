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

logger = logging.getLogger(__name__)

router = APIRouter()

# Default DB path — can be overridden via dependency injection later.
DEFAULT_DB_PATH = Path("data/indexes/metadata.sqlite")


@router.post("/search", response_model=SearchResponse)
def search_papers(request: SearchRequest) -> SearchResponse:
    """Search papers by keyword or semantic meaning.

    v0.1 supports only mode="lexical".  Vector and hybrid modes
    will be added in v0.2.
    """
    start = time.perf_counter()

    # — mode guard (v0.1) ——————————————————————————————————————————————
    if request.mode not in ("lexical",):
        raise HTTPException(
            status_code=400,
            detail=f"Search mode '{request.mode}' is not available in v0.1. "
            f"Only 'lexical' is supported.",
        )

    # — search ——————————————————————————————————————————————————————————
    results = search_lexical(
        query=request.query,
        top_k=request.top_k,
        db_path=DEFAULT_DB_PATH,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "search mode=%s query=%r top_k=%d → %d results in %.1f ms",
        request.mode,
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
