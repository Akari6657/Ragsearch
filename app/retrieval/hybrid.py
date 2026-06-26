"""
Hybrid retriever: merge lexical (BM25) and vector (cosine) results.

Strategy (min-max normalization + weighted sum):
1. Run both retrievers independently, each returning top_k candidates.
2. Normalize scores within each result set to [0, 1].
3. hybrid_score = alpha * lexical_norm + (1 - alpha) * vector_norm
4. Deduplicate by chunk_id (merge scores when both retrievers find the same chunk).
5. Sort by hybrid_score descending, return top_k.

Alpha controls the trade-off:
- alpha = 0.0 → pure semantic (vector only)
- alpha = 0.5 → equal weight (default)
- alpha = 1.0 → pure keyword (lexical only)

Usage:
    from app.retrieval.hybrid import search_hybrid
    results = search_hybrid("neural network", top_k=10, alpha=0.5)
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.schemas import SearchResult
from app.retrieval.lexical import search_lexical
from app.retrieval.vector_store import search_vector

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/indexes/metadata.sqlite")
DEFAULT_INDEX_DIR = Path("data/indexes/faiss")


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def _minmax_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] range using min-max scaling.

    For BM25 (negative scores, lower = better), this will map the
    best (most negative) score to 1.0 and the worst to 0.0.

    If all scores are identical, returns 0.5 for everything.
    """
    if not scores:
        return []

    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [0.5] * len(scores)

    return [(s - mn) / (mx - mn) for s in scores]


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


def search_hybrid(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    db_path: str | Path = DEFAULT_DB,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> list[SearchResult]:
    """Combined lexical + vector search with weighted score fusion.

    Args:
        query: Search query text.
        top_k: Final number of results to return.
        alpha: Weight for lexical scores (0-1). 1-alpha is vector weight.
               Default 0.5 gives equal weight.
        db_path: Path to metadata SQLite DB.
        index_dir: Directory with FAISS index files.

    Returns:
        Deduplicated, merged results sorted by hybrid_score (descending).
    """
    db_path = Path(db_path)
    index_dir = Path(index_dir)

    # — Fetch candidates from both retrievers —————————————————————————
    # We fetch more than top_k from each to give the merge step enough
    # material to work with.
    fetch_k = max(top_k, 20)

    lexical_results = search_lexical(query, top_k=fetch_k, db_path=db_path)
    vector_results = search_vector(query, top_k=fetch_k, db_path=db_path, index_dir=index_dir)

    if not lexical_results and not vector_results:
        return []

    # — Normalize scores separately ————————————————————————————————————
    lex_scores = [r.score for r in lexical_results]
    vec_scores = [r.score for r in vector_results]

    lex_norm = _minmax_normalize(lex_scores)
    vec_norm = _minmax_normalize(vec_scores)

    # — Merge by chunk_id ——————————————————————————————————————————————
    merged: dict[str, dict] = {}  # chunk_id → {lex_norm, vec_norm, result}

    for r, norm in zip(lexical_results, lex_norm):
        merged[r.chunk_id] = {
            "lex_norm": norm,
            "vec_norm": 0.0,
            "result": r,
            "lex_score": r.score,
            "vec_score": 0.0,
        }

    for r, norm in zip(vector_results, vec_norm):
        if r.chunk_id in merged:
            # Both retrievers found this chunk — combine
            merged[r.chunk_id]["vec_norm"] = norm
            merged[r.chunk_id]["vec_score"] = r.score
            # Carry over snippet from lexical if available
            if not merged[r.chunk_id]["result"].snippet and r.snippet:
                merged[r.chunk_id]["result"].snippet = r.snippet
        else:
            merged[r.chunk_id] = {
                "lex_norm": 0.0,
                "vec_norm": norm,
                "result": r,
                "lex_score": 0.0,
                "vec_score": r.score,
            }

    # — Compute hybrid scores ——————————————————————————————————————————
    scored: list[tuple[float, SearchResult]] = []
    for entry in merged.values():
        hybrid_score = alpha * entry["lex_norm"] + (1 - alpha) * entry["vec_norm"]
        r = entry["result"]
        # Use the hybrid score for output ordering
        r.score = round(hybrid_score, 4)
        scored.append((hybrid_score, r))

    # — Sort and truncate ——————————————————————————————————————————————
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [r for _, r in scored[:top_k]]

    logger.debug(
        "search_hybrid('%s', alpha=%.2f) → %d merged from %d lex + %d vec",
        query,
        alpha,
        len(results),
        len(lexical_results),
        len(vector_results),
    )

    return results
