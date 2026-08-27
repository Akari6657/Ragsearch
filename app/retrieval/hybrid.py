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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.config import DEFAULT_HYBRID_ALPHA, validate_hybrid_alpha
from app.core.schemas import SearchResult
from app.retrieval.lexical import search_lexical
from app.retrieval.vector_store import search_vector

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/indexes/metadata.sqlite")
DEFAULT_INDEX_DIR = Path("data/indexes/faiss")


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def _normalize_higher_is_better(scores: list[float]) -> list[float]:
    """Normalize scores where larger values are more relevant."""
    if not scores:
        return []

    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [0.5] * len(scores)

    return [(s - mn) / (mx - mn) for s in scores]


@dataclass
class _MinMaxCandidate:
    """One merged candidate while preserving its source result."""

    result: SearchResult
    lexical_normalized: float = 0.0
    vector_normalized: float = 0.0
    snippet: str = ""


def fuse_minmax_results(
    lexical_results: Sequence[SearchResult],
    vector_results: Sequence[SearchResult],
    *,
    top_k: int = 10,
    alpha: float = DEFAULT_HYBRID_ALPHA,
) -> list[SearchResult]:
    """Fuse pre-retrieved candidates without mutating either input ranking.

    The merge order and tie behavior intentionally match the original Hybrid
    implementation: lexical candidates establish insertion order, followed by
    vector-only candidates, and Python's stable sort preserves that order when
    fused scores are equal.
    """
    alpha = validate_hybrid_alpha(alpha)
    if not lexical_results and not vector_results:
        return []

    lexical_normalized = _normalize_higher_is_better(
        [result.score for result in lexical_results]
    )
    vector_normalized = _normalize_higher_is_better(
        [result.score for result in vector_results]
    )

    merged: dict[str, _MinMaxCandidate] = {}
    for result, normalized in zip(lexical_results, lexical_normalized):
        merged[result.chunk_id] = _MinMaxCandidate(
            result=result,
            lexical_normalized=normalized,
            snippet=result.snippet,
        )

    for result, normalized in zip(vector_results, vector_normalized):
        candidate = merged.get(result.chunk_id)
        if candidate is None:
            merged[result.chunk_id] = _MinMaxCandidate(
                result=result,
                vector_normalized=normalized,
                snippet=result.snippet,
            )
            continue

        candidate.vector_normalized = normalized
        if not candidate.snippet and result.snippet:
            candidate.snippet = result.snippet

    scored: list[tuple[float, SearchResult]] = []
    for candidate in merged.values():
        hybrid_score = (
            alpha * candidate.lexical_normalized
            + (1 - alpha) * candidate.vector_normalized
        )
        updates: dict[str, object] = {"score": hybrid_score}
        if candidate.snippet != candidate.result.snippet:
            updates["snippet"] = candidate.snippet
        fused_result = candidate.result.model_copy(update=updates)
        scored.append((hybrid_score, fused_result))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in scored[:top_k]]


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


def search_hybrid(
    query: str,
    top_k: int = 10,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    db_path: str | Path = DEFAULT_DB,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    lexical_query: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[SearchResult]:
    """Combined lexical + vector search with weighted score fusion.

    Args:
        query: Search query text.
        top_k: Final number of results to return.
        alpha: Weight for lexical scores (0-1). 1-alpha is vector weight.
               Default 0.5 gives equal weight.
        db_path: Path to metadata SQLite DB.
        index_dir: Directory with FAISS index files.
        lexical_query: Optional query for the BM25 branch. Dense always uses
                       ``query``. Defaults to ``query`` for existing callers.
        year_from: Inclusive earliest publication year for both branches.
        year_to: Inclusive latest publication year for both branches.

    Returns:
        Deduplicated, merged results sorted by hybrid_score (descending).
    """
    alpha = validate_hybrid_alpha(alpha)
    db_path = Path(db_path)
    index_dir = Path(index_dir)
    effective_lexical_query = query if lexical_query is None else lexical_query

    # — Fetch candidates from both retrievers —————————————————————————
    # We fetch more than top_k from each to give the merge step enough
    # material to work with.
    fetch_k = max(top_k, 20)

    lexical_results = search_lexical(
        effective_lexical_query,
        top_k=fetch_k,
        db_path=db_path,
        year_from=year_from,
        year_to=year_to,
    )
    vector_results = search_vector(
        query,
        top_k=fetch_k,
        db_path=db_path,
        index_dir=index_dir,
        year_from=year_from,
        year_to=year_to,
    )

    results = fuse_minmax_results(
        lexical_results,
        vector_results,
        top_k=top_k,
        alpha=alpha,
    )

    logger.debug(
        "search_hybrid('%s', alpha=%.2f, lexical_expanded=%s, years=%s..%s) → %d merged from %d lex + %d vec",
        query,
        alpha,
        effective_lexical_query != query,
        year_from,
        year_to,
        len(results),
        len(lexical_results),
        len(vector_results),
    )

    return results
