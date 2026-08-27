"""Fair, evaluation-only comparison of Min-max fusion and RRF.

Each query retrieves BM25 and Dense candidates once. Both fusion strategies
then consume the same immutable candidate tuples, so retrieval variation and
model latency cannot favor one fusion method over the other. This module does
not change the production Hybrid strategy.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.core.config import validate_hybrid_alpha
from app.core.schemas import SearchResult
from app.eval.retrieval_eval import (
    QUERY_TYPES,
    EvalQuery,
    deduplicate_papers,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from app.retrieval.hybrid import fuse_minmax_results


DEFAULT_CANDIDATE_DEPTH = 20
DEFAULT_FINAL_TOP_K = 10
DEFAULT_MINMAX_ALPHA = 0.5
DEFAULT_RRF_K = 60

SearchFn = Callable[[str, int], Sequence[SearchResult]]


@dataclass(frozen=True)
class FusionCandidateSet:
    """The shared candidate rankings and their measured retrieval cost."""

    lexical: tuple[SearchResult, ...]
    dense: tuple[SearchResult, ...]
    lexical_latency_ms: float
    dense_latency_ms: float
    total_latency_ms: float


@dataclass
class _RrfCandidate:
    result: SearchResult
    lexical_rank: int | None = None
    dense_rank: int | None = None
    snippet: str = ""


def retrieve_fusion_candidates(
    query: str,
    lexical_search: SearchFn,
    dense_search: SearchFn,
    *,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
) -> FusionCandidateSet:
    """Retrieve each branch exactly once and retain the original result objects."""
    if candidate_depth <= 0:
        raise ValueError("candidate_depth must be positive")

    total_started = time.perf_counter()
    lexical_started = time.perf_counter()
    lexical = tuple(lexical_search(query, candidate_depth))
    lexical_latency_ms = (time.perf_counter() - lexical_started) * 1000

    dense_started = time.perf_counter()
    dense = tuple(dense_search(query, candidate_depth))
    dense_latency_ms = (time.perf_counter() - dense_started) * 1000
    total_latency_ms = (time.perf_counter() - total_started) * 1000

    return FusionCandidateSet(
        lexical=lexical,
        dense=dense,
        lexical_latency_ms=lexical_latency_ms,
        dense_latency_ms=dense_latency_ms,
        total_latency_ms=total_latency_ms,
    )


def _validate_rrf_k(rrf_k: int) -> None:
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")


def _rrf_sort_key(candidate: _RrfCandidate, score: float) -> tuple[Any, ...]:
    ranks = tuple(
        rank
        for rank in (candidate.lexical_rank, candidate.dense_rank)
        if rank is not None
    )
    return (-score, min(ranks), sum(ranks), candidate.result.chunk_id)


def fuse_rrf_results(
    lexical_results: Sequence[SearchResult],
    dense_results: Sequence[SearchResult],
    *,
    top_k: int = DEFAULT_FINAL_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SearchResult]:
    """Fuse two rankings with equal-weight Reciprocal Rank Fusion.

    Ranks start at one and a candidate missing from a branch contributes zero.
    Exact score ties are resolved symmetrically by best rank, total rank, then
    ``chunk_id``. The returned models are copies; source scores are untouched.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    _validate_rrf_k(rrf_k)

    merged: dict[str, _RrfCandidate] = {}
    for rank, result in enumerate(lexical_results, start=1):
        if result.chunk_id not in merged:
            merged[result.chunk_id] = _RrfCandidate(
                result=result,
                lexical_rank=rank,
                snippet=result.snippet,
            )

    for rank, result in enumerate(dense_results, start=1):
        candidate = merged.get(result.chunk_id)
        if candidate is None:
            merged[result.chunk_id] = _RrfCandidate(
                result=result,
                dense_rank=rank,
                snippet=result.snippet,
            )
            continue
        if candidate.dense_rank is None:
            candidate.dense_rank = rank
        if not candidate.snippet and result.snippet:
            candidate.snippet = result.snippet

    scored: list[tuple[tuple[Any, ...], SearchResult]] = []
    for candidate in merged.values():
        score = sum(
            1.0 / (rrf_k + rank)
            for rank in (candidate.lexical_rank, candidate.dense_rank)
            if rank is not None
        )
        updates: dict[str, object] = {"score": score}
        if candidate.snippet != candidate.result.snippet:
            updates["snippet"] = candidate.snippet
        fused_result = candidate.result.model_copy(update=updates)
        scored.append((_rrf_sort_key(candidate, score), fused_result))

    scored.sort(key=lambda item: item[0])
    return [result for _, result in scored[:top_k]]


def _ranking_metrics(
    query: EvalQuery, results: Sequence[SearchResult]
) -> dict[str, Any]:
    papers = deduplicate_papers(results, limit=DEFAULT_FINAL_TOP_K)
    paper_ids = [result.paper_id for result in papers]
    relevant = set(query.relevant_paper_ids)
    first_relevant_rank = next(
        (rank for rank, paper_id in enumerate(paper_ids, start=1) if paper_id in relevant),
        None,
    )
    return {
        "query_id": query.query_id,
        "query": query.query,
        "query_type": query.query_type,
        "relevant_paper_ids": list(query.relevant_paper_ids),
        "retrieved_paper_ids": paper_ids,
        "first_relevant_rank": first_relevant_rank,
        "hit@5": bool(hit_rate_at_k(paper_ids, relevant, 5)),
        "hit@10": bool(hit_rate_at_k(paper_ids, relevant, 10)),
        "recall@5": recall_at_k(paper_ids, relevant, 5),
        "recall@10": recall_at_k(paper_ids, relevant, 10),
        "mrr@10": reciprocal_rank_at_k(paper_ids, relevant, 10),
        "ndcg@10": ndcg_at_k(paper_ids, relevant, 10),
    }


def _aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_fields = {
        "hit_rate@5": "hit@5",
        "hit_rate@10": "hit@10",
        "recall@5": "recall@5",
        "recall@10": "recall@10",
        "mrr@10": "mrr@10",
        "ndcg@10": "ndcg@10",
    }
    if not rows:
        return {"query_count": 0, **{name: 0.0 for name in metric_fields}}
    return {
        "query_count": len(rows),
        **{
            name: float(np.mean([row[field] for row in rows]))
            for name, field in metric_fields.items()
        },
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": round(float(np.mean(values)), 3) if values else 0.0,
        "p50_ms": round(float(np.percentile(values, 50)), 3) if values else 0.0,
        "p95_ms": round(float(np.percentile(values, 95)), 3) if values else 0.0,
    }


def _method_summary(
    method: str,
    rows: Sequence[dict[str, Any]],
    fusion_latencies: Sequence[float],
) -> dict[str, Any]:
    return {
        "method": method,
        "metrics": _aggregate_metrics(rows),
        "by_query_type": {
            query_type: _aggregate_metrics(
                [row for row in rows if row["query_type"] == query_type]
            )
            for query_type in QUERY_TYPES
            if any(row["query_type"] == query_type for row in rows)
        },
        "fusion_latency": _latency_summary(fusion_latencies),
        "per_query": list(rows),
    }


def _rank_for_comparison(rank: int | None) -> float:
    return float(rank) if rank is not None else float("inf")


def run_fusion_comparison(
    queries: Sequence[EvalQuery],
    lexical_search: SearchFn,
    dense_search: SearchFn,
    *,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    top_k: int = DEFAULT_FINAL_TOP_K,
    alpha: float = DEFAULT_MINMAX_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
    query_split: str = "dev",
) -> dict[str, Any]:
    """Compare Min-max and RRF using one shared retrieval pass per query."""
    if not queries:
        raise ValueError("queries must not be empty")
    if top_k != DEFAULT_FINAL_TOP_K:
        raise ValueError("top_k must be 10 for the v1.2 fusion comparison")
    if candidate_depth < top_k:
        raise ValueError("candidate_depth must be greater than or equal to top_k")
    unexpected_splits = sorted(
        {query.split for query in queries if query.split != query_split}
    )
    if unexpected_splits:
        raise ValueError(
            f"fusion comparison expected only {query_split!r} queries; "
            f"found {', '.join(unexpected_splits)}"
        )
    alpha = validate_hybrid_alpha(alpha)
    _validate_rrf_k(rrf_k)

    method_rows: dict[str, list[dict[str, Any]]] = {"minmax": [], "rrf": []}
    fusion_latencies: dict[str, list[float]] = {"minmax": [], "rrf": []}
    retrieval_latencies: dict[str, list[float]] = {
        "bm25": [],
        "dense": [],
        "total": [],
    }
    candidate_audit: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    wins = {"minmax": 0, "rrf": 0, "tie": 0}

    for query in queries:
        candidates = retrieve_fusion_candidates(
            query.query,
            lexical_search,
            dense_search,
            candidate_depth=candidate_depth,
        )
        retrieval_latencies["bm25"].append(candidates.lexical_latency_ms)
        retrieval_latencies["dense"].append(candidates.dense_latency_ms)
        retrieval_latencies["total"].append(candidates.total_latency_ms)

        started = time.perf_counter()
        minmax_results = fuse_minmax_results(
            candidates.lexical,
            candidates.dense,
            top_k=top_k,
            alpha=alpha,
        )
        fusion_latencies["minmax"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        rrf_results = fuse_rrf_results(
            candidates.lexical,
            candidates.dense,
            top_k=top_k,
            rrf_k=rrf_k,
        )
        fusion_latencies["rrf"].append((time.perf_counter() - started) * 1000)

        minmax_row = _ranking_metrics(query, minmax_results)
        rrf_row = _ranking_metrics(query, rrf_results)
        method_rows["minmax"].append(minmax_row)
        method_rows["rrf"].append(rrf_row)

        minmax_rank = _rank_for_comparison(minmax_row["first_relevant_rank"])
        rrf_rank = _rank_for_comparison(rrf_row["first_relevant_rank"])
        winner = "tie"
        if minmax_rank < rrf_rank:
            winner = "minmax"
        elif rrf_rank < minmax_rank:
            winner = "rrf"
        wins[winner] += 1

        lexical_ids = [result.chunk_id for result in candidates.lexical]
        dense_ids = [result.chunk_id for result in candidates.dense]
        lexical_set = set(lexical_ids)
        dense_set = set(dense_ids)
        union = lexical_set | dense_set
        overlap = lexical_set & dense_set
        overlap_rows.append(
            {
                "query_id": query.query_id,
                "overlap_count": len(overlap),
                "union_count": len(union),
                "jaccard": len(overlap) / len(union) if union else 0.0,
            }
        )
        candidate_audit.append(
            {
                "query_id": query.query_id,
                "bm25_chunk_ids": lexical_ids,
                "dense_chunk_ids": dense_ids,
            }
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_or_development",
        "query_count": len(queries),
        "protocol": {
            "candidate_retrieval": "BM25 and Dense each run once per measured query",
            "candidate_depth": candidate_depth,
            "final_top_k": top_k,
            "query_split": query_split,
            "ranking_unit": "paper after chunk-level fusion",
            "minmax_alpha": alpha,
            "rrf_k": rrf_k,
            "rrf_weights": "equal",
            "pairwise_win_rule": "lower first relevant rank; both missing is a tie",
        },
        "shared_retrieval_latency": {
            branch: _latency_summary(values)
            for branch, values in retrieval_latencies.items()
        },
        "candidate_overlap": {
            "mean_jaccard": float(np.mean([row["jaccard"] for row in overlap_rows])),
            "per_query": overlap_rows,
        },
        "methods": {
            method: _method_summary(
                method,
                method_rows[method],
                fusion_latencies[method],
            )
            for method in ("minmax", "rrf")
        },
        "pairwise_first_relevant_rank": {
            "minmax_wins": wins["minmax"],
            "rrf_wins": wins["rrf"],
            "ties": wins["tie"],
        },
        "candidate_audit": candidate_audit,
    }
