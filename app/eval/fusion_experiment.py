"""Fair, evaluation-only comparison of Min-max fusion and RRF.

Each query retrieves BM25 and Dense candidates once. Both fusion strategies
then consume the same immutable candidate tuples, so retrieval variation and
model latency cannot favor one fusion method over the other. This module does
not change the production Hybrid strategy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import validate_hybrid_alpha
from app.core.schemas import SearchResult
from app.eval.retrieval_eval import (
    DEFAULT_DB as DEFAULT_BENCHMARK_DB,
    DEFAULT_EVAL as DEFAULT_EVAL_PATH,
    DEFAULT_INDEX_DIR as DEFAULT_BENCHMARK_INDEX_DIR,
    OFFICIAL_DEV_COUNT,
    OFFICIAL_PAPER_COUNT,
    QUERY_TYPES,
    EvalQuery,
    build_benchmark_manifest,
    deduplicate_papers,
    hit_rate_at_k,
    load_eval_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from app.retrieval.hybrid import fuse_minmax_results
from app.retrieval.lexical import search_lexical
from app.retrieval.vector_store import search_vector


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "arxiv_cs_benchmark_v1_50000.jsonl"
DEFAULT_BASELINE_REPORT = PROJECT_ROOT / "reports" / "retrieval_baseline_v1.json"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "retrieval_fusion_dev_v1_2.json"
DEFAULT_MD_REPORT = PROJECT_ROOT / "reports" / "retrieval_fusion_dev_v1_2.md"

DEFAULT_CANDIDATE_DEPTH = 20
DEFAULT_FINAL_TOP_K = 10
DEFAULT_MINMAX_ALPHA = 0.5
DEFAULT_RRF_K = 60
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_827
DEFAULT_CONFIDENCE_LEVEL = 0.95

_BOOTSTRAP_METRICS = {
    "hit_rate@5": "hit@5",
    "hit_rate@10": "hit@10",
    "recall@5": "recall@5",
    "recall@10": "recall@10",
    "mrr@10": "mrr@10",
    "ndcg@10": "ndcg@10",
}

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


def _rows_by_query_id(
    rows: Sequence[dict[str, Any]], *, method: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{method} row has no valid query_id")
        if query_id in indexed:
            raise ValueError(f"{method} has duplicate query_id {query_id!r}")
        indexed[query_id] = row
    return indexed


def paired_bootstrap_deltas(
    minmax_rows: Sequence[dict[str, Any]],
    rrf_rows: Sequence[dict[str, Any]],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Estimate percentile CIs for paired per-query ``RRF - Min-max`` deltas."""
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if not minmax_rows or not rrf_rows:
        raise ValueError("paired bootstrap requires non-empty method rows")

    minmax_by_id = _rows_by_query_id(minmax_rows, method="minmax")
    rrf_by_id = _rows_by_query_id(rrf_rows, method="rrf")
    if set(minmax_by_id) != set(rrf_by_id):
        missing_from_rrf = sorted(set(minmax_by_id) - set(rrf_by_id))
        missing_from_minmax = sorted(set(rrf_by_id) - set(minmax_by_id))
        raise ValueError(
            "paired bootstrap query IDs differ: "
            f"missing_from_rrf={missing_from_rrf}, "
            f"missing_from_minmax={missing_from_minmax}"
        )

    query_ids = list(minmax_by_id)
    rng = np.random.default_rng(int(seed))
    sampled_indices = rng.integers(
        0,
        len(query_ids),
        size=(int(samples), len(query_ids)),
    )
    tail_percent = (1.0 - confidence_level) * 50.0
    metrics: dict[str, Any] = {}
    for metric_name, row_field in _BOOTSTRAP_METRICS.items():
        try:
            deltas = np.asarray(
                [
                    float(rrf_by_id[query_id][row_field])
                    - float(minmax_by_id[query_id][row_field])
                    for query_id in query_ids
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"paired bootstrap rows have an invalid {row_field!r} value"
            ) from exc
        sampled_means = deltas[sampled_indices].mean(axis=1)
        lower, upper = np.percentile(
            sampled_means,
            [tail_percent, 100.0 - tail_percent],
        )
        direction = "inconclusive"
        if lower > 0.0:
            direction = "rrf"
        elif upper < 0.0:
            direction = "minmax"
        metrics[metric_name] = {
            "observed_delta": float(deltas.mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "direction": direction,
        }

    return {
        "method_order": "rrf_minus_minmax",
        "interval_method": "paired_query_percentile_bootstrap",
        "query_count": len(query_ids),
        "samples": int(samples),
        "seed": int(seed),
        "confidence_level": confidence_level,
        "metrics": metrics,
    }


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
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
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
    pairwise_rows: list[dict[str, Any]] = []
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
        pairwise_rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "query_type": query.query_type,
                "winner": winner,
                "minmax_first_relevant_rank": minmax_row["first_relevant_rank"],
                "rrf_first_relevant_rank": rrf_row["first_relevant_rank"],
            }
        )

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
            "bootstrap_unit": "paired query",
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
            "per_query": pairwise_rows,
        },
        "paired_bootstrap": paired_bootstrap_deltas(
            method_rows["minmax"],
            method_rows["rrf"],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        "candidate_audit": candidate_audit,
    }


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_minmax_dev_reference(
    report_path: str | Path = DEFAULT_BASELINE_REPORT,
    *,
    alpha: float = DEFAULT_MINMAX_ALPHA,
) -> dict[str, Any]:
    """Load only the Min-max dev reference needed for reproduction checking."""
    report_path = Path(report_path)
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    sweep = report.get("dev_alpha_sweep")
    if not isinstance(sweep, list):
        raise ValueError("baseline report has no valid dev_alpha_sweep")
    matches = [
        row
        for row in sweep
        if isinstance(row, dict)
        and isinstance(row.get("alpha"), (int, float))
        and math.isclose(float(row["alpha"]), alpha, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"baseline report must contain exactly one dev row for alpha={alpha}"
        )
    row = matches[0]
    metrics = row.get("metrics")
    by_query_type = row.get("by_query_type")
    per_query = row.get("per_query")
    if (
        not isinstance(metrics, dict)
        or not isinstance(by_query_type, dict)
        or not isinstance(per_query, list)
    ):
        raise ValueError("baseline dev row has invalid metrics")
    compact_per_query: list[dict[str, Any]] = []
    for item in per_query:
        if not isinstance(item, dict):
            raise ValueError("baseline dev row has invalid per_query entries")
        compact_per_query.append(
            {
                "query_id": item.get("query_id"),
                "first_relevant_rank": item.get("first_relevant_rank"),
                **{
                    row_field: item.get(row_field)
                    for row_field in _BOOTSTRAP_METRICS.values()
                },
            }
        )

    manifest = report.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("baseline report has no valid manifest")
    return {
        "source_report": _display_path(report_path),
        "source_git_commit": manifest.get("git_commit"),
        "raw_file_sha256": manifest.get("raw_file_sha256"),
        "eval_file_sha256": manifest.get("eval_file_sha256"),
        "alpha": float(row["alpha"]),
        "metrics": metrics,
        "by_query_type": by_query_type,
        "per_query": compact_per_query,
    }


def evaluate_dev_artifact_gate(
    manifest: dict[str, Any],
    baseline_reference: dict[str, Any],
) -> dict[str, Any]:
    """Check that a dev experiment uses the frozen Benchmark v1 artifacts."""
    chunk_count = manifest.get("chunk_count")
    checks = {
        "corpus_is_arxiv_cs": manifest.get("corpus") == "arxiv_cs",
        "paper_count_is_50000": manifest.get("paper_count") == OFFICIAL_PAPER_COUNT,
        "paper_count_equals_chunk_count": (
            manifest.get("paper_count") == chunk_count
        ),
        "fts_count_matches_chunks": manifest.get("fts_row_count") == chunk_count,
        "faiss_count_matches_chunks": (
            manifest.get("faiss_vector_count") == chunk_count
        ),
        "id_map_count_matches_chunks": manifest.get("id_map_count") == chunk_count,
        "embedding_model_is_bge_m3": manifest.get("embedding_model") == "BAAI/bge-m3",
        "embedding_dim_is_1024": manifest.get("embedding_dim") == 1024,
        "raw_hash_matches_baseline": (
            bool(baseline_reference.get("raw_file_sha256"))
            and manifest.get("raw_file_sha256")
            == baseline_reference.get("raw_file_sha256")
        ),
        "eval_hash_matches_baseline": (
            bool(baseline_reference.get("eval_file_sha256"))
            and manifest.get("eval_file_sha256")
            == baseline_reference.get("eval_file_sha256")
        ),
        "git_commit_recorded": bool(manifest.get("git_commit")),
        "git_worktree_clean": manifest.get("git_dirty") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
    }


def _compare_metric_group(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
    tolerance: float,
) -> tuple[dict[str, float], list[str]]:
    deltas: dict[str, float] = {}
    failures: list[str] = []
    if observed.get("query_count") != expected.get("query_count"):
        failures.append(
            f"{label}.query_count: observed={observed.get('query_count')}, "
            f"expected={expected.get('query_count')}"
        )
    for metric_name in _BOOTSTRAP_METRICS:
        try:
            delta = float(observed[metric_name]) - float(expected[metric_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cannot compare baseline metric {label}.{metric_name}"
            ) from exc
        deltas[metric_name] = delta
        if not math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=tolerance):
            failures.append(
                f"{label}.{metric_name}: delta={delta:.17g} exceeds {tolerance}"
            )
    return deltas, failures


def compare_minmax_dev_baseline(
    report: dict[str, Any],
    baseline_reference: dict[str, Any],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Require the extracted Min-max path to reproduce the frozen dev baseline."""
    if report.get("query_count") != OFFICIAL_DEV_COUNT:
        return {
            "status": "not_applicable",
            "matches": None,
            "reason": f"requires all {OFFICIAL_DEV_COUNT} dev queries",
            "reference": baseline_reference,
        }

    minmax = report["methods"]["minmax"]
    aggregate_deltas, failures = _compare_metric_group(
        minmax["metrics"],
        baseline_reference["metrics"],
        label="aggregate",
        tolerance=tolerance,
    )
    by_query_type_deltas: dict[str, dict[str, float]] = {}
    observed_types = minmax["by_query_type"]
    expected_types = baseline_reference["by_query_type"]
    if set(observed_types) != set(expected_types):
        failures.append(
            "query type groups differ: "
            f"observed={sorted(observed_types)}, expected={sorted(expected_types)}"
        )
    for query_type in sorted(set(observed_types) & set(expected_types)):
        deltas, group_failures = _compare_metric_group(
            observed_types[query_type],
            expected_types[query_type],
            label=f"query_type.{query_type}",
            tolerance=tolerance,
        )
        by_query_type_deltas[query_type] = deltas
        failures.extend(group_failures)

    observed_rows = _rows_by_query_id(minmax["per_query"], method="observed minmax")
    expected_rows = _rows_by_query_id(
        baseline_reference["per_query"],
        method="baseline minmax",
    )
    if set(observed_rows) != set(expected_rows):
        failures.append(
            "per-query IDs differ: "
            f"observed_only={sorted(set(observed_rows) - set(expected_rows))}, "
            f"baseline_only={sorted(set(expected_rows) - set(observed_rows))}"
        )
    rank_mismatches: list[str] = []
    metric_mismatches: list[str] = []
    for query_id in sorted(set(observed_rows) & set(expected_rows)):
        observed_row = observed_rows[query_id]
        expected_row = expected_rows[query_id]
        if observed_row.get("first_relevant_rank") != expected_row.get(
            "first_relevant_rank"
        ):
            rank_mismatches.append(query_id)
        for row_field in _BOOTSTRAP_METRICS.values():
            try:
                delta = float(observed_row[row_field]) - float(expected_row[row_field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"cannot compare per-query baseline metric {query_id}.{row_field}"
                ) from exc
            if not math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=tolerance):
                metric_mismatches.append(f"{query_id}.{row_field}")
    if rank_mismatches:
        failures.append(
            "first relevant rank differs for query IDs: " + ", ".join(rank_mismatches)
        )
    if metric_mismatches:
        failures.append(
            "per-query metrics differ: " + ", ".join(metric_mismatches)
        )

    return {
        "status": "matched" if not failures else "mismatch",
        "matches": not failures,
        "tolerance": tolerance,
        "reference": baseline_reference,
        "aggregate_deltas": aggregate_deltas,
        "by_query_type_deltas": by_query_type_deltas,
        "per_query_check": {
            "query_count": len(set(observed_rows) & set(expected_rows)),
            "first_relevant_rank_mismatches": rank_mismatches,
            "metric_mismatches": metric_mismatches,
        },
        "failures": failures,
    }


def _build_decision_gate(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "dev_comparison_v1_2":
        return {
            "applicable": False,
            "recommendation": "no_decision_from_smoke_or_invalid_run",
        }

    bootstrap = report["paired_bootstrap"]["metrics"]
    pairwise = report["pairwise_first_relevant_rank"]
    criteria = {
        "ndcg_ci_lower_above_zero": bootstrap["ndcg@10"]["ci_lower"] > 0.0,
        "mrr_ci_lower_nonnegative": bootstrap["mrr@10"]["ci_lower"] >= 0.0,
        "hit_rate_observed_nonnegative": (
            bootstrap["hit_rate@10"]["observed_delta"] >= 0.0
        ),
        "rrf_query_wins_not_fewer": pairwise["rrf_wins"]
        >= pairwise["minmax_wins"],
    }
    nominate_rrf = all(criteria.values())
    return {
        "applicable": True,
        "criteria": criteria,
        "passed": nominate_rrf,
        "recommendation": (
            "nominate_rrf_for_fresh_holdout"
            if nominate_rrf
            else "retain_minmax"
        ),
        "production_changed": False,
    }


def _dev_protocol_matches(report: dict[str, Any]) -> bool:
    protocol = report["protocol"]
    bootstrap = report["paired_bootstrap"]
    selection = report["query_selection"]
    return (
        report.get("query_count") == OFFICIAL_DEV_COUNT
        and selection.get("source_dev_count") == OFFICIAL_DEV_COUNT
        and selection.get("selected_query_count") == OFFICIAL_DEV_COUNT
        and selection.get("selection") == "all_frozen_dev_queries"
        and protocol.get("query_split") == "dev"
        and protocol.get("candidate_depth") == DEFAULT_CANDIDATE_DEPTH
        and protocol.get("final_top_k") == DEFAULT_FINAL_TOP_K
        and math.isclose(
            float(protocol.get("minmax_alpha")),
            DEFAULT_MINMAX_ALPHA,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and protocol.get("rrf_k") == DEFAULT_RRF_K
        and bootstrap.get("samples") == DEFAULT_BOOTSTRAP_SAMPLES
        and bootstrap.get("seed") == DEFAULT_BOOTSTRAP_SEED
        and math.isclose(
            float(bootstrap.get("confidence_level")),
            DEFAULT_CONFIDENCE_LEVEL,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def run_dev_fusion_experiment(
    queries: Sequence[EvalQuery],
    lexical_search_fn: SearchFn,
    dense_search_fn: SearchFn,
    *,
    manifest: dict[str, Any],
    baseline_reference: dict[str, Any],
    source_dev_count: int,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    alpha: float = DEFAULT_MINMAX_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Warm retrievers once, run the comparison, and attach reproducibility gates."""
    if not queries:
        raise ValueError("queries must not be empty")
    warmup = retrieve_fusion_candidates(
        queries[0].query,
        lexical_search_fn,
        dense_search_fn,
        candidate_depth=candidate_depth,
    )
    report = run_fusion_comparison(
        queries,
        lexical_search_fn,
        dense_search_fn,
        candidate_depth=candidate_depth,
        alpha=alpha,
        rrf_k=rrf_k,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    report["manifest"] = manifest
    report["query_selection"] = {
        "split": "dev",
        "source_dev_count": source_dev_count,
        "selected_query_count": len(queries),
        "selection": (
            "all_frozen_dev_queries"
            if len(queries) == source_dev_count
            else "first_n_dev_queries_in_frozen_file_order"
        ),
        "query_types": {
            query_type: sum(query.query_type == query_type for query in queries)
            for query_type in QUERY_TYPES
        },
        "relevance_labels_per_query": {
            "min": min(len(query.relevant_paper_ids) for query in queries),
            "max": max(len(query.relevant_paper_ids) for query in queries),
            "mean": float(
                np.mean([len(query.relevant_paper_ids) for query in queries])
            ),
        },
    }
    report["warmup"] = {
        "query_id": queries[0].query_id,
        "lexical_latency_ms": round(warmup.lexical_latency_ms, 3),
        "dense_latency_ms": round(warmup.dense_latency_ms, 3),
        "total_latency_ms": round(warmup.total_latency_ms, 3),
        "excluded_from_measured_latency": True,
    }
    report["artifact_gate"] = evaluate_dev_artifact_gate(
        manifest,
        baseline_reference,
    )
    report["baseline_reproduction"] = compare_minmax_dev_baseline(
        report,
        baseline_reference,
    )

    if len(queries) != OFFICIAL_DEV_COUNT:
        report["status"] = "smoke_or_development"
    elif (
        _dev_protocol_matches(report)
        and report["artifact_gate"]["passed"]
        and report["baseline_reproduction"]["matches"]
    ):
        report["status"] = "dev_comparison_v1_2"
    else:
        report["status"] = "invalid_dev_comparison"
    report["decision_gate"] = _build_decision_gate(report)
    report["limitations"] = [
        "The 50-query dev set contains synthetic known-item queries, not human relevance judgments.",
        "The same dev split that selected Min-max alpha 0.50 is used for strategy comparison.",
        "Bootstrap intervals quantify query-sampling uncertainty only; they do not remove benchmark bias.",
        "When each query has one relevance label, HitRate and Recall are numerically identical.",
        "Latency is specific to the recorded machine, index configuration, and query sample.",
        "No Benchmark v1 test query is retrieved or used for this decision.",
        "A fresh frozen holdout is required before claiming an optimized production improvement.",
    ]
    return report


def load_selected_dev_queries(
    eval_path: str | Path,
    db_path: str | Path,
    *,
    limit: int | None = None,
) -> tuple[list[EvalQuery], int]:
    """Load the frozen set, then return only dev queries in file order."""
    queries = load_eval_queries(eval_path, db_path=db_path)
    dev_queries = [query for query in queries if query.split == "dev"]
    if len(dev_queries) != OFFICIAL_DEV_COUNT:
        raise ValueError(
            f"expected {OFFICIAL_DEV_COUNT} frozen dev queries, found {len(dev_queries)}"
        )
    if limit is None:
        return dev_queries, len(dev_queries)
    if isinstance(limit, bool) or not isinstance(limit, Integral) or not 0 < limit <= len(
        dev_queries
    ):
        raise ValueError(f"limit must be between 1 and {len(dev_queries)}")
    return dev_queries[: int(limit)], len(dev_queries)


def build_default_retrievers(
    db_path: str | Path,
    index_dir: str | Path,
) -> tuple[SearchFn, SearchFn]:
    db_path = Path(db_path)
    index_dir = Path(index_dir)

    def lexical(query: str, top_k: int) -> Sequence[SearchResult]:
        return search_lexical(query, top_k=top_k, db_path=db_path)

    def dense(query: str, top_k: int) -> Sequence[SearchResult]:
        return search_vector(
            query,
            top_k=top_k,
            db_path=db_path,
            index_dir=index_dir,
        )

    return lexical, dense


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_fusion_markdown(report: dict[str, Any]) -> str:
    """Render an answer-first technical report from measured experiment data."""
    status = report["status"]
    methods = report["methods"]
    bootstrap = report["paired_bootstrap"]
    pairwise = report["pairwise_first_relevant_rank"]
    decision = report["decision_gate"]
    minmax = methods["minmax"]["metrics"]
    rrf = methods["rrf"]["metrics"]
    ndcg_delta = bootstrap["metrics"]["ndcg@10"]
    mrr_delta = bootstrap["metrics"]["mrr@10"]

    lines = ["# CiteQuest Retrieval Fusion Dev Comparison v1.2", ""]
    if status == "dev_comparison_v1_2":
        lines.extend(
            [
                "> Dev-only strategy comparison. No Benchmark v1 test query was "
                "retrieved or used, and production retrieval was not changed.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"> Status: `{status}`. This output is a smoke or invalid run and "
                "must not be used to select a production fusion strategy.",
                "",
            ]
        )

    lines.extend(["## Technical summary", ""])
    if decision.get("recommendation") == "nominate_rrf_for_fresh_holdout":
        lines.append(
            "Fixed RRF passed the predeclared dev evidence gate and may proceed to "
            "a fresh frozen holdout; production Min-max remains unchanged."
        )
    elif decision.get("recommendation") == "retain_minmax":
        lines.append(
            "Fixed RRF did not pass the predeclared dev evidence gate. Retain the "
            "current Min-max production strategy."
        )
    else:
        lines.append(
            "This run is not eligible for a strategy decision. It verifies the "
            "experiment path only."
        )
    lines.extend(
        [
            "",
            f"RRF minus Min-max nDCG@10 was {_fmt_delta(ndcg_delta['observed_delta'])} "
            f"with a {bootstrap['confidence_level']:.0%} paired bootstrap interval "
            f"[{_fmt_delta(ndcg_delta['ci_lower'])}, "
            f"{_fmt_delta(ndcg_delta['ci_upper'])}]. MRR@10 changed by "
            f"{_fmt_delta(mrr_delta['observed_delta'])}.",
            "",
            f"First-relevant-rank outcomes were Min-max wins "
            f"**{pairwise['minmax_wins']}**, RRF wins **{pairwise['rrf_wins']}**, "
            f"and ties **{pairwise['ties']}**.",
            "",
            "## Experiment setup",
            "",
            f"- Status: `{status}`",
            f"- Corpus: `{report['manifest'].get('corpus', 'unknown')}` with "
            f"{report['manifest'].get('paper_count', 'unknown')} papers and "
            f"{report['manifest'].get('chunk_count', 'unknown')} chunks",
            f"- Query population: {report['query_count']} frozen `dev` queries "
            f"selected as `{report['query_selection']['selection']}`",
            f"- Query types: "
            f"`{json.dumps(report['query_selection']['query_types'], sort_keys=True)}`",
            f"- Relevance labels per query: "
            f"min={report['query_selection']['relevance_labels_per_query']['min']}, "
            f"max={report['query_selection']['relevance_labels_per_query']['max']}, "
            f"mean={report['query_selection']['relevance_labels_per_query']['mean']:.2f}",
            f"- Candidate depth: {report['protocol']['candidate_depth']} per branch; "
            f"final top K: {report['protocol']['final_top_k']}",
            f"- Min-max: alpha `{report['protocol']['minmax_alpha']:.2f}`; "
            f"RRF: equal weights, k `{report['protocol']['rrf_k']}`",
            f"- Uncertainty: {bootstrap['samples']:,} paired-query percentile "
            f"bootstrap samples, seed `{bootstrap['seed']}`",
            f"- Git commit: `{report['manifest'].get('git_commit', 'unknown')}`; "
            f"dirty: `{report['manifest'].get('git_dirty', 'unknown')}`",
            "",
            "## Aggregate retrieval quality",
            "",
            "| Method | HitRate@5 | HitRate@10 | Recall@10 | MRR@10 | nDCG@10 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Min-max {report['protocol']['minmax_alpha']:.2f} | "
            f"{_fmt(minmax['hit_rate@5'])} | "
            f"{_fmt(minmax['hit_rate@10'])} | {_fmt(minmax['recall@10'])} | "
            f"{_fmt(minmax['mrr@10'])} | {_fmt(minmax['ndcg@10'])} |",
            f"| RRF k={report['protocol']['rrf_k']} | {_fmt(rrf['hit_rate@5'])} | "
            f"{_fmt(rrf['hit_rate@10'])} | {_fmt(rrf['recall@10'])} | "
            f"{_fmt(rrf['mrr@10'])} | {_fmt(rrf['ndcg@10'])} |",
            f"| RRF - Min-max | "
            f"{_fmt_delta(bootstrap['metrics']['hit_rate@5']['observed_delta'])} | "
            f"{_fmt_delta(bootstrap['metrics']['hit_rate@10']['observed_delta'])} | "
            f"{_fmt_delta(bootstrap['metrics']['recall@10']['observed_delta'])} | "
            f"{_fmt_delta(mrr_delta['observed_delta'])} | "
            f"{_fmt_delta(ndcg_delta['observed_delta'])} |",
            "",
            "## Paired uncertainty",
            "",
            "Each bootstrap draw resamples query IDs once and applies the same draw "
            "to both methods. Intervals therefore estimate the mean paired "
            "`RRF - Min-max` delta.",
            "",
            "| Metric | Observed delta | CI lower | CI upper | Direction |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric_name in _BOOTSTRAP_METRICS:
        row = bootstrap["metrics"][metric_name]
        lines.append(
            f"| {metric_name} | {_fmt_delta(row['observed_delta'])} | "
            f"{_fmt_delta(row['ci_lower'])} | {_fmt_delta(row['ci_upper'])} | "
            f"{row['direction']} |"
        )

    lines.extend(
        [
            "",
            "## Results by query type",
            "",
            "| Method | Query type | N | HitRate@10 | MRR@10 | nDCG@10 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for method_name, label in (("minmax", "Min-max"), ("rrf", "RRF")):
        for query_type in QUERY_TYPES:
            row = methods[method_name]["by_query_type"].get(query_type)
            if row is None:
                continue
            lines.append(
                f"| {label} | {query_type} | {row['query_count']} | "
                f"{_fmt(row['hit_rate@10'])} | {_fmt(row['mrr@10'])} | "
                f"{_fmt(row['ndcg@10'])} |"
            )

    lines.extend(
        [
            "",
            "## Query-level differences",
            "",
            f"Min-max wins: **{pairwise['minmax_wins']}**; RRF wins: "
            f"**{pairwise['rrf_wins']}**; ties: **{pairwise['ties']}**.",
            "",
        ]
    )
    differences = [row for row in pairwise["per_query"] if row["winner"] != "tie"]
    if differences:
        lines.extend(
            [
                "| Query | Type | Winner | Min-max rank | RRF rank |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in differences:
            lines.append(
                f"| `{row['query_id']}` {_markdown_escape(row['query'])} | "
                f"{row['query_type']} | {row['winner']} | "
                f"{row['minmax_first_relevant_rank'] or '-'} | "
                f"{row['rrf_first_relevant_rank'] or '-'} |"
            )
        lines.append("")
    else:
        lines.extend(["No first-relevant-rank differences in this run.", ""])

    baseline = report["baseline_reproduction"]
    artifact_gate = report["artifact_gate"]
    lines.extend(
        [
            "## Reproducibility checks",
            "",
            f"- Artifact gate: `{'passed' if artifact_gate['passed'] else 'failed'}`",
            f"- Min-max baseline reproduction: `{baseline['status']}`",
            f"- Mean BM25/Dense candidate Jaccard overlap: "
            f"{report['candidate_overlap']['mean_jaccard']:.4f}",
        ]
    )
    if artifact_gate["failures"]:
        lines.append(
            "- Artifact failures: "
            + ", ".join(f"`{failure}`" for failure in artifact_gate["failures"])
        )
    if baseline.get("failures"):
        lines.append(
            f"- Baseline mismatches: {_markdown_escape('; '.join(baseline['failures']))}"
        )

    shared = report["shared_retrieval_latency"]
    lines.extend(
        [
            "",
            "## Warm retrieval and fusion cost",
            "",
            "Retrieval is shared by both strategies. Fusion-only timing isolates the "
            "incremental ranking cost and excludes the untimed warm-up query.",
            "",
            "| Stage | Mean ms | p50 ms | p95 ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for branch, label in (("bm25", "BM25"), ("dense", "Dense"), ("total", "Shared total")):
        row = shared[branch]
        lines.append(
            f"| {label} | {row['mean_ms']:.3f} | {row['p50_ms']:.3f} | "
            f"{row['p95_ms']:.3f} |"
        )
    for method_name, label in (("minmax", "Min-max fusion"), ("rrf", "RRF fusion")):
        row = methods[method_name]["fusion_latency"]
        lines.append(
            f"| {label} | {row['mean_ms']:.3f} | {row['p50_ms']:.3f} | "
            f"{row['p95_ms']:.3f} |"
        )

    lines.extend(["", "## Decision gate", ""])
    if decision.get("applicable"):
        lines.append(f"Recommendation: **`{decision['recommendation']}`**.")
        lines.append("")
        for criterion, passed in decision["criteria"].items():
            lines.append(f"- `{criterion}`: `{'pass' if passed else 'fail'}`")
    else:
        lines.append("No production decision is allowed from this run status.")

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
    temporary.replace(path)


def write_fusion_outputs(
    report: dict[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON_REPORT,
    markdown_path: str | Path = DEFAULT_MD_REPORT,
) -> None:
    """Atomically write independent v1.2 outputs without touching Benchmark v1."""
    _atomic_write_text(
        Path(json_path),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(Path(markdown_path), render_fusion_markdown(report))


def _validate_full_run_parameters(args: argparse.Namespace) -> None:
    expected = {
        "candidate_depth": DEFAULT_CANDIDATE_DEPTH,
        "alpha": DEFAULT_MINMAX_ALPHA,
        "rrf_k": DEFAULT_RRF_K,
        "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
    }
    mismatches = [
        f"{name}={getattr(args, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    if mismatches:
        raise ValueError(
            "full dev comparison requires the frozen protocol: " + ", ".join(mismatches)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Min-max and fixed RRF on frozen dev queries only"
    )
    parser.add_argument("--eval", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--db", default=str(DEFAULT_BENCHMARK_DB))
    parser.add_argument("--index-dir", default=str(DEFAULT_BENCHMARK_INDEX_DIR))
    parser.add_argument("--raw", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--baseline-report", default=str(DEFAULT_BASELINE_REPORT))
    parser.add_argument(
        "--limit",
        type=int,
        help="Run the first N dev queries as smoke; omitted means all 50 dev queries",
    )
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=DEFAULT_CANDIDATE_DEPTH,
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_MINMAX_ALPHA)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument("--output-json", default=str(DEFAULT_JSON_REPORT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD_REPORT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.limit is None:
        _validate_full_run_parameters(args)

    db_path = Path(args.db)
    index_dir = Path(args.index_dir)
    queries, source_dev_count = load_selected_dev_queries(
        args.eval,
        db_path,
        limit=args.limit,
    )
    baseline_reference = load_minmax_dev_reference(
        args.baseline_report,
        alpha=args.alpha,
    )
    manifest = build_benchmark_manifest(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=args.raw,
        eval_path=args.eval,
        corpus="arxiv_cs",
    )
    artifact_gate = evaluate_dev_artifact_gate(manifest, baseline_reference)
    if args.limit is None and not artifact_gate["passed"]:
        raise RuntimeError(
            "full dev comparison artifact preflight failed: "
            + ", ".join(artifact_gate["failures"])
        )

    lexical, dense = build_default_retrievers(db_path, index_dir)
    report = run_dev_fusion_experiment(
        queries,
        lexical,
        dense,
        manifest=manifest,
        baseline_reference=baseline_reference,
        source_dev_count=source_dev_count,
        candidate_depth=args.candidate_depth,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_fusion_outputs(
        report,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )
    logger.info("Fusion comparison status: %s", report["status"])
    logger.info("JSON report: %s", args.output_json)
    logger.info("Markdown report: %s", args.output_md)
    if args.limit is None and report["status"] != "dev_comparison_v1_2":
        raise RuntimeError(
            "full dev comparison failed its reproduction or protocol gate; "
            "inspect the generated report"
        )


if __name__ == "__main__":
    main()
