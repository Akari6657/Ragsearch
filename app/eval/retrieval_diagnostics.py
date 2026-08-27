"""Reproducible candidate-coverage and retrieval-failure diagnostics.

This evaluation-only runner explains *where* the frozen dev pipeline loses a
known relevant paper. It records branch scores, normalized scores, complete
fusion ranks, and optional deep-probe ranks while leaving production retrieval
unchanged. The formal run must reproduce the candidate IDs and top-10 rankings
from the frozen v1.2 fusion report.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import validate_hybrid_alpha
from app.core.schemas import SearchResult
from app.eval.fusion_experiment import (
    DEFAULT_CANDIDATE_DEPTH,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_MINMAX_ALPHA,
    DEFAULT_RRF_K,
    build_default_retrievers,
    evaluate_dev_artifact_gate,
    fuse_rrf_results,
    load_selected_dev_queries,
    retrieve_fusion_candidates,
)
from app.eval.retrieval_eval import (
    DEFAULT_DB as DEFAULT_BENCHMARK_DB,
    DEFAULT_EVAL as DEFAULT_EVAL_PATH,
    DEFAULT_INDEX_DIR as DEFAULT_BENCHMARK_INDEX_DIR,
    OFFICIAL_DEV_COUNT,
    QUERY_TYPES,
    EvalQuery,
    build_benchmark_manifest,
    deduplicate_papers,
)
from app.retrieval.hybrid import fuse_minmax_results


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "arxiv_cs_benchmark_v1_50000.jsonl"
DEFAULT_REFERENCE_REPORT = PROJECT_ROOT / "reports" / "retrieval_fusion_dev_v1_2.json"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "retrieval_diagnostics_v1_3.json"
DEFAULT_MD_REPORT = PROJECT_ROOT / "reports" / "retrieval_diagnostics_v1_3.md"
DEFAULT_DEEP_PROBE_DEPTH = 1_000
FORMAL_STATUS = "dev_diagnostics_v1_3"

SearchFn = Callable[[str, int], Sequence[SearchResult]]


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _normalize_scores(results: Sequence[SearchResult]) -> list[float]:
    """Mirror production higher-is-better Min-max normalization."""
    if not results:
        return []
    scores = [float(result.score) for result in results]
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [0.5] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _first_relevant_rank(
    results: Sequence[SearchResult], relevant_paper_ids: set[str]
) -> int | None:
    papers = deduplicate_papers(results, limit=len(results))
    return next(
        (
            rank
            for rank, result in enumerate(papers, start=1)
            if result.paper_id in relevant_paper_ids
        ),
        None,
    )


def _top_paper_ids(results: Sequence[SearchResult], top_k: int) -> list[str]:
    return [
        result.paper_id
        for result in deduplicate_papers(results[:top_k], limit=top_k)
    ]


def _branch_rows(results: Sequence[SearchResult]) -> dict[str, dict[str, Any]]:
    normalized = _normalize_scores(results)
    rows: dict[str, dict[str, Any]] = {}
    for rank, (result, normalized_score) in enumerate(
        zip(results, normalized), start=1
    ):
        if result.chunk_id in rows:
            raise ValueError(f"retriever returned duplicate chunk_id {result.chunk_id!r}")
        rows[result.chunk_id] = {
            "rank": rank,
            "raw_score": float(result.score),
            "normalized_score": float(normalized_score),
            "result": result,
        }
    return rows


def _fused_rows(results: Sequence[SearchResult]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(results, start=1):
        rows[result.chunk_id] = {
            "rank": rank,
            "score": float(result.score),
            "result": result,
        }
    return rows


def _target_branch_detail(
    paper_id: str, branch_rows: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    matches = [
        row
        for row in branch_rows.values()
        if row["result"].paper_id == paper_id
    ]
    if not matches:
        return None
    best = min(matches, key=lambda row: row["rank"])
    return {
        "rank": best["rank"],
        "raw_score": best["raw_score"],
        "normalized_score": best["normalized_score"],
        "chunk_id": best["result"].chunk_id,
    }


def _target_fused_detail(
    paper_id: str, fused_rows: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    matches = [
        row
        for row in fused_rows.values()
        if row["result"].paper_id == paper_id
    ]
    if not matches:
        return None
    best = min(matches, key=lambda row: row["rank"])
    return {
        "rank": best["rank"],
        "score": best["score"],
        "chunk_id": best["result"].chunk_id,
    }


def diagnose_query(
    query: EvalQuery,
    lexical_results: Sequence[SearchResult],
    dense_results: Sequence[SearchResult],
    *,
    top_k: int = DEFAULT_FINAL_TOP_K,
    alpha: float = DEFAULT_MINMAX_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, Any]:
    """Build a score-preserving diagnostic row from one shared candidate set."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if query.split != "dev":
        raise ValueError("retrieval diagnostics accepts dev queries only")
    alpha = validate_hybrid_alpha(alpha)

    lexical = tuple(lexical_results)
    dense = tuple(dense_results)
    lexical_rows = _branch_rows(lexical)
    dense_rows = _branch_rows(dense)
    ordered_chunk_ids = list(lexical_rows)
    ordered_chunk_ids.extend(
        chunk_id for chunk_id in dense_rows if chunk_id not in lexical_rows
    )
    union_count = len(ordered_chunk_ids)
    fusion_limit = max(union_count, 1)

    minmax_results = fuse_minmax_results(
        lexical,
        dense,
        top_k=fusion_limit,
        alpha=alpha,
    )
    rrf_results = fuse_rrf_results(
        lexical,
        dense,
        top_k=fusion_limit,
        rrf_k=rrf_k,
    )
    minmax_rows = _fused_rows(minmax_results)
    rrf_rows = _fused_rows(rrf_results)
    relevant = set(query.relevant_paper_ids)

    lexical_rank = _first_relevant_rank(lexical, relevant)
    dense_rank = _first_relevant_rank(dense, relevant)
    minmax_rank = _first_relevant_rank(minmax_results, relevant)
    rrf_rank = _first_relevant_rank(rrf_results, relevant)
    in_candidate_union = lexical_rank is not None or dense_rank is not None
    minmax_hit = minmax_rank is not None and minmax_rank <= top_k
    if minmax_hit:
        outcome = "retrieved_top_k"
    elif in_candidate_union:
        outcome = "fusion_loss"
    else:
        outcome = "candidate_miss"

    if lexical_rank is not None and dense_rank is not None:
        relevant_source = "both"
    elif lexical_rank is not None:
        relevant_source = "bm25_only"
    elif dense_rank is not None:
        relevant_source = "dense_only"
    else:
        relevant_source = "neither"

    target_details = []
    for paper_id in query.relevant_paper_ids:
        target_details.append(
            {
                "paper_id": paper_id,
                "bm25": _target_branch_detail(paper_id, lexical_rows),
                "dense": _target_branch_detail(paper_id, dense_rows),
                "minmax": _target_fused_detail(paper_id, minmax_rows),
                "rrf": _target_fused_detail(paper_id, rrf_rows),
            }
        )

    candidate_records = []
    for chunk_id in ordered_chunk_ids:
        lexical_row = lexical_rows.get(chunk_id)
        dense_row = dense_rows.get(chunk_id)
        source_result = (
            lexical_row["result"] if lexical_row is not None else dense_row["result"]
        )
        minmax_row = minmax_rows[chunk_id]
        rrf_row = rrf_rows[chunk_id]
        candidate_records.append(
            {
                "chunk_id": chunk_id,
                "paper_id": source_result.paper_id,
                "title": source_result.title,
                "is_relevant": source_result.paper_id in relevant,
                "source": (
                    "both"
                    if lexical_row is not None and dense_row is not None
                    else "bm25"
                    if lexical_row is not None
                    else "dense"
                ),
                "bm25": (
                    {
                        "rank": lexical_row["rank"],
                        "raw_score": lexical_row["raw_score"],
                        "normalized_score": lexical_row["normalized_score"],
                    }
                    if lexical_row is not None
                    else None
                ),
                "dense": (
                    {
                        "rank": dense_row["rank"],
                        "raw_score": dense_row["raw_score"],
                        "normalized_score": dense_row["normalized_score"],
                    }
                    if dense_row is not None
                    else None
                ),
                "minmax": {
                    "rank": minmax_row["rank"],
                    "score": minmax_row["score"],
                },
                "rrf": {
                    "rank": rrf_row["rank"],
                    "score": rrf_row["score"],
                },
            }
        )

    lexical_ids = [result.chunk_id for result in lexical]
    dense_ids = [result.chunk_id for result in dense]
    overlap_count = len(set(lexical_ids) & set(dense_ids))
    return {
        "query_id": query.query_id,
        "query": query.query,
        "query_type": query.query_type,
        "source_category": query.source_category,
        "relevant_paper_ids": list(query.relevant_paper_ids),
        "outcome": outcome,
        "relevant_candidate_source": relevant_source,
        "candidate_counts": {
            "bm25": len(lexical),
            "dense": len(dense),
            "overlap": overlap_count,
            "union": union_count,
        },
        "bm25_first_relevant_rank": lexical_rank,
        "dense_first_relevant_rank": dense_rank,
        "minmax_first_relevant_rank_all_candidates": minmax_rank,
        "rrf_first_relevant_rank_all_candidates": rrf_rank,
        "minmax_first_relevant_rank_at_k": (
            minmax_rank if minmax_rank is not None and minmax_rank <= top_k else None
        ),
        "rrf_first_relevant_rank_at_k": (
            rrf_rank if rrf_rank is not None and rrf_rank <= top_k else None
        ),
        "minmax_top_k_paper_ids": _top_paper_ids(minmax_results, top_k),
        "rrf_top_k_paper_ids": _top_paper_ids(rrf_results, top_k),
        "minmax_top_k_cutoff_score": (
            float(minmax_results[top_k - 1].score)
            if len(minmax_results) >= top_k
            else None
        ),
        "rrf_top_k_cutoff_score": (
            float(rrf_results[top_k - 1].score)
            if len(rrf_results) >= top_k
            else None
        ),
        "targets": target_details,
        "candidate_records": candidate_records,
        "candidate_ids": {
            "bm25_chunk_ids": lexical_ids,
            "dense_chunk_ids": dense_ids,
        },
        "deep_probe": None,
    }


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _coverage_at_depth(rows: Sequence[dict[str, Any]], depth: int) -> dict[str, Any]:
    def present(rank: int | None) -> bool:
        return rank is not None and rank <= depth

    bm25_count = sum(present(row["bm25_first_relevant_rank"]) for row in rows)
    dense_count = sum(present(row["dense_first_relevant_rank"]) for row in rows)
    either_count = sum(
        present(row["bm25_first_relevant_rank"])
        or present(row["dense_first_relevant_rank"])
        for row in rows
    )
    both_count = sum(
        present(row["bm25_first_relevant_rank"])
        and present(row["dense_first_relevant_rank"])
        for row in rows
    )
    total = len(rows)
    return {
        "depth": depth,
        "query_count": total,
        "bm25_count": bm25_count,
        "bm25_rate": _rate(bm25_count, total),
        "dense_count": dense_count,
        "dense_rate": _rate(dense_count, total),
        "either_count": either_count,
        "either_rate": _rate(either_count, total),
        "both_count": both_count,
        "both_rate": _rate(both_count, total),
    }


def _summarize_rows(
    rows: Sequence[dict[str, Any]], *, candidate_depth: int, top_k: int
) -> dict[str, Any]:
    depths = sorted({5, 10, candidate_depth})
    outcomes = Counter(row["outcome"] for row in rows)
    sources = Counter(row["relevant_candidate_source"] for row in rows)
    ranks = Counter(
        row["minmax_first_relevant_rank_at_k"]
        if row["minmax_first_relevant_rank_at_k"] is not None
        else "missing"
        for row in rows
    )
    total = len(rows)
    final_hits = outcomes["retrieved_top_k"]
    candidate_hits = final_hits + outcomes["fusion_loss"]
    return {
        "query_count": total,
        "candidate_coverage": [
            _coverage_at_depth(rows, depth) for depth in depths
        ],
        "outcomes": {
            name: {"count": outcomes[name], "rate": _rate(outcomes[name], total)}
            for name in ("retrieved_top_k", "fusion_loss", "candidate_miss")
        },
        "relevant_candidate_source": {
            name: {"count": sources[name], "rate": _rate(sources[name], total)}
            for name in ("both", "bm25_only", "dense_only", "neither")
        },
        "minmax_first_relevant_rank_distribution": {
            **{
                str(rank): ranks[rank]
                for rank in range(1, top_k + 1)
                if ranks[rank]
            },
            "missing": ranks["missing"],
        },
        "known_target_oracle": {
            "candidate_union_hit_rate": _rate(candidate_hits, total),
            "current_minmax_hit_rate": _rate(final_hits, total),
            "recoverable_fusion_loss_count": outcomes["fusion_loss"],
            "absolute_hit_rate_headroom_at_fixed_candidates": _rate(
                outcomes["fusion_loss"], total
            ),
        },
    }


def compare_with_fusion_reference(
    rows: Sequence[dict[str, Any]], reference: dict[str, Any]
) -> dict[str, Any]:
    """Prove that diagnostics reran the same v1.2 candidates and rankings."""
    audit = reference.get("candidate_audit")
    methods = reference.get("methods")
    if not isinstance(audit, list) or not isinstance(methods, dict):
        raise ValueError("fusion reference has no valid candidate or method audit")

    reference_candidates = {
        row.get("query_id"): row for row in audit if isinstance(row, dict)
    }
    reference_methods: dict[str, dict[str, dict[str, Any]]] = {}
    for method in ("minmax", "rrf"):
        per_query = methods.get(method, {}).get("per_query")
        if not isinstance(per_query, list):
            raise ValueError(f"fusion reference has no valid {method} per_query rows")
        reference_methods[method] = {
            row.get("query_id"): row for row in per_query if isinstance(row, dict)
        }

    failures: list[str] = []
    checked_query_ids: list[str] = []
    for row in rows:
        query_id = row["query_id"]
        checked_query_ids.append(query_id)
        expected_candidates = reference_candidates.get(query_id)
        if expected_candidates is None:
            failures.append(f"{query_id}: missing candidate reference")
            continue
        for branch_key in ("bm25_chunk_ids", "dense_chunk_ids"):
            observed = row["candidate_ids"][branch_key]
            expected = expected_candidates.get(branch_key)
            if observed != expected:
                failures.append(f"{query_id}: {branch_key} differs")
        for method in ("minmax", "rrf"):
            expected_method = reference_methods[method].get(query_id)
            if expected_method is None:
                failures.append(f"{query_id}: missing {method} reference")
                continue
            if row[f"{method}_top_k_paper_ids"] != expected_method.get(
                "retrieved_paper_ids"
            ):
                failures.append(f"{query_id}: {method} top-k paper IDs differ")
            if row[f"{method}_first_relevant_rank_at_k"] != expected_method.get(
                "first_relevant_rank"
            ):
                failures.append(f"{query_id}: {method} relevant rank differs")

    return {
        "status": "matched" if not failures else "mismatch",
        "matches": not failures,
        "query_count": len(rows),
        "checked_query_ids": checked_query_ids,
        "failures": failures,
    }


def load_fusion_reference(
    path: str | Path = DEFAULT_REFERENCE_REPORT,
) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "dev_comparison_v1_2":
        raise ValueError("fusion reference must be a formal v1.2 dev comparison")
    if report.get("query_count") != OFFICIAL_DEV_COUNT:
        raise ValueError(f"fusion reference must contain {OFFICIAL_DEV_COUNT} queries")
    if report.get("artifact_gate", {}).get("passed") is not True:
        raise ValueError("fusion reference artifact gate did not pass")
    if report.get("baseline_reproduction", {}).get("matches") is not True:
        raise ValueError("fusion reference did not reproduce Benchmark v1")
    protocol = report.get("protocol", {})
    if not (
        protocol.get("query_split") == "dev"
        and protocol.get("candidate_depth") == DEFAULT_CANDIDATE_DEPTH
        and protocol.get("final_top_k") == DEFAULT_FINAL_TOP_K
        and math.isclose(
            float(protocol.get("minmax_alpha")),
            DEFAULT_MINMAX_ALPHA,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and protocol.get("rrf_k") == DEFAULT_RRF_K
    ):
        raise ValueError("fusion reference does not use the frozen v1.2 protocol")
    if len(report.get("candidate_audit", [])) != OFFICIAL_DEV_COUNT:
        raise ValueError("fusion reference has an incomplete candidate audit")
    for method in ("minmax", "rrf"):
        per_query = report.get("methods", {}).get(method, {}).get("per_query", [])
        if len(per_query) != OFFICIAL_DEV_COUNT:
            raise ValueError(f"fusion reference has incomplete {method} query rows")
    return report


def evaluate_diagnostics_artifact_gate(
    manifest: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    reference_manifest = reference.get("manifest")
    if not isinstance(reference_manifest, dict):
        raise ValueError("fusion reference has no valid manifest")
    gate = evaluate_dev_artifact_gate(
        manifest,
        {
            "raw_file_sha256": reference_manifest.get("raw_file_sha256"),
            "eval_file_sha256": reference_manifest.get("eval_file_sha256"),
        },
    )
    checks = dict(gate["checks"])
    checks.update(
        {
            "reference_status_is_formal_v1_2": (
                reference.get("status") == "dev_comparison_v1_2"
            ),
            "reference_artifact_gate_passed": (
                reference.get("artifact_gate", {}).get("passed") is True
            ),
            "reference_baseline_reproduction_matched": (
                reference.get("baseline_reproduction", {}).get("matches") is True
            ),
        }
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {"passed": not failures, "checks": checks, "failures": failures}


def _attach_deep_probes(
    rows: Sequence[dict[str, Any]],
    queries_by_id: dict[str, EvalQuery],
    lexical_search: SearchFn,
    dense_search: SearchFn,
    *,
    deep_probe_depth: int,
) -> None:
    for row in rows:
        if row["outcome"] == "retrieved_top_k":
            continue
        query = queries_by_id[row["query_id"]]
        lexical = tuple(lexical_search(query.query, deep_probe_depth))
        dense = tuple(dense_search(query.query, deep_probe_depth))
        relevant = set(query.relevant_paper_ids)
        row["deep_probe"] = {
            "depth": deep_probe_depth,
            "bm25_first_relevant_rank": _first_relevant_rank(lexical, relevant),
            "dense_first_relevant_rank": _first_relevant_rank(dense, relevant),
            "diagnostic_only": True,
        }


def run_retrieval_diagnostics(
    queries: Sequence[EvalQuery],
    lexical_search: SearchFn,
    dense_search: SearchFn,
    *,
    reference: dict[str, Any],
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    top_k: int = DEFAULT_FINAL_TOP_K,
    alpha: float = DEFAULT_MINMAX_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
    deep_probe_depth: int = DEFAULT_DEEP_PROBE_DEPTH,
) -> dict[str, Any]:
    """Retrieve once per query, classify losses, and aggregate diagnostics."""
    if not queries:
        raise ValueError("queries must not be empty")
    unexpected_splits = sorted({query.split for query in queries if query.split != "dev"})
    if unexpected_splits:
        raise ValueError(
            "retrieval diagnostics expected only 'dev' queries; found "
            + ", ".join(unexpected_splits)
        )
    if candidate_depth < top_k:
        raise ValueError("candidate_depth must be greater than or equal to top_k")
    if deep_probe_depth < candidate_depth:
        raise ValueError("deep_probe_depth must be at least candidate_depth")
    alpha = validate_hybrid_alpha(alpha)

    rows: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {"bm25": [], "dense": [], "total": []}
    for query in queries:
        candidates = retrieve_fusion_candidates(
            query.query,
            lexical_search,
            dense_search,
            candidate_depth=candidate_depth,
        )
        latencies["bm25"].append(candidates.lexical_latency_ms)
        latencies["dense"].append(candidates.dense_latency_ms)
        latencies["total"].append(candidates.total_latency_ms)
        rows.append(
            diagnose_query(
                query,
                candidates.lexical,
                candidates.dense,
                top_k=top_k,
                alpha=alpha,
                rrf_k=rrf_k,
            )
        )

    _attach_deep_probes(
        rows,
        {query.query_id: query for query in queries},
        lexical_search,
        dense_search,
        deep_probe_depth=deep_probe_depth,
    )
    overall = _summarize_rows(rows, candidate_depth=candidate_depth, top_k=top_k)
    by_query_type = {
        query_type: _summarize_rows(
            [row for row in rows if row["query_type"] == query_type],
            candidate_depth=candidate_depth,
            top_k=top_k,
        )
        for query_type in QUERY_TYPES
        if any(row["query_type"] == query_type for row in rows)
    }
    failures = [row for row in rows if row["outcome"] != "retrieved_top_k"]
    semantic_failures = sum(
        row["query_type"] == "semantic_paraphrase" for row in failures
    )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_or_development",
        "query_count": len(queries),
        "protocol": {
            "query_split": "dev",
            "candidate_depth": candidate_depth,
            "final_top_k": top_k,
            "minmax_alpha": alpha,
            "rrf_k": rrf_k,
            "deep_probe_depth": deep_probe_depth,
            "ranking_unit": "paper after chunk-level fusion",
            "deep_probe_usage": "failed Min-max top-k queries only; excluded from metrics",
        },
        "retrieval_latency": {
            branch: {
                "mean_ms": round(float(np.mean(values)), 3),
                "p50_ms": round(float(np.percentile(values, 50)), 3),
                "p95_ms": round(float(np.percentile(values, 95)), 3),
            }
            for branch, values in latencies.items()
        },
        "summary": overall,
        "by_query_type": by_query_type,
        "diagnosis": {
            "candidate_miss_count": overall["outcomes"]["candidate_miss"]["count"],
            "fusion_loss_count": overall["outcomes"]["fusion_loss"]["count"],
            "semantic_paraphrase_failure_count": semantic_failures,
            "failure_count": len(failures),
            "failure_query_ids": [row["query_id"] for row in failures],
        },
        "reference_reproduction": compare_with_fusion_reference(rows, reference),
        "per_query": rows,
    }


def _formal_protocol_matches(report: dict[str, Any]) -> bool:
    protocol = report.get("protocol", {})
    return (
        protocol.get("query_split") == "dev"
        and protocol.get("candidate_depth") == DEFAULT_CANDIDATE_DEPTH
        and protocol.get("final_top_k") == DEFAULT_FINAL_TOP_K
        and math.isclose(
            float(protocol.get("minmax_alpha")),
            DEFAULT_MINMAX_ALPHA,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and protocol.get("rrf_k") == DEFAULT_RRF_K
        and protocol.get("deep_probe_depth") == DEFAULT_DEEP_PROBE_DEPTH
    )


def run_dev_retrieval_diagnostics(
    queries: Sequence[EvalQuery],
    lexical_search: SearchFn,
    dense_search: SearchFn,
    *,
    manifest: dict[str, Any],
    reference: dict[str, Any],
    artifact_gate: dict[str, Any],
    source_dev_count: int,
    formal_run: bool,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    top_k: int = DEFAULT_FINAL_TOP_K,
    alpha: float = DEFAULT_MINMAX_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
    deep_probe_depth: int = DEFAULT_DEEP_PROBE_DEPTH,
) -> dict[str, Any]:
    """Warm retrievers, run diagnostics, and assign an evidence-backed status."""
    if not queries:
        raise ValueError("queries must not be empty")
    warmup = retrieve_fusion_candidates(
        queries[0].query,
        lexical_search,
        dense_search,
        candidate_depth=candidate_depth,
    )
    report = run_retrieval_diagnostics(
        queries,
        lexical_search,
        dense_search,
        reference=reference,
        candidate_depth=candidate_depth,
        top_k=top_k,
        alpha=alpha,
        rrf_k=rrf_k,
        deep_probe_depth=deep_probe_depth,
    )
    report["manifest"] = manifest
    report["reference_report"] = {
        "path": _display_path(DEFAULT_REFERENCE_REPORT),
        "git_commit": reference.get("manifest", {}).get("git_commit"),
        "status": reference.get("status"),
    }
    report["query_selection"] = {
        "split": "dev",
        "source_dev_count": source_dev_count,
        "selected_query_count": len(queries),
        "selection": (
            "all_frozen_dev_queries"
            if formal_run
            else "first_n_dev_queries_in_frozen_file_order"
        ),
        "query_types": {
            query_type: sum(query.query_type == query_type for query in queries)
            for query_type in QUERY_TYPES
        },
    }
    report["warmup"] = {
        "query_id": queries[0].query_id,
        "lexical_latency_ms": round(warmup.lexical_latency_ms, 3),
        "dense_latency_ms": round(warmup.dense_latency_ms, 3),
        "total_latency_ms": round(warmup.total_latency_ms, 3),
        "excluded_from_measured_latency": True,
    }
    report["artifact_gate"] = artifact_gate

    if not formal_run:
        report["status"] = "smoke_or_development"
    elif (
        len(queries) == OFFICIAL_DEV_COUNT
        and source_dev_count == OFFICIAL_DEV_COUNT
        and _formal_protocol_matches(report)
        and artifact_gate.get("passed") is True
        and report["reference_reproduction"]["matches"] is True
    ):
        report["status"] = FORMAL_STATUS
    else:
        report["status"] = "invalid_dev_diagnostics"

    report["data_adequacy"] = {
        "corpus_is_sufficient_for_current_diagnosis": True,
        "additional_corpus_required_now": False,
        "dev_queries_are_sufficient_for_failure_localization": True,
        "dev_queries_are_sufficient_for_small_delta_claims": False,
        "reason": (
            "The frozen dev set separates candidate misses from fusion losses, "
            "but 50 synthetic known-item queries with one relevance label have "
            "limited statistical power and cannot judge other relevant papers."
        ),
        "future_addition": (
            "Add a fresh holdout and a small pool-judged set with graded, "
            "multi-paper relevance before claiming a promoted strategy."
        ),
    }
    report["limitations"] = [
        (
            "Known-item labels treat only the source paper as relevant, even "
            "when other retrieved papers may satisfy the query."
        ),
        (
            "The 50-query dev split is suitable for diagnosis but underpowered "
            "for small ranking deltas."
        ),
        (
            "Deep probes measure target rank at a larger retrieval depth; they "
            "do not represent a deployable candidate setting."
        ),
        "The diagnostics reuse dev evidence and must not be presented as a fresh holdout result.",
    ]
    return report


def _fmt_rate(value: float) -> str:
    return f"{value:.4f}"


def _fmt_rank(value: int | None) -> str:
    return str(value) if value is not None else "not found"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_diagnostics_markdown(report: dict[str, Any]) -> str:
    """Render an answer-first local diagnostic report."""
    summary = report["summary"]
    diagnosis = report["diagnosis"]
    oracle = summary["known_target_oracle"]
    lines = [
        "# CiteQuest Retrieval Diagnostics v1.3",
        "",
        f"Status: **`{report['status']}`**  ",
        f"Queries: **{report['query_count']}** frozen dev records",
        "",
        "## Conclusion",
        "",
        (
            f"Min-max Top-{report['protocol']['final_top_k']} misses "
            f"{diagnosis['failure_count']} known targets: "
            f"{diagnosis['candidate_miss_count']} are absent from both branch "
            f"candidate sets and {diagnosis['fusion_loss_count']} are retrieved "
            "but lost during fusion."
        ),
        "",
        (
            f"At the fixed candidate depth, a perfect known-target reranker is "
            f"bounded by {_fmt_rate(oracle['candidate_union_hit_rate'])} HitRate, "
            f"versus the current {_fmt_rate(oracle['current_minmax_hit_rate'])}."
        ),
        "",
        "No corpus expansion or index rebuild is required for the next retrieval experiment.",
        "",
        "## Candidate Coverage",
        "",
        "| Depth | BM25 | Dense | Either | Both |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["candidate_coverage"]:
        lines.append(
            f"| {row['depth']} | {_fmt_rate(row['bm25_rate'])} "
            f"| {_fmt_rate(row['dense_rate'])} | {_fmt_rate(row['either_rate'])} "
            f"| {_fmt_rate(row['both_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Query Types",
            "",
            (
                "| Query type | Queries | Candidate union coverage | Min-max "
                "HitRate | Fusion losses | Candidate misses |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    candidate_depth = report["protocol"]["candidate_depth"]
    for query_type, group in report["by_query_type"].items():
        coverage = next(
            row
            for row in group["candidate_coverage"]
            if row["depth"] == candidate_depth
        )
        lines.append(
            f"| {_escape(query_type)} | {group['query_count']} "
            f"| {_fmt_rate(coverage['either_rate'])} "
            f"| {_fmt_rate(group['outcomes']['retrieved_top_k']['rate'])} "
            f"| {group['outcomes']['fusion_loss']['count']} "
            f"| {group['outcomes']['candidate_miss']['count']} |"
        )

    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
            (
                "| ID | Type | Outcome | BM25 rank | Dense rank | Min-max full "
                "rank | Deep BM25 | Deep Dense | Query |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["per_query"]:
        if row["outcome"] == "retrieved_top_k":
            continue
        probe = row.get("deep_probe") or {}
        lines.append(
            f"| {row['query_id']} | {_escape(row['query_type'])} "
            f"| `{row['outcome']}` | {_fmt_rank(row['bm25_first_relevant_rank'])} "
            f"| {_fmt_rank(row['dense_first_relevant_rank'])} "
            f"| {_fmt_rank(row['minmax_first_relevant_rank_all_candidates'])} "
            f"| {_fmt_rank(probe.get('bm25_first_relevant_rank'))} "
            f"| {_fmt_rank(probe.get('dense_first_relevant_rank'))} "
            f"| {_escape(row['query'])} |"
        )

    fusion_losses = [
        row for row in report["per_query"] if row["outcome"] == "fusion_loss"
    ]
    if fusion_losses:
        lines.extend(
            [
                "",
                "## Fusion-Loss Scores",
                "",
                (
                    "| ID | Source | BM25 norm | Dense norm | Min-max score | "
                    "Full rank | Top-k cutoff |"
                ),
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in fusion_losses:
            target = next(
                target
                for target in row["targets"]
                if target["minmax"] is not None
            )
            bm25_norm = (
                target["bm25"]["normalized_score"]
                if target["bm25"] is not None
                else 0.0
            )
            dense_norm = (
                target["dense"]["normalized_score"]
                if target["dense"] is not None
                else 0.0
            )
            lines.append(
                f"| {row['query_id']} | `{row['relevant_candidate_source']}` "
                f"| {bm25_norm:.4f} | {dense_norm:.4f} "
                f"| {target['minmax']['score']:.4f} "
                f"| {target['minmax']['rank']} "
                f"| {row['minmax_top_k_cutoff_score']:.4f} |"
            )

    rank_distribution = summary["minmax_first_relevant_rank_distribution"]
    lines.extend(
        [
            "",
            "## Ranking Saturation",
            "",
            f"Known target at rank 1: **{rank_distribution.get('1', 0)}** queries.  ",
            f"Known target missing from Top-{report['protocol']['final_top_k']}: "
            f"**{rank_distribution['missing']}** queries.",
            "",
            "This ceiling-heavy distribution limits sensitivity to small ranking changes.",
            "",
            "## Evidence Gates",
            "",
            f"- Artifact gate: `{'pass' if report['artifact_gate']['passed'] else 'fail'}`",
            (
                "- v1.2 candidate/ranking reproduction: "
                f"`{'pass' if report['reference_reproduction']['matches'] else 'fail'}`"
            ),
            f"- Current Git commit: `{report['manifest'].get('git_commit')}`",
            f"- Git worktree clean: `{not report['manifest'].get('git_dirty')}`",
            "",
            "## Data Adequacy",
            "",
            report["data_adequacy"]["reason"],
            "",
            f"Future addition: {report['data_adequacy']['future_addition']}",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def write_diagnostics_outputs(
    report: dict[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON_REPORT,
    markdown_path: str | Path = DEFAULT_MD_REPORT,
) -> None:
    _atomic_write_text(
        Path(json_path),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(Path(markdown_path), render_diagnostics_markdown(report))


def _validate_formal_parameters(args: argparse.Namespace) -> None:
    expected = {
        "candidate_depth": DEFAULT_CANDIDATE_DEPTH,
        "top_k": DEFAULT_FINAL_TOP_K,
        "alpha": DEFAULT_MINMAX_ALPHA,
        "rrf_k": DEFAULT_RRF_K,
        "deep_probe_depth": DEFAULT_DEEP_PROBE_DEPTH,
    }
    mismatches = [
        f"{name}={getattr(args, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    if mismatches:
        raise ValueError(
            "formal dev diagnostics requires the frozen protocol: "
            + ", ".join(mismatches)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose candidate coverage and fusion losses on frozen dev queries"
    )
    parser.add_argument("--eval", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--db", default=str(DEFAULT_BENCHMARK_DB))
    parser.add_argument("--index-dir", default=str(DEFAULT_BENCHMARK_INDEX_DIR))
    parser.add_argument("--raw", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--reference-report", default=str(DEFAULT_REFERENCE_REPORT))
    parser.add_argument(
        "--limit",
        type=int,
        help="Run the first N dev queries as smoke; omitted means formal 50-query run",
    )
    parser.add_argument("--candidate-depth", type=int, default=DEFAULT_CANDIDATE_DEPTH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_MINMAX_ALPHA)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument(
        "--deep-probe-depth", type=int, default=DEFAULT_DEEP_PROBE_DEPTH
    )
    parser.add_argument("--output-json", default=str(DEFAULT_JSON_REPORT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD_REPORT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    formal_run = args.limit is None
    if formal_run:
        _validate_formal_parameters(args)

    reference = load_fusion_reference(args.reference_report)
    db_path = Path(args.db)
    index_dir = Path(args.index_dir)
    queries, source_dev_count = load_selected_dev_queries(
        args.eval,
        db_path,
        limit=args.limit,
    )
    manifest = build_benchmark_manifest(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=args.raw,
        eval_path=args.eval,
        corpus="arxiv_cs",
    )
    artifact_gate = evaluate_diagnostics_artifact_gate(manifest, reference)
    if formal_run and not artifact_gate["passed"]:
        raise RuntimeError(
            "formal diagnostics artifact preflight failed: "
            + ", ".join(artifact_gate["failures"])
        )

    lexical, dense = build_default_retrievers(db_path, index_dir)
    report = run_dev_retrieval_diagnostics(
        queries,
        lexical,
        dense,
        manifest=manifest,
        reference=reference,
        artifact_gate=artifact_gate,
        source_dev_count=source_dev_count,
        formal_run=formal_run,
        candidate_depth=args.candidate_depth,
        top_k=args.top_k,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        deep_probe_depth=args.deep_probe_depth,
    )
    report["reference_report"]["path"] = _display_path(args.reference_report)
    write_diagnostics_outputs(
        report,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )
    logger.info("Retrieval diagnostics status: %s", report["status"])
    logger.info("JSON report: %s", args.output_json)
    logger.info("Markdown report: %s", args.output_md)
    if formal_run and report["status"] != FORMAL_STATUS:
        raise RuntimeError("formal diagnostics failed its protocol or reproduction gate")


if __name__ == "__main__":
    main()
