"""Reproducible paper-level retrieval benchmark for CiteQuest.

The production retrievers rank chunks. This module keeps their behavior intact
and converts chunk rankings to unique paper rankings before computing metrics.
It also owns the Benchmark v1 protocol: warm queries, dev-only alpha tuning,
test evaluation, manifests, per-query output, and Markdown reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.runtime_info import collect_accelerator_info
from app.retrieval.hybrid import search_hybrid
from app.retrieval.lexical import search_lexical
from app.retrieval.vector_store import search_vector

logger = logging.getLogger(__name__)

DEFAULT_EVAL = PROJECT_ROOT / "data" / "eval" / "retrieval_v1.jsonl"
DEFAULT_DB = PROJECT_ROOT / "data" / "indexes" / "benchmark_v1" / "metadata.sqlite"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "benchmark_v1" / "faiss"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "benchmark_v1_manifest.json"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "retrieval_baseline_v1.json"
DEFAULT_MD_REPORT = PROJECT_ROOT / "reports" / "retrieval_baseline_v1.md"

QUERY_TYPES = ("keyword", "natural_question", "semantic_paraphrase")
SPLITS = ("dev", "test")
DEFAULT_ALPHA_VALUES = (0.20, 0.35, 0.50, 0.65, 0.80)
DEFAULT_RETRIEVAL_DEPTH = 10
OFFICIAL_PAPER_COUNT = 50_000
OFFICIAL_QUERY_COUNT = 150
OFFICIAL_DEV_COUNT = 50
OFFICIAL_TEST_COUNT = 100

SearchFn = Callable[[str, int], Sequence[Any]]
RetrieverFactory = Callable[[str, float | None], SearchFn]


class EvalDataError(ValueError):
    """Raised when a frozen retrieval evaluation file is invalid."""


@dataclass(frozen=True)
class EvalQuery:
    """Validated query record used by the benchmark runner."""

    query_id: str
    query: str
    query_type: str
    split: str
    relevant_paper_ids: tuple[str, ...]
    source_paper_id: str | None = None
    source_category: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
    temporary.replace(path)


# ---------------------------------------------------------------------------
# Frozen evaluation data
# ---------------------------------------------------------------------------


def _require_nonempty_string(record: dict[str, Any], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvalDataError(f"Line {line_number}: {field!r} must be a non-empty string")
    return value.strip()


def parse_eval_record(record: Any, *, line_number: int = 1) -> EvalQuery:
    """Validate one JSONL record without silently repairing malformed data."""
    if not isinstance(record, dict):
        raise EvalDataError(f"Line {line_number}: expected a JSON object")

    query_id = _require_nonempty_string(record, "query_id", line_number)
    query = _require_nonempty_string(record, "query", line_number)
    query_type = _require_nonempty_string(record, "query_type", line_number)
    split = _require_nonempty_string(record, "split", line_number)

    if query_type not in QUERY_TYPES:
        raise EvalDataError(
            f"Line {line_number}: query_type must be one of {', '.join(QUERY_TYPES)}"
        )
    if split not in SPLITS:
        raise EvalDataError(f"Line {line_number}: split must be one of {', '.join(SPLITS)}")

    relevant = record.get("relevant_paper_ids")
    if not isinstance(relevant, list) or not relevant:
        raise EvalDataError(
            f"Line {line_number}: 'relevant_paper_ids' must be a non-empty list"
        )
    if any(not isinstance(value, str) or not value.strip() for value in relevant):
        raise EvalDataError(
            f"Line {line_number}: every relevant paper ID must be a non-empty string"
        )
    relevant_ids = tuple(value.strip() for value in relevant)
    if len(set(relevant_ids)) != len(relevant_ids):
        raise EvalDataError(f"Line {line_number}: relevant paper IDs must be unique")

    source_paper_id = record.get("source_paper_id")
    if source_paper_id is not None:
        if not isinstance(source_paper_id, str) or not source_paper_id.strip():
            raise EvalDataError(
                f"Line {line_number}: source_paper_id must be a non-empty string or null"
            )
        source_paper_id = source_paper_id.strip()

    source_category = record.get("source_category")
    if source_category is not None:
        if not isinstance(source_category, str) or not source_category.strip():
            raise EvalDataError(
                f"Line {line_number}: source_category must be a non-empty string or null"
            )
        source_category = source_category.strip()

    return EvalQuery(
        query_id=query_id,
        query=query,
        query_type=query_type,
        split=split,
        relevant_paper_ids=relevant_ids,
        source_paper_id=source_paper_id,
        source_category=source_category,
    )


def load_eval_queries(eval_path: str | Path, db_path: str | Path | None = None) -> list[EvalQuery]:
    """Load and validate a frozen JSONL evaluation set."""
    eval_path = Path(eval_path)
    queries: list[EvalQuery] = []
    seen_query_ids: set[str] = set()
    seen_source_ids: set[str] = set()

    with open(eval_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalDataError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc

            query = parse_eval_record(raw, line_number=line_number)
            if query.query_id in seen_query_ids:
                raise EvalDataError(
                    f"Line {line_number}: duplicate query_id {query.query_id!r}"
                )
            seen_query_ids.add(query.query_id)

            if query.source_paper_id is not None:
                if query.source_paper_id in seen_source_ids:
                    raise EvalDataError(
                        f"Line {line_number}: duplicate source_paper_id "
                        f"{query.source_paper_id!r}"
                    )
                seen_source_ids.add(query.source_paper_id)
            queries.append(query)

    if not queries:
        raise EvalDataError(f"No evaluation queries found in {eval_path}")
    if db_path is not None:
        validate_relevant_paper_ids(queries, db_path)
    return queries


def _batched(values: Sequence[str], size: int = 900) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def validate_relevant_paper_ids(
    queries: Sequence[EvalQuery], db_path: str | Path
) -> None:
    """Require every relevance judgment to refer to a paper in the corpus."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise EvalDataError(f"Benchmark database not found: {db_path}")

    expected = sorted({paper_id for query in queries for paper_id in query.relevant_paper_ids})
    found: set[str] = set()
    conn = sqlite3.connect(str(db_path))
    try:
        for batch in _batched(expected):
            placeholders = ",".join("?" for _ in batch)
            try:
                rows = conn.execute(
                    f"SELECT paper_id FROM papers WHERE paper_id IN ({placeholders})",
                    tuple(batch),
                ).fetchall()
            except sqlite3.Error as exc:
                raise EvalDataError(f"Cannot validate papers in {db_path}: {exc}") from exc
            found.update(row[0] for row in rows)
    finally:
        conn.close()

    missing = sorted(set(expected) - found)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" (and {len(missing) - 10} more)"
        raise EvalDataError(
            f"{len(missing)} relevant paper ID(s) are missing from the benchmark DB: "
            f"{preview}{suffix}"
        )


# ---------------------------------------------------------------------------
# Paper-level ranking and metrics
# ---------------------------------------------------------------------------


def deduplicate_papers(results: Sequence[Any], limit: int | None = None) -> list[Any]:
    """Keep the first/highest-ranked chunk for each paper, preserving rank."""
    unique: list[Any] = []
    seen: set[str] = set()
    for result in results:
        paper_id = getattr(result, "paper_id", None)
        if not isinstance(paper_id, str) or not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        unique.append(result)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def hit_rate_at_k(pred_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Return 1 when any relevant paper occurs in top-k, otherwise 0."""
    _validate_k(k)
    return float(bool(set(pred_ids[:k]) & relevant_ids))


def recall_at_k(pred_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Return the fraction of all known relevant papers retrieved in top-k."""
    _validate_k(k)
    if not relevant_ids:
        return 0.0
    return len(set(pred_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(pred_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Return reciprocal rank of the first relevant paper within top-k."""
    _validate_k(k)
    for rank, paper_id in enumerate(pred_ids[:k], start=1):
        if paper_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pred_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute binary nDCG using all known relevant papers for the ideal DCG."""
    _validate_k(k)
    if not relevant_ids:
        return 0.0

    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, paper_id in enumerate(pred_ids[:k], start=1)
        if paper_id in relevant_ids
    )
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return float(dcg / idcg)


def _first_relevant_rank(pred_ids: Sequence[str], relevant_ids: set[str]) -> int | None:
    for rank, paper_id in enumerate(pred_ids, start=1):
        if paper_id in relevant_ids:
            return rank
    return None


def _metric_value(value: float) -> float:
    """Keep full precision for alpha selection; presentation rounds separately."""
    return float(value)


def _aggregate_per_query(per_query: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not per_query:
        return {
            "query_count": 0,
            "hit_rate@5": 0.0,
            "hit_rate@10": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr@10": 0.0,
            "ndcg@10": 0.0,
        }
    return {
        "query_count": len(per_query),
        "hit_rate@5": _metric_value(np.mean([row["hit@5"] for row in per_query])),
        "hit_rate@10": _metric_value(np.mean([row["hit@10"] for row in per_query])),
        "recall@5": _metric_value(np.mean([row["recall@5"] for row in per_query])),
        "recall@10": _metric_value(np.mean([row["recall@10"] for row in per_query])),
        "mrr@10": _metric_value(np.mean([row["mrr@10"] for row in per_query])),
        "ndcg@10": _metric_value(np.mean([row["ndcg@10"] for row in per_query])),
    }


def evaluate_method(
    method_name: str,
    queries: Sequence[EvalQuery],
    search_fn: SearchFn,
    *,
    retrieval_depth: int = DEFAULT_RETRIEVAL_DEPTH,
) -> dict[str, Any]:
    """Warm one retriever, then evaluate every query independently."""
    if not queries:
        raise ValueError(f"Cannot evaluate {method_name}: query list is empty")
    if retrieval_depth < 10:
        raise ValueError("retrieval_depth must be at least 10")

    warmup_started = time.perf_counter()
    search_fn(queries[0].query, retrieval_depth)
    warmup_ms = (time.perf_counter() - warmup_started) * 1000

    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        chunk_results = search_fn(query.query, retrieval_depth)
        latency_ms = (time.perf_counter() - started) * 1000
        paper_results = deduplicate_papers(chunk_results, limit=retrieval_depth)
        paper_ids = [result.paper_id for result in paper_results]
        relevant = set(query.relevant_paper_ids)

        latencies.append(latency_ms)
        per_query.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "query_type": query.query_type,
                "relevant_paper_ids": list(query.relevant_paper_ids),
                "retrieved_paper_ids": paper_ids,
                "first_relevant_rank": _first_relevant_rank(paper_ids, relevant),
                "hit@5": bool(hit_rate_at_k(paper_ids, relevant, 5)),
                "hit@10": bool(hit_rate_at_k(paper_ids, relevant, 10)),
                "recall@5": _metric_value(recall_at_k(paper_ids, relevant, 5)),
                "recall@10": _metric_value(recall_at_k(paper_ids, relevant, 10)),
                "mrr@10": _metric_value(reciprocal_rank_at_k(paper_ids, relevant, 10)),
                "ndcg@10": _metric_value(ndcg_at_k(paper_ids, relevant, 10)),
                "latency_ms": round(latency_ms, 3),
            }
        )

    by_query_type = {
        query_type: _aggregate_per_query(
            [row for row in per_query if row["query_type"] == query_type]
        )
        for query_type in QUERY_TYPES
        if any(row["query_type"] == query_type for row in per_query)
    }
    return {
        "method": method_name,
        "retrieval_depth": retrieval_depth,
        "warmup_query_ms": round(warmup_ms, 3),
        "metrics": _aggregate_per_query(per_query),
        "latency": {
            "mean_ms": round(float(np.mean(latencies)), 3),
            "p50_ms": round(float(np.percentile(latencies, 50)), 3),
            "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        },
        "by_query_type": by_query_type,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Benchmark protocol
# ---------------------------------------------------------------------------


def _default_retriever_factory(db_path: Path, index_dir: Path) -> RetrieverFactory:
    def factory(mode: str, alpha: float | None) -> SearchFn:
        if mode == "bm25":
            return lambda query, top_k: search_lexical(
                query, top_k=top_k, db_path=db_path
            )
        if mode == "dense":
            return lambda query, top_k: search_vector(
                query, top_k=top_k, db_path=db_path, index_dir=index_dir
            )
        if mode == "hybrid" and alpha is not None:
            return lambda query, top_k: search_hybrid(
                query,
                top_k=top_k,
                alpha=alpha,
                db_path=db_path,
                index_dir=index_dir,
            )
        raise ValueError(f"Unsupported retriever configuration: mode={mode}, alpha={alpha}")

    return factory


def select_best_alpha(dev_sweep: Sequence[dict[str, Any]]) -> float:
    """Select by nDCG@10, then MRR@10; exact ties keep sweep order."""
    if not dev_sweep:
        raise ValueError("Alpha sweep is empty")

    _, best = max(
        enumerate(dev_sweep),
        key=lambda item: (
            item[1]["metrics"]["ndcg@10"],
            item[1]["metrics"]["mrr@10"],
            -item[0],
        ),
    )
    return float(best["alpha"])


def _query_distribution(queries: Sequence[EvalQuery]) -> dict[str, Any]:
    return {
        "total": len(queries),
        "splits": dict(sorted(Counter(query.split for query in queries).items())),
        "query_types": dict(
            sorted(Counter(query.query_type for query in queries).items())
        ),
        "split_query_types": {
            split: dict(
                sorted(
                    Counter(
                        query.query_type for query in queries if query.split == split
                    ).items()
                )
            )
            for split in SPLITS
        },
    }


def _benchmark_status(manifest: dict[str, Any], queries: Sequence[EvalQuery]) -> str:
    distribution = _query_distribution(queries)
    official = (
        manifest.get("corpus") == "arxiv_cs"
        and manifest.get("paper_count") == OFFICIAL_PAPER_COUNT
        and manifest.get("paper_count") == manifest.get("chunk_count")
        and manifest.get("chunk_count") == manifest.get("fts_row_count")
        and manifest.get("chunk_count") == manifest.get("faiss_vector_count")
        and manifest.get("chunk_count") == manifest.get("id_map_count")
        and manifest.get("embedding_model") == "BAAI/bge-m3"
        and manifest.get("embedding_dim") == 1024
        and bool(manifest.get("raw_file_sha256"))
        and bool(manifest.get("eval_file_sha256"))
        and bool(manifest.get("git_commit"))
        and manifest.get("git_dirty") is False
        and distribution["total"] == OFFICIAL_QUERY_COUNT
        and distribution["splits"].get("dev") == OFFICIAL_DEV_COUNT
        and distribution["splits"].get("test") == OFFICIAL_TEST_COUNT
        and all(
            distribution["query_types"].get(query_type) == 50
            for query_type in QUERY_TYPES
        )
        and all(
            query.source_paper_id is not None
            and query.source_category is not None
            and query.source_paper_id in query.relevant_paper_ids
            for query in queries
        )
    )
    return "benchmark_v1" if official else "smoke_or_development"


def _copy_method_result(result: dict[str, Any], method_name: str) -> dict[str, Any]:
    copied = json.loads(json.dumps(result))
    copied["method"] = method_name
    copied["reused_test_run_from"] = result["method"]
    return copied


def _build_error_analysis(test_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_method = {
        method: {row["query_id"]: row for row in result["per_query"]}
        for method, result in test_results.items()
    }
    ordered_ids = [row["query_id"] for row in test_results["bm25"]["per_query"]]

    def hit(method: str, query_id: str) -> bool:
        return bool(by_method[method][query_id]["hit@10"])

    predicates: dict[str, Callable[[str], bool]] = {
        "hybrid_success_bm25_failure": lambda qid: hit("hybrid_tuned", qid)
        and not hit("bm25", qid),
        "hybrid_success_dense_failure": lambda qid: hit("hybrid_tuned", qid)
        and not hit("dense", qid),
        "dense_success_bm25_failure": lambda qid: hit("dense", qid)
        and not hit("bm25", qid),
        "bm25_success_dense_failure": lambda qid: hit("bm25", qid)
        and not hit("dense", qid),
        "all_methods_failure": lambda qid: not hit("bm25", qid)
        and not hit("dense", qid)
        and not hit("hybrid_tuned", qid),
    }

    groups: dict[str, Any] = {}
    for name, predicate in predicates.items():
        query_ids = [query_id for query_id in ordered_ids if predicate(query_id)]
        examples = []
        for query_id in query_ids[:3]:
            base = by_method["bm25"][query_id]
            examples.append(
                {
                    "query_id": query_id,
                    "query": base["query"],
                    "query_type": base["query_type"],
                    "relevant_paper_ids": base["relevant_paper_ids"],
                    "first_relevant_rank": {
                        method: by_method[method][query_id]["first_relevant_rank"]
                        for method in ("bm25", "dense", "hybrid_tuned")
                    },
                    "top_5_paper_ids": {
                        method: by_method[method][query_id]["retrieved_paper_ids"][:5]
                        for method in ("bm25", "dense", "hybrid_tuned")
                    },
                }
            )
        groups[name] = {
            "count": len(query_ids),
            "query_ids": query_ids,
            "representative_examples": examples,
        }
    return groups


def run_benchmark(
    queries: Sequence[EvalQuery],
    *,
    db_path: str | Path = DEFAULT_DB,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    manifest: dict[str, Any] | None = None,
    alpha_values: Sequence[float] = DEFAULT_ALPHA_VALUES,
    retrieval_depth: int = DEFAULT_RETRIEVAL_DEPTH,
    retriever_factory: RetrieverFactory | None = None,
) -> dict[str, Any]:
    """Run dev alpha tuning followed by one frozen test comparison."""
    dev_queries = [query for query in queries if query.split == "dev"]
    test_queries = [query for query in queries if query.split == "test"]
    if not dev_queries or not test_queries:
        raise EvalDataError("Benchmark requires at least one dev and one test query")
    if not alpha_values:
        raise ValueError("alpha_values must not be empty")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alpha_values):
        raise ValueError("Every alpha must be between 0 and 1")

    db_path = Path(db_path)
    index_dir = Path(index_dir)
    factory = retriever_factory or _default_retriever_factory(db_path, index_dir)
    manifest = manifest or {}

    dev_sweep: list[dict[str, Any]] = []
    for alpha in alpha_values:
        method = evaluate_method(
            f"hybrid_alpha_{alpha:.2f}",
            dev_queries,
            factory("hybrid", float(alpha)),
            retrieval_depth=retrieval_depth,
        )
        dev_sweep.append({"alpha": float(alpha), **method})

    selected_alpha = select_best_alpha(dev_sweep)

    test_results = {
        "bm25": evaluate_method(
            "bm25",
            test_queries,
            factory("bm25", None),
            retrieval_depth=retrieval_depth,
        ),
        "dense": evaluate_method(
            "dense",
            test_queries,
            factory("dense", None),
            retrieval_depth=retrieval_depth,
        ),
        "hybrid_0.5": evaluate_method(
            "hybrid_0.5",
            test_queries,
            factory("hybrid", 0.5),
            retrieval_depth=retrieval_depth,
        ),
    }
    if selected_alpha == 0.5:
        test_results["hybrid_tuned"] = _copy_method_result(
            test_results["hybrid_0.5"], "hybrid_tuned"
        )
    else:
        test_results["hybrid_tuned"] = evaluate_method(
            "hybrid_tuned",
            test_queries,
            factory("hybrid", selected_alpha),
            retrieval_depth=retrieval_depth,
        )

    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "status": _benchmark_status(manifest, queries),
        "manifest": manifest,
        "evaluation_set": _query_distribution(queries),
        "protocol": {
            "ranking_unit": "paper",
            "paper_deduplication": "first/highest-ranked chunk retained",
            "retrieval_depth": retrieval_depth,
            "latency": "one untimed warm-up per method, then per-query wall time",
            "alpha_selection": "dev nDCG@10, then dev MRR@10, then sweep order",
        },
        "dev_alpha_sweep": dev_sweep,
        "selected_alpha": selected_alpha,
        "test_results": test_results,
        "error_analysis": _build_error_analysis(test_results),
        "limitations": [
            "Synthetic known-item queries are generated from target title and abstract.",
            "Relevance labels are not equivalent to human judgments or a public IR benchmark.",
            "Latency is machine- and index-configuration-specific.",
        ],
    }


# ---------------------------------------------------------------------------
# Corpus manifest and reports
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def build_benchmark_manifest(
    *,
    db_path: str | Path,
    index_dir: str | Path,
    raw_path: str | Path,
    eval_path: str | Path | None = None,
    corpus: str = "arxiv_cs",
) -> dict[str, Any]:
    """Read corpus/index facts from artifacts; no benchmark value is fabricated."""
    db_path = Path(db_path)
    index_dir = Path(index_dir)
    raw_path = Path(raw_path)
    index_path = index_dir / "index.faiss"
    id_map_path = index_dir / "id_map.json"
    build_meta_path = index_dir / "build_meta.json"
    for required in (db_path, raw_path, index_path, id_map_path):
        if not required.exists():
            raise FileNotFoundError(f"Required benchmark artifact not found: {required}")

    conn = sqlite3.connect(str(db_path))
    try:
        paper_count = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        try:
            fts_row_count = int(
                conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
            )
        except sqlite3.Error as exc:
            raise RuntimeError("Benchmark database has no valid chunk_fts index") from exc
    finally:
        conn.close()

    import faiss

    index = faiss.read_index(str(index_path))
    if not index.is_trained:
        raise RuntimeError("FAISS benchmark index is not trained")
    if int(index.ntotal) != chunk_count:
        raise RuntimeError(
            f"FAISS vector count {index.ntotal} does not match chunk count {chunk_count}"
        )
    if fts_row_count != chunk_count:
        raise RuntimeError(
            f"FTS row count {fts_row_count} does not match chunk count {chunk_count}"
        )

    with open(id_map_path, "r", encoding="utf-8") as handle:
        id_map = json.load(handle)
    if not isinstance(id_map, list):
        raise RuntimeError("FAISS id_map.json must contain a JSON list")
    id_map_count = len(id_map)
    if id_map_count != chunk_count:
        raise RuntimeError(
            f"ID map count {id_map_count} does not match chunk count {chunk_count}"
        )
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT chunk_id, paper_id FROM chunks ORDER BY rowid"
        )
        for faiss_id, row in enumerate(rows):
            entry = id_map[faiss_id]
            if (
                not isinstance(entry, dict)
                or entry.get("faiss_id") != faiss_id
                or entry.get("chunk_id") != row["chunk_id"]
                or entry.get("paper_id") != row["paper_id"]
            ):
                raise RuntimeError(
                    f"FAISS ID map diverges from SQLite chunk order at position {faiss_id}"
                )
    finally:
        conn.close()

    nlist = int(index.nlist) if hasattr(index, "nlist") else None
    nprobe = min(nlist // 4, 64) if nlist is not None else None
    build_meta: dict[str, Any] = {}
    if build_meta_path.exists():
        with open(build_meta_path, "r", encoding="utf-8") as handle:
            build_meta = json.load(handle)
        if build_meta.get("num_vectors") != chunk_count:
            raise RuntimeError("FAISS build metadata vector count does not match the corpus")
        if build_meta.get("vector_dim") != int(index.d):
            raise RuntimeError("FAISS build metadata dimension does not match the index")

        db_signature = build_meta.get("db_signature")
        if isinstance(db_signature, dict) and isinstance(db_signature.get("path"), str):
            db_signature["path"] = _manifest_path(Path(db_signature["path"]))

    resolved_eval_path = Path(eval_path) if eval_path is not None else None
    if resolved_eval_path is not None and not resolved_eval_path.exists():
        raise FileNotFoundError(f"Frozen evaluation file not found: {resolved_eval_path}")

    git_status = _git_value("status", "--porcelain")
    return {
        "schema_version": 1,
        "corpus": corpus,
        "paper_count": paper_count,
        "chunk_count": chunk_count,
        "fts_row_count": fts_row_count,
        "raw_file": _manifest_path(raw_path),
        "raw_file_sha256": sha256_file(raw_path),
        "eval_file": _manifest_path(resolved_eval_path) if resolved_eval_path else None,
        "eval_file_sha256": sha256_file(resolved_eval_path) if resolved_eval_path else None,
        "database_file": _manifest_path(db_path),
        "embedding_model": build_meta.get("embedding_model"),
        "embedding_dim": int(index.d),
        "faiss_index_type": index.__class__.__name__,
        "faiss_vector_count": int(index.ntotal),
        "id_map_count": id_map_count,
        "faiss_nlist": nlist,
        "faiss_nprobe": nprobe,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "created_at": _utc_now(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "accelerator": collect_accelerator_info(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", None),
        },
        "faiss_build": build_meta or None,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown_report(result: dict[str, Any]) -> str:
    """Render a concise report entirely from measured JSON values."""
    manifest = result["manifest"]
    distribution = result["evaluation_set"]
    status = result["status"]
    title = (
        "# CiteQuest Retrieval Benchmark v1"
        if status == "benchmark_v1"
        else "# CiteQuest Retrieval Smoke / Development Run"
    )
    lines = [title, ""]
    if status != "benchmark_v1":
        lines.extend(
            [
                "> This run is not the official Benchmark v1 because it does not use the full "
                "50,000-paper corpus and 150-query frozen set.",
                "",
            ]
        )

    lines.extend(
        [
            "## Benchmark setup",
            "",
            f"- Corpus: `{manifest.get('corpus', 'unknown')}`",
            f"- Papers: {manifest.get('paper_count', 'unknown')}",
            f"- Chunks: {manifest.get('chunk_count', 'unknown')}",
            f"- Queries: {distribution['total']} "
            f"(dev={distribution['splits'].get('dev', 0)}, "
            f"test={distribution['splits'].get('test', 0)})",
            f"- Query types: `{json.dumps(distribution['query_types'], sort_keys=True)}`",
            f"- Embedding model: `{manifest.get('embedding_model') or 'unavailable'}`",
            f"- FAISS: `{manifest.get('faiss_index_type', 'unavailable')}`, "
            f"nlist={manifest.get('faiss_nlist')}, nprobe={manifest.get('faiss_nprobe')}",
            f"- Raw SHA256: `{manifest.get('raw_file_sha256', 'unavailable')}`",
            f"- Eval SHA256: `{manifest.get('eval_file_sha256', 'unavailable')}`",
            f"- Git commit: `{manifest.get('git_commit', 'unavailable')}`",
            f"- Environment: `{manifest.get('environment', {}).get('platform', 'unavailable')}`",
            "",
            "## Metric definitions",
            "",
            "- HitRate@K: fraction of queries with at least one relevant paper in top K.",
            "- Recall@K: fraction of all known relevant papers retrieved in top K.",
            "- MRR@10: mean reciprocal rank of the first relevant paper within top 10.",
            "- nDCG@10: binary normalized discounted gain using all known relevant papers.",
            "- Latency: one untimed warm-up per method, then per-query wall-clock time.",
            "",
            "## Dev alpha sweep",
            "",
            "| Alpha (lexical) | HitRate@10 | Recall@10 | MRR@10 | nDCG@10 | mean ms | p50 ms | p95 ms |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["dev_alpha_sweep"]:
        metrics = row["metrics"]
        latency = row["latency"]
        lines.append(
            f"| {row['alpha']:.2f} | {_fmt(metrics['hit_rate@10'])} | "
            f"{_fmt(metrics['recall@10'])} | {_fmt(metrics['mrr@10'])} | "
            f"{_fmt(metrics['ndcg@10'])} | {latency['mean_ms']:.3f} | "
            f"{latency['p50_ms']:.3f} | "
            f"{latency['p95_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Selected alpha: **{result['selected_alpha']:.2f}**, using dev nDCG@10 "
            "with dev MRR@10 as tie-break.",
            "",
            "## Final test results",
            "",
            "| Method | HitRate@5 | HitRate@10 | Recall@10 | MRR@10 | nDCG@10 | mean ms | p50 ms | p95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    labels = {
        "bm25": "BM25",
        "dense": "Dense",
        "hybrid_0.5": "Hybrid 0.5",
        "hybrid_tuned": f"Hybrid tuned ({result['selected_alpha']:.2f})",
    }
    for method in ("bm25", "dense", "hybrid_0.5", "hybrid_tuned"):
        row = result["test_results"][method]
        metrics = row["metrics"]
        latency = row["latency"]
        lines.append(
            f"| {labels[method]} | {_fmt(metrics['hit_rate@5'])} | "
            f"{_fmt(metrics['hit_rate@10'])} | {_fmt(metrics['recall@10'])} | "
            f"{_fmt(metrics['mrr@10'])} | {_fmt(metrics['ndcg@10'])} | "
            f"{latency['mean_ms']:.3f} | {latency['p50_ms']:.3f} | "
            f"{latency['p95_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Results by query type",
            "",
            "| Method | Query type | N | HitRate@10 | MRR@10 | nDCG@10 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in ("bm25", "dense", "hybrid_0.5", "hybrid_tuned"):
        for query_type, metrics in result["test_results"][method]["by_query_type"].items():
            lines.append(
                f"| {labels[method]} | {query_type} | {metrics['query_count']} | "
                f"{_fmt(metrics['hit_rate@10'])} | {_fmt(metrics['mrr@10'])} | "
                f"{_fmt(metrics['ndcg@10'])} |"
            )

    lines.extend(["", "## Error analysis", ""])
    for group_name, group in result["error_analysis"].items():
        lines.append(f"### `{group_name}` ({group['count']})")
        lines.append("")
        if not group["representative_examples"]:
            lines.append("No examples in this run.")
            lines.append("")
            continue
        for example in group["representative_examples"]:
            ranks = example["first_relevant_rank"]
            lines.append(
                f"- `{example['query_id']}` ({example['query_type']}): "
                f"{_markdown_escape(example['query'])} "
                f"[ranks: BM25={ranks['bm25']}, Dense={ranks['dense']}, "
                f"Hybrid={ranks['hybrid_tuned']}]"
            )
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "Benchmark v1 is a synthetic known-item retrieval benchmark generated from "
            "source-paper titles and abstracts. It is not equivalent to human relevance "
            "judgments or a standard public IR benchmark. Latency is specific to the "
            "recorded machine and index configuration.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_outputs(
    result: dict[str, Any],
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    json_report_path: str | Path = DEFAULT_JSON_REPORT,
    markdown_report_path: str | Path = DEFAULT_MD_REPORT,
) -> None:
    _atomic_write_text(
        Path(manifest_path),
        json.dumps(result["manifest"], ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        Path(json_report_path),
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(Path(markdown_report_path), render_markdown_report(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CiteQuest Retrieval Benchmark v1")
    parser.add_argument("--eval", default=str(DEFAULT_EVAL), help="Frozen eval JSONL")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Benchmark SQLite DB")
    parser.add_argument(
        "--index-dir", default=str(DEFAULT_INDEX_DIR), help="Benchmark FAISS directory"
    )
    parser.add_argument("--raw", required=True, help="Raw corpus JSONL used to build DB")
    parser.add_argument("--corpus", default="arxiv_cs", help="Corpus identifier")
    parser.add_argument(
        "--retrieval-depth", type=int, default=DEFAULT_RETRIEVAL_DEPTH
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON_REPORT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD_REPORT))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    db_path = Path(args.db)
    index_dir = Path(args.index_dir)
    queries = load_eval_queries(args.eval, db_path=db_path)
    manifest = build_benchmark_manifest(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=args.raw,
        eval_path=args.eval,
        corpus=args.corpus,
    )
    result = run_benchmark(
        queries,
        db_path=db_path,
        index_dir=index_dir,
        manifest=manifest,
        retrieval_depth=args.retrieval_depth,
    )
    write_benchmark_outputs(
        result,
        manifest_path=args.manifest,
        json_report_path=args.output_json,
        markdown_report_path=args.output_md,
    )
    logger.info("Benchmark status: %s", result["status"])
    logger.info("Selected hybrid alpha: %.2f", result["selected_alpha"])
    logger.info("JSON report: %s", args.output_json)
    logger.info("Markdown report: %s", args.output_md)


if __name__ == "__main__":
    main()
