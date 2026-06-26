"""
Retrieval evaluation: measure search quality across modes.

Metrics:
- Recall@k  — fraction of queries where ≥1 relevant doc appears in top k
- MRR       — Mean Reciprocal Rank (1 / rank of first relevant doc)
- nDCG@k    — Normalized Discounted Cumulative Gain (position-weighted)

Usage:
    python -m app.eval.retrieval_eval --eval data/eval/retrieval_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.lexical import search_lexical
from app.retrieval.vector_store import search_vector
from app.retrieval.hybrid import search_hybrid

logger = logging.getLogger(__name__)

DEFAULT_DB = PROJECT_ROOT / "data" / "indexes" / "metadata.sqlite"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "faiss"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _dcg(scores: list[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    k = min(k, len(scores))
    dcg = 0.0
    for i in range(k):
        dcg += scores[i] / np.log2(i + 2)  # i+2 because log2(1)=0 for position 1
    return dcg


def _ndcg(pred_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized DCG — relevance is binary (1 if relevant, 0 otherwise)."""
    scores = [1.0 if pid in relevant_ids else 0.0 for pid in pred_ids]
    dcg = _dcg(scores, k)

    # Ideal DCG: all relevant docs at the top
    ideal = sorted(scores, reverse=True)
    idcg = _dcg(ideal, k)

    return dcg / idcg if idcg > 0 else 0.0


def _recall(pred_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k: fraction of queries where at least 1 relevant doc is in top k."""
    k_ids = set(pred_ids[:k])
    return 1.0 if k_ids & relevant_ids else 0.0


def _mrr(pred_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant document."""
    for i, pid in enumerate(pred_ids, start=1):
        if pid in relevant_ids:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_eval(
    eval_path: Path,
    db_path: Path = DEFAULT_DB,
    index_dir: Path = DEFAULT_INDEX_DIR,
    k_values: tuple[int, ...] = (5, 10),
) -> dict:
    """Run retrieval evaluation across lexical, vector, and hybrid modes.

    Returns a dict with metrics per mode and per-query details.
    """
    # Load eval set
    queries = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            queries.append({
                "query": rec["query"],
                "relevant_paper_ids": set(rec["relevant_paper_ids"]),
            })

    if not queries:
        logger.warning("No eval queries found in %s", eval_path)
        return {}

    logger.info("Evaluating %d queries across 3 modes ...", len(queries))

    # Retrievers to evaluate: (mode_name, alpha, search_fn)
    retrievers: list[tuple[str, float | None, Callable]] = [
        ("lexical", None, lambda q, k: search_lexical(q, top_k=k, db_path=db_path)),
        (
            "vector",
            None,
            lambda q, k: search_vector(q, top_k=k, db_path=db_path, index_dir=index_dir),
        ),
        (
            "hybrid",
            0.5,
            lambda q, k: search_hybrid(
                q, top_k=k, alpha=0.5, db_path=db_path, index_dir=index_dir
            ),
        ),
    ]

    results: dict = {"queries": len(queries), "modes": {}}

    for mode_name, alpha, search_fn in retrievers:
        logger.info("  Mode: %s ...", mode_name)

        # Per-query accumulators
        recall_at: dict[int, list[float]] = {k: [] for k in k_values}
        mrr_list: list[float] = []
        ndcg_at: dict[int, list[float]] = {k: [] for k in k_values}
        latencies: list[float] = []

        for q in queries:
            t0 = time.perf_counter()
            hits = search_fn(q["query"], max(k_values))
            latencies.append((time.perf_counter() - t0) * 1000)

            paper_ids = [r.paper_id for r in hits]

            for k in k_values:
                recall_at[k].append(_recall(paper_ids, q["relevant_paper_ids"], k))
                ndcg_at[k].append(_ndcg(paper_ids, q["relevant_paper_ids"], k))

            mrr_list.append(_mrr(paper_ids, q["relevant_paper_ids"]))

        mode_result: dict = {
            "alpha": alpha,
            "latency_avg_ms": round(np.mean(latencies), 2),
        }
        for k in k_values:
            mode_result[f"recall@{k}"] = round(np.mean(recall_at[k]), 4)
            mode_result[f"ndcg@{k}"] = round(np.mean(ndcg_at[k]), 4)
        mode_result["mrr"] = round(np.mean(mrr_list), 4)

        results["modes"][mode_name] = mode_result

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument(
        "--eval",
        default=str(PROJECT_ROOT / "data" / "eval" / "retrieval_eval.jsonl"),
        help="Path to eval JSONL file",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to metadata SQLite")
    parser.add_argument(
        "--index-dir", default=str(DEFAULT_INDEX_DIR), help="Path to FAISS index directory"
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path for results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    eval_path = Path(args.eval)
    if not eval_path.exists():
        sys.exit(f"Eval file not found: {eval_path}")

    results = run_eval(eval_path, db_path=Path(args.db), index_dir=Path(args.index_dir))

    if not results:
        sys.exit("No results.")

    # — Print report ——————————————————————————————————————————————————
    print("\n" + "=" * 60)
    print(f"Retrieval Evaluation — {results['queries']} queries")
    print("=" * 60)
    for mode, metrics in results["modes"].items():
        alpha_str = f" (alpha={metrics['alpha']})" if metrics["alpha"] is not None else ""
        print(f"\n  {mode}{alpha_str}:")
        print(f"    Latency (avg): {metrics['latency_avg_ms']:.1f} ms")
        for key in sorted(metrics):
            if key.startswith("recall") or key.startswith("ndcg") or key == "mrr":
                print(f"    {key}: {metrics[key]:.4f}")
    print()

    # — Optional JSON output ———————————————————————————————————————————
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
