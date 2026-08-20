"""Run the operational acceptance suite for the 10k CiteQuest demo.

The generated report is intentionally separate from Retrieval Benchmark v1:
it validates runnable artifacts and latency, not retrieval relevance.

Usage:
    python scripts/run_demo_smoke.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.demo_smoke import (
    DEFAULT_DEMO_QUERIES,
    run_demo_smoke,
    write_demo_smoke_outputs,
)


DEFAULT_DB = PROJECT_ROOT / "data" / "indexes" / "demo10k" / "metadata.sqlite"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "demo10k" / "faiss"
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "demo_peS2o_10000.jsonl"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "demo10k_smoke.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "demo10k_smoke.md"

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the 10k demo artifacts and smoke-test BM25, Dense, "
            "Hybrid, FastAPI, and mock-LLM RAG."
        )
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Metadata SQLite path")
    parser.add_argument(
        "--index-dir", default=str(DEFAULT_INDEX_DIR), help="FAISS artifact directory"
    )
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="Source JSONL path")
    parser.add_argument(
        "--expected-papers",
        type=int,
        default=10_000,
        help="Exact expected paper count (default: 10000)",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Representative query; repeat for multiple queries",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Results per search")
    parser.add_argument(
        "--runs", type=int, default=3, help="Measured warm runs per query and method"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Hybrid lexical weight"
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip FastAPI handler and mock-RAG checks",
    )
    parser.add_argument(
        "--json-report", default=str(DEFAULT_JSON_REPORT), help="JSON output path"
    )
    parser.add_argument(
        "--markdown-report",
        default=str(DEFAULT_MARKDOWN_REPORT),
        help="Markdown output path",
    )
    return parser


def _log_failures(report: dict) -> None:
    for check in report["artifacts"].get("checks", []):
        if not check["passed"]:
            logger.error("Artifact check failed: %s (%s)", check["name"], check["detail"])
    retrieval = report.get("retrieval", {})
    for method, method_report in retrieval.get("methods", {}).items():
        for error in method_report.get("errors", []):
            logger.error("%s smoke failed: %s", method, error)
    for error in report.get("api", {}).get("errors", []):
        logger.error("API smoke failed: %s", error)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.expected_papers <= 0:
        parser.error("--expected-papers must be positive")

    queries = tuple(args.queries or DEFAULT_DEMO_QUERIES)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Running 10k demo acceptance against %s", args.db)

    try:
        report = run_demo_smoke(
            db_path=args.db,
            index_dir=args.index_dir,
            raw_path=args.raw,
            queries=queries,
            expected_papers=args.expected_papers,
            top_k=args.top_k,
            runs=args.runs,
            alpha=args.alpha,
            include_api=not args.skip_api,
        )
    except (OSError, ValueError) as exc:
        logger.error("Cannot run demo smoke: %s", exc)
        return 2

    write_demo_smoke_outputs(
        report,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    logger.info("JSON report: %s", args.json_report)
    logger.info("Markdown report: %s", args.markdown_report)
    if report["status"] != "passed":
        _log_failures(report)
        return 1

    logger.info("Demo acceptance passed. This remains an operational smoke, not Benchmark v1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
