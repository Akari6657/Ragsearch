"""Start CiteQuest and exercise its public routes over real localhost HTTP."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.http_smoke import run_http_smoke, write_http_smoke_outputs


DEFAULT_DB = PROJECT_ROOT / "data" / "indexes" / "demo10k" / "metadata.sqlite"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "demo10k" / "faiss"
DEFAULT_LOG = PROJECT_ROOT / "reports" / "demo10k_http_server.log"
DEFAULT_JSON = PROJECT_ROOT / "reports" / "demo10k_http_smoke.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "reports" / "demo10k_http_smoke.md"

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real HTTP smoke for CiteQuest")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--query", default="machine learning")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--server-log", default=str(DEFAULT_LOG))
    parser.add_argument("--json-report", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown-report", default=str(DEFAULT_MARKDOWN))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Starting real HTTP smoke on http://%s:%d", args.host, args.port)
    try:
        report = run_http_smoke(
            db_path=args.db,
            index_dir=args.index_dir,
            log_path=args.server_log,
            host=args.host,
            port=args.port,
            query=args.query,
            top_k=args.top_k,
            alpha=args.alpha,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
        )
    except (OSError, ValueError) as exc:
        logger.error("Cannot run HTTP smoke: %s", exc)
        return 2

    write_http_smoke_outputs(
        report,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
    )
    logger.info("JSON report: %s", args.json_report)
    logger.info("Markdown report: %s", args.markdown_report)
    if report["status"] != "passed":
        for error in report["errors"]:
            logger.error("HTTP smoke failed: %s", error)
        return 1
    logger.info("Real HTTP smoke passed and Uvicorn stopped cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
