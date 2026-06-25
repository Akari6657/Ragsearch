"""
Paper loader: read a JSONL file and yield paper dicts.

Part of the ingestion pipeline. This is the entry point —
all downstream modules consume its output.

Usage:
    from app.ingestion.loader import load_papers
    for paper in load_papers("data/raw/arxiv_cs_sample.jsonl"):
        ...
"""

import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def load_papers(path: str | Path) -> Iterator[dict]:
    """Read a JSONL file of paper records, yielding one dict per valid line.

    Args:
        path: Path to a .jsonl file. Each line should be a JSON object
              with paper fields (paper_id, title, abstract, ...).

    Yields:
        dict: Parsed paper record.

    Skips:
        - Empty lines
        - Lines that fail to parse as JSON (logged as warning)
        - Lines with missing or empty 'paper_id' (logged as warning)

    This function is memory-efficient: it streams one line at a time
    regardless of file size.
    """
    path = Path(path)

    total = 0
    kept = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            total += 1

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d: invalid JSON — %s", line_num, exc)
                continue

            if not isinstance(record, dict):
                logger.warning("Skipping line %d: expected dict, got %s", line_num, type(record).__name__)
                continue

            if not record.get("paper_id"):
                logger.warning("Skipping line %d: missing or empty paper_id", line_num)
                continue

            kept += 1
            yield record

    logger.info("load_papers: %d/%d records loaded from %s", kept, total, path.name)
