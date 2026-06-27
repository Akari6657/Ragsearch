"""
Paper normalizer: validate and clean raw paper records.

Takes raw dicts from loader, ensures every field is well-typed and
non-empty where required. Invalid records are dropped (return None).

This is the quality gate — downstream modules never see dirty data.

Usage:
    from app.ingestion.loader import load_papers
    from app.ingestion.normalize import normalize

    for raw in load_papers(path):
        paper = normalize(raw)
        if paper is not None:
            ...
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_str(value: Any) -> str:
    """Coerce to str and strip whitespace."""
    if not isinstance(value, str):
        return str(value) if value else ""
    return value.strip()


def _to_str_list(value: Any) -> list[str]:
    """Normalize a value into a list of non-empty strings.

    Handles common variations:
    - list[str]       → keep as-is (strip each)
    - "Author1, Author2" → split by comma
    - single str      → wrap in list
    - None / empty    → empty list
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [s.strip() for s in value if isinstance(s, str) and s.strip()]
    if isinstance(value, str):
        parts = [a.strip() for a in value.split(",") if a.strip()]
        return parts
    return []


def _to_int(value: Any, *, default: int | None = None, min_val: int | None = None) -> int | None:
    """Try to cast to int; return default on failure."""
    if value is None:
        return default
    try:
        v = int(value)
    except (ValueError, TypeError):
        return default
    if min_val is not None and v < min_val:
        return min_val
    return v


def _to_bool(value: Any) -> bool:
    """Coerce to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "t")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------

def normalize(record: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and clean a single raw paper record.

    Args:
        record: Raw dict from loader (expected to have paper_id, title, ...).

    Returns:
        Cleaned dict with guaranteed types, or None if the record is invalid
        (missing paper_id or title after cleaning).
    """
    # — required fields ——————————————————————————————————————————————
    paper_id = _clean_str(record.get("paper_id", ""))
    if not paper_id:
        logger.warning("Skipping record: empty or missing paper_id")
        return None

    title = _clean_str(record.get("title", ""))
    if not title:
        logger.warning("Skipping record %s: empty or missing title", paper_id)
        return None

    # — optional fields with defaults ——————————————————————————————————
    abstract = _clean_str(record.get("abstract", ""))
    full_text = _clean_str(record.get("full_text", ""))  # peS2o full-text
    year = _to_int(record.get("year"))
    venue = _clean_str(record.get("venue", "")) or None
    authors = _to_str_list(record.get("authors"))
    # support both "concepts" (arXiv) and "fields_of_study" (peS2o)
    concepts = _to_str_list(record.get("concepts") or record.get("fields_of_study"))
    doi = _clean_str(record.get("doi", "")) or None
    url = _clean_str(record.get("url", "")) or None
    citation_count = _to_int(record.get("citation_count", 0), default=0, min_val=0) or 0
    open_access = _to_bool(record.get("open_access", False))

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "full_text": full_text,
        "year": year,
        "venue": venue,
        "authors": authors,
        "concepts": concepts,
        "doi": doi,
        "url": url,
        "citation_count": citation_count,
        "open_access": open_access,
    }
