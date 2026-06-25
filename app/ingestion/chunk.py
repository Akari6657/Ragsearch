"""
Chunk generator: turn a normalized paper into searchable text segment(s).

In v0.1, each paper produces exactly one "metadata" chunk that combines
title, year, venue, concepts, and abstract into a single searchable text.
Future versions can add per-section or per-paragraph chunking.

Usage:
    from app.ingestion.chunk import chunk_paper
    chunk = chunk_paper(normalized_paper)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token count (≈4 chars per token for English text)."""
    return max(1, len(text) // 4)


def _build_chunk_text(paper: dict[str, Any]) -> str:
    """Assemble the searchable text from paper fields."""
    parts: list[tuple[str, str]] = [
        ("Title", paper.get("title", "")),
    ]

    year = paper.get("year")
    if year is not None:
        parts.append(("Year", str(year)))

    venue = paper.get("venue")
    if venue:
        parts.append(("Venue", venue))

    concepts = paper.get("concepts")
    if concepts:
        parts.append(("Concepts", ", ".join(concepts)))

    abstract = paper.get("abstract")
    if abstract:
        parts.append(("Abstract", abstract))

    text = "\n".join(f"{label}: {value}" for label, value in parts)
    return text


def chunk_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Create a single metadata chunk from a normalized paper.

    Args:
        paper: Normalized paper dict (from normalize.py).

    Returns:
        Chunk dict with chunk_id, paper_id, chunk_text, chunk_type, token_count.
    """
    paper_id = paper["paper_id"]
    chunk_text = _build_chunk_text(paper)

    chunk_id = f"{paper_id}_default"

    logger.debug("Chunking %s → %s (%d chars)", paper_id, chunk_id, len(chunk_text))

    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "chunk_text": chunk_text,
        "chunk_type": "metadata",
        "token_count": _estimate_tokens(chunk_text),
    }
