"""
Chunk generator: turn a normalized paper into searchable text segment(s).

Chunk text is "Title: ...\nAbstract: ..." — the semantically dense core.
Metadata fields (year, venue, concepts, authors) stay in the papers table
for filtering, not in chunk text, to keep vector embeddings clean.

v0.1: one chunk per paper.
v0.2+: per-section chunking when full text is available.

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
    """Assemble searchable text: Title + Abstract only.

    Year, venue, concepts, and authors are deliberately excluded —
    they live in the papers table for structured filtering,
    not in the chunk text that gets embedded.
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    if title and abstract:
        return f"Title: {title}\nAbstract: {abstract}"
    elif title:
        return f"Title: {title}"
    else:
        return f"Abstract: {abstract}"


def chunk_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Create a single chunk from a normalized paper.

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
