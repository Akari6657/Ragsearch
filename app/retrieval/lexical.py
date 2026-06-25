"""
Lexical retriever: BM25-ranked full-text search via SQLite FTS5.

Wraps FTS5 MATCH queries and returns typed SearchResult objects.
The API layer calls this — it never writes SQL directly.

Usage:
    from app.retrieval.lexical import search_lexical
    results = search_lexical("neural network", top_k=10)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from app.core.schemas import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL template
# ---------------------------------------------------------------------------

SEARCH_SQL = """
SELECT
    c.chunk_id,
    c.paper_id,
    p.title,
    p.year,
    p.venue,
    p.authors_json,
    bm25(chunk_fts) AS score,
    snippet(chunk_fts, 2, '<mark>', '</mark>', '...', 40) AS snippet
FROM chunk_fts c
JOIN papers p ON c.paper_id = p.paper_id
WHERE chunk_fts MATCH ?
ORDER BY score
LIMIT ?
"""

# FTS5 MATCH syntax has special characters that must be escaped.
# We wrap the user query in double-quotes for phrase search, which
# also handles most special chars safely.  For complex queries we
# sanitize minimally.
FTS5_SPECIAL = set("()*^\"-:{}[]")


def _sanitize_query(query: str) -> str:
    """Escape FTS5 special characters so user input doesn't break the MATCH syntax."""
    # Simple approach: strip FTS5 operators, then wrap in quotes for phrase match
    cleaned = "".join(ch for ch in query if ch not in FTS5_SPECIAL)
    # Wrap in quotes so it's treated as a phrase
    return f'"{cleaned.strip()}"' if cleaned.strip() else '""'


def _parse_authors(authors_json: str | None) -> list[str]:
    """Deserialize authors from the JSON column."""
    if not authors_json:
        return []
    try:
        parsed = json.loads(authors_json)
        if isinstance(parsed, list):
            return [a for a in parsed if isinstance(a, str)]
    except json.JSONDecodeError:
        pass
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_lexical(
    query: str,
    top_k: int = 10,
    db_path: str | Path = "data/indexes/metadata.sqlite",
) -> list[SearchResult]:
    """Search papers by keyword using BM25-ranked FTS5.

    Args:
        query: Free-text search query.
        top_k: Number of results to return (default 10).
        db_path: Path to the metadata SQLite database.

    Returns:
        List of SearchResult, ordered by relevance (best first).
        Returns empty list if the index hasn't been built or the query
        matches nothing.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning("Database not found: %s — run build_metadata_db.py first", db_path)
        return []

    safe_query = _sanitize_query(query)
    if not safe_query or safe_query == '""':
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Check if FTS5 table exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
        ).fetchone()
        if not exists:
            logger.warning("FTS5 index not found — run build_fts.py first")
            return []

        rows = conn.execute(SEARCH_SQL, (safe_query, top_k)).fetchall()
    except sqlite3.OperationalError as exc:
        logger.error("FTS5 search failed: %s", exc)
        return []
    finally:
        conn.close()

    results: list[SearchResult] = []
    for row in rows:
        # snippet() returns the matching text; join with the full text
        # context if needed — for now the snippet is informative enough.
        snippet = row["snippet"] or ""
        results.append(
            SearchResult(
                paper_id=row["paper_id"],
                chunk_id=row["chunk_id"],
                title=row["title"],
                year=row["year"],
                venue=row["venue"],
                authors=_parse_authors(row["authors_json"]),
                score=round(float(row["score"]), 4),
                snippet=snippet,
            )
        )

    logger.debug("search_lexical('%s', top_k=%d) → %d results", query, top_k, len(results))
    return results
