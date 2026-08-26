"""
Lexical retriever: BM25-ranked full-text search via SQLite FTS5.

Default behavior: OR search — each term contributes to BM25 independently.
Quoted phrases ("...") get a post-retrieval boost:
    boost = 1 + c * log(1 + phrase_count)
    c ≈ 0.3  →  first hit +21%, diminishing after.

Usage:
    from app.retrieval.lexical import search_lexical
    results = search_lexical('computer "neural network" optimization', top_k=10)
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from pathlib import Path

from app.core.schemas import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHRASE_BOOST_C = 0.3
"""Phrase boost coefficient. Controls how much each phrase hit increases the score."""

FETCH_MULTIPLIER = 3
"""Fetch top_k * N candidates from FTS5 to allow phrase boost to re-rank."""

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

SEARCH_SQL = """
SELECT
    c.chunk_id,
    c.paper_id,
    p.title,
    p.year,
    p.venue,
    p.authors_json,
    p.abstract,
    c.chunk_text,
    bm25(chunk_fts) AS score,
    snippet(chunk_fts, 2, '<mark>', '</mark>', '...', 40) AS snippet
FROM chunk_fts c
JOIN papers p ON c.paper_id = p.paper_id
WHERE chunk_fts MATCH ?
{year_clauses}
ORDER BY score
LIMIT ?
"""

# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

# Matches "quoted text" segments
_QUOTED_RE = re.compile(r'"([^"]*)"')

# FTS5 barewords accept Unicode alphanumerics and underscores. Extracting only
# those tokens keeps punctuation and query-language operators out of MATCH.
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _parse_query(query: str) -> tuple[str, list[str]]:
    """Parse user query into FTS5 OR terms + phrase list for post-boost.

    Args:
        query: Raw user input, e.g. 'computer "neural network" optimization'.

    Returns:
        (fts5_query, phrases) where fts5_query is safe for FTS5 MATCH
        (all words as OR terms) and phrases is a list of lowercased
        phrase strings to boost.
    """
    # Extract quoted phrases
    phrases = [m.lower() for m in _QUOTED_RE.findall(query)]

    # Tokenize instead of maintaining a partial punctuation denylist. This
    # mirrors FTS5's default treatment of punctuation and makes arbitrary user
    # input safe to pass to MATCH. Lowercasing also prevents words such as OR
    # and NEAR from being interpreted as FTS5 query operators.
    safe_words = _TOKEN_RE.findall(query.lower())

    # FTS5 treats whitespace as implicit AND. Use the explicit operator so
    # each query term can independently contribute a BM25 match.
    fts5_query = " OR ".join(safe_words) if safe_words else ""

    return fts5_query, phrases


def _count_phrases(text: str, phrases: list[str]) -> int:
    """Count total occurrences of all phrases in text (case-insensitive)."""
    if not phrases or not text:
        return 0
    text_lower = text.lower()
    total = 0
    for phrase in phrases:
        total += text_lower.count(phrase)
    return total


def _phrase_boost(phrase_count: int, c: float = PHRASE_BOOST_C) -> float:
    """Compute the score multiplier for a given number of phrase hits.

    boost = 1 + c * ln(1 + count)

    0 hits  → 1.0    (no change)
    1 hit   → 1.21
    2 hits  → 1.33
    3 hits  → 1.42
    5 hits  → 1.54
    10 hits → 1.72
    """
    if phrase_count <= 0:
        return 1.0
    return 1.0 + c * math.log(1 + phrase_count)


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
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[SearchResult]:
    """Search papers by keyword using BM25-ranked FTS5 + phrase boost.

    Workflow:
    1. Parse query into OR terms + quoted phrases.
    2. FTS5 OR search fetches top_k * N candidates.
    3. Count phrase hits in each candidate's chunk_text.
    4. Apply phrase boost to the raw lower-is-better FTS5 score.
    5. Negate it at the module boundary so public scores are higher-is-better.
    6. Re-sort and return top_k.

    Example: 'computer "neural network" optimization'
        → OR(computer, neural, network, optimization) via FTS5 BM25
        → + phrase boost for "neural network" in chunk_text
        → papers with the exact phrase rank higher.

    Args:
        query: Free-text search query. Use "..." for phrase boost.
        top_k: Number of results to return (default 10).
        db_path: Path to the metadata SQLite database.
        year_from: Inclusive earliest publication year. Unknown years are excluded.
        year_to: Inclusive latest publication year. Unknown years are excluded.

    Returns:
        List of SearchResult, ordered by relevance (best first).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning("Database not found: %s — run build_metadata_db.py first", db_path)
        return []

    fts5_query, phrases = _parse_query(query)
    if not fts5_query:
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
        ).fetchone()
        if not exists:
            logger.warning("FTS5 index not found — run build_fts.py first")
            return []

        # Fetch more candidates than needed — phrase boost will re-rank.
        # Year values remain bound parameters; only fixed SQL clauses are composed.
        fetch_k = max(top_k * FETCH_MULTIPLIER, 30)
        year_clauses: list[str] = []
        params: list[str | int] = [fts5_query]
        if year_from is not None:
            year_clauses.append("AND p.year >= ?")
            params.append(year_from)
        if year_to is not None:
            year_clauses.append("AND p.year <= ?")
            params.append(year_to)
        params.append(fetch_k)

        sql = SEARCH_SQL.format(year_clauses="\n".join(year_clauses))
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        logger.error("FTS5 search failed: %s", exc)
        return []
    finally:
        conn.close()

    # — Build results with phrase boost —————————————————————————————————
    candidates: list[tuple[float, SearchResult]] = []
    for row in rows:
        snippet = row["snippet"] or ""
        chunk_text = row["chunk_text"] or ""

        # Count phrase hits and compute boost
        phrase_count = _count_phrases(chunk_text, phrases)
        boost = _phrase_boost(phrase_count)

        raw_score = float(row["score"])
        boosted_raw_score = raw_score * boost
        relevance_score = -boosted_raw_score

        sr = SearchResult(
            paper_id=row["paper_id"],
            chunk_id=row["chunk_id"],
            title=row["title"],
            year=row["year"],
            venue=row["venue"],
            authors=_parse_authors(row["authors_json"]),
            score=relevance_score,
            snippet=snippet,
            abstract=(row["abstract"] or "")[:300],
        )
        candidates.append((relevance_score, sr))

    # — Re-sort by public relevance score and truncate ————————————————
    candidates.sort(key=lambda x: x[0], reverse=True)
    results = [sr for _, sr in candidates[:top_k]]

    logger.debug(
        "search_lexical('%s', top_k=%d, years=%s..%s) → %d candidates → %d results (%d phrases)",
        query,
        top_k,
        year_from,
        year_to,
        len(candidates),
        len(results),
        len(phrases),
    )
    return results
