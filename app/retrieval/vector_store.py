"""
Vector retriever: semantic search via FAISS + BGE-M3 embeddings.

Loads the FAISS index and ID mapping from disk, encodes the query with the
same embedding model, and returns SearchResult objects — same interface
as lexical.py so hybrid.py can call both transparently.

Usage:
    from app.retrieval.vector_store import search_vector
    results = search_vector("neural network", top_k=10)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from app.core.schemas import SearchResult
from app.retrieval.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_DB = Path("data/indexes/metadata.sqlite")
DEFAULT_INDEX_DIR = Path("data/indexes/faiss")
INDEX_FILE = "index.faiss"
ID_MAP_FILE = "id_map.json"

# ---------------------------------------------------------------------------
# Internal state (loaded once, reused across queries)
# ---------------------------------------------------------------------------

_index: Any = None  # faiss.Index
_id_map: list[dict] = []
_model: EmbeddingModel | None = None


def _load_index(index_dir: Path):
    """Lazy-load FAISS index + ID mapping + embedding model."""
    global _index, _id_map, _model

    if _index is not None:
        return  # already loaded

    import faiss

    index_path = index_dir / INDEX_FILE
    id_map_path = index_dir / ID_MAP_FILE

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Run build_faiss.py first."
        )

    logger.info("Loading FAISS index from %s ...", index_path)
    _index = faiss.read_index(str(index_path))

    with open(id_map_path, "r", encoding="utf-8") as f:
        _id_map = json.load(f)

    _model = EmbeddingModel()

    logger.info(
        "FAISS ready: %d vectors × %d dims, %d id-map entries",
        _index.ntotal,
        _index.d,
        len(_id_map),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_vector(
    query: str,
    top_k: int = 10,
    db_path: str | Path = DEFAULT_DB,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> list[SearchResult]:
    """Semantic search over paper chunks using FAISS.

    Args:
        query: Natural-language query (any length).
        top_k: Number of results.
        db_path: Path to metadata SQLite DB (for hydrating results).
        index_dir: Directory containing index.faiss and id_map.json.

    Returns:
        List of SearchResult sorted by cosine similarity (best first).
    """
    db_path = Path(db_path)
    index_dir = Path(index_dir)

    # — 1. Load index (lazy, cached) ———————————————————————————————————
    try:
        _load_index(index_dir)
    except FileNotFoundError:
        logger.warning("FAISS index missing; returning empty results.")
        return []

    # — 2. Encode query ————————————————————————————————————————————————
    query_vec = _model.encode([query], show_progress=False)  # (1, dim)
    query_vec = query_vec.astype(np.float32)

    # — 3. FAISS search —————————————————————————————————————————————————
    scores, faiss_ids = _index.search(query_vec, top_k)  # (1, k)
    scores = scores[0]
    faiss_ids = faiss_ids[0]

    # — 4. Map FAISS IDs → chunk/paper IDs —————————————————————————————
    hits: list[dict] = []
    for score, fid in zip(scores, faiss_ids):
        if fid < 0 or fid >= len(_id_map):
            continue  # FAISS returns -1 for "no more results"
        entry = _id_map[fid]
        hits.append({
            "score": float(score),
            "chunk_id": entry["chunk_id"],
            "paper_id": entry["paper_id"],
        })

    if not hits:
        return []

    # — 5. Hydrate metadata from SQLite —————————————————————————————————
    if not db_path.exists():
        logger.warning("Metadata DB not found; returning results without titles.")
        return [
            SearchResult(
                paper_id=h["paper_id"],
                chunk_id=h["chunk_id"],
                title="",
                year=None,
                venue=None,
                authors=[],
                score=h["score"],
                snippet="",
                abstract="",
            )
            for h in hits
        ]

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        chunk_ids = [h["chunk_id"] for h in hits]
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"""SELECT p.paper_id, p.title, p.year, p.venue, p.authors_json, p.abstract,
                       c.chunk_text
                FROM chunks c
                JOIN papers p ON c.paper_id = p.paper_id
                WHERE c.chunk_id IN ({placeholders})""",
            chunk_ids,
        ).fetchall()
    finally:
        conn.close()

    # Build lookup: chunk_id → metadata
    meta: dict[str, dict] = {}
    for row in rows:
        authors = []
        try:
            authors = json.loads(row["authors_json"])
        except (json.JSONDecodeError, TypeError):
            pass
        meta[row["paper_id"]] = {
            "title": row["title"],
            "year": row["year"],
            "venue": row["venue"],
            "authors": authors,
            "abstract": (row["abstract"] or "")[:300],
        }

    # — 6. Build SearchResult list ——————————————————————————————————————
    results: list[SearchResult] = []
    for h in hits:
        m = meta.get(h["paper_id"], {})
        results.append(
            SearchResult(
                paper_id=h["paper_id"],
                chunk_id=h["chunk_id"],
                title=m.get("title", ""),
                year=m.get("year"),
                venue=m.get("venue"),
                authors=m.get("authors", []),
                score=round(h["score"], 4),
                snippet="",  # FAISS has no snippet; hybrid layer can fill from lexical
                abstract=m.get("abstract", ""),
            )
        )

    logger.debug("search_vector('%s', top_k=%d) → %d results", query, top_k, len(results))
    return results
