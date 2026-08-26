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

FILTER_FETCH_MULTIPLIER = 4
MIN_FILTER_FETCH = 50
HYDRATION_BATCH_SIZE = 500

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

    # IVF: set nprobe (number of clusters to search). Higher = more accurate but slower.
    if hasattr(_index, "nprobe") and hasattr(_index, "nlist"):
        _index.nprobe = min(_index.nlist // 4, 64)
        logger.info("IVF nlist=%d, nprobe=%d", _index.nlist, _index.nprobe)

    with open(id_map_path, "r", encoding="utf-8") as f:
        _id_map = json.load(f)

    _model = EmbeddingModel()

    logger.info(
        "FAISS ready: %d vectors × %d dims, %d id-map entries",
        _index.ntotal,
        _index.d,
        len(_id_map),
    )


def _search_index(query_vec: np.ndarray, candidate_k: int) -> list[dict]:
    """Return valid FAISS hits in similarity order."""
    scores, faiss_ids = _index.search(query_vec, candidate_k)

    hits: list[dict] = []
    for score, fid in zip(scores[0], faiss_ids[0]):
        if fid < 0 or fid >= len(_id_map):
            continue
        entry = _id_map[fid]
        hits.append(
            {
                "score": float(score),
                "chunk_id": entry["chunk_id"],
                "paper_id": entry["paper_id"],
            }
        )
    return hits


def _hydrate_metadata(
    conn: sqlite3.Connection,
    hits: list[dict],
    *,
    year_from: int | None = None,
    year_to: int | None = None,
) -> dict[str, dict]:
    """Load metadata for hits, optionally retaining only an inclusive year range."""
    metadata: dict[str, dict] = {}
    year_clauses: list[str] = []
    year_params: list[int] = []
    if year_from is not None:
        year_clauses.append("AND p.year >= ?")
        year_params.append(year_from)
    if year_to is not None:
        year_clauses.append("AND p.year <= ?")
        year_params.append(year_to)

    chunk_ids = [hit["chunk_id"] for hit in hits]
    for offset in range(0, len(chunk_ids), HYDRATION_BATCH_SIZE):
        batch = chunk_ids[offset : offset + HYDRATION_BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch)
        sql = f"""SELECT c.chunk_id, p.paper_id, p.title, p.year, p.venue,
                          p.authors_json, p.abstract
                   FROM chunks c
                   JOIN papers p ON c.paper_id = p.paper_id
                   WHERE c.chunk_id IN ({placeholders})
                   {' '.join(year_clauses)}"""
        rows = conn.execute(sql, [*batch, *year_params]).fetchall()

        for row in rows:
            authors = []
            try:
                parsed_authors = json.loads(row["authors_json"])
                if isinstance(parsed_authors, list):
                    authors = [author for author in parsed_authors if isinstance(author, str)]
            except (json.JSONDecodeError, TypeError):
                pass
            metadata[row["chunk_id"]] = {
                "title": row["title"],
                "year": row["year"],
                "venue": row["venue"],
                "authors": authors,
                "abstract": (row["abstract"] or "")[:300],
            }
    return metadata


def _build_results(
    hits: list[dict],
    metadata: dict[str, dict],
    *,
    include_missing_metadata: bool,
) -> list[SearchResult]:
    """Build results in FAISS rank order from hydrated metadata."""
    results: list[SearchResult] = []
    for hit in hits:
        item = metadata.get(hit["chunk_id"])
        if item is None and not include_missing_metadata:
            continue
        item = item or {}
        results.append(
            SearchResult(
                paper_id=hit["paper_id"],
                chunk_id=hit["chunk_id"],
                title=item.get("title", ""),
                year=item.get("year"),
                venue=item.get("venue"),
                authors=item.get("authors", []),
                score=hit["score"],
                snippet="",
                abstract=item.get("abstract", ""),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_vector(
    query: str,
    top_k: int = 10,
    db_path: str | Path = DEFAULT_DB,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[SearchResult]:
    """Semantic search over paper chunks using FAISS.

    Args:
        query: Natural-language query (any length).
        top_k: Number of results.
        db_path: Path to metadata SQLite DB (for hydrating results).
        index_dir: Directory containing index.faiss and id_map.json.
        year_from: Inclusive earliest publication year. Unknown years are excluded.
        year_to: Inclusive latest publication year. Unknown years are excluded.

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

    # — 3. Handle missing metadata —————————————————————————————————————
    has_year_filter = year_from is not None or year_to is not None
    if not db_path.exists():
        if has_year_filter:
            logger.warning("Metadata DB not found; cannot apply publication-year filter.")
            return []

        hits = _search_index(query_vec, top_k)
        logger.warning("Metadata DB not found; returning results without titles.")
        return _build_results(hits, {}, include_missing_metadata=True)

    # — 4. Search and hydrate ———————————————————————————————————————————
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if not has_year_filter:
            hits = _search_index(query_vec, top_k)
            metadata = _hydrate_metadata(conn, hits)
            results = _build_results(hits, metadata, include_missing_metadata=True)
        else:
            total_candidates = min(int(_index.ntotal), len(_id_map))
            if total_candidates <= 0:
                return []

            candidate_k = min(
                total_candidates,
                max(top_k * FILTER_FETCH_MULTIPLIER, MIN_FILTER_FETCH),
            )
            seen_chunk_ids: set[str] = set()
            matching_metadata: dict[str, dict] = {}
            results: list[SearchResult] = []

            while candidate_k > 0:
                hits = _search_index(query_vec, candidate_k)
                new_hits = [
                    hit for hit in hits if hit["chunk_id"] not in seen_chunk_ids
                ]
                seen_chunk_ids.update(hit["chunk_id"] for hit in new_hits)
                matching_metadata.update(
                    _hydrate_metadata(
                        conn,
                        new_hits,
                        year_from=year_from,
                        year_to=year_to,
                    )
                )
                results = _build_results(
                    hits,
                    matching_metadata,
                    include_missing_metadata=False,
                )[:top_k]

                if len(results) >= top_k or candidate_k >= total_candidates:
                    break

                next_candidate_k = min(total_candidates, candidate_k * 2)
                if next_candidate_k == candidate_k or not new_hits:
                    break
                candidate_k = next_candidate_k
    finally:
        conn.close()

    logger.debug(
        "search_vector('%s', top_k=%d, years=%s..%s) → %d results",
        query,
        top_k,
        year_from,
        year_to,
        len(results),
    )
    return results
