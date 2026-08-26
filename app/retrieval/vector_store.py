"""
Vector retriever: semantic search via FAISS + BGE-M3 embeddings.

Loads and validates the requested FAISS index and ID mapping, caches one
artifact set by path and file signature, encodes the query with the matching
embedding model, and returns SearchResult objects — the same interface as
lexical.py so hybrid.py can call both transparently.

Usage:
    from app.retrieval.vector_store import search_vector
    results = search_vector("neural network", top_k=10)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from app.core.schemas import SearchResult
from app.retrieval.embeddings import DEFAULT_MODEL_NAME, EmbeddingModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_DB = Path("data/indexes/metadata.sqlite")
DEFAULT_INDEX_DIR = Path("data/indexes/faiss")
INDEX_FILE = "index.faiss"
ID_MAP_FILE = "id_map.json"
BUILD_META_FILE = "build_meta.json"

FILTER_FETCH_MULTIPLIER = 4
MIN_FILTER_FETCH = 50
HYDRATION_BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Internal cache state
# ---------------------------------------------------------------------------

class FaissArtifactError(RuntimeError):
    """Raised when persisted FAISS artifacts are internally inconsistent."""


@dataclass(frozen=True)
class _FileSignature:
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _IndexCacheKey:
    index_dir: Path
    index_file: _FileSignature
    id_map_file: _FileSignature
    build_meta_file: _FileSignature | None


@dataclass(frozen=True)
class _LoadedIndex:
    key: _IndexCacheKey
    index: Any
    id_map: tuple[dict[str, Any], ...]
    model_name: str
    dimension: int


@dataclass(frozen=True)
class _LoadedModel:
    key: tuple[str, int]
    model: EmbeddingModel


_cache_lock = Lock()
_index_cache: _LoadedIndex | None = None
_model_cache: _LoadedModel | None = None


def _file_signature(path: Path) -> _FileSignature:
    stat = path.stat()
    return _FileSignature(
        device=stat.st_dev,
        inode=stat.st_ino,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _build_cache_key(index_dir: Path) -> _IndexCacheKey:
    resolved_dir = index_dir.expanduser().resolve()
    index_path = resolved_dir / INDEX_FILE
    id_map_path = resolved_dir / ID_MAP_FILE
    build_meta_path = resolved_dir / BUILD_META_FILE

    for path in (index_path, id_map_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"FAISS artifact not found at {path}. Run build_faiss.py first."
            )

    return _IndexCacheKey(
        index_dir=resolved_dir,
        index_file=_file_signature(index_path),
        id_map_file=_file_signature(id_map_path),
        build_meta_file=(
            _file_signature(build_meta_path) if build_meta_path.is_file() else None
        ),
    )


def _read_json(path: Path, *, label: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FaissArtifactError(f"Cannot read {label} at {path}: {exc}") from exc


def _validate_id_map(
    raw_id_map: Any,
    *,
    expected_count: int,
    path: Path,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_id_map, list):
        raise FaissArtifactError(f"FAISS ID map at {path} must be a JSON list")
    if len(raw_id_map) != expected_count:
        raise FaissArtifactError(
            "FAISS artifact count mismatch: "
            f"index has {expected_count} vectors but ID map has {len(raw_id_map)} entries"
        )

    for faiss_id, entry in enumerate(raw_id_map):
        if (
            not isinstance(entry, dict)
            or entry.get("faiss_id") != faiss_id
            or not isinstance(entry.get("chunk_id"), str)
            or not isinstance(entry.get("paper_id"), str)
        ):
            raise FaissArtifactError(
                f"Invalid FAISS ID-map entry at position {faiss_id} in {path}"
            )
    return tuple(raw_id_map)


def _read_build_metadata(
    path: Path,
    *,
    index_count: int,
    index_dimension: int,
) -> str:
    if not path.is_file():
        return DEFAULT_MODEL_NAME

    metadata = _read_json(path, label="FAISS build metadata")
    if not isinstance(metadata, dict):
        raise FaissArtifactError(f"FAISS build metadata at {path} must be a JSON object")

    recorded_count = metadata.get("num_vectors")
    if recorded_count is not None and recorded_count != index_count:
        raise FaissArtifactError(
            "FAISS build metadata count mismatch: "
            f"metadata records {recorded_count!r}, index has {index_count}"
        )

    recorded_dimension = metadata.get("vector_dim")
    if recorded_dimension is not None and recorded_dimension != index_dimension:
        raise FaissArtifactError(
            "FAISS build metadata dimension mismatch: "
            f"metadata records {recorded_dimension!r}, index has {index_dimension}"
        )

    model_name = metadata.get("embedding_model", DEFAULT_MODEL_NAME)
    if not isinstance(model_name, str) or not model_name.strip():
        raise FaissArtifactError(
            f"FAISS build metadata at {path} has an invalid embedding_model"
        )
    return model_name


def _load_index(index_dir: Path) -> _LoadedIndex:
    """Return a validated cached index for the requested artifact directory."""
    global _index_cache

    with _cache_lock:
        key = _build_cache_key(index_dir)
        if _index_cache is not None and _index_cache.key == key:
            return _index_cache

        import faiss

        index_path = key.index_dir / INDEX_FILE
        id_map_path = key.index_dir / ID_MAP_FILE
        build_meta_path = key.index_dir / BUILD_META_FILE

        logger.info("Loading FAISS index from %s ...", index_path)
        index = faiss.read_index(str(index_path))
        index_count = int(index.ntotal)
        index_dimension = int(index.d)

        if index_dimension <= 0:
            raise FaissArtifactError(
                f"FAISS index at {index_path} has invalid dimension {index_dimension}"
            )
        if hasattr(index, "is_trained") and not index.is_trained:
            raise FaissArtifactError(f"FAISS index at {index_path} is not trained")

        raw_id_map = _read_json(id_map_path, label="FAISS ID map")
        id_map = _validate_id_map(
            raw_id_map,
            expected_count=index_count,
            path=id_map_path,
        )
        model_name = _read_build_metadata(
            build_meta_path,
            index_count=index_count,
            index_dimension=index_dimension,
        )

        # IVF: set nprobe (number of clusters to search). Higher is more
        # accurate but slower.
        if hasattr(index, "nprobe") and hasattr(index, "nlist"):
            index.nprobe = min(index.nlist // 4, 64)
            logger.info("IVF nlist=%d, nprobe=%d", index.nlist, index.nprobe)

        loaded = _LoadedIndex(
            key=key,
            index=index,
            id_map=id_map,
            model_name=model_name,
            dimension=index_dimension,
        )
        _index_cache = loaded

        logger.info(
            "FAISS ready: %d vectors × %d dims, %d id-map entries, model=%s",
            index_count,
            index_dimension,
            len(id_map),
            model_name,
        )
        return loaded


def _get_embedding_model(model_name: str, dimension: int) -> EmbeddingModel:
    """Reuse one embedding wrapper only for a compatible model and dimension."""
    global _model_cache

    key = (model_name, dimension)
    with _cache_lock:
        if _model_cache is not None and _model_cache.key == key:
            return _model_cache.model

        model = EmbeddingModel(model_name=model_name)
        model_dimension = int(model.dim)
        if model_dimension != dimension:
            raise FaissArtifactError(
                "Embedding model dimension mismatch: "
                f"model {model_name!r} reports {model_dimension}, index expects {dimension}"
            )
        _model_cache = _LoadedModel(key=key, model=model)
        return model


def _reset_cache() -> None:
    """Clear runtime caches; intended for isolated tests and process teardown."""
    global _index_cache, _model_cache

    with _cache_lock:
        _index_cache = None
        _model_cache = None


def _search_index(
    loaded: _LoadedIndex,
    query_vec: np.ndarray,
    candidate_k: int,
) -> list[dict]:
    """Return valid FAISS hits in similarity order."""
    scores, faiss_ids = loaded.index.search(query_vec, candidate_k)

    hits: list[dict] = []
    for score, fid in zip(scores[0], faiss_ids[0]):
        if fid < 0 or fid >= len(loaded.id_map):
            continue
        entry = loaded.id_map[fid]
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
        loaded = _load_index(index_dir)
    except FileNotFoundError:
        logger.warning("FAISS index missing; returning empty results.")
        return []

    # — 2. Encode query ————————————————————————————————————————————————
    model = _get_embedding_model(loaded.model_name, loaded.dimension)
    query_vec = np.asarray(
        model.encode([query], show_progress=False),
        dtype=np.float32,
    )
    if query_vec.shape != (1, loaded.dimension):
        raise FaissArtifactError(
            "Embedding dimension mismatch: "
            f"model {loaded.model_name!r} returned shape {query_vec.shape}, "
            f"index expects (1, {loaded.dimension})"
        )

    # — 3. Handle missing metadata —————————————————————————————————————
    has_year_filter = year_from is not None or year_to is not None
    if not db_path.exists():
        if has_year_filter:
            logger.warning("Metadata DB not found; cannot apply publication-year filter.")
            return []

        hits = _search_index(loaded, query_vec, top_k)
        logger.warning("Metadata DB not found; returning results without titles.")
        return _build_results(hits, {}, include_missing_metadata=True)

    # — 4. Search and hydrate ———————————————————————————————————————————
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if not has_year_filter:
            hits = _search_index(loaded, query_vec, top_k)
            metadata = _hydrate_metadata(conn, hits)
            results = _build_results(hits, metadata, include_missing_metadata=True)
        else:
            total_candidates = min(int(loaded.index.ntotal), len(loaded.id_map))
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
                hits = _search_index(loaded, query_vec, candidate_k)
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
