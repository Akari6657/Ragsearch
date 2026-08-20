"""Build a resumable FAISS vector index from SQLite chunks.

The expensive embedding phase is checkpointed to a NumPy memmap. Re-running
the same command resumes from the last completed chunk, while ``--restart``
explicitly discards an incompatible or unwanted checkpoint.

Usage:
    python scripts/build_faiss.py --db data/indexes/metadata.sqlite
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embeddings import DEFAULT_MODEL_NAME, EmbeddingModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and build defaults
# ---------------------------------------------------------------------------

DEFAULT_DB = PROJECT_ROOT / "data" / "indexes" / "metadata.sqlite"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "indexes" / "faiss"
INDEX_FILE = "index.faiss"
ID_MAP_FILE = "id_map.json"
BUILD_META_FILE = "build_meta.json"
WORK_DIR = ".build"
STATE_FILE = "state.json"
EMBEDDINGS_FILE = "embeddings.npy"

STATE_VERSION = 1
DEFAULT_ENCODER_BATCH_SIZE = 16
DEFAULT_CHECKPOINT_SIZE = 256
DEFAULT_FAISS_ADD_BATCH_SIZE = 4096
MAX_TRAIN_VECTORS = 100_000
TRAINING_POINTS_PER_CENTROID = 39


# ---------------------------------------------------------------------------
# Small persistence helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file, then atomically replace."""
    temp_path = path.with_name(f".{path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _database_signature(db_path: Path) -> dict[str, Any]:
    """Return enough corpus identity information to validate a checkpoint."""
    if not db_path.exists():
        raise FileNotFoundError(f"Metadata database not found: {db_path}")

    stat = db_path.stat()
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """SELECT COUNT(*), COALESCE(MIN(rowid), 0), COALESCE(MAX(rowid), 0)
               FROM chunks"""
        ).fetchone()
        first = conn.execute(
            "SELECT chunk_id FROM chunks ORDER BY rowid LIMIT 1"
        ).fetchone()
        last = conn.execute(
            "SELECT chunk_id FROM chunks ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    return {
        "path": str(db_path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "chunk_count": int(row[0]),
        "min_rowid": int(row[1]),
        "max_rowid": int(row[2]),
        "first_chunk_id": first[0] if first else None,
        "last_chunk_id": last[0] if last else None,
    }


def _load_state(state_path: Path) -> dict[str, Any]:
    with open(state_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_state(
    state: dict[str, Any],
    *,
    db_signature: dict[str, Any],
    model_name: str,
    vector_dim: int,
) -> None:
    mismatches: list[str] = []
    if state.get("version") != STATE_VERSION:
        mismatches.append("checkpoint version")
    if state.get("db_signature") != db_signature:
        mismatches.append("metadata database")
    if state.get("model_name") != model_name:
        mismatches.append("embedding model")
    if state.get("vector_dim") != vector_dim:
        mismatches.append("embedding dimension")

    if mismatches:
        fields = ", ".join(mismatches)
        raise RuntimeError(
            f"Existing FAISS checkpoint does not match the current {fields}. "
            "Re-run with --restart to discard it."
        )


# ---------------------------------------------------------------------------
# Resumable embedding phase
# ---------------------------------------------------------------------------


def _open_embedding_checkpoint(
    *,
    output_dir: Path,
    db_signature: dict[str, Any],
    model_name: str,
    vector_dim: int,
    restart: bool,
) -> tuple[np.memmap, dict[str, Any], Path]:
    work_dir = output_dir / WORK_DIR
    state_path = work_dir / STATE_FILE
    embeddings_path = work_dir / EMBEDDINGS_FILE

    if restart and work_dir.exists():
        logger.info("Discarding checkpoint at %s", work_dir)
        shutil.rmtree(work_dir)

    if state_path.exists():
        state = _load_state(state_path)
        _validate_state(
            state,
            db_signature=db_signature,
            model_name=model_name,
            vector_dim=vector_dim,
        )
        if not embeddings_path.exists():
            raise RuntimeError(
                f"Checkpoint state exists but {embeddings_path} is missing. "
                "Re-run with --restart."
            )
        embeddings = np.load(embeddings_path, mmap_mode="r+")
        expected_shape = (db_signature["chunk_count"], vector_dim)
        if embeddings.shape != expected_shape or embeddings.dtype != np.float32:
            raise RuntimeError(
                "Checkpoint embedding file has the wrong shape or dtype. "
                "Re-run with --restart."
            )
        logger.info(
            "Resuming embedding checkpoint: %d/%d chunks complete",
            state["completed_chunks"],
            db_signature["chunk_count"],
        )
        return embeddings, state, state_path

    if work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete checkpoint files found in {work_dir}. Re-run with --restart."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(db_signature["chunk_count"], vector_dim),
    )
    state = {
        "version": STATE_VERSION,
        "status": "encoding",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "db_signature": db_signature,
        "model_name": model_name,
        "vector_dim": vector_dim,
        "completed_chunks": 0,
        "last_rowid": None,
        "encoding_seconds": 0.0,
    }
    _atomic_write_json(state_path, state)
    logger.info(
        "Created embedding checkpoint for %d chunks at %s",
        db_signature["chunk_count"],
        embeddings_path,
    )
    return embeddings, state, state_path


def _encode_chunks(
    *,
    db_path: Path,
    embeddings: np.memmap,
    state: dict[str, Any],
    state_path: Path,
    model: Any,
    encoder_batch_size: int,
    checkpoint_size: int,
) -> None:
    total_chunks = int(state["db_signature"]["chunk_count"])
    completed = int(state["completed_chunks"])
    last_rowid = state.get("last_rowid")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        while completed < total_chunks:
            if last_rowid is None:
                rows = conn.execute(
                    """SELECT rowid, chunk_text FROM chunks
                       ORDER BY rowid LIMIT ?""",
                    (checkpoint_size,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT rowid, chunk_text FROM chunks
                       WHERE rowid > ? ORDER BY rowid LIMIT ?""",
                    (last_rowid, checkpoint_size),
                ).fetchall()

            if not rows:
                raise RuntimeError(
                    "Chunk rows ended before the checkpoint's expected corpus size. "
                    "The metadata database may have changed."
                )

            texts = [row["chunk_text"] for row in rows]
            batch_started = time.perf_counter()
            vectors = model.encode(
                texts,
                batch_size=encoder_batch_size,
                show_progress=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            expected_shape = (len(rows), embeddings.shape[1])
            if vectors.shape != expected_shape:
                raise RuntimeError(
                    f"Embedding model returned shape {vectors.shape}; expected {expected_shape}."
                )
            if not np.isfinite(vectors).all():
                raise RuntimeError("Embedding model returned NaN or infinite values.")

            end = completed + len(rows)
            embeddings[completed:end] = vectors
            embeddings.flush()

            completed = end
            last_rowid = int(rows[-1]["rowid"])
            state.update(
                {
                    "updated_at": _utc_now(),
                    "completed_chunks": completed,
                    "last_rowid": last_rowid,
                    "encoding_seconds": round(
                        float(state.get("encoding_seconds", 0.0))
                        + (time.perf_counter() - batch_started),
                        3,
                    ),
                }
            )
            _atomic_write_json(state_path, state)
            logger.info(
                "Embedding progress: %d/%d chunks (%.1f%%)",
                completed,
                total_chunks,
                completed * 100 / total_chunks,
            )
    finally:
        conn.close()

    state.update({"status": "embeddings_complete", "updated_at": _utc_now()})
    _atomic_write_json(state_path, state)


# ---------------------------------------------------------------------------
# FAISS and mapping phase
# ---------------------------------------------------------------------------


def _build_index(
    vectors: np.memmap,
    *,
    add_batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    import faiss

    n_vectors, dim = vectors.shape
    if n_vectors < TRAINING_POINTS_PER_CENTROID:
        index = faiss.IndexFlatIP(dim)
        index_type = "IndexFlatIP"
        nlist = None
        train_vectors = 0
    else:
        target_nlist = max(1, int(4 * math.sqrt(n_vectors)))
        target_nlist = min(target_nlist, 65536)
        desired_training = max(10_000, target_nlist * TRAINING_POINTS_PER_CENTROID)
        train_vectors = min(n_vectors, desired_training, MAX_TRAIN_VECTORS)
        nlist = min(target_nlist, max(1, train_vectors // TRAINING_POINTS_PER_CENTROID))

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer,
            dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )

        if train_vectors == n_vectors:
            training_data = vectors
        else:
            rng = np.random.default_rng(42)
            sample_ids = np.sort(rng.choice(n_vectors, size=train_vectors, replace=False))
            training_data = np.asarray(vectors[sample_ids], dtype=np.float32)

        logger.info(
            "Training IVF index (nlist=%d, training_vectors=%d/%d)...",
            nlist,
            train_vectors,
            n_vectors,
        )
        index.train(training_data)
        del training_data
        index_type = "IndexIVFFlat"

    for start in range(0, n_vectors, add_batch_size):
        end = min(start + add_batch_size, n_vectors)
        index.add(np.asarray(vectors[start:end], dtype=np.float32))
        logger.info(
            "FAISS add progress: %d/%d vectors (%.1f%%)",
            end,
            n_vectors,
            end * 100 / n_vectors,
        )

    return index, {
        "index_type": index_type,
        "nlist": nlist,
        "training_vectors": train_vectors,
    }


def _write_id_map(db_path: Path, destination: Path) -> int:
    """Stream the ordered FAISS ID mapping without retaining it in memory."""
    temp_path = destination.with_name(f".{destination.name}.tmp")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    count = 0
    try:
        cursor = conn.execute(
            "SELECT chunk_id, paper_id FROM chunks ORDER BY rowid"
        )
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write("[")
            for row in cursor:
                if count:
                    handle.write(",")
                json.dump(
                    {
                        "faiss_id": count,
                        "chunk_id": row["chunk_id"],
                        "paper_id": row["paper_id"],
                    },
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                count += 1
            handle.write("]\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        conn.close()

    os.replace(temp_path, destination)
    return count


def _clear_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Public build entry point
# ---------------------------------------------------------------------------


def build_faiss(
    db_path: Path,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    embedding_model: Any | None = None,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    checkpoint_size: int = DEFAULT_CHECKPOINT_SIZE,
    faiss_add_batch_size: int = DEFAULT_FAISS_ADD_BATCH_SIZE,
    restart: bool = False,
) -> tuple[int, int]:
    """Build and atomically save a resumable FAISS index.

    Returns ``(num_vectors, vector_dim)``. Partial embeddings remain under
    ``<output_dir>/.build`` after an interruption and are removed after a
    successful final index is written.
    """
    if encoder_batch_size <= 0:
        raise ValueError("encoder_batch_size must be positive")
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    if faiss_add_batch_size <= 0:
        raise ValueError("faiss_add_batch_size must be positive")

    db_path = db_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_started = time.perf_counter()

    db_signature = _database_signature(db_path)
    n_vectors = int(db_signature["chunk_count"])
    if n_vectors == 0:
        logger.warning("No chunks found in database.")
        return 0, 0

    model = embedding_model or EmbeddingModel(model_name=model_name)
    vector_dim = int(model.dim)
    embeddings, state, state_path = _open_embedding_checkpoint(
        output_dir=output_dir,
        db_signature=db_signature,
        model_name=model_name,
        vector_dim=vector_dim,
        restart=restart,
    )

    _encode_chunks(
        db_path=db_path,
        embeddings=embeddings,
        state=state,
        state_path=state_path,
        model=model,
        encoder_batch_size=encoder_batch_size,
        checkpoint_size=checkpoint_size,
    )
    del model
    _clear_accelerator_cache()

    if _database_signature(db_path) != db_signature:
        raise RuntimeError(
            "Metadata database changed during embedding. Re-run with --restart."
        )

    index, index_meta = _build_index(
        embeddings,
        add_batch_size=faiss_add_batch_size,
    )
    if index.ntotal != n_vectors:
        raise RuntimeError(
            f"FAISS contains {index.ntotal} vectors; expected {n_vectors}."
        )

    import faiss

    index_path = output_dir / INDEX_FILE
    index_temp_path = output_dir / f".{INDEX_FILE}.tmp"
    faiss.write_index(index, str(index_temp_path))

    id_map_path = output_dir / ID_MAP_FILE
    id_map_count = _write_id_map(db_path, id_map_path)
    if id_map_count != n_vectors:
        raise RuntimeError(
            f"ID map contains {id_map_count} entries; expected {n_vectors}."
        )

    os.replace(index_temp_path, index_path)
    build_meta = {
        "version": 1,
        "status": "complete",
        "created_at": _utc_now(),
        "db_signature": db_signature,
        "embedding_model": model_name,
        "vector_dim": vector_dim,
        "num_vectors": n_vectors,
        "encoder_batch_size": encoder_batch_size,
        "checkpoint_size": checkpoint_size,
        "faiss_add_batch_size": faiss_add_batch_size,
        "encoding_seconds": state.get("encoding_seconds", 0.0),
        "build_run_seconds": round(time.perf_counter() - build_started, 3),
        **index_meta,
    }
    _atomic_write_json(output_dir / BUILD_META_FILE, build_meta)

    logger.info(
        "FAISS %s built: %d vectors, dim=%d, nlist=%s",
        index_meta["index_type"],
        n_vectors,
        vector_dim,
        index_meta["nlist"],
    )
    logger.info(
        "Index saved to %s (%.1f MB)",
        index_path,
        index_path.stat().st_size / (1024 * 1024),
    )
    logger.info("ID map saved to %s (%d entries)", id_map_path, id_map_count)

    del embeddings
    shutil.rmtree(output_dir / WORK_DIR)
    return n_vectors, vector_dim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resumable FAISS vector index")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to metadata SQLite DB")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for FAISS files",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformers embedding model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_ENCODER_BATCH_SIZE,
        help="Embedding model batch size",
    )
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=DEFAULT_CHECKPOINT_SIZE,
        help="Chunks encoded between durable checkpoints",
    )
    parser.add_argument(
        "--faiss-add-batch-size",
        type=int,
        default=DEFAULT_FAISS_ADD_BATCH_SIZE,
        help="Vectors added to FAISS per batch",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard an existing partial checkpoint and rebuild embeddings",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    n, dim = build_faiss(
        Path(args.db),
        Path(args.output_dir),
        model_name=args.model,
        encoder_batch_size=args.batch_size,
        checkpoint_size=args.checkpoint_size,
        faiss_add_batch_size=args.faiss_add_batch_size,
        restart=args.restart,
    )
    if n > 0:
        logger.info("Done! %d vectors x %d dimensions", n, dim)


if __name__ == "__main__":
    main()
