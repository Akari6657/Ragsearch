"""
Build a FAISS vector index from chunk texts in the metadata SQLite DB.

Workflow:
1. Read all chunks from the chunks table
2. Encode chunk_text with BAAI/bge-m3
3. Build a FAISS IndexIVFFlat (IVF clustering for fast approximate search)
4. Save the index + an ID mapping file (faiss_id → chunk_id → paper_id)

Usage:
    python scripts/build_faiss.py --db data/indexes/metadata.sqlite
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DB = PROJECT_ROOT / "data" / "indexes" / "metadata.sqlite"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "indexes" / "faiss"
INDEX_FILE = "index.faiss"
ID_MAP_FILE = "id_map.json"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_faiss(db_path: Path, output_dir: Path) -> tuple[int, int]:
    """Build and save a FAISS index.

    Returns (num_vectors, vector_dim).
    """
    # — 1. Load chunks from SQLite ——————————————————————————————————————
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chunk_id, paper_id, chunk_text FROM chunks ORDER BY rowid"
    ).fetchall()
    conn.close()

    if not rows:
        logger.warning("No chunks found in database.")
        return 0, 0

    chunk_ids = [r["chunk_id"] for r in rows]
    paper_ids = [r["paper_id"] for r in rows]
    texts = [r["chunk_text"] for r in rows]

    logger.info("Loaded %d chunks from database", len(texts))

    # — 2. Encode ————————————————————————————————————————————————————————
    model = EmbeddingModel()
    vectors = model.encode(texts, batch_size=4)  # small batch for GPU memory

    # — 3. Build FAISS index (IVF) ————————————————————————————————————
    import faiss

    dim = vectors.shape[1]
    n_vectors = vectors.shape[0]

    # nlist = 4 × sqrt(N) is the standard heuristic for IVF clustering
    nlist = max(1, int(4 * math.sqrt(n_vectors)))
    nlist = min(nlist, 65536)  # FAISS upper bound for nlist

    quantizer = faiss.IndexFlatIP(dim)  # exact IP for coarse assignment
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    logger.info("Training IVF index (nlist=%d, vectors=%d)...", nlist, n_vectors)
    index.train(vectors)
    index.add(vectors)

    logger.info("FAISS IndexIVFFlat built: %d vectors, dim=%d, nlist=%d",
                index.ntotal, dim, nlist)

    # — 4. Save index ————————————————————————————————————————————————————
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / INDEX_FILE
    faiss.write_index(index, str(index_path))
    logger.info("Index saved to %s (%.1f KB)", index_path, index_path.stat().st_size / 1024)

    # — 5. Save ID mapping ———————————————————————————————————————————————
    id_map = [
        {"faiss_id": i, "chunk_id": chunk_ids[i], "paper_id": paper_ids[i]}
        for i in range(len(chunk_ids))
    ]
    id_map_path = output_dir / ID_MAP_FILE
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)
    logger.info("ID map saved to %s (%d entries)", id_map_path, len(id_map))

    return index.ntotal, dim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to metadata SQLite DB")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for FAISS files"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    n, dim = build_faiss(Path(args.db), Path(args.output_dir))
    if n > 0:
        logger.info("Done! %d vectors × %d dimensions", n, dim)


if __name__ == "__main__":
    main()
