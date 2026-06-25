"""
CiteQuest-RAG FastAPI application entry point.

Usage:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from fastapi import FastAPI

from app.api.routes_search import router as search_router

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CiteQuest-RAG",
    description="Academic Search + Citation-grounded RAG + Lightweight Research Agent",
    version="0.1.0",
)

app.include_router(search_router)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path("data/indexes/metadata.sqlite")


def _check_indexes(db_path: Path) -> dict[str, bool]:
    """Report which indexes are available."""
    status = {
        "metadata_db": db_path.exists(),
        "fts5": False,
        "faiss": False,
    }
    if status["metadata_db"]:
        try:
            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
            ).fetchone()
            status["fts5"] = row is not None
            conn.close()
        except sqlite3.Error:
            pass

    faiss_path = db_path.parent / "faiss" / "index.faiss"
    status["faiss"] = faiss_path.exists()

    return status


@app.get("/health")
def health():
    """Return service status and index availability."""
    indexes = _check_indexes(DEFAULT_DB_PATH)
    all_ready = indexes["metadata_db"] and indexes["fts5"]

    return {
        "status": "healthy" if all_ready else "degraded",
        "version": app.version,
        "indexes": indexes,
    }
