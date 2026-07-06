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
from fastapi.staticfiles import StaticFiles

from app.api.routes_search import router as search_router
from app.api.routes_ask import router as ask_router

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CiteQuest-RAG",
    description="Academic Search + Citation-grounded RAG + AI Overview",
    version="0.5.0",
)

app.include_router(search_router)
app.include_router(ask_router)

# Serve frontend static files
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

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
    faiss_dir = db_path.parent / "faiss"
    faiss_index = faiss_dir / "index.faiss"
    faiss_id_map = faiss_dir / "id_map.json"

    status = {
        "metadata_db": db_path.exists(),
        "fts5": False,
        "faiss_index": faiss_index.exists(),
        "faiss_id_map": faiss_id_map.exists(),
        "faiss": False,
    }
    if status["metadata_db"]:
        conn = None
        try:
            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
            ).fetchone()
            status["fts5"] = row is not None
        except sqlite3.Error:
            pass
        finally:
            if conn is not None:
                conn.close()

    status["faiss"] = status["faiss_index"] and status["faiss_id_map"]

    return status


@app.get("/health")
def health():
    """Return service status and index availability."""
    indexes = _check_indexes(DEFAULT_DB_PATH)
    capabilities = {
        "lexical_search": indexes["metadata_db"] and indexes["fts5"],
        "vector_search": indexes["metadata_db"] and indexes["faiss"],
        "hybrid_search": indexes["metadata_db"] and indexes["fts5"] and indexes["faiss"],
        "rag": indexes["metadata_db"] and indexes["fts5"],
    }

    if capabilities["lexical_search"] and capabilities["vector_search"]:
        status = "healthy"
    elif capabilities["lexical_search"]:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "version": app.version,
        "indexes": indexes,
        "capabilities": capabilities,
    }
