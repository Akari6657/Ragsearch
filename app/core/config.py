"""Runtime path configuration for CiteQuest-RAG.

The default app uses ``data/indexes``.  Environment variables let us point the
same API at a smaller demo index without changing code or rebuilding the whole
local benchmark database.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DB_PATH = Path("data/indexes/metadata.sqlite")
DEFAULT_FAISS_DIR = Path("data/indexes/faiss")

DB_PATH_ENV = "CITEQUEST_DB_PATH"
FAISS_DIR_ENV = "CITEQUEST_FAISS_DIR"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    return Path(value).expanduser()


def get_db_path() -> Path:
    """Return the metadata SQLite path for the current process."""
    return _path_from_env(DB_PATH_ENV, DEFAULT_DB_PATH)


def get_faiss_dir() -> Path:
    """Return the FAISS artifact directory for the current process."""
    return _path_from_env(FAISS_DIR_ENV, DEFAULT_FAISS_DIR)
