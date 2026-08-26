"""Runtime path configuration for CiteQuest-RAG.

The default app uses ``data/indexes``.  Environment variables let us point the
same API at a smaller demo index without changing code or rebuilding the whole
local benchmark database.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

DEFAULT_DB_PATH = Path("data/indexes/metadata.sqlite")
DEFAULT_FAISS_DIR = Path("data/indexes/faiss")
DEFAULT_HYBRID_ALPHA = 0.5

DB_PATH_ENV = "CITEQUEST_DB_PATH"
FAISS_DIR_ENV = "CITEQUEST_FAISS_DIR"
HYBRID_ALPHA_ENV = "CITEQUEST_HYBRID_ALPHA"


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


def validate_hybrid_alpha(value: object, *, source: str = "alpha") -> float:
    """Return a finite Hybrid weight in the inclusive range [0, 1]."""
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} must be a finite number between 0 and 1; got {value!r}"
        ) from exc

    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"{source} must be a finite number between 0 and 1; got {value!r}"
        )
    return alpha


def get_hybrid_alpha() -> float:
    """Return the configured production default for Hybrid retrieval."""
    value = os.getenv(HYBRID_ALPHA_ENV)
    if value is None:
        return DEFAULT_HYBRID_ALPHA
    return validate_hybrid_alpha(value, source=HYBRID_ALPHA_ENV)


def resolve_hybrid_alpha(request_alpha: float | None) -> float:
    """Resolve request override first, then runtime configuration."""
    if request_alpha is None:
        return get_hybrid_alpha()
    return validate_hybrid_alpha(request_alpha, source="request alpha")
