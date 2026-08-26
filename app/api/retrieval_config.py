"""API-level resolution for retrieval runtime configuration."""

from __future__ import annotations

from fastapi import HTTPException

from app.core.config import resolve_hybrid_alpha


def resolve_request_hybrid_alpha(
    mode: str,
    request_alpha: float | None,
) -> float | None:
    """Resolve Hybrid alpha or return None when the mode does not use it."""
    if mode != "hybrid":
        return None

    try:
        return resolve_hybrid_alpha(request_alpha)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "INVALID_HYBRID_ALPHA_CONFIGURATION",
                "message": str(exc),
            },
        ) from exc
