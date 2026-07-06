"""Tests for runtime path configuration."""

from __future__ import annotations

import sqlite3

from app.core.config import DEFAULT_DB_PATH, DEFAULT_FAISS_DIR, get_db_path, get_faiss_dir
from app.main import health


def test_default_paths(monkeypatch):
    monkeypatch.delenv("CITEQUEST_DB_PATH", raising=False)
    monkeypatch.delenv("CITEQUEST_FAISS_DIR", raising=False)

    assert get_db_path() == DEFAULT_DB_PATH
    assert get_faiss_dir() == DEFAULT_FAISS_DIR


def test_env_paths(monkeypatch, tmp_path):
    db_path = tmp_path / "demo" / "metadata.sqlite"
    faiss_dir = tmp_path / "demo" / "faiss"

    monkeypatch.setenv("CITEQUEST_DB_PATH", str(db_path))
    monkeypatch.setenv("CITEQUEST_FAISS_DIR", str(faiss_dir))

    assert get_db_path() == db_path
    assert get_faiss_dir() == faiss_dir


def test_health_uses_env_paths(monkeypatch, tmp_path):
    db_path = tmp_path / "demo" / "metadata.sqlite"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE chunk_fts(id TEXT)")
    conn.commit()
    conn.close()

    faiss_dir = tmp_path / "demo" / "faiss"
    faiss_dir.mkdir()
    (faiss_dir / "index.faiss").write_text("fake", encoding="utf-8")
    (faiss_dir / "id_map.json").write_text("[]", encoding="utf-8")

    monkeypatch.setenv("CITEQUEST_DB_PATH", str(db_path))
    monkeypatch.setenv("CITEQUEST_FAISS_DIR", str(faiss_dir))

    response = health()

    assert response["status"] == "healthy"
    assert response["paths"] == {
        "metadata_db": str(db_path),
        "faiss_dir": str(faiss_dir),
    }
    assert response["capabilities"]["lexical_search"] is True
    assert response["capabilities"]["vector_search"] is True
    assert response["capabilities"]["hybrid_search"] is True
