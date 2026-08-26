"""Tests for app.retrieval.vector_store — FAISS search."""

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.retrieval.vector_store import search_vector


class FakeEmbeddingModel:
    """Small deterministic query encoder used by vector-store unit tests."""

    def encode(self, texts, *, show_progress=False, **kwargs):
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, sum(ord(ch) for ch in text) % 1024] = 1.0
        return vectors


@pytest.fixture
def test_index_dir():
    """Build a tiny FAISS index with 5 vectors for testing."""
    import faiss

    tmp_dir = Path(tempfile.mkdtemp(prefix="faiss_test_"))
    dim = 1024

    # Create random normalized vectors
    rng = np.random.RandomState(42)
    vecs = rng.randn(5, dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(tmp_dir / "index.faiss"))

    id_map = [
        {"faiss_id": 0, "chunk_id": "A_default", "paper_id": "A"},
        {"faiss_id": 1, "chunk_id": "B_default", "paper_id": "B"},
        {"faiss_id": 2, "chunk_id": "C_default", "paper_id": "C"},
        {"faiss_id": 3, "chunk_id": "D_default", "paper_id": "D"},
        {"faiss_id": 4, "chunk_id": "E_default", "paper_id": "E"},
    ]
    with open(tmp_dir / "id_map.json", "w") as f:
        json.dump(id_map, f)

    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def test_db_path(test_index_dir):
    """Create a SQLite DB with papers and chunks for the test vectors."""
    db_path = test_index_dir / "metadata.sqlite"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY, title TEXT, abstract TEXT DEFAULT '',
        year INTEGER, venue TEXT, authors_json TEXT DEFAULT '[]',
        concepts_json TEXT DEFAULT '[]', doi TEXT, url TEXT,
        citation_count INTEGER DEFAULT 0, open_access INTEGER DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL,
        chunk_text TEXT NOT NULL, chunk_type TEXT DEFAULT 'metadata',
        token_count INTEGER DEFAULT 0
    )""")

    years = {"A": 2018, "B": 2019, "C": 2020, "D": 2021, "E": None}
    for pid, year in years.items():
        conn.execute(
            "INSERT OR IGNORE INTO papers VALUES (?, ?, '', ?, NULL, '[]', '[]', NULL, NULL, 0, 1)",
            (pid, f"Paper {pid}", year),
        )
        conn.execute(
            "INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, 'metadata', 10)",
            (f"{pid}_default", pid, f"Text for paper {pid}"),
        )
    conn.commit()
    conn.close()
    return db_path


class TestVectorSearch:
    @pytest.fixture(autouse=True)
    def _reset_globals(self, monkeypatch):
        """Reset vector_store module globals before each test."""
        import app.retrieval.vector_store as vs
        vs._index = None
        vs._id_map = []
        vs._model = None
        monkeypatch.setattr(vs, "EmbeddingModel", FakeEmbeddingModel)

    def test_returns_results(self, test_db_path, test_index_dir):
        results = search_vector(
            "neural network", top_k=3, db_path=test_db_path, index_dir=test_index_dir
        )
        assert len(results) > 0
        assert len(results) <= 3
        assert all(r.paper_id for r in results)
        assert all(r.title for r in results)

    def test_top_k_respected(self, test_db_path, test_index_dir):
        results = search_vector(
            "anything", top_k=2, db_path=test_db_path, index_dir=test_index_dir
        )
        assert len(results) == 2

    def test_score_keeps_faiss_precision(self, test_db_path, test_index_dir):
        import app.retrieval.vector_store as vs

        precise_score = np.float32(0.12345679)

        class PreciseFakeIndex:
            def search(self, query_vec, top_k):
                return (
                    np.asarray([[precise_score]], dtype=np.float32),
                    np.asarray([[0]], dtype=np.int64),
                )

        vs._index = PreciseFakeIndex()
        vs._id_map = [
            {"faiss_id": 0, "chunk_id": "A_default", "paper_id": "A"}
        ]
        vs._model = FakeEmbeddingModel()

        results = search_vector(
            "precise score", top_k=1, db_path=test_db_path, index_dir=test_index_dir
        )

        assert results[0].score == pytest.approx(float(precise_score))
        assert results[0].score != round(results[0].score, 4)

    def test_year_bounds_are_inclusive_and_unknown_years_are_excluded(
        self, test_db_path, test_index_dir
    ):
        results = search_vector(
            "year filter",
            top_k=5,
            db_path=test_db_path,
            index_dir=test_index_dir,
            year_from=2019,
            year_to=2020,
        )

        assert {result.paper_id for result in results} == {"B", "C"}
        assert all(result.year in {2019, 2020} for result in results)

    def test_one_sided_year_filters(self, test_db_path, test_index_dir):
        newer = search_vector(
            "newer",
            top_k=5,
            db_path=test_db_path,
            index_dir=test_index_dir,
            year_from=2021,
        )
        older = search_vector(
            "older",
            top_k=5,
            db_path=test_db_path,
            index_dir=test_index_dir,
            year_to=2019,
        )

        assert [result.paper_id for result in newer] == ["D"]
        assert {result.paper_id for result in older} == {"A", "B"}
        assert all(result.paper_id != "E" for result in [*newer, *older])

    def test_year_filter_adaptively_expands_faiss_candidates(self, test_db_path):
        import app.retrieval.vector_store as vs

        class ProgressiveFakeIndex:
            ntotal = 80

            def __init__(self):
                self.search_sizes = []

            def search(self, query_vec, candidate_k):
                self.search_sizes.append(candidate_k)
                ids = np.arange(candidate_k, dtype=np.int64)
                scores = 1.0 - ids.astype(np.float32) / 100.0
                return scores[None, :], ids[None, :]

        conn = sqlite3.connect(str(test_db_path))
        try:
            for index in range(80):
                paper_id = f"P{index}"
                year = 2024 if index >= 55 else 2010
                conn.execute(
                    "INSERT INTO papers VALUES (?, ?, '', ?, NULL, '[]', '[]', NULL, NULL, 0, 1)",
                    (paper_id, f"Paper {index}", year),
                )
                conn.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, 'metadata', 10)",
                    (f"{paper_id}_default", paper_id, f"Text for {paper_id}"),
                )
            conn.commit()
        finally:
            conn.close()

        fake_index = ProgressiveFakeIndex()
        vs._index = fake_index
        vs._id_map = [
            {
                "faiss_id": index,
                "chunk_id": f"P{index}_default",
                "paper_id": f"P{index}",
            }
            for index in range(80)
        ]
        vs._model = FakeEmbeddingModel()

        results = search_vector(
            "adaptive filter",
            top_k=3,
            db_path=test_db_path,
            index_dir=test_db_path,
            year_from=2024,
            year_to=2024,
        )

        assert fake_index.search_sizes == [50, 80]
        assert [result.paper_id for result in results] == ["P55", "P56", "P57"]
        assert all(result.year == 2024 for result in results)

    def test_missing_index(self, test_db_path):
        results = search_vector(
            "test", top_k=5, db_path=test_db_path, index_dir="/nonexistent"
        )
        assert results == []
