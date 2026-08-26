"""Tests for app.retrieval.vector_store — FAISS search."""

import json
import shutil
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.retrieval.vector_store import search_vector


class FakeEmbeddingModel:
    """Small deterministic query encoder used by vector-store unit tests."""

    dim = 1024

    def __init__(self, model_name=None):
        self.model_name = model_name

    def encode(self, texts, *, show_progress=False, **kwargs):
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, sum(ord(ch) for ch in text) % self.dim] = 1.0
        return vectors


def _write_index_artifacts(index_dir, paper_ids, *, dimension=1024):
    """Write a deterministic flat index and matching ID map."""
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.zeros((len(paper_ids), dimension), dtype=np.float32)
    for row in range(len(paper_ids)):
        vectors[row, row % dimension] = 1.0

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "id_map.json").write_text(
        json.dumps(
            [
                {
                    "faiss_id": faiss_id,
                    "chunk_id": f"{paper_id}_default",
                    "paper_id": paper_id,
                }
                for faiss_id, paper_id in enumerate(paper_ids)
            ]
        ),
        encoding="utf-8",
    )


def _write_metadata_db(db_path, paper_ids):
    """Write the metadata rows required to hydrate test search hits."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY, title TEXT, abstract TEXT DEFAULT '',
            year INTEGER, venue TEXT, authors_json TEXT DEFAULT '[]'
        )""")
        conn.execute("""CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL
        )""")
        for paper_id in paper_ids:
            conn.execute(
                "INSERT INTO papers VALUES (?, ?, '', 2024, NULL, '[]')",
                (paper_id, f"Paper {paper_id}"),
            )
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?)",
                (f"{paper_id}_default", paper_id, f"Text for {paper_id}"),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def reset_vector_cache(monkeypatch):
    """Keep each test isolated from module-level runtime cache state."""
    import app.retrieval.vector_store as vs

    vs._reset_cache()
    monkeypatch.setattr(vs, "EmbeddingModel", FakeEmbeddingModel)
    yield
    vs._reset_cache()


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

    def test_score_keeps_faiss_precision(
        self, monkeypatch, test_db_path, test_index_dir
    ):
        import app.retrieval.vector_store as vs

        precise_score = np.float32(0.12345679)

        class PreciseFakeIndex:
            def search(self, query_vec, top_k):
                return (
                    np.asarray([[precise_score]], dtype=np.float32),
                    np.asarray([[0]], dtype=np.int64),
                )

        loaded = SimpleNamespace(
            index=PreciseFakeIndex(),
            id_map=(
                {"faiss_id": 0, "chunk_id": "A_default", "paper_id": "A"},
            ),
            model_name="test-model",
            dimension=1024,
        )
        monkeypatch.setattr(vs, "_load_index", lambda index_dir: loaded)

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

    def test_year_filter_adaptively_expands_faiss_candidates(
        self, monkeypatch, test_db_path
    ):
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
        loaded = SimpleNamespace(
            index=fake_index,
            id_map=tuple(
                {
                    "faiss_id": index,
                    "chunk_id": f"P{index}_default",
                    "paper_id": f"P{index}",
                }
                for index in range(80)
            ),
            model_name="test-model",
            dimension=1024,
        )
        monkeypatch.setattr(vs, "_load_index", lambda index_dir: loaded)

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


class TestIndexCache:
    def test_switching_paths_returns_each_index_results(self, tmp_path, monkeypatch):
        import app.retrieval.vector_store as vs

        index_a = tmp_path / "a" / "faiss"
        index_b = tmp_path / "b" / "faiss"
        db_a = tmp_path / "a" / "metadata.sqlite"
        db_b = tmp_path / "b" / "metadata.sqlite"
        _write_index_artifacts(index_a, ["A"])
        _write_index_artifacts(index_b, ["B"])
        _write_metadata_db(db_a, ["A"])
        _write_metadata_db(db_b, ["B"])

        created_models = []

        class CountingEmbeddingModel(FakeEmbeddingModel):
            def __init__(self, model_name=None):
                super().__init__(model_name=model_name)
                created_models.append(self)

        monkeypatch.setattr(vs, "EmbeddingModel", CountingEmbeddingModel)

        loaded_a = vs._load_index(index_a)
        result_a = search_vector("query", 1, db_a, index_a)
        result_b = search_vector("query", 1, db_b, index_b)

        assert [result.paper_id for result in result_a] == ["A"]
        assert [result.paper_id for result in result_b] == ["B"]
        assert len(created_models) == 1
        assert vs._index_cache.key.index_dir == index_b.resolve()

        query_vec = np.zeros((1, 1024), dtype=np.float32)
        query_vec[0, 0] = 1.0
        assert vs._search_index(loaded_a, query_vec, 1)[0]["paper_id"] == "A"

    def test_unchanged_path_reuses_loaded_index(self, tmp_path, monkeypatch):
        import faiss
        import app.retrieval.vector_store as vs

        index_dir = tmp_path / "faiss"
        _write_index_artifacts(index_dir, ["A"])
        real_read_index = faiss.read_index
        read_paths = []

        def counted_read_index(path):
            read_paths.append(path)
            return real_read_index(path)

        monkeypatch.setattr(faiss, "read_index", counted_read_index)

        first = vs._load_index(index_dir)
        second = vs._load_index(index_dir / ".." / "faiss")

        assert first is second
        assert read_paths == [str(index_dir.resolve() / "index.faiss")]

    def test_replacing_artifacts_at_same_path_reloads(self, tmp_path, monkeypatch):
        import faiss
        import app.retrieval.vector_store as vs

        index_dir = tmp_path / "active" / "faiss"
        replacement_dir = tmp_path / "replacement" / "faiss"
        db_path = tmp_path / "metadata.sqlite"
        _write_index_artifacts(index_dir, ["A"])
        _write_index_artifacts(replacement_dir, ["B"])
        _write_metadata_db(db_path, ["A", "B"])

        real_read_index = faiss.read_index
        read_paths = []

        def counted_read_index(path):
            read_paths.append(path)
            return real_read_index(path)

        monkeypatch.setattr(faiss, "read_index", counted_read_index)

        first = search_vector("query", 1, db_path, index_dir)
        (replacement_dir / "index.faiss").replace(index_dir / "index.faiss")
        (replacement_dir / "id_map.json").replace(index_dir / "id_map.json")
        second = search_vector("query", 1, db_path, index_dir)

        assert [result.paper_id for result in first] == ["A"]
        assert [result.paper_id for result in second] == ["B"]
        assert len(read_paths) == 2

    def test_concurrent_first_load_reads_artifacts_once(self, tmp_path, monkeypatch):
        import faiss
        import app.retrieval.vector_store as vs

        index_dir = tmp_path / "faiss"
        _write_index_artifacts(index_dir, ["A"])
        real_read_index = faiss.read_index
        read_paths = []

        def slow_read_index(path):
            read_paths.append(path)
            time.sleep(0.02)
            return real_read_index(path)

        monkeypatch.setattr(faiss, "read_index", slow_read_index)

        with ThreadPoolExecutor(max_workers=4) as executor:
            loaded = list(executor.map(lambda _: vs._load_index(index_dir), range(4)))

        assert len(read_paths) == 1
        assert all(item is loaded[0] for item in loaded)

    def test_mismatched_index_and_id_map_is_rejected(self, tmp_path):
        import faiss
        import app.retrieval.vector_store as vs

        index_dir = tmp_path / "faiss"
        index_dir.mkdir()
        index = faiss.IndexFlatIP(4)
        index.add(np.eye(2, 4, dtype=np.float32))
        faiss.write_index(index, str(index_dir / "index.faiss"))
        (index_dir / "id_map.json").write_text(
            json.dumps(
                [{"faiss_id": 0, "chunk_id": "A_default", "paper_id": "A"}]
            ),
            encoding="utf-8",
        )

        with pytest.raises(vs.FaissArtifactError, match="count mismatch"):
            vs._load_index(index_dir)

        assert vs._index_cache is None

    def test_build_metadata_dimension_mismatch_is_rejected(self, tmp_path):
        import app.retrieval.vector_store as vs

        index_dir = tmp_path / "faiss"
        _write_index_artifacts(index_dir, ["A"], dimension=4)
        (index_dir / "build_meta.json").write_text(
            json.dumps(
                {
                    "embedding_model": "test-model",
                    "vector_dim": 8,
                    "num_vectors": 1,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(vs.FaissArtifactError, match="dimension mismatch"):
            vs._load_index(index_dir)

    def test_embedding_model_dimension_mismatch_is_rejected(
        self, tmp_path, monkeypatch
    ):
        import app.retrieval.vector_store as vs

        index_dir = tmp_path / "faiss"
        db_path = tmp_path / "metadata.sqlite"
        _write_index_artifacts(index_dir, ["A"])
        _write_metadata_db(db_path, ["A"])

        class WrongDimensionModel(FakeEmbeddingModel):
            dim = 8

        monkeypatch.setattr(vs, "EmbeddingModel", WrongDimensionModel)

        with pytest.raises(vs.FaissArtifactError, match="model dimension mismatch"):
            search_vector("query", 1, db_path, index_dir)
