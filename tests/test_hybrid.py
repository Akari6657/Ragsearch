"""Tests for app.retrieval.hybrid — score fusion."""

import pytest

from app.core.schemas import SearchResult
from app.retrieval.hybrid import (
    _normalize_higher_is_better,
    _normalize_lower_is_better,
    search_hybrid,
)


def _result(chunk_id: str, paper_id: str, score: float) -> SearchResult:
    return SearchResult(
        paper_id=paper_id,
        chunk_id=chunk_id,
        title=f"Paper {paper_id}",
        year=None,
        venue=None,
        authors=[],
        score=score,
        snippet="",
        abstract="",
    )


class TestNormalizeHigherIsBetter:
    def test_normal_case(self):
        scores = [10.0, 20.0, 30.0]
        norm = _normalize_higher_is_better(scores)
        assert norm == [0.0, 0.5, 1.0]

    def test_identical(self):
        norm = _normalize_higher_is_better([5.0, 5.0, 5.0])
        assert norm == [0.5, 0.5, 0.5]

    def test_empty(self):
        assert _normalize_higher_is_better([]) == []

    def test_single(self):
        norm = _normalize_higher_is_better([42.0])
        assert norm == [0.5]


class TestNormalizeLowerIsBetter:
    def test_bm25_direction(self):
        """SQLite FTS5 bm25 scores are lower-is-better."""
        scores = [-20.0, -10.0, -2.0]
        norm = _normalize_lower_is_better(scores)
        assert norm[0] == 1.0
        assert norm[1] == pytest.approx(0.4444, abs=1e-4)
        assert norm[2] == 0.0

    def test_identical(self):
        norm = _normalize_lower_is_better([-5.0, -5.0, -5.0])
        assert norm == [0.5, 0.5, 0.5]

    def test_empty(self):
        assert _normalize_lower_is_better([]) == []


class TestHybridMerge:
    def test_lexical_bm25_direction_affects_ranking(self, monkeypatch):
        """The best BM25 hit should stay best when lexical weight dominates."""
        lexical = [
            _result("A_chunk", "A", -20.0),  # best lexical
            _result("B_chunk", "B", -2.0),   # worst lexical
        ]
        vector = [
            _result("B_chunk", "B", 0.9),    # best vector
            _result("C_chunk", "C", 0.1),    # worst vector
        ]

        monkeypatch.setattr("app.retrieval.hybrid.search_lexical", lambda *a, **k: lexical)
        monkeypatch.setattr("app.retrieval.hybrid.search_vector", lambda *a, **k: vector)

        results = search_hybrid("test query", top_k=3, alpha=0.7)

        assert [r.chunk_id for r in results] == ["A_chunk", "B_chunk", "C_chunk"]
        assert results[0].score == pytest.approx(0.7)
        assert results[1].score == pytest.approx(0.3)

    def test_alpha_zero_uses_vector_scores(self, monkeypatch):
        lexical = [
            _result("A_chunk", "A", -20.0),
            _result("B_chunk", "B", -2.0),
        ]
        vector = [
            _result("B_chunk", "B", 0.9),
            _result("C_chunk", "C", 0.1),
        ]

        monkeypatch.setattr("app.retrieval.hybrid.search_lexical", lambda *a, **k: lexical)
        monkeypatch.setattr("app.retrieval.hybrid.search_vector", lambda *a, **k: vector)

        results = search_hybrid("test query", top_k=3, alpha=0.0)

        assert [r.chunk_id for r in results] == ["B_chunk", "A_chunk", "C_chunk"]
