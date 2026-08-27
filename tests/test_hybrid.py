"""Tests for app.retrieval.hybrid — score fusion."""

import pytest

from app.core.schemas import SearchResult
from app.retrieval.hybrid import (
    _normalize_higher_is_better,
    fuse_minmax_results,
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


class TestHybridMerge:
    @pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
    def test_invalid_alpha_is_rejected(self, alpha):
        with pytest.raises(ValueError, match="between 0 and 1"):
            search_hybrid("test query", alpha=alpha)

    def test_higher_is_better_lexical_score_affects_ranking(self, monkeypatch):
        """The best public lexical score should stay best when it dominates."""
        lexical = [
            _result("A_chunk", "A", 20.0),  # best lexical
            _result("B_chunk", "B", 2.0),   # worst lexical
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

    def test_pure_fusion_preserves_inputs_and_existing_ranking(self):
        lexical = [
            _result("A_chunk", "A", 20.0),
            _result("B_chunk", "B", 2.0),
        ]
        vector = [
            _result("B_chunk", "B", 0.9).model_copy(
                update={"snippet": "dense fallback"}
            ),
            _result("C_chunk", "C", 0.1),
        ]
        lexical_before = [result.model_dump() for result in lexical]
        vector_before = [result.model_dump() for result in vector]

        results = fuse_minmax_results(lexical, vector, top_k=3, alpha=0.7)

        assert [result.chunk_id for result in results] == [
            "A_chunk",
            "B_chunk",
            "C_chunk",
        ]
        assert [result.score for result in results] == pytest.approx([0.7, 0.3, 0.0])
        assert results[1].snippet == "dense fallback"
        assert [result.model_dump() for result in lexical] == lexical_before
        assert [result.model_dump() for result in vector] == vector_before
        assert all(
            fused is not source
            for fused in results
            for source in (*lexical, *vector)
            if fused.chunk_id == source.chunk_id
        )

    def test_pure_fusion_preserves_lexical_first_ties(self):
        lexical = [_result("B_chunk", "B", 1.0)]
        vector = [_result("A_chunk", "A", 1.0)]

        results = fuse_minmax_results(lexical, vector, top_k=2, alpha=0.5)

        assert [result.chunk_id for result in results] == ["B_chunk", "A_chunk"]

    def test_alpha_zero_uses_vector_scores(self, monkeypatch):
        lexical = [
            _result("A_chunk", "A", 20.0),
            _result("B_chunk", "B", 2.0),
        ]
        vector = [
            _result("B_chunk", "B", 0.9),
            _result("C_chunk", "C", 0.1),
        ]

        monkeypatch.setattr("app.retrieval.hybrid.search_lexical", lambda *a, **k: lexical)
        monkeypatch.setattr("app.retrieval.hybrid.search_vector", lambda *a, **k: vector)

        results = search_hybrid("test query", top_k=3, alpha=0.0)

        assert [r.chunk_id for r in results] == ["B_chunk", "A_chunk", "C_chunk"]

    def test_hybrid_score_keeps_full_precision(self, monkeypatch):
        lexical = [
            _result("A_chunk", "A", 3.0),
            _result("B_chunk", "B", 2.0),
            _result("C_chunk", "C", 1.0),
        ]
        vector = [
            _result("C_chunk", "C", 0.3),
            _result("B_chunk", "B", 0.2),
            _result("A_chunk", "A", 0.1),
        ]

        monkeypatch.setattr("app.retrieval.hybrid.search_lexical", lambda *a, **k: lexical)
        monkeypatch.setattr("app.retrieval.hybrid.search_vector", lambda *a, **k: vector)

        results = search_hybrid("test query", top_k=3, alpha=1 / 3)

        assert [r.chunk_id for r in results] == ["C_chunk", "B_chunk", "A_chunk"]
        assert results[0].score == pytest.approx(2 / 3)
        assert results[0].score != round(results[0].score, 4)

    def test_optional_lexical_query_only_changes_bm25_branch(self, monkeypatch):
        observed = {}

        def fake_lexical(query, **kwargs):
            observed["lexical_query"] = query
            return []

        def fake_vector(query, **kwargs):
            observed["vector_query"] = query
            return []

        monkeypatch.setattr("app.retrieval.hybrid.search_lexical", fake_lexical)
        monkeypatch.setattr("app.retrieval.hybrid.search_vector", fake_vector)

        search_hybrid(
            "为什么 GAN 训练不稳定",
            lexical_query="为什么 GAN 训练不稳定 GAN mode collapse convergence",
        )

        assert observed == {
            "lexical_query": "为什么 GAN 训练不稳定 GAN mode collapse convergence",
            "vector_query": "为什么 GAN 训练不稳定",
        }

    def test_default_lexical_query_preserves_existing_behavior(self, monkeypatch):
        observed = {}

        def fake_lexical(query, **kwargs):
            observed["lexical"] = query
            return []

        def fake_vector(query, **kwargs):
            observed["vector"] = query
            return []

        monkeypatch.setattr("app.retrieval.hybrid.search_lexical", fake_lexical)
        monkeypatch.setattr("app.retrieval.hybrid.search_vector", fake_vector)

        search_hybrid("same query")

        assert observed == {"lexical": "same query", "vector": "same query"}

    def test_year_range_is_forwarded_to_both_branches(self, monkeypatch):
        observed = {}

        def fake_lexical(query, **kwargs):
            observed["lexical"] = kwargs
            return []

        def fake_vector(query, **kwargs):
            observed["vector"] = kwargs
            return []

        monkeypatch.setattr("app.retrieval.hybrid.search_lexical", fake_lexical)
        monkeypatch.setattr("app.retrieval.hybrid.search_vector", fake_vector)

        search_hybrid("filtered query", year_from=2020, year_to=2024)

        assert observed["lexical"]["year_from"] == 2020
        assert observed["lexical"]["year_to"] == 2024
        assert observed["vector"]["year_from"] == 2020
        assert observed["vector"]["year_to"] == 2024
