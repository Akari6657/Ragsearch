"""Tests for app.retrieval.hybrid — score fusion."""

import math

import pytest

from app.retrieval.hybrid import _minmax_normalize, search_hybrid


class TestMinMaxNormalize:
    def test_normal_case(self):
        scores = [10.0, 20.0, 30.0]
        norm = _minmax_normalize(scores)
        assert norm == [0.0, 0.5, 1.0]

    def test_reverse_order(self):
        """BM25 scores are negative — best = most negative → 1.0."""
        scores = [-10.0, -5.0, -1.0]
        norm = _minmax_normalize(scores)
        assert norm[0] == 0.0  # -10 is worst
        assert norm[2] == 1.0  # -1 is best

    def test_identical(self):
        norm = _minmax_normalize([5.0, 5.0, 5.0])
        assert norm == [0.5, 0.5, 0.5]

    def test_empty(self):
        assert _minmax_normalize([]) == []

    def test_single(self):
        norm = _minmax_normalize([42.0])
        assert norm == [0.5]


class TestHybridAlpha:
    def test_alpha_zero_pure_vector(self):
        """alpha=0 → pure semantic, lexical scores ignored."""
        # This is an integration test — if FAISS is built, hybrid at alpha=0
        # should match vector-only results.
        # We just verify the function runs without error.
        results = search_hybrid("test query", top_k=3, alpha=0.0)
        assert isinstance(results, list)

    def test_alpha_one_pure_lexical(self):
        results = search_hybrid("test query", top_k=3, alpha=1.0)
        assert isinstance(results, list)

    def test_alpha_half_balanced(self):
        results = search_hybrid("test query", top_k=3, alpha=0.5)
        assert isinstance(results, list)

    def test_invalid_alpha_clamped(self):
        """Alpha outside [0,1] should still work (Pydantic validates at API layer)."""
        results = search_hybrid("test query", top_k=3, alpha=1.5)
        assert isinstance(results, list)
