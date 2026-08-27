"""Tests for the evaluation-only Min-max versus RRF comparison."""

from __future__ import annotations

import pytest

from app.core.schemas import SearchResult
from app.eval import fusion_experiment
from app.eval.fusion_experiment import fuse_rrf_results, run_fusion_comparison
from app.eval.retrieval_eval import EvalQuery


def _result(chunk_id: str, score: float, *, snippet: str = "") -> SearchResult:
    return SearchResult(
        paper_id=chunk_id.split("_")[0],
        chunk_id=chunk_id,
        title=f"Paper {chunk_id}",
        year=2025,
        venue="arXiv",
        authors=[],
        score=score,
        snippet=snippet,
        abstract="",
    )


def _query(
    query_id: str, query_type: str = "keyword", *, split: str = "dev"
) -> EvalQuery:
    return EvalQuery(
        query_id=query_id,
        query=f"query {query_id}",
        query_type=query_type,
        split=split,
        relevant_paper_ids=(query_id.upper(),),
        source_paper_id=query_id.upper(),
        source_category="Computer Science",
    )


def test_rrf_matches_hand_calculation_and_preserves_sources():
    lexical = [
        _result("A_chunk", 9.0),
        _result("B_chunk", 8.0),
        _result("C_chunk", 7.0),
    ]
    dense = [
        _result("B_chunk", 0.9, snippet="dense B"),
        _result("D_chunk", 0.8),
        _result("A_chunk", 0.7),
    ]
    lexical_before = [result.model_dump() for result in lexical]
    dense_before = [result.model_dump() for result in dense]

    results = fuse_rrf_results(lexical, dense, top_k=4, rrf_k=60)

    assert [result.chunk_id for result in results] == [
        "B_chunk",
        "A_chunk",
        "D_chunk",
        "C_chunk",
    ]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61 + 1 / 63)
    assert results[2].score == pytest.approx(1 / 62)
    assert results[0].snippet == "dense B"
    assert [result.model_dump() for result in lexical] == lexical_before
    assert [result.model_dump() for result in dense] == dense_before


def test_rrf_tie_break_is_deterministic_and_branch_symmetric():
    lexical_first = fuse_rrf_results(
        [_result("B_chunk", 2.0)],
        [_result("A_chunk", 0.8)],
        top_k=2,
    )
    dense_first = fuse_rrf_results(
        [_result("A_chunk", 2.0)],
        [_result("B_chunk", 0.8)],
        top_k=2,
    )

    assert [result.chunk_id for result in lexical_first] == ["A_chunk", "B_chunk"]
    assert [result.chunk_id for result in dense_first] == ["A_chunk", "B_chunk"]


@pytest.mark.parametrize("rrf_k", [0, -1, 1.5, True])
def test_rrf_rejects_invalid_k(rrf_k):
    with pytest.raises(ValueError, match="positive integer"):
        fuse_rrf_results([], [], rrf_k=rrf_k)


def test_comparison_retrieves_once_and_reuses_exact_candidate_tuples(monkeypatch):
    queries = [
        _query("p1", "keyword"),
        _query("p2", "natural_question"),
        _query("p3", "semantic_paraphrase"),
    ]
    lexical_sources = {
        query.query: [_result(f"{query.query_id.upper()}_chunk", 10.0)]
        for query in queries
    }
    dense_sources = {
        query.query: [_result(f"{query.query_id.upper()}_chunk", 0.9)]
        for query in queries
    }
    lexical_before = {
        key: [result.model_dump() for result in values]
        for key, values in lexical_sources.items()
    }
    dense_before = {
        key: [result.model_dump() for result in values]
        for key, values in dense_sources.items()
    }
    lexical_calls = []
    dense_calls = []

    def lexical_search(text, top_k):
        lexical_calls.append((text, top_k))
        return lexical_sources[text]

    def dense_search(text, top_k):
        dense_calls.append((text, top_k))
        return dense_sources[text]

    observed_inputs = []
    real_minmax = fusion_experiment.fuse_minmax_results
    real_rrf = fusion_experiment.fuse_rrf_results

    def tracked_minmax(lexical, dense, **kwargs):
        observed_inputs.append(("minmax", id(lexical), id(dense)))
        return real_minmax(lexical, dense, **kwargs)

    def tracked_rrf(lexical, dense, **kwargs):
        observed_inputs.append(("rrf", id(lexical), id(dense)))
        return real_rrf(lexical, dense, **kwargs)

    monkeypatch.setattr(fusion_experiment, "fuse_minmax_results", tracked_minmax)
    monkeypatch.setattr(fusion_experiment, "fuse_rrf_results", tracked_rrf)

    report = run_fusion_comparison(queries, lexical_search, dense_search)

    assert lexical_calls == [(query.query, 20) for query in queries]
    assert dense_calls == [(query.query, 20) for query in queries]
    for index in range(0, len(observed_inputs), 2):
        assert observed_inputs[index][0] == "minmax"
        assert observed_inputs[index + 1][0] == "rrf"
        assert observed_inputs[index][1:] == observed_inputs[index + 1][1:]
    assert report["query_count"] == 3
    assert report["methods"]["minmax"]["metrics"]["hit_rate@10"] == 1.0
    assert report["methods"]["rrf"]["metrics"]["hit_rate@10"] == 1.0
    assert report["pairwise_first_relevant_rank"] == {
        "minmax_wins": 0,
        "rrf_wins": 0,
        "ties": 3,
    }
    assert report["protocol"]["query_split"] == "dev"
    assert {
        key: [result.model_dump() for result in values]
        for key, values in lexical_sources.items()
    } == lexical_before
    assert {
        key: [result.model_dump() for result in values]
        for key, values in dense_sources.items()
    } == dense_before


def test_comparison_rejects_unexpected_split_before_retrieval():
    calls = []

    def search(text, top_k):
        calls.append((text, top_k))
        return []

    with pytest.raises(ValueError, match="expected only 'dev'"):
        run_fusion_comparison([_query("p1", split="test")], search, search)

    assert calls == []
