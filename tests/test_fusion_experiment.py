"""Tests for the evaluation-only Min-max versus RRF comparison."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from app.core.schemas import SearchResult
from app.eval import fusion_experiment
from app.eval.fusion_experiment import (
    compare_minmax_dev_baseline,
    evaluate_dev_artifact_gate,
    fuse_rrf_results,
    paired_bootstrap_deltas,
    render_fusion_markdown,
    run_dev_fusion_experiment,
    run_fusion_comparison,
    write_fusion_outputs,
)
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


def _metric_row(query_id: str, value: float) -> dict:
    return {
        "query_id": query_id,
        "hit@5": value,
        "hit@10": value,
        "recall@5": value,
        "recall@10": value,
        "mrr@10": value,
        "ndcg@10": value,
    }


def test_paired_bootstrap_uses_query_deltas_and_is_deterministic():
    minmax = [_metric_row("q1", 0.0), _metric_row("q2", 0.25)]
    rrf = [_metric_row("q2", 1.25), _metric_row("q1", 1.0)]

    first = paired_bootstrap_deltas(minmax, rrf, samples=200, seed=7)
    second = paired_bootstrap_deltas(minmax, rrf, samples=200, seed=7)

    assert first == second
    assert first["method_order"] == "rrf_minus_minmax"
    assert first["query_count"] == 2
    for metric in first["metrics"].values():
        assert metric == {
            "observed_delta": 1.0,
            "ci_lower": 1.0,
            "ci_upper": 1.0,
            "direction": "rrf",
        }


def test_paired_bootstrap_rejects_unpaired_query_ids():
    with pytest.raises(ValueError, match="query IDs differ"):
        paired_bootstrap_deltas(
            [_metric_row("q1", 0.0)],
            [_metric_row("q2", 1.0)],
        )


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
    pairwise = report["pairwise_first_relevant_rank"]
    assert {key: pairwise[key] for key in ("minmax_wins", "rrf_wins", "ties")} == {
        "minmax_wins": 0,
        "rrf_wins": 0,
        "ties": 3,
    }
    assert [row["winner"] for row in pairwise["per_query"]] == ["tie"] * 3
    assert report["paired_bootstrap"]["query_count"] == 3
    assert report["paired_bootstrap"]["seed"] == 20_260_827
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


def _summary(query_count: int, value: float = 1.0) -> dict:
    return {
        "query_count": query_count,
        "hit_rate@5": value,
        "hit_rate@10": value,
        "recall@5": value,
        "recall@10": value,
        "mrr@10": value,
        "ndcg@10": value,
    }


def _baseline_reference() -> dict:
    return {
        "source_report": "reports/reference.json",
        "source_git_commit": "baseline-commit",
        "raw_file_sha256": "raw-hash",
        "eval_file_sha256": "eval-hash",
        "alpha": 0.5,
        "metrics": _summary(50),
        "by_query_type": {
            "keyword": _summary(17),
            "natural_question": _summary(17),
            "semantic_paraphrase": _summary(16),
        },
        "per_query": [
            {
                **_metric_row(f"p{index:02d}", 1.0),
                "first_relevant_rank": 1,
            }
            for index in range(50)
        ],
    }


def _manifest(*, dirty: bool = False) -> dict:
    return {
        "corpus": "arxiv_cs",
        "paper_count": 50_000,
        "chunk_count": 50_000,
        "fts_row_count": 50_000,
        "faiss_vector_count": 50_000,
        "id_map_count": 50_000,
        "embedding_model": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "raw_file_sha256": "raw-hash",
        "eval_file_sha256": "eval-hash",
        "git_commit": "experiment-commit",
        "git_dirty": dirty,
    }


def _frozen_dev_queries() -> list[EvalQuery]:
    query_types = ("keyword", "natural_question", "semantic_paraphrase")
    return [
        _query(f"p{index:02d}", query_types[index % len(query_types)])
        for index in range(50)
    ]


@pytest.fixture(scope="module")
def formal_experiment_report():
    queries = _frozen_dev_queries()
    calls = {"lexical": [], "dense": []}

    def lexical_search(text, top_k):
        calls["lexical"].append((text, top_k))
        paper_id = text.removeprefix("query ").upper()
        return [_result(f"{paper_id}_chunk", 10.0)]

    def dense_search(text, top_k):
        calls["dense"].append((text, top_k))
        paper_id = text.removeprefix("query ").upper()
        return [_result(f"{paper_id}_chunk", 0.9)]

    report = run_dev_fusion_experiment(
        queries,
        lexical_search,
        dense_search,
        manifest=_manifest(),
        baseline_reference=_baseline_reference(),
        source_dev_count=50,
    )
    return report, calls


def test_formal_dev_experiment_requires_artifacts_and_reproduces_baseline(
    formal_experiment_report,
):
    report, calls = formal_experiment_report

    assert report["status"] == "dev_comparison_v1_2"
    assert report["artifact_gate"]["passed"] is True
    assert report["baseline_reproduction"]["status"] == "matched"
    assert report["baseline_reproduction"]["per_query_check"] == {
        "query_count": 50,
        "first_relevant_rank_mismatches": [],
        "metric_mismatches": [],
    }
    assert report["decision_gate"]["recommendation"] == "retain_minmax"
    assert report["decision_gate"]["production_changed"] is False
    assert len(calls["lexical"]) == 51  # one warm-up plus 50 measured queries
    assert len(calls["dense"]) == 51
    assert report["warmup"]["excluded_from_measured_latency"] is True


def test_dirty_artifact_fails_gate_and_smoke_cannot_make_decision():
    gate = evaluate_dev_artifact_gate(_manifest(dirty=True), _baseline_reference())
    assert gate["passed"] is False
    assert gate["failures"] == ["git_worktree_clean"]

    queries = _frozen_dev_queries()[:3]

    def search(text, top_k):
        paper_id = text.removeprefix("query ").upper()
        return [_result(f"{paper_id}_chunk", 1.0)]

    report = run_dev_fusion_experiment(
        queries,
        search,
        search,
        manifest=_manifest(dirty=True),
        baseline_reference=_baseline_reference(),
        source_dev_count=50,
        bootstrap_samples=100,
    )

    assert report["status"] == "smoke_or_development"
    assert report["baseline_reproduction"]["status"] == "not_applicable"
    assert report["decision_gate"] == {
        "applicable": False,
        "recommendation": "no_decision_from_smoke_or_invalid_run",
    }


def test_baseline_comparison_reports_exact_metric_mismatch(formal_experiment_report):
    report, _ = formal_experiment_report
    changed = deepcopy(report)
    changed["methods"]["minmax"]["metrics"]["ndcg@10"] = 0.9

    reproduction = compare_minmax_dev_baseline(changed, _baseline_reference())

    assert reproduction["status"] == "mismatch"
    assert reproduction["matches"] is False
    assert reproduction["aggregate_deltas"]["ndcg@10"] == pytest.approx(-0.1)
    assert any("aggregate.ndcg@10" in failure for failure in reproduction["failures"])


def test_fusion_report_outputs_are_independent_and_answer_first(
    tmp_path, formal_experiment_report
):
    report, _ = formal_experiment_report
    json_path = tmp_path / "fusion.json"
    markdown_path = tmp_path / "fusion.md"

    write_fusion_outputs(report, json_path=json_path, markdown_path=markdown_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_fusion_markdown(report)
    assert markdown.startswith("# CiteQuest Retrieval Fusion Dev Comparison v1.2")
    assert "## Technical summary" in markdown
    assert "No Benchmark v1 test query" in markdown
    assert "Recommendation: **`retain_minmax`**" in markdown
