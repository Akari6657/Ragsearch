"""Tests for v1.3 candidate-coverage and failure diagnostics."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from app.core.schemas import SearchResult
from app.eval.retrieval_diagnostics import (
    FORMAL_STATUS,
    compare_with_fusion_reference,
    diagnose_query,
    evaluate_diagnostics_artifact_gate,
    load_fusion_reference,
    render_diagnostics_markdown,
    run_dev_retrieval_diagnostics,
    run_retrieval_diagnostics,
    write_diagnostics_outputs,
)
from app.eval.retrieval_eval import EvalQuery


def _result(paper_id: str, score: float) -> SearchResult:
    return SearchResult(
        paper_id=paper_id,
        chunk_id=f"{paper_id}_chunk0",
        title=f"Paper {paper_id}",
        year=2026,
        venue="arXiv",
        authors=[],
        score=score,
        snippet="",
        abstract="",
    )


def _query(
    query_id: str,
    *,
    relevant_paper_id: str | None = None,
    query_type: str = "keyword",
    split: str = "dev",
) -> EvalQuery:
    paper_id = relevant_paper_id or query_id.upper()
    return EvalQuery(
        query_id=query_id,
        query=f"query {query_id}",
        query_type=query_type,
        split=split,
        relevant_paper_ids=(paper_id,),
        source_paper_id=paper_id,
        source_category="Computer Science",
    )


def _reference_from_rows(rows: list[dict]) -> dict:
    return {
        "status": "dev_comparison_v1_2",
        "query_count": len(rows),
        "candidate_audit": [
            {"query_id": row["query_id"], **row["candidate_ids"]} for row in rows
        ],
        "methods": {
            method: {
                "per_query": [
                    {
                        "query_id": row["query_id"],
                        "retrieved_paper_ids": row[f"{method}_top_k_paper_ids"],
                        "first_relevant_rank": row[
                            f"{method}_first_relevant_rank_at_k"
                        ],
                    }
                    for row in rows
                ]
            }
            for method in ("minmax", "rrf")
        },
        "protocol": {
            "query_split": "dev",
            "candidate_depth": 20,
            "final_top_k": 10,
            "minmax_alpha": 0.5,
            "rrf_k": 60,
        },
        "manifest": {
            "git_commit": "fusion-commit",
            "raw_file_sha256": "raw-hash",
            "eval_file_sha256": "eval-hash",
        },
        "artifact_gate": {"passed": True},
        "baseline_reproduction": {"matches": True},
    }


def _perfect_reference(queries: list[EvalQuery]) -> dict:
    rows = []
    for query in queries:
        result = _result(query.relevant_paper_ids[0], 1.0)
        rows.append(diagnose_query(query, [result], [result]))
    reference = _reference_from_rows(rows)
    reference["query_count"] = 50
    return reference


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
        "git_commit": "diagnostics-commit",
        "git_dirty": dirty,
    }


def test_diagnose_query_records_scores_and_classifies_fusion_loss():
    query = _query("q1", relevant_paper_id="A")
    lexical = [_result("B", 10.0), _result("A", 5.0), _result("C", 0.0)]
    dense = [_result("B", 1.0), _result("C", 0.9), _result("D", 0.8)]
    lexical_before = [result.model_dump() for result in lexical]
    dense_before = [result.model_dump() for result in dense]

    row = diagnose_query(query, lexical, dense, top_k=1, alpha=0.5)

    assert row["outcome"] == "fusion_loss"
    assert row["relevant_candidate_source"] == "bm25_only"
    assert row["bm25_first_relevant_rank"] == 2
    assert row["dense_first_relevant_rank"] is None
    assert row["minmax_first_relevant_rank_all_candidates"] == 2
    assert row["minmax_first_relevant_rank_at_k"] is None
    assert row["minmax_top_k_paper_ids"] == ["B"]
    target = row["targets"][0]
    assert target["bm25"] == {
        "rank": 2,
        "raw_score": 5.0,
        "normalized_score": 0.5,
        "chunk_id": "A_chunk0",
    }
    assert target["dense"] is None
    assert target["minmax"]["score"] == pytest.approx(0.25)
    assert row["minmax_top_k_cutoff_score"] == pytest.approx(1.0)
    assert [result.model_dump() for result in lexical] == lexical_before
    assert [result.model_dump() for result in dense] == dense_before


def test_run_diagnostics_deep_probes_failures_only_and_summarizes_headroom():
    queries = [
        _query("q1", relevant_paper_id="A"),
        _query("q2", relevant_paper_id="Z", query_type="semantic_paraphrase"),
    ]
    regular = {
        "query q1": ([_result("A", 2.0)], [_result("A", 0.9)]),
        "query q2": (
            [_result("B", 2.0), _result("C", 1.0)],
            [_result("D", 0.9), _result("E", 0.8)],
        ),
    }
    expected_rows = [
        diagnose_query(query, *regular[query.query], top_k=2) for query in queries
    ]
    reference = _reference_from_rows(expected_rows)
    calls = {"bm25": [], "dense": []}

    def lexical(text, top_k):
        calls["bm25"].append((text, top_k))
        if top_k == 4 and text == "query q2":
            return [
                _result("B", 4.0),
                _result("C", 3.0),
                _result("Z", 2.0),
                _result("F", 1.0),
            ]
        return regular[text][0]

    def dense(text, top_k):
        calls["dense"].append((text, top_k))
        return regular[text][1]

    report = run_retrieval_diagnostics(
        queries,
        lexical,
        dense,
        reference=reference,
        candidate_depth=2,
        top_k=2,
        deep_probe_depth=4,
    )

    assert report["reference_reproduction"]["matches"] is True
    assert report["summary"]["outcomes"] == {
        "retrieved_top_k": {"count": 1, "rate": 0.5},
        "fusion_loss": {"count": 0, "rate": 0.0},
        "candidate_miss": {"count": 1, "rate": 0.5},
    }
    assert report["summary"]["known_target_oracle"] == {
        "candidate_union_hit_rate": 0.5,
        "current_minmax_hit_rate": 0.5,
        "recoverable_fusion_loss_count": 0,
        "absolute_hit_rate_headroom_at_fixed_candidates": 0.0,
    }
    assert report["per_query"][0]["deep_probe"] is None
    assert report["per_query"][1]["deep_probe"] == {
        "depth": 4,
        "bm25_first_relevant_rank": 3,
        "dense_first_relevant_rank": None,
        "diagnostic_only": True,
    }
    assert calls["bm25"] == [("query q1", 2), ("query q2", 2), ("query q2", 4)]
    assert calls["dense"] == [("query q1", 2), ("query q2", 2), ("query q2", 4)]


def test_reference_comparison_reports_exact_candidate_mismatch():
    query = _query("q1", relevant_paper_id="A")
    row = diagnose_query(query, [_result("A", 1.0)], [_result("A", 0.9)])
    reference = _reference_from_rows([row])
    changed = deepcopy(row)
    changed["candidate_ids"]["bm25_chunk_ids"] = ["different_chunk0"]

    result = compare_with_fusion_reference([changed], reference)

    assert result["matches"] is False
    assert result["failures"] == ["q1: bm25_chunk_ids differs"]


def test_diagnostics_rejects_test_split_before_retrieval():
    calls = []

    def search(text, top_k):
        calls.append((text, top_k))
        return []

    with pytest.raises(ValueError, match="expected only 'dev'"):
        run_retrieval_diagnostics(
            [_query("q1", split="test")],
            search,
            search,
            reference={"candidate_audit": [], "methods": {}},
        )

    assert calls == []


def test_artifact_gate_is_anchored_to_formal_fusion_reference():
    reference = _perfect_reference([_query("q1")])

    passed = evaluate_diagnostics_artifact_gate(_manifest(), reference)
    dirty = evaluate_diagnostics_artifact_gate(_manifest(dirty=True), reference)

    assert passed["passed"] is True
    assert dirty["passed"] is False
    assert dirty["failures"] == ["git_worktree_clean"]


@pytest.fixture(scope="module")
def formal_diagnostics_report():
    query_types = ("keyword", "natural_question", "semantic_paraphrase")
    queries = [
        _query(f"q{index:02d}", query_type=query_types[index % 3])
        for index in range(50)
    ]
    reference = _perfect_reference(queries)
    calls = {"bm25": [], "dense": []}

    def lexical(text, top_k):
        calls["bm25"].append((text, top_k))
        return [_result(text.removeprefix("query ").upper(), 2.0)]

    def dense(text, top_k):
        calls["dense"].append((text, top_k))
        return [_result(text.removeprefix("query ").upper(), 0.9)]

    report = run_dev_retrieval_diagnostics(
        queries,
        lexical,
        dense,
        manifest=_manifest(),
        reference=reference,
        artifact_gate={"passed": True, "checks": {}, "failures": []},
        source_dev_count=50,
        formal_run=True,
    )
    return report, calls, reference


def test_formal_diagnostics_requires_full_reproduction(formal_diagnostics_report):
    report, calls, _ = formal_diagnostics_report

    assert report["status"] == FORMAL_STATUS
    assert report["reference_reproduction"]["matches"] is True
    assert report["summary"]["outcomes"]["retrieved_top_k"]["count"] == 50
    assert report["summary"]["minmax_first_relevant_rank_distribution"] == {
        "1": 50,
        "missing": 0,
    }
    assert len(calls["bm25"]) == 51
    assert len(calls["dense"]) == 51
    assert report["data_adequacy"]["additional_corpus_required_now"] is False


def test_explicit_smoke_cannot_receive_formal_status(formal_diagnostics_report):
    _, _, reference = formal_diagnostics_report
    queries = [_query("q00")]

    def search(text, top_k):
        return [_result("Q00", 1.0)]

    report = run_dev_retrieval_diagnostics(
        queries,
        search,
        search,
        manifest=_manifest(dirty=True),
        reference=reference,
        artifact_gate={
            "passed": False,
            "checks": {"git_worktree_clean": False},
            "failures": ["git_worktree_clean"],
        },
        source_dev_count=50,
        formal_run=False,
    )

    assert report["status"] == "smoke_or_development"


def test_load_reference_and_write_answer_first_reports(
    tmp_path, formal_diagnostics_report
):
    report, _, reference = formal_diagnostics_report
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    assert load_fusion_reference(reference_path) == reference

    json_path = tmp_path / "diagnostics.json"
    markdown_path = tmp_path / "diagnostics.md"
    write_diagnostics_outputs(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_diagnostics_markdown(report)
    assert markdown.startswith("# CiteQuest Retrieval Diagnostics v1.3")
    assert "## Conclusion" in markdown
    assert "## Candidate Coverage" in markdown
    assert "No corpus expansion or index rebuild" in markdown
