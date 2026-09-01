"""Tests for the same-vector FAISS IVF approximation probe."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.eval.ann_probe_experiment import (
    FORMAL_STATUS,
    compare_current_candidates_with_reference,
    evaluate_ann_artifact_gate,
    finalize_ann_probe_report,
    load_diagnostics_reference,
    render_ann_probe_markdown,
    run_ann_probe_comparison,
    write_ann_probe_outputs,
)
from app.eval.retrieval_eval import EvalQuery


class FakeIvfIndex:
    """Small deterministic IVF-shaped test double keyed by vector marker."""

    def __init__(self, rankings, *, dimension=2, nlist=4, total=4):
        self.rankings = rankings
        self.d = dimension
        self.nlist = nlist
        self.ntotal = total
        self.nprobe = 1
        self.calls = []

    def search(self, vectors, top_k):
        marker = int(vectors[0, 0])
        self.calls.append((marker, self.nprobe, top_k, tuple(vectors[0])))
        order = list(self.rankings.get((marker, self.nprobe), range(self.ntotal)))
        ids = order[:top_k]
        scores = [1.0 - rank * 0.01 for rank in range(len(ids))]
        if len(ids) < top_k:
            padding = top_k - len(ids)
            ids.extend([-1] * padding)
            scores.extend([-float("inf")] * padding)
        return (
            np.asarray([scores], dtype=np.float32),
            np.asarray([ids], dtype=np.int64),
        )


def _query(
    query_id: str,
    paper_id: str,
    *,
    query_type: str = "keyword",
    split: str = "dev",
) -> EvalQuery:
    return EvalQuery(
        query_id=query_id,
        query=f"query {query_id}",
        query_type=query_type,
        split=split,
        relevant_paper_ids=(paper_id,),
        source_paper_id=paper_id,
        source_category="Computer Science",
    )


def _id_map(count: int) -> list[dict]:
    return [
        {
            "faiss_id": index,
            "chunk_id": f"P{index}_chunk0",
            "paper_id": f"P{index}",
        }
        for index in range(count)
    ]


def _reference(queries, current_orders, *, formal=False):
    per_query = []
    for query, order in zip(queries, current_orders):
        paper_ids = [f"P{index}" for index in order]
        relevant = query.relevant_paper_ids[0]
        rank = next(
            (index for index, paper_id in enumerate(paper_ids, start=1) if paper_id == relevant),
            None,
        )
        per_query.append(
            {
                "query_id": query.query_id,
                "candidate_ids": {
                    "dense_chunk_ids": [f"P{index}_chunk0" for index in order]
                },
                "dense_first_relevant_rank": rank,
            }
        )
    return {
        "status": "dev_diagnostics_v1_3",
        "query_count": 50 if formal else len(queries),
        "protocol": {"candidate_depth": 20},
        "artifact_gate": {"passed": True},
        "reference_reproduction": {"matches": True},
        "manifest": {
            "git_commit": "diagnostics-commit",
            "raw_file_sha256": "raw-hash",
            "eval_file_sha256": "eval-hash",
        },
        "per_query": per_query,
    }


def _manifest(*, dirty=False):
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
        "faiss_index_type": "IndexIVFFlat",
        "faiss_nlist": 894,
        "faiss_nprobe": 64,
        "git_commit": "ann-probe-commit",
        "git_dirty": dirty,
    }


def test_same_vectors_isolate_nprobe_and_recover_a_target():
    queries = [
        _query("q1", "P0"),
        _query("q2", "P3", query_type="semantic_paraphrase"),
    ]
    rankings = {
        (0, 2): [0, 1, 2, 3],
        (0, 4): [1, 0, 2, 3],
        (1, 2): [1, 2, 0, 3],
        (1, 4): [3, 1, 2, 0],
    }
    index = FakeIvfIndex(rankings)
    vectors = np.asarray([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
    current_orders = [[0, 1], [1, 2]]
    reference = _reference(queries, current_orders)

    report = run_ann_probe_comparison(
        queries,
        vectors,
        index,
        _id_map(4),
        reference=reference,
        current_nprobe=2,
        exhaustive_nprobe=4,
        candidate_depth=2,
        deep_probe_depth=4,
    )

    assert report["reference_reproduction"]["matches"] is True
    assert report["methods"]["current"]["metrics"]["hit_rate@20"] == 0.5
    assert report["methods"]["exhaustive"]["metrics"]["hit_rate@20"] == 1.0
    assert report["target_rank_comparison"]["top20"] == {
        "current_wins": 1,
        "exhaustive_wins": 1,
        "ties": 0,
        "targets_recovered_by_exhaustive": 1,
        "targets_lost_by_exhaustive": 0,
    }
    assert report["per_query"][1]["current"][
        "first_relevant_rank_at_deep_probe"
    ] == 4
    assert report["per_query"][1]["exhaustive"][
        "first_relevant_rank_at_candidate_depth"
    ] == 1

    measured = [call for call in index.calls if call[2] == 2][2:]
    for marker in (0, 1):
        vectors_seen = [call[3] for call in measured if call[0] == marker]
        assert vectors_seen == [tuple(vectors[marker])] * 2
    assert [(call[0], call[1]) for call in measured] == [
        (0, 2),
        (0, 4),
        (1, 4),
        (1, 2),
    ]


def test_reference_comparison_reports_current_candidate_drift():
    queries = [_query("q1", "P0")]
    index = FakeIvfIndex({(0, 2): [0, 1], (0, 4): [0, 1]})
    reference = _reference(queries, [[0, 1]])
    report = run_ann_probe_comparison(
        queries,
        np.asarray([[0.0, 0.5]], dtype=np.float32),
        index,
        _id_map(4),
        reference=reference,
        current_nprobe=2,
        exhaustive_nprobe=4,
        candidate_depth=2,
        deep_probe_depth=4,
    )
    report["per_query"][0]["current"]["chunk_ids"] = ["changed_chunk0"]

    comparison = compare_current_candidates_with_reference(
        report["per_query"], reference
    )

    assert comparison["matches"] is False
    assert comparison["failures"] == ["q1: current Dense candidate IDs differ"]


def test_probe_rejects_non_dev_split_before_search():
    index = FakeIvfIndex({})
    with pytest.raises(ValueError, match="expected only 'dev'"):
        run_ann_probe_comparison(
            [_query("q1", "P0", split="test")],
            np.asarray([[0.0, 0.5]], dtype=np.float32),
            index,
            _id_map(4),
            reference={"per_query": []},
            current_nprobe=2,
            exhaustive_nprobe=4,
            candidate_depth=2,
            deep_probe_depth=4,
        )
    assert index.calls == []


def test_artifact_gate_requires_clean_frozen_ivf_configuration():
    queries = [_query(f"q{index:02d}", f"P{index}") for index in range(50)]
    orders = [list(range(20)) for _ in queries]
    reference = _reference(queries, orders, formal=True)

    passed = evaluate_ann_artifact_gate(
        _manifest(), reference, current_nprobe=64, exhaustive_nprobe=894
    )
    dirty = evaluate_ann_artifact_gate(
        _manifest(dirty=True),
        reference,
        current_nprobe=64,
        exhaustive_nprobe=894,
    )

    assert passed["passed"] is True
    assert dirty["passed"] is False
    assert dirty["failures"] == ["git_worktree_clean"]


@pytest.fixture(scope="module")
def formal_report():
    queries = [
        _query(
            f"q{index:02d}",
            f"P{index}",
            query_type=("keyword", "natural_question", "semantic_paraphrase")[
                index % 3
            ],
        )
        for index in range(50)
    ]
    id_map = _id_map(50)
    current_orders = [
        [index, *[other for other in range(50) if other != index][:19]]
        for index in range(50)
    ]
    rankings = {}
    for index, order in enumerate(current_orders):
        rankings[(index, 64)] = order
        rankings[(index, 894)] = order
    index = FakeIvfIndex(rankings, nlist=894, total=50)
    vectors = np.asarray([[float(i), 0.5] for i in range(50)], dtype=np.float32)
    reference = _reference(queries, current_orders, formal=True)
    report = run_ann_probe_comparison(
        queries,
        vectors,
        index,
        id_map,
        reference=reference,
        current_nprobe=64,
        exhaustive_nprobe=894,
    )
    report = finalize_ann_probe_report(
        report,
        manifest=_manifest(),
        reference=reference,
        reference_path="reports/retrieval_diagnostics_v1_3.json",
        artifact_gate={"passed": True, "checks": {}, "failures": []},
        encoding={"same_vector_reused_across_conditions": True},
        source_dev_count=50,
        formal_run=True,
    )
    return report, reference


def test_formal_status_retains_nprobe_when_exhaustive_recovers_nothing(formal_report):
    report, _ = formal_report

    assert report["status"] == FORMAL_STATUS
    assert report["reference_reproduction"]["matches"] is True
    assert report["candidate_overlap"]["mean_current_recall_of_exhaustive"] == 1.0
    assert report["decision"] == {
        "applicable": True,
        "rule": (
            "Run an nprobe quality/latency sweep only if exhaustive IVF recovers "
            "at least one known target into Dense Top-20; otherwise retain 64 "
            "and investigate query/document representation."
        ),
        "recommendation": "retain_nprobe_64_and_test_representation",
        "production_changed": False,
    }


def test_smoke_cannot_make_a_configuration_decision(formal_report):
    report, reference = formal_report
    smoke = finalize_ann_probe_report(
        dict(report),
        manifest=_manifest(dirty=True),
        reference=reference,
        reference_path="reports/retrieval_diagnostics_v1_3.json",
        artifact_gate={"passed": False, "failures": ["git_worktree_clean"]},
        encoding={"same_vector_reused_across_conditions": True},
        source_dev_count=50,
        formal_run=False,
    )

    assert smoke["status"] == "smoke_or_development"
    assert smoke["decision"]["recommendation"] == (
        "no_decision_from_smoke_or_invalid_run"
    )


def test_load_reference_and_write_answer_first_report(tmp_path, formal_report):
    report, reference = formal_report
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    assert load_diagnostics_reference(reference_path) == reference

    json_path = tmp_path / "ann.json"
    markdown_path = tmp_path / "ann.md"
    write_ann_probe_outputs(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_ann_probe_markdown(report)
    assert markdown.startswith("# CiteQuest ANN Approximation Probe v1.3.1")
    assert "## Dense Quality" in markdown
    assert "No production configuration was changed" in markdown
