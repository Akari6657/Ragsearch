"""Tests for the paper-level Retrieval Benchmark v1 protocol."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import pytest

from app.eval.retrieval_eval import (
    _benchmark_status,
    EvalDataError,
    EvalQuery,
    build_benchmark_manifest,
    deduplicate_papers,
    evaluate_method,
    hit_rate_at_k,
    load_eval_queries,
    ndcg_at_k,
    parse_eval_record,
    recall_at_k,
    reciprocal_rank_at_k,
    render_markdown_report,
    run_benchmark,
    select_best_alpha,
)


@dataclass
class Hit:
    paper_id: str
    chunk_id: str


def query(
    query_id: str = "q0001",
    *,
    split: str = "test",
    query_type: str = "keyword",
    relevant: tuple[str, ...] = ("P1",),
) -> EvalQuery:
    return EvalQuery(
        query_id=query_id,
        query=f"search text {query_id}",
        query_type=query_type,
        split=split,
        relevant_paper_ids=relevant,
        source_paper_id=relevant[0],
        source_category="Information Retrieval",
    )


def test_paper_deduplication_preserves_highest_ranked_chunk():
    hits = [
        Hit("P1", "P1_chunk0"),
        Hit("P1", "P1_chunk1"),
        Hit("P2", "P2_chunk0"),
    ]

    unique = deduplicate_papers(hits)

    assert [(hit.paper_id, hit.chunk_id) for hit in unique] == [
        ("P1", "P1_chunk0"),
        ("P2", "P2_chunk0"),
    ]


def test_hit_rate_and_standard_recall_are_distinct_for_multiple_relevant_papers():
    predicted = ["P2", "X", "P1"]
    relevant = {"P1", "P2", "P3"}

    assert hit_rate_at_k(predicted, relevant, 3) == 1.0
    assert recall_at_k(predicted, relevant, 3) == pytest.approx(2 / 3)


def test_mrr_is_cut_off_at_k():
    predicted = [f"X{i}" for i in range(10)] + ["R"]

    assert reciprocal_rank_at_k(["X", "R"], {"R"}, 10) == 0.5
    assert reciprocal_rank_at_k(predicted, {"R"}, 10) == 0.0


def test_ndcg_ideal_uses_all_known_relevant_papers():
    predicted = ["X", "R1"]
    relevant = {"R1", "R2"}
    expected = (1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)

    assert ndcg_at_k(predicted, relevant, 10) == pytest.approx(expected)
    assert ndcg_at_k(["R1", "R2"], relevant, 10) == 1.0


def test_metrics_handle_empty_results_and_relevant_outside_k():
    assert hit_rate_at_k([], {"R"}, 5) == 0.0
    assert recall_at_k([], {"R"}, 5) == 0.0
    assert reciprocal_rank_at_k([], {"R"}, 10) == 0.0
    assert ndcg_at_k([], {"R"}, 10) == 0.0

    predicted = ["X1", "X2", "R"]
    assert hit_rate_at_k(predicted, {"R"}, 2) == 0.0
    assert recall_at_k(predicted, {"R"}, 2) == 0.0


@pytest.mark.parametrize(
    "record, message",
    [
        ({}, "query_id"),
        (
            {
                "query_id": "q1",
                "query": "valid query",
                "query_type": "unknown",
                "split": "test",
                "relevant_paper_ids": ["P1"],
            },
            "query_type",
        ),
        (
            {
                "query_id": "q1",
                "query": "valid query",
                "query_type": "keyword",
                "split": "test",
                "relevant_paper_ids": [],
            },
            "relevant_paper_ids",
        ),
    ],
)
def test_malformed_eval_record_is_rejected(record, message):
    with pytest.raises(EvalDataError, match=message):
        parse_eval_record(record)


def test_missing_relevant_paper_id_is_rejected(tmp_path):
    db_path = tmp_path / "metadata.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE papers (paper_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO papers VALUES ('P1')")
    conn.commit()
    conn.close()

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "query_id": "q0001",
                "query": "missing target query",
                "query_type": "keyword",
                "split": "test",
                "relevant_paper_ids": ["MISSING"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalDataError, match="MISSING"):
        load_eval_queries(eval_path, db_path=db_path)


def test_evaluate_method_computes_rank_after_paper_deduplication():
    calls = []

    def search_fn(text, top_k):
        calls.append((text, top_k))
        return [
            Hit("X", "X_chunk0"),
            Hit("X", "X_chunk1"),
            Hit("P1", "P1_chunk0"),
        ]

    result = evaluate_method("fake", [query()], search_fn, retrieval_depth=10)

    assert len(calls) == 2  # one untimed warm-up and one measured query
    assert result["per_query"][0]["first_relevant_rank"] == 2
    assert result["per_query"][0]["mrr@10"] == 0.5


def test_alpha_selection_is_deterministic_and_uses_required_tie_breaks():
    sweep = [
        {"alpha": 0.2, "metrics": {"ndcg@10": 0.7, "mrr@10": 0.6}},
        {"alpha": 0.35, "metrics": {"ndcg@10": 0.8, "mrr@10": 0.5}},
        {"alpha": 0.5, "metrics": {"ndcg@10": 0.8, "mrr@10": 0.7}},
        {"alpha": 0.65, "metrics": {"ndcg@10": 0.8, "mrr@10": 0.7}},
    ]

    assert select_best_alpha(sweep) == 0.5


def test_full_protocol_uses_dev_for_tuning_and_marks_small_run_as_smoke():
    queries = [
        query("dev1", split="dev", relevant=("D",)),
        query("test1", split="test", query_type="keyword", relevant=("T1",)),
        query(
            "test2",
            split="test",
            query_type="natural_question",
            relevant=("T2",),
        ),
        query(
            "test3",
            split="test",
            query_type="semantic_paraphrase",
            relevant=("T3",),
        ),
    ]
    calls: list[tuple[str, float | None, str]] = []

    def factory(mode, alpha):
        def search_fn(text, top_k):
            calls.append((mode, alpha, text))
            target = {"search text dev1": "D", "search text test1": "T1",
                      "search text test2": "T2", "search text test3": "T3"}[text]
            if mode == "hybrid" and alpha == 0.2:
                return [Hit(target, f"{target}_chunk")]
            if mode == "bm25" and target == "T1":
                return [Hit(target, f"{target}_chunk")]
            if mode == "dense" and target == "T2":
                return [Hit(target, f"{target}_chunk")]
            return [Hit("MISS", f"MISS_{mode}_{alpha}")]

        return search_fn

    result = run_benchmark(
        queries,
        manifest={"corpus": "arxiv_cs", "paper_count": 4},
        alpha_values=(0.2, 0.5),
        retrieval_depth=10,
        retriever_factory=factory,
    )

    assert result["selected_alpha"] == 0.2
    assert result["status"] == "smoke_or_development"
    assert result["test_results"]["bm25"]["metrics"]["query_count"] == 3
    assert all(
        text == "search text dev1"
        for mode, alpha, text in calls
        if mode == "hybrid" and alpha == 0.5 and text.startswith("search text dev")
    )
    markdown = render_markdown_report(result)
    assert "Smoke / Development Run" in markdown
    assert "does not satisfy the exact official Benchmark v1" in markdown
    assert "Selected alpha: **0.20**" in markdown


def test_manifest_reads_and_cross_checks_real_artifacts(tmp_path):
    import faiss
    import numpy as np

    db_path = tmp_path / "metadata.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE papers (paper_id TEXT PRIMARY KEY)")
    conn.execute(
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, paper_id TEXT, chunk_text TEXT)"
    )
    conn.execute("CREATE TABLE chunk_fts (chunk_id TEXT)")
    conn.executemany("INSERT INTO papers VALUES (?)", [("P1",), ("P2",)])
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?)",
        [("C1", "P1", "text one"), ("C2", "P2", "text two")],
    )
    conn.executemany("INSERT INTO chunk_fts VALUES (?)", [("C1",), ("C2",)])
    conn.commit()
    conn.close()

    index_dir = tmp_path / "faiss"
    index_dir.mkdir()
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "id_map.json").write_text(
        json.dumps(
            [
                {"faiss_id": 0, "chunk_id": "C1", "paper_id": "P1"},
                {"faiss_id": 1, "chunk_id": "C2", "paper_id": "P2"},
            ]
        ),
        encoding="utf-8",
    )
    (index_dir / "build_meta.json").write_text(
        json.dumps(
            {"embedding_model": "fake", "vector_dim": 2, "num_vectors": 2}
        ),
        encoding="utf-8",
    )
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text('{"paper_id":"P1"}\n{"paper_id":"P2"}\n', encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"query_id":"q1"}\n', encoding="utf-8")

    manifest = build_benchmark_manifest(
        db_path=db_path,
        index_dir=index_dir,
        raw_path=raw_path,
        eval_path=eval_path,
        corpus="smoke",
    )

    assert manifest["paper_count"] == 2
    assert manifest["chunk_count"] == 2
    assert manifest["fts_row_count"] == 2
    assert manifest["faiss_vector_count"] == 2
    assert manifest["id_map_count"] == 2
    assert manifest["embedding_model"] == "fake"
    assert len(manifest["raw_file_sha256"]) == 64
    assert len(manifest["eval_file_sha256"]) == 64


def test_official_gate_requires_exactly_one_chunk_per_paper():
    queries = [
        EvalQuery(
            query_id=f"q{index + 1:04d}",
            query=f"benchmark query {index + 1}",
            query_type=("keyword", "natural_question", "semantic_paraphrase")[
                index % 3
            ],
            split="dev" if index < 50 else "test",
            relevant_paper_ids=(f"P{index + 1}",),
            source_paper_id=f"P{index + 1}",
            source_category="Computer Science",
        )
        for index in range(150)
    ]
    manifest = {
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
        "git_commit": "commit",
        "git_dirty": False,
    }

    assert _benchmark_status(manifest, queries) == "benchmark_v1"

    manifest.update(
        chunk_count=50_001,
        fts_row_count=50_001,
        faiss_vector_count=50_001,
        id_map_count=50_001,
    )
    assert _benchmark_status(manifest, queries) == "smoke_or_development"
