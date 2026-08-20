"""Tests for deterministic Retrieval Benchmark v1 query construction."""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.rag.llm_provider import LLMProvider, LLMResponse
from scripts.build_retrieval_eval import (
    QueryTarget,
    TargetPaper,
    build_eval_set,
    build_query_plan,
    generate_query,
    load_eligible_targets,
    validate_generated_query,
)


class SequenceProvider(LLMProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system="", user="", **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        return LLMResponse(text=self.responses.pop(0), model="fake-generator")


class TypeAwareProvider(LLMProvider):
    def generate(self, system="", user="", **kwargs):
        if "Query type: keyword" in user:
            query = "robust methods for noisy prediction"
        elif "Query type: natural_question" in user:
            query = "How can models remain reliable under noisy supervision?"
        else:
            query = "reliable prediction techniques using imperfect training labels"
        return LLMResponse(text=json.dumps({"query": query}), model="fake-generator")


def make_db(path, count=12):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            authors_json TEXT NOT NULL,
            concepts_json TEXT NOT NULL
        )"""
    )
    abstract = " ".join(f"word{i}" for i in range(80))
    categories = ["AI", "IR", "CV"]
    for index in range(count):
        conn.execute(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?)",
            (
                f"P{index:02d}",
                f"Unique Research Topic Number {index}",
                abstract,
                json.dumps([f"Author {index}"]),
                json.dumps([categories[index % len(categories)]]),
            ),
        )
    conn.commit()
    conn.close()


def test_query_plan_split_and_type_assignment_are_deterministic(tmp_path):
    db_path = tmp_path / "metadata.sqlite"
    make_db(db_path)
    eligible = load_eligible_targets(db_path)

    plan_a = build_query_plan(eligible, size=9, dev_size=3, seed=42)
    plan_b = build_query_plan(eligible, size=9, dev_size=3, seed=42)

    assert plan_a == plan_b
    assert [target.split for target in plan_a].count("dev") == 3
    assert [target.split for target in plan_a].count("test") == 6
    assert {kind: [target.query_type for target in plan_a].count(kind) for kind in (
        "keyword", "natural_question", "semantic_paraphrase"
    )} == {"keyword": 3, "natural_question": 3, "semantic_paraphrase": 3}
    assert max(
        [target.query_type for target in plan_a[:3]].count(kind)
        for kind in ("keyword", "natural_question", "semantic_paraphrase")
    ) == 1
    assert {target.paper.source_category for target in plan_a} == {"AI", "IR", "CV"}


def test_duplicate_normalized_titles_and_short_abstracts_are_ineligible(tmp_path):
    db_path = tmp_path / "metadata.sqlite"
    make_db(db_path, count=3)
    conn = sqlite3.connect(db_path)
    long_abstract = " ".join("word" for _ in range(80))
    conn.execute(
        "INSERT INTO papers VALUES ('D1', 'Same: Title!', ?, '[]', '[\"AI\"]')",
        (long_abstract,),
    )
    conn.execute(
        "INSERT INTO papers VALUES ('D2', 'same title', ?, '[]', '[\"AI\"]')",
        (long_abstract,),
    )
    conn.execute(
        "INSERT INTO papers VALUES ('SHORT', 'A Different Title', 'too short', '[]', '[\"AI\"]')"
    )
    conn.commit()
    conn.close()

    ids = {paper.paper_id for paper in load_eligible_targets(db_path)}

    assert "D1" not in ids
    assert "D2" not in ids
    assert "SHORT" not in ids


def test_query_generation_retries_invalid_title_copy_at_low_temperature():
    paper = TargetPaper(
        paper_id="2301.00001",
        title="Learning Robust Representations from Noisy Labels",
        abstract="An abstract about robust learning with imperfect annotations.",
        authors=("Ada Researcher",),
        source_category="Machine Learning",
    )
    target = QueryTarget("q0001", "dev", "keyword", paper)
    provider = SequenceProvider(
        [
            '{"query": "Learning Robust Representations from Noisy Labels"}',
            '{"query": "robust learning under imperfect supervision"}',
        ]
    )

    generated, model = generate_query(provider, target)

    assert generated == "robust learning under imperfect supervision"
    assert model == "fake-generator"
    assert len(provider.calls) == 2
    assert provider.calls[0]["temperature"] == 0.0


def test_generated_query_constraints_cover_id_author_and_question_form():
    paper = TargetPaper(
        paper_id="2301.00001",
        title="A Novel Method for Graph Learning",
        abstract="Useful abstract.",
        authors=("Ada Researcher",),
        source_category="AI",
    )
    target = QueryTarget("q1", "dev", "natural_question", paper)

    reasons = validate_generated_query("Ada Researcher 2301.00001 graph learning", target)

    assert any("author" in reason for reason in reasons)
    assert any("paper ID" in reason for reason in reasons)
    assert any("question mark" in reason for reason in reasons)


def test_build_eval_set_freezes_valid_file_and_refuses_accidental_overwrite(tmp_path):
    db_path = tmp_path / "metadata.sqlite"
    output_path = tmp_path / "retrieval_v1.jsonl"
    make_db(db_path, count=6)

    records = build_eval_set(
        db_path=db_path,
        output_path=output_path,
        provider=TypeAwareProvider(),
        size=3,
        dev_size=1,
        seed=42,
    )

    assert len(records) == 3
    assert output_path.exists()
    assert not (tmp_path / "retrieval_v1.jsonl.partial").exists()
    assert sorted(record["query_type"] for record in records) == [
        "keyword", "natural_question", "semantic_paraphrase"
    ]
    with pytest.raises(FileExistsError, match="Frozen evaluation set"):
        build_eval_set(
            db_path=db_path,
            output_path=output_path,
            provider=TypeAwareProvider(),
            size=3,
            dev_size=1,
            seed=42,
        )
