"""Tests for scripts.sample_corpus."""

from __future__ import annotations

import json

import pytest

from scripts.sample_corpus import sample_corpus


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_sample_corpus_writes_requested_size(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "sample.jsonl"
    _write_jsonl(input_path, [{"paper_id": f"p{i}"} for i in range(10)])

    valid_seen, written = sample_corpus(input_path, output_path, size=4, seed=7)

    assert valid_seen == 10
    assert written == 4
    assert len(_read_jsonl(output_path)) == 4


def test_sample_corpus_is_deterministic_for_seed(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_a = tmp_path / "sample_a.jsonl"
    output_b = tmp_path / "sample_b.jsonl"
    _write_jsonl(input_path, [{"paper_id": f"p{i}"} for i in range(20)])

    sample_corpus(input_path, output_a, size=5, seed=42)
    sample_corpus(input_path, output_b, size=5, seed=42)

    assert output_a.read_text(encoding="utf-8") == output_b.read_text(encoding="utf-8")


def test_sample_corpus_writes_all_when_input_is_smaller(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "sample.jsonl"
    _write_jsonl(input_path, [{"paper_id": "p1"}, {"paper_id": "p2"}])

    valid_seen, written = sample_corpus(input_path, output_path, size=10, seed=1)

    assert valid_seen == 2
    assert written == 2
    assert [r["paper_id"] for r in _read_jsonl(output_path)] == ["p1", "p2"]


def test_sample_corpus_rejects_non_positive_size(tmp_path):
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, [{"paper_id": "p1"}])

    with pytest.raises(ValueError, match="size"):
        sample_corpus(input_path, tmp_path / "sample.jsonl", size=0)
