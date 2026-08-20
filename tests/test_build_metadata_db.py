"""Tests for clean, reproducible metadata database builds."""

from __future__ import annotations

import json
import sqlite3

import pytest

from scripts.build_metadata_db import build_db


def _write_papers(path, paper_ids):
    with open(path, "w", encoding="utf-8") as handle:
        for paper_id in paper_ids:
            handle.write(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "title": f"Title {paper_id}",
                        "abstract": f"Abstract {paper_id}",
                    }
                )
                + "\n"
            )


def test_build_db_refuses_implicit_merge_and_overwrite_replaces_corpus(tmp_path):
    first_raw = tmp_path / "first.jsonl"
    second_raw = tmp_path / "second.jsonl"
    db_path = tmp_path / "metadata.sqlite"
    _write_papers(first_raw, ["P1", "P2"])
    _write_papers(second_raw, ["P3"])

    assert build_db(first_raw, db_path) == (2, 2)
    with pytest.raises(FileExistsError, match="--overwrite"):
        build_db(second_raw, db_path)

    assert build_db(second_raw, db_path, overwrite=True) == (1, 1)
    conn = sqlite3.connect(db_path)
    try:
        paper_ids = [row[0] for row in conn.execute("SELECT paper_id FROM papers")]
    finally:
        conn.close()
    assert paper_ids == ["P3"]

