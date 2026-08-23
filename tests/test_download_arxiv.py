"""Tests for deterministic and resumable arXiv corpus downloads."""

from __future__ import annotations

import json
from datetime import date

import pytest

from scripts.download_arxiv import (
    CATEGORY_BALANCED_STRATEGY,
    CS_CATEGORIES,
    DOWNLOAD_CATEGORY_FIELD,
    _balanced_category_targets,
    _load_existing_state,
    _year_windows,
)


def test_default_categories_are_unique_and_queryable_cs_categories():
    assert len(CS_CATEGORIES) == len(set(CS_CATEGORIES))
    assert "cs.SY" not in CS_CATEGORIES
    assert "cs.DC" in CS_CATEGORIES


def test_balanced_category_targets_are_exact_and_deterministic():
    assert _balanced_category_targets(["cs.AI", "cs.CL", "cs.IR"], 8) == {
        "cs.AI": 3,
        "cs.CL": 3,
        "cs.IR": 2,
    }


def test_year_windows_are_inclusive_and_newest_first():
    assert _year_windows(2024, date(2026, 8, 23)) == [
        ("202601010000", "202608232359"),
        ("202501010000", "202512312359"),
        ("202401010000", "202412312359"),
    ]


def test_load_existing_category_balanced_state(tmp_path):
    output = tmp_path / "papers.jsonl"
    records = [
        {
            "paper_id": "P1",
            "primary_category": "cs.AI",
            DOWNLOAD_CATEGORY_FIELD: "cs.AI",
        },
        {
            "paper_id": "P2",
            "primary_category": "cs.AI",
            DOWNLOAD_CATEGORY_FIELD: "cs.AI",
        },
        {
            "paper_id": "P3",
            "primary_category": "cs.IR",
            DOWNLOAD_CATEGORY_FIELD: "cs.IR",
        },
    ]
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    paper_ids, counts = _load_existing_state(
        output,
        strategy=CATEGORY_BALANCED_STRATEGY,
        categories=["cs.AI", "cs.IR"],
    )

    assert paper_ids == {"P1", "P2", "P3"}
    assert counts == {"cs.AI": 2, "cs.IR": 1}


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"paper_id": "P1"}], "Invalid or missing"),
        (
            [
                {
                    "paper_id": "P1",
                    "primary_category": "cs.AI",
                    DOWNLOAD_CATEGORY_FIELD: "cs.AI",
                },
                {
                    "paper_id": "P1",
                    "primary_category": "cs.AI",
                    DOWNLOAD_CATEGORY_FIELD: "cs.AI",
                },
            ],
            "Duplicate paper_id",
        ),
        (
            [
                {
                    "paper_id": "P1",
                    "primary_category": "cs.CL",
                    DOWNLOAD_CATEGORY_FIELD: "cs.AI",
                }
            ],
            "primary_category does not match",
        ),
    ],
)
def test_load_existing_state_rejects_incompatible_resume(tmp_path, records, message):
    output = tmp_path / "papers.jsonl"
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _load_existing_state(
            output,
            strategy=CATEGORY_BALANCED_STRATEGY,
            categories=["cs.AI"],
        )
