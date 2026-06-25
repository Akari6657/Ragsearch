"""Tests for app.ingestion.normalize — field validation and cleaning."""

import pytest

from app.ingestion.normalize import normalize, _clean_str, _to_str_list, _to_int, _to_bool


class TestHelpers:
    def test_clean_str(self):
        assert _clean_str("  hello  ") == "hello"
        assert _clean_str(42) == "42"
        assert _clean_str(None) == ""

    def test_to_str_list_list(self):
        assert _to_str_list(["A", "B"]) == ["A", "B"]
        assert _to_str_list(["  A  ", "B"]) == ["A", "B"]

    def test_to_str_list_comma_string(self):
        assert _to_str_list("Alice, Bob, Carol") == ["Alice", "Bob", "Carol"]

    def test_to_str_list_single_string(self):
        assert _to_str_list("Alice") == ["Alice"]

    def test_to_str_list_none(self):
        assert _to_str_list(None) == []

    def test_to_int_valid(self):
        assert _to_int("2023") == 2023
        assert _to_int(2023) == 2023

    def test_to_int_invalid(self):
        assert _to_int("notanumber") is None
        assert _to_int(None) is None

    def test_to_int_min_val(self):
        assert _to_int(-5, min_val=0) == 0
        assert _to_int(5, min_val=0) == 5

    def test_to_bool(self):
        assert _to_bool(True) is True
        assert _to_bool(False) is False
        assert _to_bool("true") is True
        assert _to_bool("1") is True
        assert _to_bool(1) is True
        assert _to_bool("false") is False
        assert _to_bool(0) is False


class TestNormalize:
    def test_minimal_valid(self):
        result = normalize({"paper_id": "X", "title": "Y"})
        assert result is not None
        assert result["paper_id"] == "X"
        assert result["title"] == "Y"
        assert result["abstract"] == ""
        assert result["year"] is None
        assert result["authors"] == []

    def test_missing_paper_id(self):
        assert normalize({"title": "Y"}) is None

    def test_empty_paper_id(self):
        assert normalize({"paper_id": "", "title": "Y"}) is None
        assert normalize({"paper_id": "   ", "title": "Y"}) is None

    def test_missing_title(self):
        assert normalize({"paper_id": "X"}) is None

    def test_empty_title(self):
        assert normalize({"paper_id": "X", "title": ""}) is None
        assert normalize({"paper_id": "X", "title": "   "}) is None

    def test_full_record(self):
        result = normalize({
            "paper_id": "W123",
            "title": "  A Great Paper  ",
            "abstract": "  Important findings.  ",
            "year": 2023,
            "venue": "SIGIR 2023",
            "authors": ["Alice", "Bob"],
            "concepts": ["Information Retrieval"],
            "doi": "10.1234/foo",
            "url": "https://example.org",
            "citation_count": 42,
            "open_access": True,
        })
        assert result is not None
        assert result["title"] == "A Great Paper"
        assert result["abstract"] == "Important findings."
        assert result["year"] == 2023
        assert result["venue"] == "SIGIR 2023"
        assert result["authors"] == ["Alice", "Bob"]
        assert result["concepts"] == ["Information Retrieval"]
        assert result["citation_count"] == 42
        assert result["open_access"] is True

    def test_string_year(self):
        result = normalize({"paper_id": "X", "title": "Y", "year": "2023"})
        assert result["year"] == 2023

    def test_invalid_year(self):
        result = normalize({"paper_id": "X", "title": "Y", "year": "not-a-year"})
        assert result["year"] is None

    def test_comma_separated_authors(self):
        result = normalize({"paper_id": "X", "title": "Y", "authors": "A, B, C"})
        assert result["authors"] == ["A", "B", "C"]

    def test_string_concepts(self):
        result = normalize({"paper_id": "X", "title": "Y", "concepts": "ML"})
        assert result["concepts"] == ["ML"]

    def test_negative_citation_count(self):
        result = normalize({"paper_id": "X", "title": "Y", "citation_count": -5})
        assert result["citation_count"] == 0

    def test_empty_venue_becomes_none(self):
        result = normalize({"paper_id": "X", "title": "Y", "venue": ""})
        assert result["venue"] is None

    def test_empty_doi_becomes_none(self):
        result = normalize({"paper_id": "X", "title": "Y", "doi": ""})
        assert result["doi"] is None
