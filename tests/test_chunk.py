"""Tests for app.ingestion.chunk — text segment generation."""

from app.ingestion.chunk import chunk_paper, _build_chunk_text


class TestBuildChunkText:
    def test_all_fields(self):
        text = _build_chunk_text({
            "title": "A Great Paper",
            "year": 2023,
            "venue": "SIGIR",
            "concepts": ["IR", "ML"],
            "abstract": "We present a new approach.",
        })
        assert "Title: A Great Paper" in text
        assert "Year: 2023" in text
        assert "Venue: SIGIR" in text
        assert "Concepts: IR, ML" in text
        assert "Abstract: We present a new approach." in text

    def test_minimal_fields(self):
        text = _build_chunk_text({
            "title": "Lonely Paper",
        })
        assert "Title: Lonely Paper" in text
        assert "Year:" not in text
        assert "Abstract:" not in text

    def test_missing_optional(self):
        text = _build_chunk_text({
            "title": "T",
            "year": None,
            "venue": None,
            "concepts": [],
            "abstract": "",
        })
        # Only title should appear
        assert text == "Title: T"


class TestChunkPaper:
    def test_chunk_id_format(self):
        chunk = chunk_paper({"paper_id": "W123", "title": "Test Paper"})
        assert chunk["chunk_id"] == "W123_default"

    def test_chunk_type(self):
        chunk = chunk_paper({"paper_id": "X", "title": "Y"})
        assert chunk["chunk_type"] == "metadata"

    def test_token_count(self):
        chunk = chunk_paper({
            "paper_id": "X",
            "title": "A" * 100,
            "abstract": "B" * 300,
        })
        # Rough estimate: ~400 chars / 4 = 100 tokens
        assert chunk["token_count"] >= 80
        assert chunk["token_count"] <= 120
