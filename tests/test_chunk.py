"""Tests for app.ingestion.chunk — text segment generation."""

from app.ingestion.chunk import chunk_paper, _build_chunk_text


class TestBuildChunkText:
    def test_title_and_abstract(self):
        text = _build_chunk_text({
            "title": "A Great Paper",
            "abstract": "We present a new approach.",
        })
        assert text == "Title: A Great Paper\nAbstract: We present a new approach."

    def test_metadata_not_in_chunk(self):
        """Year, venue, concepts should NOT appear in chunk text — only Title + Abstract."""
        text = _build_chunk_text({
            "title": "Test",
            "year": 2023,
            "venue": "SIGIR",
            "concepts": ["IR", "ML"],
            "abstract": "Findings.",
        })
        assert "Year" not in text
        assert "SIGIR" not in text
        assert "IR" not in text
        assert "Title: Test" in text
        assert "Abstract: Findings." in text

    def test_title_only(self):
        text = _build_chunk_text({"title": "Lonely Paper"})
        assert text == "Title: Lonely Paper"

    def test_abstract_only(self):
        text = _build_chunk_text({"title": "", "abstract": "Results here."})
        assert text == "Abstract: Results here."


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
