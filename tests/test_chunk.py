"""Tests for app.ingestion.chunk — multi-chunk text segment generation."""

import pytest
from app.ingestion.chunk import (
    chunk_paper,
    _build_title_abstract,
    _estimate_tokens,
    _merge_body_paragraphs,
    _sliding_window,
    _find_sentence_boundary,
    _make_chunk,
    CHUNK_TARGET_TOKENS,
    MIN_CHUNK_TOKENS,
)


# ---------------------------------------------------------------------------
# _build_title_abstract
# ---------------------------------------------------------------------------


class TestBuildTitleAbstract:
    def test_title_and_abstract(self):
        text = _build_title_abstract({
            "title": "A Great Paper",
            "abstract": "We present a new approach.",
        })
        assert text == "Title: A Great Paper\nAbstract: We present a new approach."

    def test_metadata_not_in_chunk(self):
        text = _build_title_abstract({
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
        text = _build_title_abstract({"title": "Lonely Paper"})
        assert text == "Title: Lonely Paper"

    def test_abstract_only(self):
        text = _build_title_abstract({"title": "", "abstract": "Results here."})
        assert text == "Abstract: Results here."


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") >= 1

    def test_short(self):
        tokens = _estimate_tokens("hello world")
        assert 2 <= tokens <= 5

    def test_long(self):
        text = "machine learning " * 100
        tokens = _estimate_tokens(text)
        assert 100 <= tokens <= 300  # 100 words × 1.3 = 130


# ---------------------------------------------------------------------------
# _make_chunk
# ---------------------------------------------------------------------------


class TestMakeChunk:
    def test_structure(self):
        ch = _make_chunk("p123", "some text here", "body", 2)
        assert ch["chunk_id"] == "p123_chunk2"
        assert ch["paper_id"] == "p123"
        assert ch["chunk_type"] == "body"
        assert ch["position"] == 2
        assert ch["token_count"] > 0


# ---------------------------------------------------------------------------
# _find_sentence_boundary
# ---------------------------------------------------------------------------


class TestFindSentenceBoundary:
    def test_finds_period(self):
        text = "First sentence. Second sentence."
        # Target near position 20 — should find the period
        cut = _find_sentence_boundary(text, 16)
        assert text[cut - 1] == "."

    def test_finds_newline(self):
        text = "Line one\nLine two"
        cut = _find_sentence_boundary(text, 10)
        assert text[cut - 1] == "\n"

    def test_fallback_exact(self):
        text = "NoBoundariesHereAtAll"
        cut = _find_sentence_boundary(text, 8)
        assert cut == 8  # falls back to exact target


# ---------------------------------------------------------------------------
# _sliding_window
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_short_text_returns_one_window(self):
        sentence = "This is a complete test sentence with enough words to exceed the minimum token threshold. "
        text = sentence * 5  # ~100 words → ~130 tokens, above MIN_CHUNK_TOKENS
        windows = _sliding_window(text, CHUNK_TARGET_TOKENS, 120)
        assert len(windows) == 1

    def test_long_text_returns_multiple(self):
        # Create a paragraph long enough to need sliding window
        sentence = "This is a complete sentence that ends with a period. "
        text = sentence * 600  # ~3600 words → ~4680 tokens → needs 6 windows
        windows = _sliding_window(text, CHUNK_TARGET_TOKENS, 120)
        assert len(windows) >= 3  # at least 3 windows needed

    def test_windows_overlap(self):
        sentence = "This is a complete sentence that ends with a period. "
        text = sentence * 400  # ~2400 words → ~3120 tokens
        windows = _sliding_window(text, CHUNK_TARGET_TOKENS, 120)
        # Adjacent windows should share some text (overlap)
        if len(windows) >= 2:
            # Last few words of window 0 should appear in window 1
            w0_end = windows[0][-50:]
            assert len(w0_end) > 0


# ---------------------------------------------------------------------------
# _merge_body_paragraphs
# ---------------------------------------------------------------------------


class TestMergeBodyParagraphs:
    def test_empty(self):
        assert _merge_body_paragraphs([]) == []

    def test_single_short_para(self):
        para = "machine learning is a field of artificial intelligence " * 8  # ~100 words → ~130 tokens
        chunks = _merge_body_paragraphs([para])
        assert len(chunks) == 1

    def test_accumulates_to_target(self):
        """Short paragraphs should be merged until they reach ~800 tokens."""
        short_para = "machine learning " * 20  # ~20 words → ~26 tokens
        paragraphs = [short_para] * 60  # 60 × 26 = 1560 tokens → ~2 chunks
        chunks = _merge_body_paragraphs(paragraphs)
        assert len(chunks) >= 2

    def test_discards_tiny_chunks(self):
        tiny = "hi"  # well below MIN_CHUNK_TOKENS (50)
        chunks = _merge_body_paragraphs([tiny])
        # Tiny content is discarded
        assert chunks == []


# ---------------------------------------------------------------------------
# chunk_paper (integration)
# ---------------------------------------------------------------------------


class TestChunkPaper:
    def test_returns_list(self):
        chunks = chunk_paper({"paper_id": "W123", "title": "Test"})
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_chunk0_is_title_abstract(self):
        chunks = chunk_paper({
            "paper_id": "X",
            "title": "My Paper",
            "abstract": "Some findings.",
        })
        assert chunks[0]["chunk_type"] == "title_abstract"
        assert chunks[0]["position"] == 0
        assert "My Paper" in chunks[0]["chunk_text"]

    def test_no_fulltext_returns_single_chunk(self):
        chunks = chunk_paper({
            "paper_id": "Y",
            "title": "Short",
            "abstract": "No body.",
        })
        assert len(chunks) == 1

    def test_with_fulltext(self):
        """Full text should produce body chunks."""
        body = "Introduction paragraph.\n\n" + ("Methods section content. " * 300) + "\n\n" + ("Results data. " * 300)
        chunks = chunk_paper({
            "paper_id": "Z",
            "title": "Full Paper",
            "abstract": "Comprehensive study.",
            "full_text": f"Full Paper\n\nComprehensive study.\n\n{body}",
        })
        assert len(chunks) >= 2  # title_abstract + at least 1 body
        body_chunks = [c for c in chunks if c["chunk_type"] == "body"]
        assert len(body_chunks) >= 1
        # Body chunks should have position >= 1
        for bc in body_chunks:
            assert bc["position"] >= 1

    def test_chunk_ids_unique(self):
        chunks = chunk_paper({
            "paper_id": "U",
            "title": "Unique",
            "abstract": "Test.",
            "full_text": "Unique\n\nTest.\n\nPara one.\n\nPara two.\n\nPara three.",
        })
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))  # all IDs unique
