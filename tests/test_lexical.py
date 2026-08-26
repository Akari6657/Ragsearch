"""Tests for app.retrieval.lexical — BM25 keyword search."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from app.retrieval.lexical import search_lexical, _parse_query


# ---------------------------------------------------------------------------
# Fixture: tiny test database with FTS5
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db():
    """Create a temporary SQLite database with 5 papers, chunks, and FTS5 index."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    db_path = Path(tmp.name)
    tmp.close()

    conn = sqlite3.connect(str(db_path))

    # Papers table
    conn.execute("""
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT DEFAULT '',
            year INTEGER,
            venue TEXT,
            authors_json TEXT DEFAULT '[]',
            concepts_json TEXT DEFAULT '[]',
            doi TEXT,
            url TEXT,
            citation_count INTEGER DEFAULT 0,
            open_access INTEGER DEFAULT 0
        )
    """)

    # Chunks table
    conn.execute("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_type TEXT DEFAULT 'metadata',
            token_count INTEGER DEFAULT 0
        )
    """)

    # Insert test data
    papers = [
        ("P1", "Neural Networks for Image Recognition", "A paper about neural networks.", 2020, "CVPR"),
        ("P2", "Graph Algorithms in Social Networks", "Graph theory applied to social media.", 2019, "WWW"),
        ("P3", "Information Retrieval with BM25", "A survey of retrieval models.", 2021, "SIGIR"),
        ("P4", "Quantum Computing for Machine Learning", "Quantum approaches to ML.", 2022, "Nature"),
        ("P5", "Deep Learning for NLP Tasks", "Using transformers for text.", 2023, "ACL"),
    ]
    for pid, title, abstract, year, venue in papers:
        conn.execute(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, '[]', '[]', NULL, NULL, 0, 1)",
            (pid, title, abstract, year, venue),
        )
        chunk_text = f"Title: {title}\nAbstract: {abstract}"
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, 'metadata', ?)",
            (f"{pid}_default", pid, chunk_text, len(chunk_text) // 4),
        )

    # FTS5
    conn.execute("""
        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            chunk_id UNINDEXED,
            paper_id UNINDEXED,
            title,
            chunk_text
        )
    """)
    conn.execute("""
        INSERT INTO chunk_fts(chunk_id, paper_id, title, chunk_text)
        SELECT c.chunk_id, c.paper_id, p.title, c.chunk_text
        FROM chunks c JOIN papers p ON c.paper_id = p.paper_id
    """)
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseQuery:
    def test_simple_or(self):
        fts5, phrases = _parse_query("neural network")
        assert fts5 == "neural OR network"
        assert phrases == []

    def test_quoted_phrase(self):
        fts5, phrases = _parse_query('"neural network"')
        assert fts5 == "neural OR network"
        assert phrases == ["neural network"]

    def test_mixed(self):
        fts5, phrases = _parse_query('computer "neural network" optimization')
        assert fts5 == "computer OR neural OR network OR optimization"
        assert phrases == ["neural network"]

    def test_special_chars(self):
        fts5, phrases = _parse_query("test (query)")
        assert "(" not in fts5
        assert ")" not in fts5

    def test_sentence_punctuation_is_tokenized_safely(self):
        fts5, phrases = _parse_query(
            "Adaptive decoding, scheduling, and multi-step training."
        )
        assert fts5 == (
            "adaptive OR decoding OR scheduling OR and OR multi OR step OR training"
        )
        assert phrases == []

    def test_fts5_operator_words_are_lowercased(self):
        fts5, _ = _parse_query("OR AND NOT NEAR")
        assert fts5 == "or OR and OR not OR near"

    def test_only_punctuation_is_empty(self):
        fts5, phrases = _parse_query("... , ! ?")
        assert fts5 == ""
        assert phrases == []

    def test_empty(self):
        fts5, phrases = _parse_query("")
        assert fts5 == ""
        assert phrases == []


class TestSearchLexical:
    def test_multi_term_query_uses_or_semantics(self, test_db):
        results = search_lexical("neural graph", top_k=5, db_path=test_db)
        paper_ids = {result.paper_id for result in results}

        assert {"P1", "P2"}.issubset(paper_ids)

    def test_exact_match(self, test_db):
        results = search_lexical("neural networks", top_k=5, db_path=test_db)
        assert len(results) >= 1
        assert results[0].paper_id == "P1"

    def test_public_scores_are_higher_is_better(self, test_db):
        results = search_lexical(
            "neural graph learning", top_k=5, db_path=test_db
        )
        scores = [result.score for result in results]

        assert len(scores) >= 2
        assert all(score >= 0.0 for score in scores)
        assert scores == sorted(scores, reverse=True)

    def test_sentence_punctuation_does_not_break_search(self, test_db):
        results = search_lexical(
            "Neural networks, image recognition.", top_k=5, db_path=test_db
        )
        assert results
        assert results[0].paper_id == "P1"

    def test_partial_match(self, test_db):
        results = search_lexical("graph", top_k=5, db_path=test_db)
        assert any(r.paper_id == "P2" for r in results)

    def test_no_match(self, test_db):
        results = search_lexical("dinosaur paleontology", top_k=5, db_path=test_db)
        assert results == []

    def test_top_k_limit(self, test_db):
        results = search_lexical("learning", top_k=2, db_path=test_db)
        assert len(results) <= 2

    def test_empty_query(self, test_db):
        results = search_lexical("", top_k=5, db_path=test_db)
        assert results == []

    def test_missing_db(self):
        results = search_lexical("anything", top_k=5, db_path="/nonexistent/db.sqlite")
        assert results == []

    def test_result_fields(self, test_db):
        results = search_lexical("neural", top_k=1, db_path=test_db)
        assert len(results) == 1
        r = results[0]
        assert r.paper_id
        assert r.title
        assert r.score is not None
        assert isinstance(r.score, float)
