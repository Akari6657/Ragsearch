"""Tests for app.api.routes_search readiness behavior."""

from types import SimpleNamespace

from fastapi import HTTPException

from app.api import routes_search
from app.core.schemas import SearchRequest, SearchResult


def test_faiss_ready_requires_index_and_id_map(tmp_path):
    assert routes_search._faiss_ready(tmp_path) is False

    (tmp_path / "index.faiss").write_text("fake", encoding="utf-8")
    assert routes_search._faiss_ready(tmp_path) is False

    (tmp_path / "id_map.json").write_text("[]", encoding="utf-8")
    assert routes_search._faiss_ready(tmp_path) is True


def test_vector_search_requires_faiss(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)

    request = SearchRequest(query="GAN image generation", mode="vector", top_k=3)

    try:
        routes_search.search_papers(request)
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail["error_code"] == "INDEX_NOT_READY"
    else:
        raise AssertionError("Expected vector search to fail when FAISS is missing")


def test_hybrid_search_requires_faiss(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)

    request = SearchRequest(query="GAN image generation", mode="hybrid", top_k=3)

    try:
        routes_search.search_papers(request)
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail["error_code"] == "INDEX_NOT_READY"
    else:
        raise AssertionError("Expected hybrid search to fail when FAISS is missing")


def test_lexical_search_does_not_require_faiss(monkeypatch, tmp_path):
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)
    monkeypatch.setattr(routes_search, "search_lexical", lambda *args, **kwargs: [])

    request = SearchRequest(query="GAN image generation", mode="lexical", top_k=3)
    response = routes_search.search_papers(request)

    assert response.mode == "lexical"
    assert response.total_results == 0
    assert response.results == []


def test_paper_deduplication_preserves_retrieval_rank(monkeypatch, tmp_path):
    ranked_results = [
        SearchResult(
            paper_id="P1",
            chunk_id="P1_best",
            title="Paper one",
            year=2024,
            venue=None,
            authors=[],
            score=0.1,
            abstract="Best-ranked chunk",
        ),
        SearchResult(
            paper_id="P1",
            chunk_id="P1_later",
            title="Paper one",
            year=2024,
            venue=None,
            authors=[],
            score=0.9,
            abstract="Later chunk with an incomparable score",
        ),
        SearchResult(
            paper_id="P2",
            chunk_id="P2_only",
            title="Paper two",
            year=2023,
            venue=None,
            authors=[],
            score=0.05,
            abstract="Second paper",
        ),
    ]

    monkeypatch.setattr(routes_search, "get_db_path", lambda: tmp_path / "metadata.sqlite")
    monkeypatch.setattr(routes_search, "get_faiss_dir", lambda: tmp_path)
    monkeypatch.setattr(
        routes_search, "search_lexical", lambda *args, **kwargs: ranked_results
    )
    monkeypatch.setattr(
        routes_search,
        "route_query",
        lambda query: SimpleNamespace(should_rag=False, reason="test"),
    )

    response = routes_search.search_papers(
        SearchRequest(query="ranked chunks", mode="lexical", top_k=3)
    )

    assert response.total_results == 2
    assert [result.chunk_id for result in response.results] == ["P1_best", "P2_only"]
