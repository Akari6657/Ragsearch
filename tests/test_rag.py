"""Tests for RAG modules — all use MockLLM, no real API calls."""

import pytest

from app.rag.citation import extract_citations, verify_citations
from app.rag.context_builder import build_evidence, _estimate_tokens
from app.rag.llm_provider import MockLLMProvider, create_provider, LLMResponse
from app.rag.prompt import build_prompts
from app.core.schemas import SearchResult


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_basic(self):
        assert extract_citations("结论 [1] 和 [2] 表明了这一点。") == [1, 2]

    def test_deduplicate(self):
        assert extract_citations("[1] [1] [2] [1]") == [1, 2]

    def test_no_citations(self):
        assert extract_citations("没有引用。") == []

    def test_large_numbers(self):
        assert extract_citations("参见 [1] 和 [123]") == [1, 123]

    def test_not_citation_brackets(self):
        # Python list syntax should not be caught
        assert extract_citations("list = [1, 2, 3]") == []


class TestVerifyCitations:
    def _map(self, ids):
        return [{"citation_id": i, "title": f"Paper {i}", "paper_id": f"P{i}"} for i in ids]

    def test_all_valid(self):
        r = verify_citations("如 [1][2] 所示。", self._map([1, 2, 3]))
        assert r.valid is True

    def test_invalid_id(self):
        r = verify_citations("如 [1] 和 [5] 所示。", self._map([1, 2]))
        assert r.valid is False
        assert 5 in r.invalid_ids

    def test_no_citations(self):
        r = verify_citations("没有引用。", self._map([1, 2]))
        assert r.valid is False
        assert len(r.warnings) > 0


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


class TestBuildPrompts:
    def test_returns_tuple(self):
        system, user = build_prompts("[1] Test evidence", "测试问题")
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_chinese_content(self):
        system, user = build_prompts("[1] 证据", "问题？")
        assert "证据" in user
        assert "问题？" in user

    def test_evidence_in_user(self):
        _, user = build_prompts("[1] GAN paper abstract here", "什么是GAN？")
        assert "[1] GAN paper abstract here" in user
        assert "什么是GAN？" in user


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 1

    def test_short(self):
        assert _estimate_tokens("hello world") == 7  # 15 chars / 1.5 = 10 -> max(1, 10)


class TestBuildEvidence:
    def _make_result(self, chunk_id, paper_id, title, **kwargs):
        return SearchResult(
            paper_id=paper_id,
            chunk_id=chunk_id,
            title=title,
            year=kwargs.get("year"),
            venue=kwargs.get("venue"),
            authors=kwargs.get("authors", []),
            score=kwargs.get("score", 0.5),
            snippet=kwargs.get("snippet", ""),
        )

    def test_empty_results(self):
        text, cmap = build_evidence([])
        assert text == ""
        assert cmap == []

    def test_formats_citation_ids(self):
        """build_evidence should assign [1], [2] IDs."""
        r1 = self._make_result("C1", "P1", "Paper One")
        r2 = self._make_result("C2", "P2", "Paper Two")
        text, cmap = build_evidence([r1, r2])
        assert "[1]" in text
        assert "[2]" in text
        assert cmap[0]["citation_id"] == 1
        assert cmap[1]["citation_id"] == 2

    def test_token_budget(self):
        """With a tiny token budget, only 1 result should be included."""
        r1 = self._make_result("C1", "P1", "X" * 200)
        r2 = self._make_result("C2", "P2", "Y" * 200)
        text, cmap = build_evidence([r1, r2], max_tokens=10)
        assert "[1]" in text
        assert "[2]" not in text  # budget was hit
        assert len(cmap) == 1


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------


class TestMockLLM:
    def test_returns_fixed_response(self):
        llm = MockLLMProvider("固定回答 [1]。")
        r = llm.generate(system="sys", user="question")
        assert r.text == "固定回答 [1]。"
        assert r.model == "mock"
        assert r.latency_ms == 0.0

    def test_default_response(self):
        llm = MockLLMProvider()
        r = llm.generate(system="sys", user="question")
        assert len(r.text) > 0


class TestCreateProvider:
    def test_mock_when_no_key(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        provider = create_provider()
        assert isinstance(provider, MockLLMProvider)
