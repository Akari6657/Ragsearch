"""Tests for app.rag.router — query classification."""

import pytest

from app.rag.router import (
    route_query,
    _extract_signals,
    _rule_classify,
    _detect_language,
)


class TestDetectLanguage:
    def test_chinese(self):
        assert _detect_language("为什么GAN训练不稳定？") == "zh"

    def test_english(self):
        assert _detect_language("attention mechanism") == "en"

    def test_mixed(self):
        assert _detect_language("对比BERT和GPT") == "zh"


class TestExtractSignals:
    def test_chinese_question(self):
        s = _extract_signals("什么是注意力机制")
        assert s["has_question_words"] is True
        assert s["language"] == "zh"

    def test_english_question(self):
        s = _extract_signals("What is attention mechanism?")
        assert s["has_question_words"] is True
        assert s["language"] == "en"

    def test_keywords(self):
        s = _extract_signals("GAN image generation")
        assert s["is_pure_keywords"] is True
        assert s["has_question_words"] is False


class TestRuleClassify:
    def test_question_word_triggers_rag(self):
        r = _rule_classify({"token_count": 6, "has_question_words": True,
                            "has_comparison_words": False, "is_pure_keywords": False,
                            "language": "zh"})
        assert r is not None
        assert r.should_rag is True

    def test_short_keywords_no_rag(self):
        r = _rule_classify({"token_count": 3, "has_question_words": False,
                            "has_comparison_words": False, "is_pure_keywords": True,
                            "language": "en"})
        assert r is not None
        assert r.should_rag is False

    def test_ultra_short_no_rag(self):
        r = _rule_classify({"token_count": 2, "has_question_words": False,
                            "has_comparison_words": False, "is_pure_keywords": False,
                            "language": "en"})
        assert r is not None
        assert r.should_rag is False


class TestRouteQuery:
    def test_chinese_question(self):
        r = route_query("为什么GAN训练不稳定？")
        assert r.should_rag is True
        assert r.needs_rewrite is True

    def test_short_keywords(self):
        r = route_query("GAN image generation")
        assert r.should_rag is False

    def test_force_rag(self):
        r = route_query("GAN", force_rag=True)
        assert r.should_rag is True
