"""
Query router: decide whether to trigger RAG (AI Overview) for a query.

Two-level cascade:
  Level 1: Rule-based (< 1ms) — question/comparison keywords, token count
  Level 2: LLM fallback (~0.5s) — lightweight YES/NO classification

Usage:
    from app.rag.router import route_query
    result = route_query("为什么GAN训练不稳定？")
    if result.should_rag:
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

QUESTION_WORDS_EN: set[str] = {
    "what", "how", "why", "when", "where", "who", "which",
    "explain", "describe", "define", "compare",
}

QUESTION_WORDS_ZH: set[str] = {
    "什么", "怎么", "怎样", "如何", "为什么", "为何",
    "哪", "谁", "何时", "解释", "说明", "定义", "对比", "比较",
}

COMPARISON_WORDS_EN: set[str] = {
    "compare", "vs", "versus", "difference", "differences",
    "contrast", "better", "worse", "advantage", "drawback",
}

COMPARISON_WORDS_ZH: set[str] = {
    "对比", "比较", "区别", "差异", "优劣",
    "哪个好", "更好", "优缺点", "vs",
}

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RouterResult:
    """Routing decision for a query."""

    should_rag: bool
    """Whether to trigger RAG (AI Overview)."""

    confidence: float
    """Confidence in the decision (0.0–1.0)."""

    reason: str
    """Human-readable explanation of the decision."""

    language: str
    """Detected language: 'en' or 'zh'."""

    needs_rewrite: bool
    """Whether query rewriting would improve retrieval."""


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def _is_chinese(char: str) -> bool:
    return "一" <= char <= "鿿"


def _detect_language(text: str) -> str:
    """Detect if query is Chinese or English."""
    cjk_count = sum(1 for c in text if _is_chinese(c))
    if cjk_count > 0:
        return "zh"
    return "en"


def _extract_signals(query: str) -> dict[str, Any]:
    """Extract structured signals from a raw query string."""
    query_lower = query.lower()
    language = _detect_language(query)

    # Token count: space-split for English, character-based for Chinese
    tokens = query_lower.split()
    cjk_chars = sum(1 for c in query if _is_chinese(c))
    if language == "zh":
        # Chinese: each CJK character ≈ 1 word
        n_tokens = cjk_chars
    else:
        n_tokens = len(tokens)

    tokens_set = set(tokens)

    has_question = bool(
        tokens_set & QUESTION_WORDS_EN
        or any(w in query_lower for w in QUESTION_WORDS_ZH)
    )
    has_comparison = bool(
        tokens_set & COMPARISON_WORDS_EN
        or any(w in query_lower for w in COMPARISON_WORDS_ZH)
    )

    # Heuristic: if the query has no question/comparison words and is short,
    # it's likely just keywords
    has_verbs = False
    for w in tokens:
        if w.endswith(("ing", "ed", "ize", "ise", "ate", "ify")):
            has_verbs = True
            break

    is_pure_keywords = not has_question and not has_comparison and not has_verbs

    return {
        "token_count": n_tokens,
        "has_question_words": has_question,
        "has_comparison_words": has_comparison,
        "is_pure_keywords": is_pure_keywords,
        "language": language,
    }


# ---------------------------------------------------------------------------
# Level 1: Rule-based
# ---------------------------------------------------------------------------


def _rule_classify(signals: dict[str, Any]) -> RouterResult | None:
    """Apply deterministic rules. Returns None if no rule matches."""
    n = signals["token_count"]
    has_q = signals["has_question_words"]
    has_c = signals["has_comparison_words"]
    is_kw = signals["is_pure_keywords"]
    lang = signals["language"]

    # Rule 1: Question words + > 3 tokens → RAG
    if has_q and n > 3:
        return RouterResult(
            should_rag=True,
            confidence=0.95,
            reason=f"rule:question_word(n={n})",
            language=lang,
            needs_rewrite=True,
        )

    # Rule 2: Comparison words + > 4 tokens → RAG
    if has_c and n > 4:
        return RouterResult(
            should_rag=True,
            confidence=0.90,
            reason=f"rule:comparison(n={n})",
            language=lang,
            needs_rewrite=True,
        )

    # Rule 3: Pure keywords (short, no question/comparison/verb) → NO RAG
    if is_kw and n <= 4:
        return RouterResult(
            should_rag=False,
            confidence=0.95,
            reason=f"rule:pure_keywords(n={n})",
            language=lang,
            needs_rewrite=False,
        )

    # Rule 4: Ultra-short (≤ 2 words) → NO RAG
    if n <= 2:
        return RouterResult(
            should_rag=False,
            confidence=0.95,
            reason=f"rule:ultra_short(n={n})",
            language=lang,
            needs_rewrite=False,
        )

    return None  # No rule matched → Level 2


# ---------------------------------------------------------------------------
# Level 2: LLM fallback
# ---------------------------------------------------------------------------


_LLM_CLASSIFY_PROMPT = """你是一个查询路由器。判断用户输入是否需要 AI 来生成总结回答。

需要 AI 总结的情况：
- 用户在提问（有疑问词或隐含问题）
- 用户想了解某个概念的详细解释
- 用户在做对比分析

不需要 AI 总结的情况：
- 用户只是想搜索论文
- 用户输入的是关键词/名词短语
- 用户想浏览某个领域的最新论文

用户输入："{query}"

只需回复 YES 或 NO。"""


def _llm_classify(query: str) -> RouterResult:
    """Use DeepSeek to classify ambiguous queries."""
    from app.rag.llm_provider import create_provider

    llm = create_provider()
    prompt = _LLM_CLASSIFY_PROMPT.format(query=query)

    try:
        response = llm.generate(user=prompt)
        answer = response.text.strip().upper()
        should_rag = answer.startswith("YES")
        lang = _detect_language(query)

        return RouterResult(
            should_rag=should_rag,
            confidence=0.75 if should_rag else 0.80,
            reason=f"llm:answer={answer[:20]}",
            language=lang,
            needs_rewrite=should_rag,  # Only rewrite if RAG is triggered
        )
    except Exception:
        logger.warning("LLM classification failed; defaulting to no RAG")
        return RouterResult(
            should_rag=False,
            confidence=0.50,
            reason="llm:error→default_no",
            language=_detect_language(query),
            needs_rewrite=False,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_query(query: str, force_rag: bool = False) -> RouterResult:
    """Classify a query and decide whether to trigger RAG.

    Args:
        query: Raw user query string.
        force_rag: If True, bypass routing and always trigger RAG.
                   Used when the user explicitly requests AI Overview.

    Returns:
        RouterResult with should_rag, confidence, reason, language, needs_rewrite.
    """
    if force_rag:
        lang = _detect_language(query)
        return RouterResult(
            should_rag=True,
            confidence=1.0,
            reason="force:user_requested",
            language=lang,
            needs_rewrite=True,
        )

    # Level 1: Rules
    signals = _extract_signals(query)
    result = _rule_classify(signals)

    if result is not None:
        logger.debug("Router Level1: %s", result.reason)
        return result

    # Level 2: LLM
    logger.info("Router → Level 2 (LLM) for query: %s", query[:60])
    result = _llm_classify(query)
    logger.info("Router Level2: %s", result.reason)
    return result
