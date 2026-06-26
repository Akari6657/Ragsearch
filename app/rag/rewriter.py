"""
Query rewriter: extract English academic keywords to improve lexical recall.

When a query is in Chinese (or mixed), the LLM extracts English keywords
that are then used alongside the original query for hybrid retrieval.
This gives FTS5 lexical search something to match against.

Usage:
    from app.rag.rewriter import rewrite_query
    keywords = rewrite_query("为什么GAN训练不稳定？")
    # → "GAN training mode collapse instability convergence"
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿"


def detect_language(text: str) -> str:
    """Return 'zh' if text contains CJK characters, otherwise 'en'."""
    for c in text:
        if _is_cjk(c):
            return "zh"
    return "en"


# ---------------------------------------------------------------------------
# Keyword extraction prompt
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """从以下查询中提取 3-5 个英文学术关键词，用于搜索论文。只返回关键词，用空格分隔，不要其他内容。

查询："{query}"

关键词："""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rewrite_query(query: str) -> str:
    """Extract English academic keywords from a query.

    If the query is already in English, returns the original query as-is.
    If Chinese/mixed, calls LLM to extract English keywords.

    Args:
        query: Raw user query (any language).

    Returns:
        Space-separated English keywords suitable for lexical search.
    """
    lang = detect_language(query)

    # English queries don't need rewriting for lexical search
    if lang == "en":
        logger.debug("rewrite: English query, using as-is")
        return query

    # Chinese/mixed: ask LLM to extract English keywords
    from app.rag.llm_provider import create_provider

    llm = create_provider()
    prompt = _REWRITE_PROMPT.format(query=query)

    try:
        response = llm.generate(user=prompt)
        keywords = response.text.strip()
        logger.info("rewrite: '%s' → '%s'", query[:60], keywords[:80])
        return keywords if keywords else query
    except Exception:
        logger.warning("rewrite: LLM call failed, using original query")
        return query
