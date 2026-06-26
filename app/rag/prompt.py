"""
RAG prompt templates for citation-grounded question answering.

Prompts are in Chinese — the UI and agent responses are Chinese.

The LLM is instructed to:
- Answer based on the provided evidence, with reasonable inference allowed.
- Mark every factual claim with a citation marker [1], [2], [3].
- Distinguish evidence-based facts from inferences using "(基于证据推断)".
- Never invent paper titles, authors, or specific claims.
"""

from __future__ import annotations

SYSTEM_PROMPT = """你是一个学术研究助手。你的任务是基于提供的论文证据来回答问题。

规则：
1. **以证据为基础。** 优先使用提供的证据回答问题。如果需要基于证据做合理推断
   或补充背景知识来形成完整回答，可以在证据基础上适当扩展，但扩展部分必须在
   句末标注 "(基于证据推断)"。

2. **证据不足时先作答再说明。** 即使证据不够完整，也先根据已有证据回答已知部分，
   然后在最后如实说明哪些方面证据不足。不要直接拒绝回答。

3. **每个论断都要引用。** 来自证据的事实陈述后面必须跟上引用标记 [1]、[2]、[3]。

4. **不要编造。** 不要编造论文标题、作者、年份或任何不在证据中的具体内容。
   如果不确定，请如实说明。

5. **用与问题相同的语言回答。** 保持简洁直接，直接回应问题。

6. **严格使用引用格式。** 引用标记必须使用方括号，如 [1]、[2]。"""

USER_PROMPT_TEMPLATE = """证据：
{evidence}

问题：{question}

请回答（带引用）："""


def build_prompts(evidence: str, question: str) -> tuple[str, str]:
    """Build system and user prompts for a chat completion call.

    Args:
        evidence: Formatted evidence block with [N] markers per chunk.
        question: The user's natural-language question (in Chinese).

    Returns:
        (system_prompt, user_prompt) tuple ready for the LLM provider.
    """
    system = SYSTEM_PROMPT
    user = USER_PROMPT_TEMPLATE.format(evidence=evidence, question=question)
    return system, user
