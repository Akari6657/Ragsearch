"""
RAG ask API route for CiteQuest-RAG.

POST /ask — ask a question, get a citation-grounded answer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.core.schemas import AskRequest, AskResponse
from app.rag.answer import answer_question

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Ask a question and receive a citation-grounded answer.

    Evidence is retrieved via hybrid search, formatted with citation IDs,
    and sent to an LLM that answers strictly based on the provided evidence.
    """
    logger.info("ask question=%r top_k=%d mode=%s", request.question, request.top_k, request.retrieval_mode)

    return answer_question(
        question=request.question,
        top_k=request.top_k,
        retrieval_mode=request.retrieval_mode,
        alpha=request.alpha,
    )
