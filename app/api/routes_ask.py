"""
RAG ask API route for CiteQuest-RAG.

POST /ask        — ask a question, get a citation-grounded answer.
POST /ask/stream — same, but via SSE with real-time phase updates.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.schemas import AskRequest, AskResponse
from app.rag.answer import answer_question, answer_question_stream

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


@router.post("/ask/stream")
async def ask_question_stream(request: AskRequest):
    """Ask a question with SSE streaming — real-time phase updates.

    Returns ``text/event-stream`` with these event types:

    - ``status`` — pipeline phase change (retrieving/organizing/generating/verifying)
    - ``result`` — final AskResponse as JSON
    - ``done``   — stream end marker
    """
    logger.info("ask/stream question=%r top_k=%d mode=%s", request.question, request.top_k, request.retrieval_mode)

    return StreamingResponse(
        answer_question_stream(
            question=request.question,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            alpha=request.alpha,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
