"""
Full RAG pipeline: retrieve → build evidence → call LLM → verify citations.

Usage:
    from app.rag.answer import answer_question
    response = answer_question("神经网络如何优化？", top_k=5)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from app.core.schemas import AskRequest, AskResponse, CitationInfo
from app.rag.citation import verify_citations
from app.rag.context_builder import build_evidence
from app.rag.llm_provider import create_provider
from app.rag.prompt import build_prompts
from app.rag.rewriter import rewrite_query
from app.retrieval.hybrid import search_hybrid
from app.retrieval.lexical import search_lexical

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/indexes/metadata.sqlite")
DEFAULT_INDEX_DIR = Path("data/indexes/faiss")


def answer_question(
    question: str,
    top_k: int = 8,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.3,
    use_rewrite: bool = True,
    db_path: str | Path = DEFAULT_DB,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> AskResponse:
    """Answer a question with citation-grounded RAG.

    Pipeline:
    1. (Optional) Rewrite query → extract English keywords.
    2. Retrieve evidence: hybrid search on original query + lexical
       search on rewritten keywords → merge + deduplicate.
    3. Format evidence block with [N] citation IDs.
    4. Build system + user prompts.
    5. Call LLM to generate an answer.
    6. Verify that citation markers are valid.

    Args:
        question: Natural-language question.
        top_k: Number of evidence chunks to retrieve.
        retrieval_mode: 'lexical', 'vector', or 'hybrid'.
        alpha: Hybrid weight (0=vector, 1=lexical).
        use_rewrite: If True, extract English keywords to enhance retrieval.

    Returns:
        AskResponse with answer text, citations, validity, and latency.
    """
    t0 = time.perf_counter()
    db_path = Path(db_path)
    index_dir = Path(index_dir)

    # — 1. Retrieve evidence (with optional rewrite) ———————————————————
    if retrieval_mode == "hybrid":
        results = search_hybrid(
            question, top_k=top_k, alpha=alpha, db_path=db_path, index_dir=index_dir
        )
    elif retrieval_mode == "vector":
        from app.retrieval.vector_store import search_vector
        results = search_vector(question, top_k=top_k, db_path=db_path, index_dir=index_dir)
    else:
        results = search_lexical(question, top_k=top_k, db_path=db_path)

    # Rewrite-enhanced retrieval: add lexical results for English keywords.
    # Lexical results are appended after the primary results (which are
    # already relevance-sorted). We don't re-sort because hybrid scores
    # (positive, higher=better) and BM25 scores (negative, lower=better)
    # are on incompatible scales.
    if use_rewrite:
        keywords = rewrite_query(question)
        if keywords and keywords != question:
            kw_results = search_lexical(keywords, top_k=top_k, db_path=db_path)
            seen = {r.chunk_id for r in results}
            for r in kw_results:
                if r.chunk_id not in seen:
                    results.append(r)
                    seen.add(r.chunk_id)

    logger.info("Retrieved %d evidence chunks for question: %s", len(results), question[:60])

    # — 2. Build evidence context ———————————————————————————————————————
    evidence_text, citation_map = build_evidence(results, db_path=db_path)

    if not evidence_text:
        elapsed = (time.perf_counter() - t0) * 1000
        return AskResponse(
            question=question,
            answer="未找到相关证据，无法回答该问题。",
            citations=[],
            citation_valid=True,
            citation_warnings=[],
            latency_ms=round(elapsed, 2),
        )

    # — 3. Build prompts ————————————————————————————————————————————————
    system, user = build_prompts(evidence_text, question)

    # — 4. Call LLM —————————————————————————————————————————————————————
    llm = create_provider()
    llm_response = llm.generate(system=system, user=user)

    logger.info("LLM generated %d chars in %.0f ms", len(llm_response.text), llm_response.latency_ms)

    # — 5. Verify citations —————————————————————————————————————————————
    cit_result = verify_citations(llm_response.text, citation_map)

    # Build CitationInfo list
    citations = [
        CitationInfo(
            citation_id=c["citation_id"],
            paper_id=c["paper_id"],
            chunk_id=c["chunk_id"],
            title=c["title"],
            url=c.get("url"),
        )
        for c in citation_map
    ]

    elapsed = (time.perf_counter() - t0) * 1000

    logger.info(
        "answer_question complete: citations=%d/%d valid=%s latency=%.0f ms",
        len(cit_result.cited_ids),
        len(citation_map),
        cit_result.valid,
        elapsed,
    )

    return AskResponse(
        question=question,
        answer=llm_response.text,
        citations=citations,
        citation_valid=cit_result.valid,
        citation_warnings=cit_result.warnings,
        latency_ms=round(elapsed, 2),
    )
