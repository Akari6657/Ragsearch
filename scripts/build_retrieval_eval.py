"""Build the frozen Retrieval Benchmark v1 query set.

Target-paper sampling, dev/test assignment, and query-type assignment are
deterministic. Query text is generated once with an OpenAI-compatible provider
and then frozen as JSONL; normal benchmark evaluation only reads that file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.retrieval_eval import QUERY_TYPES, load_eval_queries
from app.rag.llm_provider import LLMProvider, MockLLMProvider, create_provider

logger = logging.getLogger(__name__)

DEFAULT_DB = PROJECT_ROOT / "data" / "indexes" / "benchmark_v1" / "metadata.sqlite"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "retrieval_v1.jsonl"
DEFAULT_SIZE = 150
DEFAULT_DEV_SIZE = 50
DEFAULT_SEED = 42
DEFAULT_MIN_ABSTRACT_WORDS = 60
GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 1024
MAX_GENERATION_ATTEMPTS = 3
MAX_TITLE_TOKEN_OVERLAP = 0.8

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_QUESTION_STARTS = {
    "how",
    "what",
    "when",
    "where",
    "which",
    "why",
    "can",
    "does",
    "do",
    "is",
    "are",
}


@dataclass(frozen=True)
class TargetPaper:
    paper_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    source_category: str


@dataclass(frozen=True)
class QueryTarget:
    query_id: str
    split: str
    query_type: str
    paper: TargetPaper


def normalize_title(title: str) -> str:
    """Normalize titles for duplicate detection and copy checks."""
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(_WORD_RE.findall(normalized))


def _parse_string_list(raw: Any) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def load_eligible_targets(
    db_path: str | Path,
    *,
    min_abstract_words: int = DEFAULT_MIN_ABSTRACT_WORDS,
) -> list[TargetPaper]:
    """Load papers with useful abstracts and globally unique normalized titles."""
    if min_abstract_words <= 0:
        raise ValueError("min_abstract_words must be positive")

    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT paper_id, title, abstract, authors_json, concepts_json
               FROM papers ORDER BY paper_id"""
        ).fetchall()
    finally:
        conn.close()

    title_counts = Counter(
        normalize_title(row["title"] or "")
        for row in rows
        if normalize_title(row["title"] or "")
    )
    eligible: list[TargetPaper] = []
    for row in rows:
        paper_id = (row["paper_id"] or "").strip()
        title = (row["title"] or "").strip()
        abstract = (row["abstract"] or "").strip()
        normalized_title = normalize_title(title)
        if not paper_id or not normalized_title or title_counts[normalized_title] != 1:
            continue
        if len(_WORD_RE.findall(abstract)) < min_abstract_words:
            continue

        authors = tuple(_parse_string_list(row["authors_json"]))
        concepts = _parse_string_list(row["concepts_json"])
        source_category = concepts[0] if concepts else "Uncategorized"
        eligible.append(
            TargetPaper(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                source_category=source_category,
            )
        )
    return eligible


def stratified_sample(
    candidates: Sequence[TargetPaper], *, size: int, seed: int = DEFAULT_SEED
) -> list[TargetPaper]:
    """Round-robin shuffled categories so broad CS areas remain represented."""
    if size <= 0:
        raise ValueError("size must be positive")
    if len(candidates) < size:
        raise ValueError(
            f"Need {size} eligible papers, but the database only provides {len(candidates)}"
        )

    groups: dict[str, list[TargetPaper]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.source_category].append(candidate)

    rng = random.Random(seed)
    categories = sorted(groups)
    rng.shuffle(categories)
    for category in categories:
        rng.shuffle(groups[category])

    selected: list[TargetPaper] = []
    offsets = {category: 0 for category in categories}
    while len(selected) < size:
        made_progress = False
        for category in categories:
            offset = offsets[category]
            if offset >= len(groups[category]):
                continue
            selected.append(groups[category][offset])
            offsets[category] += 1
            made_progress = True
            if len(selected) == size:
                break
        if not made_progress:
            raise RuntimeError("Category sampler exhausted candidates unexpectedly")
    return selected


def build_query_plan(
    candidates: Sequence[TargetPaper],
    *,
    size: int = DEFAULT_SIZE,
    dev_size: int = DEFAULT_DEV_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[QueryTarget]:
    """Create deterministic source-paper, split, and query-type assignments."""
    if dev_size <= 0 or dev_size >= size:
        raise ValueError("dev_size must be positive and smaller than size")
    papers = stratified_sample(candidates, size=size, seed=seed)

    # Shuffle balanced labels independently from the category round-robin. This
    # prevents a category's position in that round-robin from determining its
    # query type while keeping both splits as even as mathematically possible.
    type_rng = random.Random(seed + 1)

    balanced = [QUERY_TYPES[index % len(QUERY_TYPES)] for index in range(size)]
    dev_types = balanced[:dev_size]
    test_types = balanced[dev_size:]
    type_rng.shuffle(dev_types)
    type_rng.shuffle(test_types)
    query_types = dev_types + test_types
    return [
        QueryTarget(
            query_id=f"q{index + 1:04d}",
            split="dev" if index < dev_size else "test",
            query_type=query_types[index],
            paper=paper,
        )
        for index, paper in enumerate(papers)
    ]


def _prompt_for(target: QueryTarget, feedback: str | None = None) -> tuple[str, str]:
    type_instructions = {
        "keyword": (
            "Write a short realistic academic keyword query of 3-12 words. "
            "It must not be a full sentence or end with a question mark. "
            "Do not merely remove or reorder words from the title; rephrase at least "
            "one central idea."
        ),
        "natural_question": (
            "Write a self-contained natural-language research question of 5-30 words. "
            "It must end with a question mark."
        ),
        "semantic_paraphrase": (
            "Write a 5-30 word search query that paraphrases the paper's main problem or "
            "contribution with substantially different wording from the title."
        ),
    }
    system = (
        "You construct known-item academic retrieval benchmarks. Return exactly one JSON "
        "object with one string field named query and no markdown. The query must describe "
        "information supported by the supplied abstract. Never mention authors, arXiv IDs, "
        "or copy the paper title."
    )
    user = (
        f"Query type: {target.query_type}\n"
        f"Instructions: {type_instructions[target.query_type]}\n\n"
        f"Paper title: {target.paper.title}\n"
        f"Paper abstract: {target.paper.abstract}"
    )
    if feedback:
        user += f"\n\nThe previous output was rejected because: {feedback}. Generate a new query."
    return system, user


def parse_generated_query(text: str) -> str:
    """Parse a strict JSON response, with a small plain-text compatibility fallback."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("query"), str):
        return payload["query"].strip()

    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    first_line = re.sub(r"^(?:query\s*:\s*)", "", first_line, flags=re.IGNORECASE)
    return first_line.strip().strip('"')


def _contains_title_ngram(query: str, title: str, n: int = 5) -> bool:
    query_tokens = normalize_title(query).split()
    title_tokens = normalize_title(title).split()
    # A single-word title is often also the unavoidable topic term. For short
    # multi-word titles, check the complete title; otherwise use an n-gram.
    if len(title_tokens) < 2:
        return False
    window_size = min(n, len(title_tokens))
    query_windows = {
        tuple(query_tokens[start : start + window_size])
        for start in range(len(query_tokens) - window_size + 1)
    }
    return any(
        tuple(title_tokens[start : start + window_size]) in query_windows
        for start in range(len(title_tokens) - window_size + 1)
    )


def _has_excessive_title_token_overlap(query: str, title: str) -> bool:
    """Reject title-derived queries while preserving normal topic-term overlap."""
    query_tokens = normalize_title(query).split()
    title_tokens = set(normalize_title(title).split())
    if len(query_tokens) < 5 or len(title_tokens) < 5:
        return False
    overlap = sum(token in title_tokens for token in query_tokens)
    return overlap / len(query_tokens) >= MAX_TITLE_TOKEN_OVERLAP


def validate_generated_query(query: str, target: QueryTarget) -> list[str]:
    """Return deterministic rejection reasons for generated query text."""
    reasons: list[str] = []
    words = _WORD_RE.findall(query)
    normalized_query = normalize_title(query)
    if not query.strip():
        return ["the query is empty"]
    if len(query) > 300:
        reasons.append("the query is longer than 300 characters")
    if len(words) < 3:
        reasons.append("the query has fewer than three words")
    if normalized_query == normalize_title(target.paper.title):
        reasons.append("the query copies the full paper title")
    if _contains_title_ngram(query, target.paper.title):
        reasons.append("the query copies a multi-word phrase from the title")
    if _has_excessive_title_token_overlap(query, target.paper.title):
        reasons.append("the query reuses too many words from the title")

    normalized_with_spaces = f" {normalized_query} "
    if normalize_title(target.paper.paper_id) in normalized_query:
        reasons.append("the query mentions the paper ID")
    for author in target.paper.authors:
        normalized_author = normalize_title(author)
        if normalized_author and f" {normalized_author} " in normalized_with_spaces:
            reasons.append("the query mentions an author")
            break

    if target.query_type == "keyword":
        if len(words) > 12:
            reasons.append("a keyword query must have at most 12 words")
        if query.rstrip().endswith("?"):
            reasons.append("a keyword query must not be a question")
    elif target.query_type == "natural_question":
        if len(words) > 30:
            reasons.append("a natural question must have at most 30 words")
        if not query.rstrip().endswith("?"):
            reasons.append("a natural question must end with a question mark")
        if words and words[0].casefold() not in _QUESTION_STARTS:
            reasons.append("a natural question must start with a question word")
    elif target.query_type == "semantic_paraphrase" and len(words) > 30:
        reasons.append("a semantic paraphrase must have at most 30 words")
    return reasons


def generate_query(
    provider: LLMProvider,
    target: QueryTarget,
    *,
    max_attempts: int = MAX_GENERATION_ATTEMPTS,
) -> tuple[str, str]:
    """Generate and validate one query, retrying bounded invalid outputs."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    feedback: str | None = None
    last_reasons: list[str] = []
    for _ in range(max_attempts):
        system, user = _prompt_for(target, feedback)
        response = provider.generate(
            system=system,
            user=user,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=GENERATION_MAX_TOKENS,
        )
        query = parse_generated_query(response.text)
        last_reasons = validate_generated_query(query, target)
        if not last_reasons:
            return query, response.model
        feedback = "; ".join(last_reasons)
    raise RuntimeError(
        f"Could not generate valid query for {target.query_id} after {max_attempts} "
        f"attempts: {'; '.join(last_reasons)}"
    )


def _record_for(target: QueryTarget, query: str, model: str) -> dict[str, Any]:
    return {
        "query_id": target.query_id,
        "query": query,
        "query_type": target.query_type,
        "split": target.split,
        "relevant_paper_ids": [target.paper.paper_id],
        "source_paper_id": target.paper.paper_id,
        "source_category": target.paper.source_category,
        "generation_model": model,
        "generation_temperature": GENERATION_TEMPERATURE,
        "generation_max_tokens": GENERATION_MAX_TOKENS,
    }


def _load_partial(path: Path, plan: Sequence[QueryTarget]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid partial file at line {line_number}") from exc
            if len(records) >= len(plan):
                raise RuntimeError("Partial eval file contains more records than the query plan")
            target = plan[len(records)]
            expected = {
                "query_id": target.query_id,
                "query_type": target.query_type,
                "split": target.split,
                "source_paper_id": target.paper.paper_id,
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError(
                    "Partial eval file does not match the deterministic query plan; "
                    "remove it or use --overwrite"
                )
            records.append(record)
    return records


def build_eval_set(
    *,
    db_path: str | Path,
    output_path: str | Path,
    provider: LLMProvider,
    size: int = DEFAULT_SIZE,
    dev_size: int = DEFAULT_DEV_SIZE,
    seed: int = DEFAULT_SEED,
    min_abstract_words: int = DEFAULT_MIN_ABSTRACT_WORDS,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Generate a resumable JSONL file, then atomically freeze it."""
    db_path = Path(db_path)
    output_path = Path(output_path)
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Frozen evaluation set already exists: {output_path}. "
            "Use --overwrite only to intentionally replace it."
        )
    if overwrite:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    eligible = load_eligible_targets(
        db_path, min_abstract_words=min_abstract_words
    )
    plan = build_query_plan(eligible, size=size, dev_size=dev_size, seed=seed)
    records = _load_partial(partial_path, plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Eligible papers: %d; query plan: %d; resuming at %d",
        len(eligible),
        len(plan),
        len(records),
    )

    with open(partial_path, "a", encoding="utf-8") as handle:
        for target in plan[len(records) :]:
            query, model = generate_query(provider, target)
            record = _record_for(target, query, model)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            records.append(record)
            logger.info(
                "Generated %s (%s, %s): %s",
                target.query_id,
                target.split,
                target.query_type,
                query,
            )

    load_eval_queries(partial_path, db_path=db_path)
    partial_path.replace(output_path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen Retrieval Benchmark v1 queries")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Benchmark metadata DB")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Frozen JSONL path")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--dev-size", type=int, default=DEFAULT_DEV_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-abstract-words", type=int, default=DEFAULT_MIN_ABSTRACT_WORDS
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace an existing frozen eval set",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    provider = create_provider()
    if isinstance(provider, MockLLMProvider):
        sys.exit(
            "LLM_API_KEY is required to construct the frozen eval set; the mock provider "
            "must not generate benchmark queries."
        )

    records = build_eval_set(
        db_path=args.db,
        output_path=args.output,
        provider=provider,
        size=args.size,
        dev_size=args.dev_size,
        seed=args.seed,
        min_abstract_words=args.min_abstract_words,
        overwrite=args.overwrite,
    )
    distribution = Counter(record["query_type"] for record in records)
    logger.info("Frozen %d queries at %s", len(records), args.output)
    logger.info("Query type distribution: %s", dict(sorted(distribution.items())))


if __name__ == "__main__":
    main()
