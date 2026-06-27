"""
Download CS papers from arXiv API (free, no key needed).

Fetches recent papers from major CS categories and saves as JSONL
in our standard format. Much more reliable than UnarXive streaming.

Usage:
    python scripts/download_arxiv.py --size 5000

Categories covered:
    cs.AI, cs.LG, cs.CL, cs.CV, cs.IR, cs.CR, cs.DB, cs.DS,
    cs.NE, cs.SE, cs.RO, cs.SY
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Major CS categories to fetch from
CS_CATEGORIES = [
    "cs.AI",   # Artificial Intelligence
    "cs.LG",   # Machine Learning
    "cs.CL",   # Computation and Language (NLP)
    "cs.CV",   # Computer Vision
    "cs.IR",   # Information Retrieval
    "cs.CR",   # Cryptography and Security
    "cs.DB",   # Databases
    "cs.NE",   # Neural and Evolutionary Computing
    "cs.SE",   # Software Engineering
    "cs.RO",   # Robotics
]

# Map arXiv categories to concept labels
CATEGORY_LABELS = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.IR": "Information Retrieval",
    "cs.CR": "Cryptography and Security",
    "cs.DB": "Databases",
    "cs.DS": "Data Structures and Algorithms",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.SE": "Software Engineering",
    "cs.RO": "Robotics",
    "cs.SY": "Systems and Control",
    "cs.DC": "Distributed Parallel and Cluster Computing",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IT": "Information Theory",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.NI": "Networking and Internet Architecture",
    "cs.SI": "Social and Information Networks",
    "stat.ML": "Machine Learning",
}


def _map_to_paper(result) -> dict | None:
    """Convert an arxiv.Result to our standard paper dict."""
    try:
        arxiv_id = result.entry_id.split("/")[-1]
        # Remove version suffix from ID (e.g. "2301.00001v2" → "2301.00001")
        if "v" in arxiv_id and arxiv_id.split("v")[-1].isdigit():
            arxiv_id = arxiv_id[: arxiv_id.rindex("v")]
    except (AttributeError, ValueError):
        return None

    title = (result.title or "").strip()
    if not title:
        return None

    abstract = (result.summary or "").strip()
    authors = [a.name for a in (result.authors or []) if a.name]
    year = result.published.year if result.published else None

    # Venue from journal_ref or comment
    venue = None
    if result.journal_ref:
        venue = str(result.journal_ref).strip()
    elif result.comment:
        comment = str(result.comment).strip()
        if len(comment) < 120:
            venue = comment

    # Map categories to concept labels
    concepts = []
    seen = set()
    for cat in (result.categories or []):
        cat_str = str(cat)
        label = CATEGORY_LABELS.get(cat_str, cat_str)
        if label not in seen:
            concepts.append(label)
            seen.add(label)

    doi = str(result.doi).strip() if result.doi else None
    url = str(result.pdf_url).strip() if result.pdf_url else f"https://arxiv.org/abs/{arxiv_id}"
    citation_count = 0  # arXiv API doesn't provide this

    return {
        "paper_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "venue": venue,
        "authors": authors,
        "concepts": concepts,
        "doi": doi,
        "url": url,
        "citation_count": citation_count,
        "open_access": True,  # arXiv papers are OA
    }


def main():
    parser = argparse.ArgumentParser(description="Download CS papers from arXiv API")
    parser.add_argument(
        "--size", type=int, default=5000, help="Target number of papers (default: 5000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "arxiv_cs_sample.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(CS_CATEGORIES),
        help="Comma-separated arXiv categories",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2024,
        help="Minimum publication year (default: 2024)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    categories = [c.strip() for c in args.categories.split(",")]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing papers for dedup
    existing_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    existing_ids.add(rec.get("paper_id", ""))
                except json.JSONDecodeError:
                    pass
        logger.info("Existing papers in output: %d", len(existing_ids))

    logger.info("Fetching from arXiv: %s", ", ".join(categories))
    logger.info("Target: %d papers (min year: %d)", args.size, args.min_year)

    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=3)
    written = 0
    skipped_old = 0
    skipped_dup = 0
    checked = 0

    # Build category query
    category_query = " OR ".join(f"cat:{c}" for c in categories)
    search = arxiv.Search(
        query=category_query,
        max_results=args.size * 3,  # Fetch more than needed (filter by year)
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    with open(output_path, "a" if existing_ids else "w", encoding="utf-8") as f:
        try:
            for result in client.results(search):
                checked += 1

                if result.published and result.published.year < args.min_year:
                    skipped_old += 1
                    continue

                paper = _map_to_paper(result)
                if paper is None:
                    continue

                if paper["paper_id"] in existing_ids:
                    skipped_dup += 1
                    continue

                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
                existing_ids.add(paper["paper_id"])
                written += 1

                if written % 100 == 0:
                    logger.info(
                        "  %d written (checked %d, skipped %d old + %d dup)",
                        written, checked, skipped_old, skipped_dup,
                    )

                if written >= args.size:
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted. Progress saved.")

    logger.info(
        "Done! Wrote %d papers (checked %d, skipped %d old + %d dup)",
        written, checked, skipped_old, skipped_dup,
    )
    logger.info("Output: %s (%.0f KB)", output_path, output_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
