"""
Download a CS paper sample from UnarXive 2024 on Hugging Face.

UnarXive 2024 contains ~608K CS papers with full metadata and text.
This script:
1. Streams the dataset (no full download needed)
2. Filters to CS papers only (arxiv categories starting with "cs.")
3. Maps fields to OpenAlex-style format
4. Saves the first N records as JSONL to data/raw/

Usage:
    python scripts/download_corpus.py --size 1000
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so we can run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_year(metadata: dict, paper_id: str) -> int | None:
    """Extract publication year from metadata or arxiv ID.

    ArXiv IDs before 2007 are formatted differently (e.g., 'cs/0212040').
    For 2007+, the ID format is 'YYMM.NNNNN' (e.g., '1807.04467' = 2018).
    """
    # Try explicit year from versions
    versions = metadata.get("versions", [])
    if versions:
        try:
            first_version = versions[0]
            if isinstance(first_version, dict):
                created = first_version.get("created", "")
                if created:
                    return int(created[:4])
            elif isinstance(first_version, list) and len(first_version) >= 3:
                return int(first_version[2][:4])
        except (ValueError, IndexError, KeyError):
            pass

    # Fallback: parse from arxiv ID (YYMM format, works for 2007+)
    try:
        yy = int(paper_id[:2])
        return 2000 + yy
    except (ValueError, IndexError):
        return None


def extract_venue(metadata: dict) -> str | None:
    """Extract venue from journal-ref or comments."""
    journal_ref = metadata.get("journal-ref")
    if journal_ref and journal_ref != "null":
        # journal-ref often looks like: "SIGIR 2023" or "J. Mach. Learn. Res. 23 (2022)"
        return journal_ref.strip()
    return None


def extract_concepts(metadata: dict) -> list[str]:
    """Convert arxiv categories to concept labels.

    ArXiv category format: 'cs.IR cs.CL' or 'stat.ML cs.LG'
    We keep only cs.* categories and convert to human-readable labels.
    """
    CATEGORY_LABELS = {
        "cs.AI": "Artificial Intelligence",
        "cs.CL": "Computation and Language",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computers and Society",
        "cs.CR": "Cryptography and Security",
        "cs.DS": "Data Structures and Algorithms",
        "cs.DB": "Databases",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DC": "Distributed Parallel and Cluster Computing",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory",
        "cs.GT": "Computer Science and Game Theory",
        "cs.GR": "Graphics",
        "cs.AR": "Hardware Architecture",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MS": "Mathematical Software",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.NI": "Networking and Internet Architecture",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NA": "Numerical Analysis",
        "cs.OS": "Operating Systems",
        "cs.OH": "Other Computer Science",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SI": "Social and Information Networks",
        "cs.SE": "Software Engineering",
        "cs.SD": "Sound",
        "cs.SC": "Symbolic Computation",
        "cs.CE": "Computational Engineering Finance and Science",
        "cs.CG": "Computational Geometry",
        "cs.CC": "Computational Complexity",
    }

    categories_raw = metadata.get("categories", "")
    if not categories_raw:
        return []

    concepts = []
    for cat in categories_raw.split():
        cat = cat.strip()
        if cat.startswith("cs."):
            label = CATEGORY_LABELS.get(cat, cat)
            concepts.append(label)
    return concepts


def normalize_record(paper_id: str, metadata: dict, abstract_text: str) -> dict | None:
    """Convert UnarXive record to OpenAlex-style paper dict.

    Returns None if the record should be skipped.
    """
    categories = metadata.get("categories", "")
    if not categories or not any(c.strip().startswith("cs.") for c in categories.split()):
        return None  # Skip non-CS papers

    authors_raw = metadata.get("authors", "")
    authors = [a.strip() for a in authors_raw.split(",") if a.strip()] if authors_raw else []

    year = extract_year(metadata, paper_id)
    venue = extract_venue(metadata)
    concepts = extract_concepts(metadata)

    # Build a stable paper_id
    doi = metadata.get("doi")
    arxiv_id = metadata.get("id", paper_id)
    if doi and doi != "null":
        paper_doi = doi
    else:
        paper_doi = None

    return {
        "paper_id": arxiv_id,
        "title": metadata.get("title", "").strip() or "Untitled",
        "abstract": abstract_text.strip() if abstract_text else "",
        "year": year,
        "venue": venue,
        "authors": authors,
        "concepts": concepts,
        "doi": paper_doi,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "citation_count": metadata.get("cited_by_count", 0) or 0,
        "open_access": True,  # arxiv papers are open-access
    }


def main():
    parser = argparse.ArgumentParser(description="Download CS paper sample from UnarXive 2024")
    parser.add_argument(
        "--size", type=int, default=1000, help="Number of CS papers to download (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/raw/arxiv_cs_sample.jsonl)",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "raw" / "arxiv_cs_sample.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.size} CS papers from UnarXive 2024...")
    print(f"Output: {output_path}")
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Please install 'datasets': pip install datasets")

    ds = load_dataset("ines-besrour/unarxive_2024", split="train", streaming=True)

    written = 0
    skipped_cs = 0
    skipped_error = 0
    checked = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in ds:
            checked += 1

            try:
                inner = json.loads(rec["jsonl"])
            except json.JSONDecodeError:
                skipped_error += 1
                continue

            paper_id = inner.get("paper_id", "")
            metadata = inner.get("metadata", {})
            if not metadata:
                skipped_error += 1
                continue

            # Extract abstract text
            abstract = inner.get("abstract", {})
            if isinstance(abstract, dict):
                abstract_text = abstract.get("text", "")
            else:
                abstract_text = str(abstract) if abstract else ""

            paper = normalize_record(paper_id, metadata, abstract_text)
            if paper is None:
                skipped_cs += 1
                continue

            f.write(json.dumps(paper, ensure_ascii=False) + "\n")
            written += 1

            if written % 100 == 0:
                print(f"  Progress: {written}/{args.size} papers written "
                      f"(checked {checked}, skipped {skipped_cs} non-CS, {skipped_error} errors)")

            if written >= args.size:
                break

    print()
    print(f"Done! Wrote {written} papers to {output_path}")
    print(f"  Total checked: {checked}")
    print(f"  Skipped non-CS: {skipped_cs}")
    print(f"  Skipped errors: {skipped_error}")
    print(f"\nSample ID: {args.size}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
