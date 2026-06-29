"""
Build the SQLite metadata database from a JSONL paper file.

Runs the full ingestion pipeline (load → normalize → chunk) and writes
everything into data/indexes/metadata.sqlite.

Usage:
    python scripts/build_metadata_db.py \\
        --input data/raw/arxiv_cs_sample.jsonl \\
        --db data/indexes/metadata.sqlite
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_papers
from app.ingestion.normalize import normalize
from app.ingestion.chunk import chunk_paper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id   TEXT PRIMARY KEY,
    title      TEXT    NOT NULL,
    abstract   TEXT    NOT NULL DEFAULT '',
    full_text  TEXT    NOT NULL DEFAULT '',
    year       INTEGER,
    venue      TEXT,
    authors_json TEXT  NOT NULL DEFAULT '[]',
    concepts_json TEXT NOT NULL DEFAULT '[]',
    doi        TEXT,
    url        TEXT,
    citation_count INTEGER NOT NULL DEFAULT 0,
    open_access    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    paper_id   TEXT    NOT NULL,
    chunk_text TEXT    NOT NULL,
    chunk_type TEXT    NOT NULL DEFAULT 'metadata',
    token_count INTEGER NOT NULL DEFAULT 0,
    position   INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS idx_papers_year      ON papers (year);
"""

INSERT_PAPER = """
INSERT OR REPLACE INTO papers
    (paper_id, title, abstract, full_text, year, venue, authors_json, concepts_json,
     doi, url, citation_count, open_access)
VALUES
    (:paper_id, :title, :abstract, :full_text, :year, :venue, :authors_json, :concepts_json,
     :doi, :url, :citation_count, :open_access)
"""

INSERT_CHUNK = """
INSERT OR REPLACE INTO chunks
    (chunk_id, paper_id, chunk_text, chunk_type, token_count, position)
VALUES
    (:chunk_id, :paper_id, :chunk_text, :chunk_type, :token_count, :position)
"""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_db(input_path: Path, db_path: Path) -> tuple[int, int]:
    """Run ingestion pipeline and populate SQLite.

    Returns (num_papers, num_chunks).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(CREATE_TABLES)

    num_papers = 0
    num_chunks = 0

    try:
        for raw in load_papers(input_path):
            paper = normalize(raw)
            if paper is None:
                continue

            chunks = chunk_paper(paper)  # now returns list[dict]

            # Serialize list fields to JSON for SQLite storage
            paper_row = {
                **paper,
                "authors_json": json.dumps(paper.get("authors", []), ensure_ascii=False),
                "concepts_json": json.dumps(paper.get("concepts", []), ensure_ascii=False),
                "open_access": 1 if paper.get("open_access") else 0,
            }

            conn.execute(INSERT_PAPER, paper_row)
            for ch in chunks:
                ch["position"] = ch.get("position", 0)
                conn.execute(INSERT_CHUNK, ch)

            num_papers += 1
            num_chunks += len(chunks)

            if num_papers % 100 == 0:
                logger.info("  Inserted %d papers/chunks...", num_papers)

        conn.commit()
    finally:
        conn.close()

    return num_papers, num_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build metadata SQLite database")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "raw" / "arxiv_cs_sample.jsonl"),
        help="Input JSONL file (default: data/raw/arxiv_cs_sample.jsonl)",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "indexes" / "metadata.sqlite"),
        help="Output SQLite database (default: data/indexes/metadata.sqlite)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = Path(args.input)
    db_path = Path(args.db)

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    logger.info("Building metadata DB from %s", input_path.name)
    np, nc = build_db(input_path, db_path)
    logger.info("Done! %d papers, %d chunks → %s", np, nc, db_path)


if __name__ == "__main__":
    main()
