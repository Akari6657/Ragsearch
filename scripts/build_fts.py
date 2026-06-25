"""
Build the SQLite FTS5 full-text index from the chunks table.

Creates a virtual FTS5 table that indexes title and chunk_text,
enabling fast BM25-ranked keyword search.

Usage:
    python scripts/build_fts.py --db data/indexes/metadata.sqlite
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

# The FTS5 table mirrors chunks but only indexes searchable columns.
# chunk_id and paper_id are UNINDEXED — they are stored for lookup but
# not searched.  title and chunk_text are the indexed content.
#
# We use a CONTENT table (chunks) so FTS5 doesn't duplicate the text
# storage — it only stores the inverted index.

CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id  UNINDEXED,
    paper_id  UNINDEXED,
    title,
    chunk_text
);
"""

POPULATE_FTS = """
INSERT INTO chunk_fts(chunk_id, paper_id, title, chunk_text)
SELECT
    c.chunk_id,
    c.paper_id,
    COALESCE(p.title, ''),
    c.chunk_text
FROM chunks c
LEFT JOIN papers p ON c.paper_id = p.paper_id
"""

# After building, verify counts match.
COUNT_FTS = "SELECT COUNT(*) FROM chunk_fts"
COUNT_CHUNKS = "SELECT COUNT(*) FROM chunks"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_fts(db_path: Path) -> int:
    """Create and populate the FTS5 index.

    Returns the number of indexed documents.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}\nRun build_metadata_db.py first.")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")

    try:
        # Check if FTS table already exists — if so, rebuild from scratch
        conn.execute("DROP TABLE IF EXISTS chunk_fts")

        conn.execute(CREATE_FTS)
        conn.execute(POPULATE_FTS)

        fts_count = conn.execute(COUNT_FTS).fetchone()[0]
        chunk_count = conn.execute(COUNT_CHUNKS).fetchone()[0]

        if fts_count != chunk_count:
            logger.warning(
                "FTS count (%d) != chunk count (%d) — some chunks may not be indexed",
                fts_count,
                chunk_count,
            )

        conn.commit()
    finally:
        conn.close()

    return fts_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build FTS5 full-text index")
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "indexes" / "metadata.sqlite"),
        help="Path to metadata SQLite database",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    db_path = Path(args.db)
    logger.info("Building FTS5 index on %s", db_path)
    count = build_fts(db_path)
    logger.info("Done! %d documents indexed in chunk_fts", count)


if __name__ == "__main__":
    main()
