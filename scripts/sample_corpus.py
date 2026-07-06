"""Sample a small demo JSONL corpus from a larger local corpus.

The sampler uses deterministic reservoir sampling, so it can work with large
JSONL files while keeping memory bounded by the requested sample size.

Usage:
    python scripts/sample_corpus.py \
        --input data/raw/peS2o_cs_fulltext_50000.jsonl \
        --output data/raw/demo_peS2o_1000.jsonl \
        --size 1000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/raw/peS2o_cs_fulltext_50000.jsonl")
DEFAULT_OUTPUT = Path("data/raw/demo_peS2o_1000.jsonl")


def sample_corpus(input_path: Path, output_path: Path, size: int, seed: int = 42) -> tuple[int, int]:
    """Sample up to ``size`` valid JSONL records from ``input_path``.

    Invalid JSON lines are skipped.  The returned tuple is:
        (valid_records_seen, records_written)
    """
    if size <= 0:
        raise ValueError("size must be a positive integer")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rng = random.Random(seed)
    reservoir: list[str] = []
    valid_seen = 0
    invalid_seen = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                invalid_seen += 1
                logger.warning("Skipping invalid JSON on line %d", line_no)
                continue

            valid_seen += 1
            normalized_line = stripped + "\n"

            if len(reservoir) < size:
                reservoir.append(normalized_line)
                continue

            j = rng.randrange(valid_seen)
            if j < size:
                reservoir[j] = normalized_line

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(reservoir)

    logger.info(
        "Sampled %d records from %d valid lines (%d invalid skipped) -> %s",
        len(reservoir),
        valid_seen,
        invalid_seen,
        output_path,
    )

    return valid_seen, len(reservoir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a demo JSONL corpus")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL corpus")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output demo JSONL path")
    parser.add_argument("--size", type=int, default=1000, help="Number of records to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    valid_seen, written = sample_corpus(
        input_path=Path(args.input),
        output_path=Path(args.output),
        size=args.size,
        seed=args.seed,
    )

    logger.info("Done. valid_seen=%d written=%d", valid_seen, written)


if __name__ == "__main__":
    main()
