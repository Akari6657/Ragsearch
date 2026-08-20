"""Tests for the resumable FAISS build script."""

import json
import sqlite3

import numpy as np
import pytest

from scripts.build_faiss import build_faiss


class FakeEmbeddingModel:
    dim = 8

    def __init__(self, *, fail_on_call: int | None = None):
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.encoded_texts: list[str] = []

    def encode(self, texts, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated encoder interruption")

        self.encoded_texts.extend(texts)
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, int(text.rsplit(" ", 1)[-1]) % self.dim] = 1.0
        return vectors


def _create_chunk_db(path, count: int):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL
        )"""
    )
    for index in range(count):
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?)",
            (f"chunk-{index}", f"paper-{index // 2}", f"chunk text {index}"),
        )
    conn.commit()
    conn.close()


def test_builds_index_mapping_and_metadata_in_batches(tmp_path):
    import faiss

    db_path = tmp_path / "metadata.sqlite"
    output_dir = tmp_path / "faiss"
    _create_chunk_db(db_path, count=80)

    count, dim = build_faiss(
        db_path,
        output_dir,
        model_name="fake-embedding-model",
        embedding_model=FakeEmbeddingModel(),
        checkpoint_size=17,
        faiss_add_batch_size=19,
    )

    index = faiss.read_index(str(output_dir / "index.faiss"))
    id_map = json.loads((output_dir / "id_map.json").read_text())
    metadata = json.loads((output_dir / "build_meta.json").read_text())

    assert (count, dim) == (80, 8)
    assert index.ntotal == 80
    assert index.d == 8
    assert index.is_trained
    assert len(id_map) == 80
    assert id_map[0] == {
        "faiss_id": 0,
        "chunk_id": "chunk-0",
        "paper_id": "paper-0",
    }
    assert id_map[-1]["faiss_id"] == 79
    assert metadata["status"] == "complete"
    assert metadata["num_vectors"] == 80
    assert metadata["index_type"] == "IndexIVFFlat"
    assert not (output_dir / ".build").exists()


def test_resumes_after_encoder_interruption(tmp_path):
    db_path = tmp_path / "metadata.sqlite"
    output_dir = tmp_path / "faiss"
    _create_chunk_db(db_path, count=7)

    with pytest.raises(RuntimeError, match="simulated encoder interruption"):
        build_faiss(
            db_path,
            output_dir,
            model_name="fake-embedding-model",
            embedding_model=FakeEmbeddingModel(fail_on_call=2),
            checkpoint_size=3,
        )

    state = json.loads((output_dir / ".build" / "state.json").read_text())
    assert state["completed_chunks"] == 3
    assert state["status"] == "encoding"

    resumed_model = FakeEmbeddingModel()
    count, dim = build_faiss(
        db_path,
        output_dir,
        model_name="fake-embedding-model",
        embedding_model=resumed_model,
        checkpoint_size=3,
    )

    assert (count, dim) == (7, 8)
    assert resumed_model.encoded_texts == [
        "chunk text 3",
        "chunk text 4",
        "chunk text 5",
        "chunk text 6",
    ]
    assert not (output_dir / ".build").exists()
