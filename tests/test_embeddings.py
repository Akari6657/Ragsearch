"""Tests for app.retrieval.embeddings — embedding model."""

import numpy as np
import pytest

from app.retrieval.embeddings import EmbeddingModel


class FakeEmbeddingBackend:
    """Deterministic 1024-dim embedding backend for offline unit tests."""

    def get_embedding_dimension(self) -> int:
        return 1024

    def encode(
        self,
        texts,
        *,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)

        for i, text in enumerate(texts):
            text_l = text.lower()
            if any(token in text_l for token in ("neural", "network", "deep", "learning", "training")):
                vectors[i, 0] = 1.0
            elif any(token in text_l for token in ("banana", "fruit", "smoothie")):
                vectors[i, 1] = 1.0
            else:
                idx = sum(ord(ch) for ch in text_l) % 1024
                vectors[i, idx] = 1.0

        if normalize_embeddings and len(texts):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)

        return vectors


@pytest.fixture(scope="module")
def model():
    """Use an offline fake backend; real model loading belongs in integration tests."""
    return EmbeddingModel(model=FakeEmbeddingBackend())


class TestEmbeddingModel:
    def test_dim(self, model):
        assert model.dim == 1024

    def test_encode_empty(self, model):
        vecs = model.encode([], show_progress=False)
        assert vecs.shape == (0, 1024)

    def test_encode_single(self, model):
        vecs = model.encode(["test"], show_progress=False)
        assert vecs.shape == (1, 1024)
        assert vecs.dtype == np.float32

    def test_encode_batch(self, model):
        vecs = model.encode(["a", "b", "c"], show_progress=False)
        assert vecs.shape == (3, 1024)

    def test_normalized(self, model):
        """Vectors should be L2-normalized (norms ≈ 1.0)."""
        vecs = model.encode(["test sentence"], show_progress=False)
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 0.01

    def test_semantic_similarity(self, model):
        """Similar sentences should be closer than dissimilar ones."""
        vecs = model.encode(
            ["neural network optimization", "deep learning training", "banana fruit smoothie"],
            show_progress=False,
        )
        sim_nn_dl = np.dot(vecs[0], vecs[1])
        sim_nn_banana = np.dot(vecs[0], vecs[2])
        assert sim_nn_dl > sim_nn_banana
