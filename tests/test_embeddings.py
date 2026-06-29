"""Tests for app.retrieval.embeddings — embedding model."""

import numpy as np
import pytest

from app.retrieval.embeddings import EmbeddingModel


@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests in this module."""
    return EmbeddingModel()


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
