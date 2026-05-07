"""Tests for text embedding preprocessing/postprocessing.

The embeddinggemma-300m model outputs a pre-pooled ``sentence_embedding``
of shape (batch, 768). Only L2 normalization is needed client-side.
"""

import numpy as np
import pytest

from triton_shared.inference.text_embedding import (
    embed_postprocess,
    l2_normalize,
)


class TestL2Normalize:
    def test_unit_norm(self):
        emb = np.array([[3.0, 4.0]], dtype=np.float32)
        result = l2_normalize(emb)
        np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0)
        np.testing.assert_allclose(result[0], [0.6, 0.8])

    def test_zero_vector(self):
        emb = np.array([[0.0, 0.0]], dtype=np.float32)
        result = l2_normalize(emb)
        np.testing.assert_allclose(result, [[0.0, 0.0]])

    def test_batch(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = l2_normalize(emb)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0)
        np.testing.assert_allclose(np.linalg.norm(result[1]), 1.0)


class TestEmbedPostprocess:
    def test_returns_python_floats(self):
        """embed_postprocess takes pre-pooled (batch, dim) and returns list[list[float]]."""
        emb = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        result = embed_postprocess(emb)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)
        # Each row is unit-length
        np.testing.assert_allclose(
            np.linalg.norm(np.array(result[0])), 1.0, atol=1e-6
        )
        np.testing.assert_allclose(
            np.linalg.norm(np.array(result[1])), 1.0, atol=1e-6
        )
        # Values match expected L2-normalized outputs
        np.testing.assert_allclose(result[0], [0.6, 0.8])
        np.testing.assert_allclose(result[1], [0.0, 1.0])

    def test_single_embedding(self):
        emb = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = embed_postprocess(emb)
        assert len(result) == 1
        assert len(result[0]) == 3
        norm = np.linalg.norm(np.array(result[0]))
        np.testing.assert_allclose(norm, 1.0)

    def test_768_dim_typical_case(self):
        """Simulates the real model output: batch=3, dim=768, already pooled."""
        rng = np.random.default_rng(42)
        emb = rng.normal(size=(3, 768)).astype(np.float32)
        result = embed_postprocess(emb)
        assert len(result) == 3
        for row in result:
            assert len(row) == 768
            norm = np.linalg.norm(np.array(row))
            np.testing.assert_allclose(norm, 1.0, atol=1e-5)
