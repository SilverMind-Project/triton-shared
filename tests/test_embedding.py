"""Tests for CLIP ViT-L/14 preprocessing and postprocessing."""

from __future__ import annotations

import numpy as np
from PIL import Image

from triton_shared.inference.embedding import (
    CLIP_EMBEDDING_DIM,
    CLIP_INPUT_SIZE,
    CLIP_MEAN,
    CLIP_STD,
    clip_postprocess,
    clip_preprocess,
)


def _make_image(w: int = 300, h: int = 200) -> Image.Image:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_clip_preprocess_output_shape() -> None:
    img = _make_image()
    tensor = clip_preprocess(img)
    assert tensor.shape == (3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE)
    assert tensor.dtype == np.float32


def test_clip_preprocess_landscape() -> None:
    """Landscape image: shortest side is height, resize so height=224."""
    img = _make_image(w=400, h=200)
    tensor = clip_preprocess(img)
    assert tensor.shape == (3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE)


def test_clip_preprocess_portrait() -> None:
    """Portrait image: shortest side is width, resize so width=224."""
    img = _make_image(w=200, h=400)
    tensor = clip_preprocess(img)
    assert tensor.shape == (3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE)


def test_clip_postprocess_unit_norm() -> None:
    vec = np.random.default_rng(0).random(CLIP_EMBEDDING_DIM).astype(np.float32)
    result = clip_postprocess(vec)
    assert len(result) == CLIP_EMBEDDING_DIM
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-5


def test_clip_postprocess_zero_vector() -> None:
    vec = np.zeros(CLIP_EMBEDDING_DIM, dtype=np.float32)
    result = clip_postprocess(vec)
    assert len(result) == CLIP_EMBEDDING_DIM
    assert all(v == 0.0 for v in result)


def test_clip_mean_std_constants() -> None:
    assert len(CLIP_MEAN) == 3
    assert len(CLIP_STD) == 3
    assert all(0.0 < s < 1.0 for s in CLIP_STD)
