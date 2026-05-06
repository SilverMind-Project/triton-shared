"""CLIP ViT-L/14 image preprocessing and postprocessing.

Shared between SAS and CTS for CLIP vision encoder inference via Triton.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

# CLIP ViT-L/14 normalization constants (OpenAI).
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

CLIP_INPUT_SIZE = 224
CLIP_EMBEDDING_DIM = 768


def clip_preprocess(
    image: Image.Image,
    target_size: int = CLIP_INPUT_SIZE,
) -> npt.NDArray[np.float32]:
    """Preprocess a PIL image for CLIP ViT-L/14 vision encoder.

    1. Resize shortest side to *target_size* (bicubic)
    2. Center-crop to target_size×target_size
    3. Convert to float32 [0, 1], CHW layout
    4. Normalize with CLIP mean/std

    Returns:
        (3, target_size, target_size) float32 tensor.
    """
    # Resize so the shortest side matches target_size (bicubic).
    w, h = image.size
    if w < h:
        new_w = target_size
        new_h = round(h * target_size / w)
    else:
        new_h = target_size
        new_w = round(w * target_size / h)
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # Center-crop to target_size×target_size.
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    cropped = resized.crop((left, top, left + target_size, top + target_size))

    # Convert to numpy float32 CHW, scale to [0, 1].
    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)  # HWC → CHW

    # Normalize with CLIP mean/std.
    mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
    return (chw - mean) / std


def clip_postprocess(
    output: npt.NDArray[np.float32],
) -> list[float]:
    """L2-normalize the CLIP vision encoder output and convert to Python float list."""
    vec = np.asarray(output, dtype=np.float32).flatten()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def clip_batch_preprocess(
    images: list[Image.Image],
    target_size: int = CLIP_INPUT_SIZE,
) -> npt.NDArray[np.float32]:
    """Preprocess a batch of PIL images for CLIP."""
    if not images:
        return np.empty((0, 3, target_size, target_size), dtype=np.float32)
    tensors = [clip_preprocess(img, target_size) for img in images]
    return np.stack(tensors)
