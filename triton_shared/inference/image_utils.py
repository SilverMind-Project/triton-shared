"""Shared image preprocessing utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image


def letterbox_image(
    image: Image.Image,
    target: int,
    fill: tuple[int, int, int] = (0, 0, 0),
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> tuple[Image.Image, int, int, float]:
    """Scale image to fit *target*×*target*, padding the shorter side with *fill*.

    Returns:
        (padded_image, pad_x, pad_y, scale) where pad_x/pad_y are the pixel
        offsets of the original image within the canvas and scale is the ratio
        applied to both dimensions.
    """
    w, h = image.size
    scale = target / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    scaled = image.resize((new_w, new_h), resample)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas = Image.new("RGB", (target, target), fill)
    canvas.paste(scaled, (pad_x, pad_y))
    return canvas, pad_x, pad_y, scale
