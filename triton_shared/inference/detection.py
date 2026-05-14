"""YOLO26L person detection: preprocess and decode.

YOLO26L uses a NMS-Free (end-to-end) architecture. The ONNX graph bakes NMS
into the model, so no post-processing NMS is needed client-side.

Output tensor "output0" shape [batch, 300, 6]:
  dim 0: batch
  dim 1: 300 maximum detections per image (post-NMS)
  dim 2: 6 = x1, y1, x2, y2 (letterbox pixel space), confidence, class_id
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

from triton_shared.inference.image_utils import letterbox_image
from triton_shared.inference.schemas import DetectionBox

# Constants matching person-detector/config.pbtxt
DETECTOR_MODEL_NAME = "person-detector"
DETECTOR_INPUT_SIZE = 640
DETECTOR_PERSON_CLASS = 0
DETECTOR_CONF_THRESHOLD = 0.25


def letterbox_preprocess(
    image: npt.NDArray[np.uint8],
    target: int = DETECTOR_INPUT_SIZE,
) -> tuple[npt.NDArray[np.float32], int, int, float]:
    """Letterbox-resize image to target×target, return (chw_fp32, pad_x, pad_y, scale)."""
    pil = Image.fromarray(image)
    canvas, pad_x, pad_y, scale = letterbox_image(
        pil, target, fill=(114, 114, 114), resample=Image.Resampling.BILINEAR
    )
    float_chw = np.asarray(canvas, dtype=np.float32) / 255.0
    return float_chw.transpose(2, 0, 1), pad_x, pad_y, scale


def decode_output(
    raw: npt.NDArray[np.float32],
    orig_h: int,
    orig_w: int,
    pad_x: int,
    pad_y: int,
    scale: float,
    conf_threshold: float = DETECTOR_CONF_THRESHOLD,
    person_class: int = DETECTOR_PERSON_CLASS,
) -> list[DetectionBox]:
    """Decode YOLO26L NMS-free output0 tensor (300, 6) → DetectionBox list.

    Columns: x1, y1, x2, y2 (letterbox pixel space), confidence, class_id.
    Converts letterbox pixel coords to normalised original-image coordinates.
    """
    conf_col = raw[:, 4]
    class_col = raw[:, 5]
    mask = (conf_col > conf_threshold) & (class_col.round().astype(np.int32) == person_class)
    filtered = raw[mask]

    if filtered.shape[0] == 0:
        return []

    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    x1 = np.clip((filtered[:, 0] - pad_x) / new_w, 0.0, 1.0)
    y1 = np.clip((filtered[:, 1] - pad_y) / new_h, 0.0, 1.0)
    x2 = np.clip((filtered[:, 2] - pad_x) / new_w, 0.0, 1.0)
    y2 = np.clip((filtered[:, 3] - pad_y) / new_h, 0.0, 1.0)
    scores = filtered[:, 4]

    return [
        DetectionBox(
            x1=float(x1[i]),
            y1=float(y1[i]),
            x2=float(x2[i]),
            y2=float(y2[i]),
            confidence=float(scores[i]),
        )
        for i in range(filtered.shape[0])
    ]


def letterbox_batch(
    images: list[npt.NDArray[np.uint8]],
    target: int = DETECTOR_INPUT_SIZE,
) -> tuple[npt.NDArray[np.float32], list[tuple[int, int, int, int, float]]]:
    """Letterbox-resize a batch of RGB images, return (NCHW_tensor, metadata).

    Each metadata tuple is (orig_h, orig_w, pad_x, pad_y, scale).
    Returns an empty (0, 3, target, target) tensor for an empty input list.
    """
    if not images:
        return np.empty((0, 3, target, target), dtype=np.float32), []

    preprocessed: list[npt.NDArray[np.float32]] = []
    meta: list[tuple[int, int, int, int, float]] = []
    for img in images:
        tensor, px, py, scale = letterbox_preprocess(img, target)
        preprocessed.append(tensor)
        meta.append((img.shape[0], img.shape[1], px, py, scale))
    return np.stack(preprocessed), meta
