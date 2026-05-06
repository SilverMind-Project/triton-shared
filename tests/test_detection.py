"""Tests for YOLO26L detection preprocessing and decode."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from triton_shared.inference.detection import (
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_INPUT_SIZE,
    DETECTOR_PERSON_CLASS,
    decode_output,
    letterbox_batch,
    letterbox_preprocess,
)
from triton_shared.inference.schemas import DetectionBox


def _make_image(h: int = 120, w: int = 160) -> npt.NDArray[np.uint8]:
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _make_yolo_output(
    batch: int,
    x1: float = 270.0,
    y1: float = 165.0,
    x2: float = 370.0,
    y2: float = 315.0,
    conf: float = 0.9,
    class_id: float = 0.0,
) -> npt.NDArray[np.float32]:
    """Construct a fake YOLO26L NMS-free output0 tensor [batch, 300, 6]."""
    out = np.zeros((batch, 300, 6), dtype=np.float32)
    out[:, 0, 0] = x1
    out[:, 0, 1] = y1
    out[:, 0, 2] = x2
    out[:, 0, 3] = y2
    out[:, 0, 4] = conf
    out[:, 0, 5] = class_id
    return out


# ---------------------------------------------------------------------------
# DetectionBox
# ---------------------------------------------------------------------------


def test_detection_box_area() -> None:
    b = DetectionBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8, confidence=0.9)
    assert abs(b.area - 0.24) < 1e-6


def test_detection_box_zero_area() -> None:
    b = DetectionBox(x1=0.5, y1=0.5, x2=0.5, y2=0.5, confidence=0.5)
    assert b.area == 0.0


# ---------------------------------------------------------------------------
# letterbox_preprocess
# ---------------------------------------------------------------------------


def test_letterbox_square_output() -> None:
    img = _make_image(480, 640)
    tensor, _px, _py, _scale = letterbox_preprocess(img, 640)
    assert tensor.shape == (3, 640, 640)
    assert tensor.dtype == np.float32


def test_letterbox_values_in_range() -> None:
    img = _make_image(100, 200)
    tensor, *_ = letterbox_preprocess(img, 640)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_letterbox_scale_preserves_aspect_ratio() -> None:
    img = _make_image(480, 640)
    _tensor, pad_x, pad_y, scale = letterbox_preprocess(img, 640)
    # 640 is the longer side, so scale should be 640/640 = 1.0
    assert scale == 1.0
    assert pad_x == 0
    # 480 height → resized to 480, padding = (640 - 480) / 2 = 80
    assert pad_y == 80


def test_letterbox_batch() -> None:
    images = [_make_image(100, 200), _make_image(300, 400)]
    batch, meta = letterbox_batch(images)
    assert batch.shape == (2, 3, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE)
    assert batch.dtype == np.float32
    assert len(meta) == 2
    for m in meta:
        assert len(m) == 5  # orig_h, orig_w, pad_x, pad_y, scale


def test_letterbox_batch_empty() -> None:
    batch, meta = letterbox_batch([])
    assert batch.shape == (0, 3, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE)
    assert meta == []


# ---------------------------------------------------------------------------
# decode_output
# ---------------------------------------------------------------------------


def test_decode_finds_person() -> None:
    raw = _make_yolo_output(1)[0]  # single sample (300, 6)
    boxes = decode_output(raw, orig_h=480, orig_w=640, pad_x=0, pad_y=80, scale=1.0)
    assert len(boxes) == 1
    b = boxes[0]
    assert 0.0 <= b.x1 <= b.x2 <= 1.0
    assert 0.0 <= b.y1 <= b.y2 <= 1.0
    assert b.confidence >= DETECTOR_CONF_THRESHOLD


def test_decode_empty_on_low_confidence() -> None:
    raw = np.zeros((300, 6), dtype=np.float32)
    raw[0, 4] = 0.1  # below threshold
    raw[0, 5] = float(DETECTOR_PERSON_CLASS)
    boxes = decode_output(raw, orig_h=480, orig_w=640, pad_x=0, pad_y=0, scale=1.0)
    assert boxes == []


def test_decode_filters_non_person_class() -> None:
    raw = _make_yolo_output(1, conf=0.9, class_id=1.0)[0]
    boxes = decode_output(raw, orig_h=480, orig_w=640, pad_x=0, pad_y=0, scale=1.0)
    assert boxes == []


def test_decode_normalizes_coordinates() -> None:
    """Detection at (270, 165, 370, 315) in 640x640 letterbox for 480x640 original.

    scale=1.0, pad_x=0, pad_y=80 (480→640, 80px padding top+bottom).
    Expected: x1=270/640≈0.422, y1=(165-80)/480≈0.177, x2=370/640≈0.578, y2=(315-80)/480≈0.490
    """
    raw = _make_yolo_output(1, x1=270, y1=165, x2=370, y2=315, conf=0.9, class_id=0.0)[0]
    boxes = decode_output(raw, orig_h=480, orig_w=640, pad_x=0, pad_y=80, scale=1.0)
    assert len(boxes) == 1
    b = boxes[0]
    assert abs(b.x1 - 0.421875) < 0.01
    assert abs(b.y1 - 0.177083) < 0.01
    assert abs(b.x2 - 0.578125) < 0.01
    assert abs(b.y2 - 0.489583) < 0.01


