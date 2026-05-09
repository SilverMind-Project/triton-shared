"""Florence-2-large image preprocessing and task prompt utilities.

Shared between SAS and CTS for Florence-2 scene description via Triton.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

# Florence-2 task prompts (see microsoft/Florence-2-large documentation).
TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
TASK_CAPTION = "<CAPTION>"
TASK_OBJECT_DETECTION = "<OD>"

# Florence-2 image preprocessing constants.
# Matches the preprocessor_config.json from onnx-community/Florence-2-large.
# Resize shortest side to input_size, rescale to [0, 1], then ImageNet normalize.
FLORENCE_INPUT_SIZE = 768
FLORENCE_MEAN = (0.485, 0.456, 0.406)
FLORENCE_STD = (0.229, 0.224, 0.225)


def florence_preprocess(
    image: Image.Image,
    input_size: int = FLORENCE_INPUT_SIZE,
) -> npt.NDArray[np.float32]:
    """Preprocess a PIL image for Florence-2-large (ONNX community export).

    1. Resize shortest side to *input_size* (bicubic)
    2. Convert to float32 [0, 1], CHW layout
    3. Normalize with ImageNet mean/std

    Returns:
        (1, 3, H, W) float32 tensor in NCHW format.
    """
    # Resize so the shortest side matches input_size.
    w, h = image.size
    if w < h:
        new_w = input_size
        new_h = round(h * input_size / w)
    else:
        new_h = input_size
        new_w = round(w * input_size / h)
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # Convert to float32 CHW, scale to [0, 1].
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)  # HWC → CHW

    # Normalize.
    mean = np.array(FLORENCE_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(FLORENCE_STD, dtype=np.float32).reshape(3, 1, 1)
    normalized = (chw - mean) / std

    # Add batch dimension.
    return np.expand_dims(normalized, axis=0)  # (1, 3, H, W)


def tokenize_task_prompt(
    tokenizer: object,  # tokenizers.Tokenizer
    task: str = TASK_DETAILED_CAPTION,
) -> list[int]:
    """Tokenize a Florence-2 task prompt using a ``tokenizers.Tokenizer`` instance.

    Returns:
        List of token IDs for the task prompt.
    """
    encoding = tokenizer.encode(task)
    return encoding.ids
