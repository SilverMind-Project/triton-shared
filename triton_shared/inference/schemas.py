"""Shared inference data types.

These are distinct from domain types — they carry raw model outputs that
services translate into domain objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DetectionBox:
    """One bounding box in normalised image coordinates [0, 1]."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


#: 768-dim L2-normalised appearance embedding from SOLIDER-REID or CLIP.
Embedding = npt.NDArray[np.float32]
