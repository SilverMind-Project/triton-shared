"""Structural interface for a Triton inference client.

Tests inject a mock that satisfies this Protocol without importing tritonclient.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class TritonClientProtocol(Protocol):
    """Minimal Triton client surface used by all inference wrappers."""

    async def infer(
        self,
        model_name: str,
        inputs: list[tuple[str, npt.NDArray[np.float32]]],
        output_names: list[str],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Run a batched inference request and return named output tensors."""
        ...

    async def is_model_ready(self, model_name: str) -> bool:
        """Return True if Triton reports the model as ready."""
        ...
