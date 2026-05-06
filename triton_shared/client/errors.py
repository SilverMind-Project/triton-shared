"""Triton client error types."""

from __future__ import annotations


class TritonError(RuntimeError):
    """Raised when a Triton inference request fails."""

    def __init__(self, model_name: str, detail: str) -> None:
        super().__init__(f"Triton inference failed for '{model_name}': {detail}")
        self.model_name = model_name
        self.detail = detail


class ModelNotReadyError(TritonError):
    """Raised when the requested Triton model is not ready."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name, "model is not ready")
