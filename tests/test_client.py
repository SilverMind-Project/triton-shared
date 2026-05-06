"""Tests for the Triton client module — no GPU or tritonclient required."""

from __future__ import annotations

import numpy as np

from triton_shared.client import TritonClientProtocol, TritonError, ModelNotReadyError
from triton_shared.client.grpc import TritonGrpcClient


def test_triton_error() -> None:
    err = TritonError("test-model", "something went wrong")
    assert "test-model" in str(err)
    assert "something went wrong" in str(err)
    assert err.model_name == "test-model"


def test_model_not_ready_error() -> None:
    err = ModelNotReadyError("test-model")
    assert "not ready" in str(err)
    assert err.model_name == "test-model"


def test_grpc_client_not_open_raises() -> None:
    client = TritonGrpcClient("localhost:8001")
    try:
        import asyncio

        async def _call() -> None:
            return await client.infer(
                "test",
                [("input", np.zeros((1, 3, 640, 640), dtype=np.float32))],
                ["output"],
            )

        asyncio.run(_call())
        assert False, "should have raised"
    except RuntimeError as e:
        assert "not open" in str(e)


def test_grpc_client_protocol_satisfaction() -> None:
    """TritonGrpcClient structurally satisfies TritonClientProtocol (runtime check)."""
    client = TritonGrpcClient("localhost:8001")
    # Not opened, so can't call infer — but the type is structurally compatible.
    assert isinstance(client, TritonClientProtocol)
