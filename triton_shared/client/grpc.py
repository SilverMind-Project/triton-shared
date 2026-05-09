"""Async Triton Inference Server gRPC client.

tritonclient has no type stubs; the library import is deferred to the
first call to keep the module importable without the extra installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# Map numpy dtype to Triton dtype string.
_NP_TO_TRITON_DTYPE: dict[type, str] = {
    np.float32: "FP32",
    np.float64: "FP64",
    np.float16: "FP16",
    np.int64: "INT64",
    np.int32: "INT32",
    np.int16: "INT16",
    np.int8: "INT8",
    np.uint64: "UINT64",
    np.uint32: "UINT32",
    np.uint16: "UINT16",
    np.uint8: "UINT8",
    np.bool_: "BOOL",
}


def _triton_dtype(arr: np.ndarray) -> str:
    """Return the Triton dtype string for a numpy array."""
    return _NP_TO_TRITON_DTYPE.get(arr.dtype.type, "FP32")


class TritonGrpcClient:
    """Async gRPC Triton client.  Use as an async context manager.

    Example::

        async with TritonGrpcClient("localhost:8701") as client:
            outputs = await client.infer("person-detector", inputs, ["output0"])
    """

    def __init__(self, url: str, timeout_ms: int = 150) -> None:
        self._url = url
        self._timeout_ms = timeout_ms
        self._client: Any = None  # tritonclient.grpc.aio.InferenceServerClient

    async def __aenter__(self) -> TritonGrpcClient:
        self._client = self._make_client()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def infer(
        self,
        model_name: str,
        inputs: list[tuple[str, npt.NDArray[Any]]],
        output_names: list[str],
    ) -> dict[str, npt.NDArray[Any]]:
        """Run a batched inference request on Triton."""
        client = self._require_client()
        tc = self._triton_common()

        triton_inputs: list[Any] = []
        for name, array in inputs:
            arr = np.ascontiguousarray(array)
            dtype_str = _triton_dtype(arr)
            inp: Any = tc.InferInput(name, list(arr.shape), dtype_str)
            inp.set_data_from_numpy(arr)
            triton_inputs.append(inp)

        triton_outputs: list[Any] = [tc.InferRequestedOutput(n) for n in output_names]

        result: Any = await client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
            client_timeout=self._timeout_ms / 1000.0,
        )

        return {name: np.asarray(result.as_numpy(name)) for name in output_names}

    async def is_model_ready(self, model_name: str) -> bool:
        client = self._require_client()
        return bool(await client.is_model_ready(model_name))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_client(self) -> Any:
        try:
            import tritonclient.grpc.aio as _aio
        except ImportError as exc:
            raise ImportError(
                "tritonclient is not installed. Run: uv sync --extra triton"
            ) from exc
        return _aio.InferenceServerClient(url=self._url, verbose=False)

    @staticmethod
    def _triton_common() -> Any:
        try:
            import tritonclient.grpc as _tc
        except ImportError as exc:
            raise ImportError(
                "tritonclient is not installed. Run: uv sync --extra triton"
            ) from exc
        return _tc

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError(
                "TritonGrpcClient is not open. Use 'async with TritonGrpcClient(...)'."
            )
        return self._client
