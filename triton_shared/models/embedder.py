"""Thin model wrappers composing TritonClientProtocol + processing functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from triton_shared.client.protocol import TritonClientProtocol
from triton_shared.inference.text_embedding import (
    embed_postprocess,
    load_tokenizer,
    tokenize,
)


class TextEmbedder:
    """Embed sentences via a text embedding model on Triton.

    Mirrors the shape of ``ClipEmbedder`` (planned): constructor takes a
    Triton client, model name, tokenizer path, and optional config. The
    ``embed`` method returns L2-normalized float32 arrays.
    """

    def __init__(
        self,
        client: TritonClientProtocol,
        model_name: str,
        tokenizer_path: str | Path,
        max_seq_len: int = 2048,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._tokenizer = load_tokenizer(tokenizer_path)
        self._max_seq_len = max_seq_len

    async def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return (len(texts), DIM) L2-normalized float32 array.

        Raises TritonError on inference failure.
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        input_ids, attention_mask = tokenize(
            self._tokenizer, texts, max_seq_len=self._max_seq_len
        )

        outputs = await self._client.infer(
            model_name=self._model_name,
            inputs=[
                ("input_ids", input_ids.astype(np.float32)),
                ("attention_mask", attention_mask.astype(np.float32)),
            ],
            output_names=["sentence_embedding"],
        )

        sentence_embedding = outputs["sentence_embedding"]
        embeddings = embed_postprocess(sentence_embedding)
        return np.array(embeddings, dtype=np.float32)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Returns a Python list[float]."""
        arr = await self.embed([text])
        return arr[0].tolist()

    async def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text chunks. Returns list[list[float]]."""
        arr = await self.embed(texts)
        return arr.tolist()
