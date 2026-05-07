"""Text embedding preprocessing and postprocessing for embeddinggemma-300m.

Tokenization via HuggingFace ``tokenizers`` (fast, no PyTorch). The model
outputs a pre-pooled ``sentence_embedding`` (shape [batch, 768]), so only
L2 normalization is needed client-side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

TEXT_EMBEDDING_DIM = 768
_MAX_SEQ_LEN = 2048


def load_tokenizer(tokenizer_path: str | Path) -> object:
    """Load a HuggingFace ``tokenizers.Tokenizer`` from a JSON file.

    Returns the Tokenizer instance (imported lazily so ``tokenizers`` is
    an optional dependency only required by callers that use text embedding).
    """
    from tokenizers import Tokenizer  # noqa: E402

    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))


def tokenize(
    tokenizer: object,  # tokenizers.Tokenizer
    texts: list[str],
    max_seq_len: int = _MAX_SEQ_LEN,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Tokenize a batch of texts.

    Returns:
        (input_ids, attention_mask) as int64 arrays of shape (batch, max_seq_len).
        Sequences longer than *max_seq_len* are truncated; shorter ones are
        padded to the right with the tokenizer's pad token (or 0).
    """
    batch_ids: list[list[int]] = []
    for text in texts:
        enc = tokenizer.encode(text)
        ids = enc.ids[:max_seq_len]
        # Pad to max_seq_len on the right
        if len(ids) < max_seq_len:
            pad_id = tokenizer.token_to_id("[PAD]") if tokenizer.token_to_id("[PAD]") is not None else 0
            ids = ids + [pad_id] * (max_seq_len - len(ids))
        batch_ids.append(ids)

    input_ids = np.array(batch_ids, dtype=np.int64)
    attention_mask = np.where(input_ids != (tokenizer.token_to_id("[PAD]") if tokenizer.token_to_id("[PAD]") is not None else 0), 1, 0).astype(np.int64)
    return input_ids, attention_mask


def l2_normalize(embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """L2-normalize each row of *embeddings*.

    Args:
        embeddings: shape (batch, dim)

    Returns:
        L2-normalized array, same shape.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def embed_postprocess(
    sentence_embedding: npt.NDArray[np.float32],
) -> list[list[float]]:
    """L2-normalize the pre-pooled sentence_embedding from the model.

    The embeddinggemma-300m ONNX model outputs a ``sentence_embedding``
    tensor already pooled to shape (batch, 768). Only L2 normalization
    is needed.

    Args:
        sentence_embedding: shape (batch, dim), pre-pooled by the model.

    Returns:
        List of lists, shape (batch, dim) as Python floats.
    """
    normalized = l2_normalize(sentence_embedding)
    return normalized.tolist()
