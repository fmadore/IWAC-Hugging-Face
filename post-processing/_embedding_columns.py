"""Dataset-side plumbing shared by the two semantic-embedding scripts.

``semantic_embedding.py`` (text) and ``semantic_embedding_images.py`` (images)
read and write their vectors the same way: ``float32`` numpy arrays in memory
(a Python float list costs ~8x more), and a typed ``list<float32>`` column with
nulls for missing rows on the dataset. The encoder itself lives in
``_gemma_encoder``.

Unlike ``_embedding_utils`` (pure, dependency-free helpers), this module may
know about PyArrow and the shape of a ``datasets.Dataset``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import pyarrow as pa

from _embedding_utils import is_empty_embedding


def _as_chunked(values) -> pa.ChunkedArray:
    return values if isinstance(values, pa.ChunkedArray) else pa.chunked_array([values])


def column_vectors(ds, column: str) -> List[Optional[np.ndarray]]:
    """One ``float32`` array (or ``None``) per row, read straight from Arrow.

    Arrays of a ``float32`` column are views of the dataset's own buffers, so
    reading existing vectors costs no per-value Python objects.
    """
    if column not in ds.column_names:
        return [None] * len(ds)
    out: List[Optional[np.ndarray]] = []
    for chunk in _as_chunked(ds.with_format("arrow")[column]).chunks:
        if pa.types.is_null(chunk.type):
            out.extend([None] * len(chunk))
            continue
        if pa.types.is_fixed_size_list(chunk.type) or pa.types.is_large_list(chunk.type):
            chunk = chunk.cast(pa.list_(chunk.type.value_type))
        # ``offsets`` follow the slice; ``values`` is the unsliced child.
        values = chunk.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        offsets = chunk.offsets.to_numpy()
        valid = chunk.is_valid().to_numpy(zero_copy_only=False)
        out.extend(values[offsets[i]:offsets[i + 1]] if valid[i] else None for i in range(len(chunk)))
    return out


def build_embedding_array(all_embeddings: Sequence[Any]) -> pa.Array:
    """Typed ``list<float32>`` array with ``None`` for empty embeddings.

    Building the column explicitly avoids PyArrow type-inference issues when
    embeddings are sparse (all-null or mixed null/list columns). float32 is the
    canonical storage type (``iwac_common.schema.EMBEDDING_VALUE_TYPE``); the
    write gateway would cast to it anyway. Accepts arrays or lists; values are
    concatenated once rather than converted element by element.
    """
    present = np.array([not is_empty_embedding(e) for e in all_embeddings], dtype=bool)
    parts = [np.asarray(e, dtype=np.float32).ravel()
             for e, keep in zip(all_embeddings, present) if keep]
    lengths = np.zeros(len(present), dtype=np.int64)
    lengths[present] = [len(p) for p in parts]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    values = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()), pa.array(values, type=pa.float32()),
        type=pa.list_(pa.float32()), mask=pa.array(~present),
    )


def set_embedding_column(ds, column: str, all_embeddings: List[Any]):
    """Return ``ds`` with ``column`` replaced/added as a typed embedding column."""
    pa_array = build_embedding_array(all_embeddings)
    if column in ds.column_names:
        return ds.remove_columns([column]).add_column(column, pa_array)
    return ds.add_column(column, pa_array)


__all__ = [
    "column_vectors",
    "build_embedding_array",
    "set_embedding_column",
]
