"""Dataset-side plumbing shared by the two semantic-embedding scripts.

``semantic_embedding.py`` (text) and ``semantic_embedding_images.py`` (images)
write their vectors the same way: a typed ``list<float32>`` column with nulls
for missing rows, and a resume-cache restore keyed by row id. The encoder
itself lives in ``_gemma_encoder``.

Unlike ``_embedding_utils`` (pure, dependency-free helpers), this module may
know about PyArrow and the shape of a ``datasets.Dataset``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pyarrow as pa

from _embedding_utils import cached_value, is_empty_embedding


def restore_from_cache(
    all_embeddings: List[Any],
    row_ids: Sequence[Any],
    cache: Dict[str, Any],
    fingerprints: Sequence[str],
) -> int:
    """Overwrite rows whose cached vector was computed from their current
    input; return the count.

    Mutates ``all_embeddings`` in place. Cache keys are stringified row ids and
    entries carry an input fingerprint (``_embedding_utils.make_entry``). An
    entry for input that has changed since — or a legacy entry with no
    fingerprint — is not restored, so the row is embedded afresh instead of
    receiving a vector of some earlier version of its input.
    """
    restored = 0
    for i, (oid, fingerprint) in enumerate(zip(row_ids, fingerprints)):
        vector = cached_value(cache, oid, fingerprint)
        if vector is not None:
            all_embeddings[i] = vector
            restored += 1
    return restored


def build_embedding_array(all_embeddings: List[Any]) -> pa.Array:
    """Typed ``list<float32>`` array with ``None`` for empty embeddings.

    Building the column explicitly avoids PyArrow type-inference issues when
    embeddings are sparse (all-null or mixed null/list columns). float32 is the
    canonical storage type (``iwac_common.schema.EMBEDDING_VALUE_TYPE``); the
    write gateway would cast to it anyway.
    """
    return pa.array(
        [None if is_empty_embedding(e) else e for e in all_embeddings],
        type=pa.list_(pa.float32()),
    )


def set_embedding_column(ds, column: str, all_embeddings: List[Any]):
    """Return ``ds`` with ``column`` replaced/added as a typed embedding column."""
    pa_array = build_embedding_array(all_embeddings)
    if column in ds.column_names:
        return ds.remove_columns([column]).add_column(column, pa_array)
    return ds.add_column(column, pa_array)


__all__ = [
    "restore_from_cache",
    "build_embedding_array",
    "set_embedding_column",
]
