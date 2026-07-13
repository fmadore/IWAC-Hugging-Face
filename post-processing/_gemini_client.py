"""Shared Gemini-embedding plumbing for the two semantic-embedding scripts.

``semantic_embedding.py`` (text) and ``semantic_embedding_images.py``
(multimodal) share the same operational shell around different API payloads:
the 429/backoff retry ladder, the resume-cache restore loop, and the typed
PyArrow column build. Those shared pieces live here; each script keeps its
own API-call specifics (``task_type`` for text, image ``Content`` parts for
images) and its own model/batch constants.

Unlike ``_embedding_utils`` (pure, dependency-free helpers), this module may
know about PyArrow and the shape of a ``datasets.Dataset``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Sequence

import pyarrow as pa

from _embedding_utils import is_empty_embedding

logger = logging.getLogger(__name__)

# --- Retry ladder (identical across text and image embedding) ---
MAX_RETRIES = 6
BASE_RETRY_DELAY = 5  # seconds


def call_with_retry(embed_call: Callable[[], List[List[float]]]) -> List[List[float]]:
    """Run a Gemini embed call with exponential backoff for rate limiting.

    ``embed_call`` is a zero-argument callable performing the actual
    ``client.models.embed_content(...)`` request and returning the list of
    embedding vectors. 429/RESOURCE_EXHAUSTED errors always back off and
    retry; other errors retry with the same backoff until the last attempt,
    which re-raises.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return embed_call()
        except Exception as e:  # noqa: BLE001
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate limited (attempt {attempt + 1}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                wait = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def restore_from_cache(
    all_embeddings: List[Any],
    row_ids: Sequence[Any],
    cache: Dict[str, Any],
) -> int:
    """Overwrite rows whose ``o:id`` has a cached vector; return the count.

    Mutates ``all_embeddings`` in place. Cache keys are stringified row ids
    (the resume caches written by ``_embedding_utils.save_cache``).
    """
    restored = 0
    for i, oid in enumerate(row_ids):
        oid_str = str(oid)
        if oid_str in cache:
            all_embeddings[i] = cache[oid_str]
            restored += 1
    return restored


def build_embedding_array(all_embeddings: List[Any]) -> pa.Array:
    """Typed ``list<float64>`` array with ``None`` for empty embeddings.

    Building the column explicitly avoids PyArrow type-inference issues when
    embeddings are sparse (all-null or mixed null/list columns).
    """
    return pa.array(
        [None if is_empty_embedding(e) else e for e in all_embeddings],
        type=pa.list_(pa.float64()),
    )


def set_embedding_column(ds, column: str, all_embeddings: List[Any]):
    """Return ``ds`` with ``column`` replaced/added as a typed embedding column."""
    pa_array = build_embedding_array(all_embeddings)
    if column in ds.column_names:
        return ds.remove_columns([column]).add_column(column, pa_array)
    return ds.add_column(column, pa_array)


__all__ = [
    "MAX_RETRIES",
    "BASE_RETRY_DELAY",
    "call_with_retry",
    "restore_from_cache",
    "build_embedding_array",
    "set_embedding_column",
]
