"""Pure helpers for embedding workflows: chunking, mean-pooling, and a
gzipped on-disk cache keyed by ``o:id``.

Extracted from ``semantic_embedding.py`` to keep that script focused on
orchestration. None of these helpers know about Gemini, datasets, or the
script's CLI surface; they're plain utilities.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


def load_cache(cache_file: Path) -> Dict[str, List[float]]:
    """Load cached embeddings from a gzipped JSON file.

    Returns an empty dict if the file is missing or unreadable.
    """
    if not cache_file.exists():
        return {}
    try:
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} cached embeddings from {cache_file}")
        return data
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load cache ({e}), starting fresh.")
        return {}


def save_cache(cache: Dict[str, List[float]], cache_file: Path) -> None:
    """Atomically write the cache to a gzipped JSON file.

    Creates the parent directory if needed; writes to a ``.tmp.gz`` sibling
    and renames on success so a crash mid-write doesn't corrupt the cache.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp.gz")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(cache, f)
        tmp.replace(cache_file)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to save cache: {e}")
        if tmp.exists():
            tmp.unlink()


def delete_cache(cache_file: Path) -> None:
    """Remove the cache file. Silent if it doesn't exist."""
    try:
        if cache_file.exists():
            cache_file.unlink()
            logger.info("Cache file deleted after successful push.")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to delete cache: {e}")


def is_empty_embedding(emb: Any) -> bool:
    """Return ``True`` for missing, empty, or all-zero embedding values."""
    if emb is None:
        return True
    if isinstance(emb, list):
        if len(emb) == 0:
            return True
        if all(x == 0.0 for x in emb):
            return True
    return False


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split ``text`` into overlapping fixed-size character chunks.

    Short texts pass through as a single-element list. Long texts are split
    at ``chunk_size`` boundaries with ``overlap`` characters of overlap
    between consecutive chunks to preserve context continuity.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """Mean-pool a list of equal-dimension embedding vectors."""
    if len(embeddings) == 1:
        return embeddings[0]
    dim = len(embeddings[0])
    averaged = [0.0] * dim
    for emb in embeddings:
        for j in range(dim):
            averaged[j] += emb[j]
    n = len(embeddings)
    return [v / n for v in averaged]


__all__ = [
    "load_cache",
    "save_cache",
    "delete_cache",
    "is_empty_embedding",
    "chunk_text",
    "average_embeddings",
]
