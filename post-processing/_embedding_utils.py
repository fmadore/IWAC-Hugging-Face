"""Pure helpers for embedding workflows: chunking, mean-pooling, and a
JSON/transactional SQLite cache keyed by ``o:id``, whose entries carry a
hash of the input they were computed from.

Extracted from ``semantic_embedding.py`` to keep that script focused on
orchestration. None of these helpers know about Gemini, datasets, or the
script's CLI surface; they're plain utilities. The cache helpers store any
JSON-serializable value — ``lemmatize_update_hf.py`` caches string pairs
through them, the embedding scripts cache float vectors.

Note on stored embeddings: vectors are cached and pushed to the Hub RAW,
NOT L2-normalized. Consumers computing cosine similarity must normalize at
read time (``related_articles.py`` does).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence


logger = logging.getLogger(__name__)


def cache_fingerprint(model: str, dimensionality: int, task_type: str) -> str:
    """Filesystem-safe slug identifying the embedding configuration.

    Embedding resume caches MUST embed this in their filename: a cache
    written at one (model, dim, task) must never be restored into a run
    with different parameters — that silently mixes vector spaces.
    """
    safe_model = model.replace("/", "-")
    return f"{safe_model}_{dimensionality}d_{task_type.lower()}"


def repo_slug(repo_id: str) -> str:
    """Filesystem-safe form of a repository id, for resume-cache filenames.

    A scratch run (``IWAC_HF_PRIVATE_REPO``) and a production run must never
    share a cache: identical ``o:id`` values would restore one repo's results
    into the other.
    """
    return repo_id.replace("/", "__")


def input_fingerprint(*parts: Any) -> str:
    """Stable hash of everything a cached value was computed from.

    A configuration fingerprint in the filename proves the cache came from the
    same model; it says nothing about whether the row's *text* is still the
    text that was embedded. Every cache entry therefore carries this hash of
    its input (text plus any per-row processing settings such as chunking), and
    a restore only reuses an entry whose hash matches the current input.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:32]


def make_entry(value: Any, fingerprint: str) -> Dict[str, Any]:
    """A cache entry tied to its input: ``{"h": fingerprint, "v": value}``."""
    return {"h": fingerprint, "v": value}


def cached_value(cache: Dict[str, Any], key: Any, fingerprint: str) -> Any:
    """The cached value for ``key`` if it was computed from this exact input.

    Returns ``None`` for a missing entry, an entry computed from other input,
    and a legacy entry that carries no fingerprint (it cannot prove its input,
    so it is recomputed rather than trusted).
    """
    entry = cache.get(str(key))
    if isinstance(entry, dict) and entry.get("h") == fingerprint:
        return entry.get("v")
    return None


def load_cache(cache_file: Path) -> Dict[str, Any]:
    """Load a legacy gzipped JSON cache or transactional SQLite cache.

    Returns an empty dict if the file is missing or unreadable.
    """
    if not cache_file.exists():
        return {}
    try:
        if cache_file.suffix == ".sqlite3":
            with sqlite3.connect(cache_file) as connection:
                return {key: json.loads(value) for key, value in connection.execute(
                    "SELECT key, value FROM entries"
                )}
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} cached embeddings from {cache_file}")
        return data
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load cache ({e}), starting fresh.")
        return {}


def save_cache(cache: Dict[str, Any], cache_file: Path) -> None:
    """Atomically checkpoint entries (SQLite), or replace a legacy gzip file.

    Creates the parent directory if needed; writes to a ``.tmp.gz`` sibling
    and renames on success so a crash mid-write doesn't corrupt the cache.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.suffix == ".sqlite3":
        # The caller supplies only newly completed entries. Transactions make
        # each checkpoint atomic without reserializing every previous vector.
        with sqlite3.connect(cache_file) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.executemany(
                "INSERT OR REPLACE INTO entries(key, value) VALUES (?, ?)",
                ((str(key), json.dumps(value, allow_nan=False)) for key, value in cache.items()),
            )
        return
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
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def average_embeddings(
    embeddings: List[List[float]],
    weights: Sequence[float] | None = None,
) -> List[float]:
    """Weighted mean-pool a list of equal-dimension embedding vectors.

    ``weights`` should be the chunk lengths (in characters) so a short tail
    chunk doesn't count as much as a full-size one; the overlap region is a
    second-order effect once chunks are length-weighted. Falls back to a
    plain mean when ``weights`` is omitted. The result is NOT L2-normalized.
    """
    if len(embeddings) == 1:
        return list(embeddings[0])

    import numpy as np

    mat = np.asarray(embeddings, dtype=np.float64)
    if weights is None:
        pooled = mat.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(embeddings):
            raise ValueError(
                f"weights length {len(w)} != embeddings length {len(embeddings)}"
            )
        total = w.sum()
        if total <= 0:
            pooled = mat.mean(axis=0)
        else:
            pooled = (mat * w[:, None]).sum(axis=0) / total
    return pooled.tolist()


__all__ = [
    "cache_fingerprint",
    "repo_slug",
    "input_fingerprint",
    "make_entry",
    "cached_value",
    "load_cache",
    "save_cache",
    "delete_cache",
    "is_empty_embedding",
    "chunk_text",
    "average_embeddings",
]
