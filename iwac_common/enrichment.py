"""Persisted provenance for computed columns.

A resume cache is temporary. These per-row fingerprints survive a successful
push and prevent a missing-only run from mixing inputs or processing versions.
Unknown provenance is deliberately recomputed, never inferred from vector size.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

PROVENANCE_OUTPUTS = (
    "embedding_OCR", "embedding_tableOfContents", "embedding_image",
    "lemma_text", "Richesse_Lexicale_OCR", "Lisibilite_OCR",
)
PRIVATE_PROVENANCE_COLUMNS = frozenset(
    f"{column}_{suffix}" for column in PROVENANCE_OUTPUTS
    for suffix in ("input_hash", "config_hash", "config_json")
)


def private_provenance_columns() -> frozenset[str]:
    return PRIVATE_PROVENANCE_COLUMNS


def fingerprint(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_fingerprint(settings: Mapping[str, Any]) -> str:
    return fingerprint(dict(settings))


def config_json(settings: Mapping[str, Any]) -> str:
    return json.dumps(dict(settings), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def provenance_columns(column: str) -> tuple[str, str]:
    return f"{column}_input_hash", f"{column}_config_hash"


def compatible_rows(ds, column: str, inputs: Sequence[str], configuration: str) -> list[bool]:
    input_col, config_col = provenance_columns(column)
    if input_col not in ds.column_names or config_col not in ds.column_names:
        return [False] * len(inputs)
    return [
        previous_input == current_input and previous_config == configuration
        for previous_input, current_input, previous_config in zip(
            ds[input_col], inputs, ds[config_col], strict=True
        )
    ]


def set_provenance(ds, column: str, inputs: Sequence[str], configuration: str,
                   completed: Sequence[bool], settings: Mapping[str, Any] | None = None):
    """Attach typed, nullable hashes; failures never acquire valid provenance."""
    import pyarrow as pa

    input_col, config_col = provenance_columns(column)
    values = {
        input_col: [value if done else None for value, done in zip(inputs, completed, strict=True)],
        config_col: [configuration if done else None for done in completed],
    }
    if settings is not None:
        if config_fingerprint(settings) != configuration:
            raise ValueError("Provenance configuration hash does not match its settings")
        description = config_json(settings)
        values[f"{column}_config_json"] = [description if done else None for done in completed]
    for name, value in values.items():
        if name in ds.column_names:
            ds = ds.remove_columns(name)
        ds = ds.add_column(name, pa.array(value, type=pa.string()))
    return ds


def invalidate_columns(ds, columns: Sequence[str], changed: Sequence[bool]):
    """Clear dependent outputs while retaining their declared Arrow types."""
    import pyarrow as pa

    if not any(changed):
        return ds
    for column in columns:
        if column not in ds.column_names:
            continue
        values = [None if stale else value for value, stale in zip(ds[column], changed, strict=True)]
        arrow_type = ds.data.column(column).type
        ds = ds.remove_columns(column).add_column(column, pa.array(values, type=arrow_type))
    return ds
