"""Canonical IWAC subset registry and lightweight dataframe contracts.

The same seven subset names and Omeka resource-class ids used to be repeated
in upload scripts, the public publisher, the local mirror downloader, tests,
and documentation.  This module is the machine-readable source of truth for
those stable facts.  It deliberately does *not* auto-approve public columns:
``public_columns.json`` remains a reviewed rights allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class SubsetDefinition:
    name: str
    resource_class_ids: tuple[int, ...]
    content_columns: tuple[str, ...] = ()
    embedding_columns: Mapping[str, int] | None = None


SUBSETS: dict[str, SubsetDefinition] = {
    "articles": SubsetDefinition(
        "articles", (36,),
        ("OCR", "lemma_text", "lemma_nostop"),
        {"embedding_OCR": 768},
    ),
    "publications": SubsetDefinition(
        "publications", (60,),
        ("OCR", "lemma_text", "lemma_nostop"),
        {"embedding_tableOfContents": 768},
    ),
    "index": SubsetDefinition(
        "index", (9, 94, 96, 54, 244)
    ),
    "references": SubsetDefinition(
        "references", (35, 43, 88, 40, 82, 178, 52, 77, 305),
        ("OCR", "lemma_text", "lemma_nostop"),
        {"embedding_OCR": 768},
    ),
    "audiovisual": SubsetDefinition(
        "audiovisual", (38,), ("OCR",)
    ),
    "documents": SubsetDefinition(
        "documents", (49,), ("OCR", "lemma_text", "lemma_nostop")
    ),
    "images": SubsetDefinition(
        "images", (58,), embedding_columns={"embedding_image": 768}
    ),
}

ALL_CONFIGS: tuple[str, ...] = tuple(SUBSETS)
CONTENT_COLUMNS: dict[str, list[str]] = {
    name: list(spec.content_columns)
    for name, spec in SUBSETS.items()
    if spec.content_columns
}


class DataContractError(ValueError):
    """A dataframe violates an invariant required for a safe Hub write."""


def validate_ids(df: pd.DataFrame, *, label: str = "dataset") -> None:
    """Require a non-null, unique canonical ``o:id`` column."""
    if "o:id" not in df.columns:
        raise DataContractError(f"{label} is missing the required 'o:id' column")
    if df["o:id"].isna().any():
        raise DataContractError(f"{label} contains null 'o:id' values")
    ids = df["o:id"].astype(str)
    duplicated = ids[ids.duplicated()].unique()
    if len(duplicated):
        sample = ", ".join(duplicated[:5])
        raise DataContractError(
            f"{label} contains duplicated 'o:id' values (e.g. {sample})"
        )


def validate_embedding_dimensions(
    df: pd.DataFrame, config_name: str, *, allow_empty: bool = True
) -> None:
    """Validate non-empty embedding values against the subset contract."""
    spec = SUBSETS.get(config_name)
    if spec is None:
        raise DataContractError(f"Unknown IWAC subset: {config_name!r}")
    for column, expected in (spec.embedding_columns or {}).items():
        if column not in df.columns:
            continue
        for row_id, value in zip(df["o:id"], df[column]):
            if value is None:
                continue
            try:
                size = len(value)
            except TypeError as exc:
                raise DataContractError(
                    f"{config_name}.{column} for o:id={row_id} is not a vector"
                ) from exc
            if size == 0 and allow_empty:
                continue
            if size != expected:
                raise DataContractError(
                    f"{config_name}.{column} for o:id={row_id} has dimension "
                    f"{size}, expected {expected}"
                )


def validate_frame(df: pd.DataFrame, config_name: str) -> None:
    """Run the inexpensive contracts shared by every subset write."""
    if config_name not in SUBSETS:
        raise DataContractError(f"Unknown IWAC subset: {config_name!r}")
    validate_ids(df, label=f"'{config_name}' dataframe")
    validate_embedding_dimensions(df, config_name)


def validate_dataset(ds, config_name: str) -> None:
    """Validate a ``datasets.Dataset`` without materializing all columns."""
    spec = SUBSETS.get(config_name)
    if spec is None:
        raise DataContractError(f"Unknown IWAC subset: {config_name!r}")
    if "o:id" not in ds.column_names:
        raise DataContractError(f"'{config_name}' dataset is missing 'o:id'")
    ids = ds["o:id"]
    if any(value is None for value in ids):
        raise DataContractError(f"'{config_name}' dataset contains null 'o:id' values")
    normalized = [str(value) for value in ids]
    if len(normalized) != len(set(normalized)):
        raise DataContractError(f"'{config_name}' dataset contains duplicate 'o:id' values")
    for column, expected in (spec.embedding_columns or {}).items():
        if column not in ds.column_names:
            continue
        for row_id, value in zip(ids, ds[column]):
            if value is None:
                continue
            try:
                size = len(value)
            except TypeError as exc:
                raise DataContractError(
                    f"{config_name}.{column} for o:id={row_id} is not a vector"
                ) from exc
            if size == 0:
                continue
            if size != expected:
                raise DataContractError(
                    f"{config_name}.{column} for o:id={row_id} has dimension "
                    f"{size}, expected {expected}"
                )


__all__ = [
    "SubsetDefinition",
    "SUBSETS",
    "ALL_CONFIGS",
    "CONTENT_COLUMNS",
    "DataContractError",
    "validate_ids",
    "validate_embedding_dimensions",
    "validate_frame",
    "validate_dataset",
]
