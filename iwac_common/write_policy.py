"""Publication policy shared by projection preparation and the write boundary.

A column allowlist is a reviewed policy, not evidence that an item's private
values are public. Item visibility and per-property restrictions travel with the
private mirror. Missing visibility metadata requires an Omeka refresh.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CONTENT_COLUMNS, DERIVED_FROM, dataset_to_pandas


class PublicationPolicyError(ValueError):
    """A write cannot establish the required publication/privacy contract."""


# These computed outputs were deliberately included in the existing public
# dataset policy. Retaining them is a documented disclosure decision, NOT a
# claim that embeddings cannot reveal source information. Directly private
# Omeka values (including AI summaries/justifications) always override this set.
REVIEWED_DERIVED_COLUMNS = frozenset({
    "embedding_OCR", "embedding_tableOfContents", "embedding_image",
    "nb_mots", "Richesse_Lexicale_OCR", "Lisibilite_OCR",
    "lda_topic_id", "lda_topic_prob", "lda_topic_label", "lda_topic_topk",
    "lda_model_name", "related_articles", "hijri_year", "hijri_month", "hijri_day",
})


def strict_public_flag(value, *, field: str) -> bool:
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise PublicationPolicyError(f"{field} must be a boolean, got {value!r}")


def _private_fields(value) -> set[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise PublicationPolicyError("private_fields must be a list of mapped column names")
    return set(value)


def restricted_columns(row, config_name: str) -> set[str]:
    """Columns withheld for one public item, including unreviewed derivatives."""
    private = _private_fields(row["private_fields"])
    restricted = set(private)
    if "OCR" in private or not strict_public_flag(row.get("OCR_is_public"), field="OCR_is_public"):
        restricted.update(CONTENT_COLUMNS.get(config_name, ()))
    for source, derived in DERIVED_FROM.get(config_name, {}).items():
        if source in private:
            restricted.update(c for c in derived if c not in REVIEWED_DERIVED_COLUMNS)
    # A private property must never hide the policy controls themselves.
    if private & {"o:id", "item_is_public", "private_fields", "OCR_is_public"}:
        raise PublicationPolicyError("private_fields contains a publication control column")
    return restricted


def _empty(value) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value) == 0
    if isinstance(value, str):
        return value == ""
    return isinstance(value, float) and pd.isna(value)


def _provenance_columns() -> set[str]:
    from .enrichment import private_provenance_columns
    return set(private_provenance_columns())


def prepare_public_projection(df: pd.DataFrame, config_name: str) -> tuple[pd.DataFrame, int]:
    required = {"item_is_public", "private_fields"}
    if not required.issubset(df.columns):
        raise PublicationPolicyError(
            f"'{config_name}' lacks visibility metadata {sorted(required - set(df.columns))}; "
            "refresh this subset from Omeka before publishing"
        )
    keep = df["item_is_public"].map(lambda v: strict_public_flag(v, field="item_is_public"))
    result = df.loc[keep].copy()
    result = result.drop(columns=list(set(result.columns) & _provenance_columns()))
    if any(c in result.columns for c in CONTENT_COLUMNS.get(config_name, ())):
        if "OCR_is_public" not in result.columns:
            raise PublicationPolicyError(f"'{config_name}' lacks OCR_is_public")
    for index, row in result.iterrows():
        for column in restricted_columns(row, config_name) & set(result.columns):
            value = result.at[index, column]
            result.at[index, column] = "" if isinstance(value, str) else None
    return result, int((~keep).sum())


def validate_public_projection(ds, config_name: str) -> None:
    """Check the *already projected* data immediately before committing it."""
    from .repos import load_public_columns

    allow = load_public_columns().get(config_name, set())
    unknown = set(ds.column_names) - allow
    if unknown:
        raise PublicationPolicyError(f"Unreviewed public columns in '{config_name}': {sorted(unknown)}")
    provenance = set(ds.column_names) & _provenance_columns()
    if provenance:
        raise PublicationPolicyError(f"Private provenance columns cannot be published: {sorted(provenance)}")
    df = dataset_to_pandas(ds)
    required = {"item_is_public", "private_fields"}
    if not required.issubset(df.columns):
        raise PublicationPolicyError(f"'{config_name}' lacks item/property visibility metadata")
    if any(c in df.columns for c in CONTENT_COLUMNS.get(config_name, ())):
        if "OCR_is_public" not in df.columns:
            raise PublicationPolicyError(f"'{config_name}' lacks OCR_is_public")
    for row in df.to_dict("records"):
        if not strict_public_flag(row["item_is_public"], field="item_is_public"):
            raise PublicationPolicyError(f"Private item {row['o:id']} remains in public projection")
        for column in restricted_columns(row, config_name) & set(df.columns):
            if not _empty(row[column]):
                raise PublicationPolicyError(f"Private value {config_name}/{row['o:id']}/{column} remains in projection")
