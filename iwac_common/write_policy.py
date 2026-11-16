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
    "embedding_OCR", "embedding_abstract", "embedding_tableOfContents", "embedding_image",
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


#: Columns whose values name other items of the same subset, serialised by
#: ``related_articles.py`` as ``"<o:id>:<cosine>|<o:id>:<cosine>|..."``. A
#: per-row mask cannot protect them: a public row would still name a private
#: item's id and its similarity. They may only reference published ids.
NEIGHBOUR_ID_COLUMNS = ("related_articles",)


def parse_neighbours(value) -> list[tuple[str, str]]:
    """``(o:id, cosine text)`` pairs of one neighbour-list value; [] when empty.

    Anything that is not the stage's exact serialisation is refused rather than
    guessed at: an unparseable value cannot be shown to reference only
    published items.
    """
    if _empty(value):
        return []
    if not isinstance(value, str):
        raise PublicationPolicyError(f"Neighbour list must be a string, got {type(value).__name__}")
    pairs = []
    for entry in value.split("|"):
        item_id, sep, score = entry.rpartition(":")
        if not sep or not item_id.strip() or not score.strip():
            raise PublicationPolicyError(f"Malformed neighbour entry {entry!r}")
        try:
            float(score)
        except ValueError as exc:
            raise PublicationPolicyError(f"Malformed neighbour score in {entry!r}") from exc
        pairs.append((item_id.strip(), score))
    return pairs


def strip_unpublished_neighbours(values: pd.Series, published_ids: set[str]) -> tuple[pd.Series, int]:
    """Drop every neighbour entry whose id is not published; return (values, n_removed).

    A row left with no neighbour becomes null, as rows without an embedding are.
    """
    removed = 0

    def keep_published(value):
        nonlocal removed
        pairs = parse_neighbours(value)
        if not pairs:
            return value
        kept = [(item_id, score) for item_id, score in pairs if item_id in published_ids]
        removed += len(pairs) - len(kept)
        if len(kept) == len(pairs):
            return value
        return "|".join(f"{item_id}:{score}" for item_id, score in kept) or None

    # An explicit object Series: a pandas string dtype would turn None into NaN.
    kept = pd.Series([keep_published(v) for v in values.tolist()], index=values.index, dtype=object)
    return kept, removed


def prepare_public_projection(df: pd.DataFrame, config_name: str) -> tuple[pd.DataFrame, int]:
    """Public rows only, private values masked, neighbour lists restricted to
    published ids. The number of stripped neighbour entries is recorded in
    ``result.attrs["neighbour_entries_removed"]``."""
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
    # An item that turned private after its neighbours were computed must not
    # remain named by a public row.
    published_ids = set(result["o:id"].astype(str)) if "o:id" in result.columns else set()
    removed = 0
    for column in NEIGHBOUR_ID_COLUMNS:
        if column in result.columns:
            result[column], n = strip_unpublished_neighbours(result[column], published_ids)
            removed += n
    result.attrs["neighbour_entries_removed"] = removed
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
    published_ids = set(df["o:id"].astype(str))
    neighbour_columns = [c for c in NEIGHBOUR_ID_COLUMNS if c in df.columns]
    for row in df.to_dict("records"):
        if not strict_public_flag(row["item_is_public"], field="item_is_public"):
            raise PublicationPolicyError(f"Private item {row['o:id']} remains in public projection")
        for column in restricted_columns(row, config_name) & set(df.columns):
            if not _empty(row[column]):
                raise PublicationPolicyError(f"Private value {config_name}/{row['o:id']}/{column} remains in projection")
        for column in neighbour_columns:
            unpublished = [i for i, _ in parse_neighbours(row[column]) if i not in published_ids]
            if unpublished:
                raise PublicationPolicyError(
                    f"{config_name}/{row['o:id']}/{column} names unpublished item(s) {unpublished[:5]}"
                )
