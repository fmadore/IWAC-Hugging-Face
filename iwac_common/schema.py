"""Canonical IWAC subset registry and lightweight dataframe contracts.

The same seven subset names and Omeka resource-class ids used to be repeated
in upload scripts, the public publisher, the local mirror downloader, tests,
and documentation.  This module is the machine-readable source of truth for
those stable facts.  It deliberately does *not* auto-approve public columns:
``public_columns.json`` remains a reviewed rights allowlist.

It also owns the **canonical column types** that every write is conformed to
(:func:`conform_dataset`, called by the write gateway). They exist because a
pandas round trip silently turns a nullable integer column into ``float64``:
``lda_topic_id``, ``nb_pages`` and ``hijri_*`` reached the public dataset as
floats that way, while ``images`` (no nulls) kept ``int64`` for the very same
``hijri_*`` columns. Declaring the types once and enforcing them at the single
write path fixes every writer at the same time, including the ones that do not
exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .sentiment_panel import PANEL, consensus_columns, numeric_subjectivite_columns

#: Stored element type of every embedding vector. Gemini returns values of
#: float32 precision; storing them as float64 doubled the bytes of the largest
#: columns in the dataset without adding information (cosine similarity is
#: unaffected at 768 dimensions).
EMBEDDING_VALUE_TYPE = "float32"

_HIJRI = ("hijri_year", "hijri_month", "hijri_day")


@dataclass(frozen=True)
class SubsetDefinition:
    name: str
    resource_class_ids: tuple[int, ...]
    content_columns: tuple[str, ...] = ()
    embedding_columns: Mapping[str, int] | None = None
    #: Columns stored as (nullable) ``int64``. Conformed at write time, so a
    #: float that crept in through pandas is cast back — and a genuinely
    #: fractional value fails the write instead of being truncated.
    int_columns: tuple[str, ...] = ()


SUBSETS: dict[str, SubsetDefinition] = {
    "articles": SubsetDefinition(
        "articles", (36,),
        ("OCR", "lemma_text", "lemma_nostop"),
        {"embedding_OCR": 768},
        ("nb_pages", *_HIJRI, "pub_year", "nb_mots", "lda_topic_id",
         # Generation-1 subjectivité: the 1-5 integer. Generation-2 columns
         # hold labels and are strings.
         *numeric_subjectivite_columns()),
    ),
    "publications": SubsetDefinition(
        "publications", (60,),
        ("OCR", "lemma_text", "lemma_nostop"),
        {"embedding_tableOfContents": 768},
        ("nb_pages", *_HIJRI, "pub_year", "nb_mots", "lda_topic_id"),
    ),
    "index": SubsetDefinition(
        "index", (9, 94, 96, 54, 244),
        int_columns=("frequency",),
    ),
    "references": SubsetDefinition(
        "references", (35, 43, 88, 40, 82, 178, 52, 77, 305),
        ("OCR", "lemma_text", "lemma_nostop"),
        {"embedding_OCR": 768},
        # chapter/edition/nb_pages/page_* stay strings here on purpose: the
        # source holds ranges such as "12-15".
        ("pub_year", "nb_mots", "lda_topic_id"),
    ),
    "audiovisual": SubsetDefinition(
        "audiovisual", (38,), ("OCR",),
        int_columns=(*_HIJRI, "pub_year", "duration_seconds", "nb_mots"),
    ),
    "documents": SubsetDefinition(
        "documents", (49,), ("OCR", "lemma_text", "lemma_nostop"),
        int_columns=("nb_pages", *_HIJRI, "pub_year", "nb_mots"),
    ),
    "images": SubsetDefinition(
        "images", (58,), embedding_columns={"embedding_image": 768},
        int_columns=(*_HIJRI, "pub_year"),
    ),
}

ALL_CONFIGS: tuple[str, ...] = tuple(SUBSETS)

_OCR_METRICS = ("nb_mots", "Richesse_Lexicale_OCR", "Lisibilite_OCR")
_LEMMAS = ("lemma_text", "lemma_nostop")
_LDA = ("lda_topic_id", "lda_topic_prob", "lda_topic_label", "lda_topic_topk",
        "lda_model_name")

#: Which computed columns are derived from which source column, per subset.
#: The upload compares each source column with the Hub copy; where a row's
#: source changed, its derived values describe an older version of the item.
#: Only Hub-only (post-processed) columns are ever affected — a column the
#: mapper produces is fresh by construction. Raw sentiment annotations are
#: read from Omeka (generation 1 is frozen history), not recomputed from OCR.
#: Their consensus descendants do depend on changed raw annotation labels.
DERIVED_FROM: dict[str, dict[str, tuple[str, ...]]] = {
    "articles": {
        "OCR": ("embedding_OCR", *_LEMMAS, *_OCR_METRICS, *_LDA, "related_articles"),
        "title": ("embedding_OCR", "related_articles"),
        "language": (*_LEMMAS, "Richesse_Lexicale_OCR", "Lisibilite_OCR", *_LDA),
        "pub_date": _HIJRI,
    },
    "publications": {
        "OCR": (*_LEMMAS, *_OCR_METRICS, *_LDA),
        "title": ("embedding_tableOfContents", "related_articles"),
        "language": (*_LEMMAS, "Richesse_Lexicale_OCR", "Lisibilite_OCR", *_LDA),
        "tableOfContents": ("embedding_tableOfContents", "related_articles"),
        "pub_date": _HIJRI,
    },
    "references": {
        "OCR": ("embedding_OCR", *_LEMMAS, *_OCR_METRICS, *_LDA),
        "title": ("embedding_OCR",),
        "language": (*_LEMMAS, "Richesse_Lexicale_OCR", "Lisibilite_OCR", *_LDA),
    },
    "documents": {
        "OCR": (*_LEMMAS, *_OCR_METRICS),
        "language": (*_LEMMAS, "Richesse_Lexicale_OCR", "Lisibilite_OCR"),
        "pub_date": _HIJRI,
    },
    "audiovisual": {
        "OCR": _OCR_METRICS,
        "language": ("Richesse_Lexicale_OCR", "Lisibilite_OCR"),
        "pub_date": _HIJRI,
    },
    "images": {
        "image_url": ("embedding_image",),
        "thumbnail": ("embedding_image",),
        "pub_date": _HIJRI,
    },
    "index": {},
}

# Consensus depends on the annotation instrument's actual label inputs, even
# when OCR itself did not change. The frozen generation remains independent.
# Legacy generic columns have no trustworthy generation marker, so any changed
# model input invalidates the corresponding legacy dimension conservatively.
for _model in PANEL:
    _consensus = consensus_columns(_model.generation)
    for _suffix, _legacy in (
        ("polarite", "consensus_polarite"),
        ("centralite_islam_musulmans", "consensus_centralite"),
        ("subjectivite_score", "consensus_subjectivite_score"),
    ):
        DERIVED_FROM["articles"][_model.column(_suffix)] = (
            _consensus[_legacy], _consensus["sentiment_disagreement"],
            _consensus["instrument_id"], _legacy, "sentiment_disagreement",
        )

# Provenance cannot remain attached to an invalidated output. Keep these
# dependencies here so every upload adapter has the same invalidation rule.
for _sources in DERIVED_FROM.values():
    for _source, _outputs in list(_sources.items()):
        _sources[_source] = tuple(dict.fromkeys([
            *_outputs,
            *(f"{column}_{suffix}" for column in _outputs
              for suffix in ("input_hash", "config_hash", "config_json")),
        ]))

# Omeka value properties -> flat columns containing or exposing those values.
# Privacy is conservative for joined multilingual/multivalued properties: if
# any nonempty value is private (or lacks a flag), all mapped columns are
# marked private. The full mirror keeps all values; the public projector masks
# the named columns. These source mappings deliberately do not decide whether
# non-reconstructive computed derivatives are approved for publication.
SOURCE_FIELD_COLUMNS = {
    "dcterms:identifier": ("identifier",),
    "dcterms:title": ("title", "Titre"),
    "dcterms:creator": ("author", "author_ids", "creator", "creator_ids"),
    "dcterms:publisher": ("newspaper", "newspaper_ids", "publisher", "publisher_ids"),
    "dcterms:date": ("pub_date", "pub_year", "pub_date_precision", "date", *_HIJRI),
    "bibo:shortDescription": ("descriptionAI", "descriptionAI_en"),
    "dcterms:subject": ("subject", "subject_ids"),
    "dcterms:spatial": ("spatial", "spatial_ids"),
    "dcterms:language": ("language",),
    "bibo:numPages": ("nb_pages",),
    "fabio:hasURL": ("URL",),
    "dcterms:source": ("source",),
    "bibo:content": ("OCR",),
    "bibo:issue": ("issue",),
    "dcterms:tableOfContents": ("tableOfContents",),
    "dcterms:contributor": ("contributor",),
    "dcterms:type": ("type",),
    "dcterms:rights": ("rights",),
    "dcterms:description": ("description", "Description"),
    "bibo:volume": ("volume",),
    "dcterms:isPartOf": ("is_part_of", "Partie de"),
    "dcterms:extent": ("extent", "duration_seconds"),
    "dcterms:medium": ("medium",),
    "bibo:authorList": ("author", "author_ids"),
    "bibo:editorList": ("editor", "editor_ids"),
    "bibo:reviewOf": ("review_of",),
    "dcterms:alternative": ("book_title", "Titre alternatif"),
    "bibo:chapter": ("chapter",),
    "dcterms:abstract": ("abstract", "abstract_en"),
    "bibo:edition": ("edition",),
    "bibo:pageStart": ("page_start",),
    "bibo:pageEnd": ("page_end",),
    "dcterms:provenance": ("provenance",),
    "bibo:doi": ("doi", "URL"),
    "curation:coordinates": ("coordinates", "Coordonnées", "latitude", "longitude"),
    "dcterms:created": ("Date création",),
    "dcterms:relation": ("Relation",),
    "dcterms:isReplacedBy": ("Remplacé par",),
    "dcterms:hasPart": ("A une partie",),
    "foaf:firstName": ("Prénom",),
    "foaf:lastName": ("Nom",),
    "foaf:gender": ("Genre",),
    "foaf:birthday": ("Naissance",),
}

#: Country-specific Omeka item sets, per subset → the canonical country label
#: (the un-accented ``Benin``/``Nigeria`` that country_mapper emits). Format
#: collections (e.g. audiovisual 2183/2184) are absent on purpose: they group
#: by medium, not place, and must never resolve a country.
COUNTRY_ITEM_SETS: dict[str, dict[int, str]] = {
    "documents": {23452: "Benin", 23453: "Burkina Faso", 26327: "Togo"},
    "references": {
        2193: "Benin", 2212: "Burkina Faso", 2217: "Côte d'Ivoire",
        2222: "Niger", 2225: "Nigeria", 2228: "Togo",
    },
    "audiovisual": {
        2194: "Benin",           # Vidéos YouTube (Bénin)
        108260: "Burkina Faso",  # Vidéos YouTube (Burkina Faso)
    },
    "images": {
        2192: "Benin", 2211: "Burkina Faso", 2216: "Côte d'Ivoire",
        2220: "Niger", 2227: "Togo",
    },
}
CONTENT_COLUMNS: dict[str, list[str]] = {
    name: list(spec.content_columns)
    for name, spec in SUBSETS.items()
    if spec.content_columns
}


class DataContractError(ValueError):
    """A dataframe violates an invariant required for a safe Hub write."""


def _spec(config_name: str) -> SubsetDefinition:
    spec = SUBSETS.get(config_name)
    if spec is None:
        raise DataContractError(f"Unknown IWAC subset: {config_name!r}")
    return spec


def validate_ids(df: pd.DataFrame, *, label: str = "dataset") -> None:
    """Require a non-null, non-blank, unique canonical ``o:id`` column."""
    if "o:id" not in df.columns:
        raise DataContractError(f"{label} is missing the required 'o:id' column")
    if df["o:id"].isna().any():
        raise DataContractError(f"{label} contains null 'o:id' values")
    ids = df["o:id"].astype(str)
    if ids.str.strip().eq("").any():
        raise DataContractError(f"{label} contains blank 'o:id' values")
    duplicated = ids[ids.duplicated()].unique()
    if len(duplicated):
        sample = ", ".join(duplicated[:5])
        raise DataContractError(
            f"{label} contains duplicated 'o:id' values (e.g. {sample})"
        )


def _is_missing(value: object) -> bool:
    """True for a scalar null, including the ``NaN`` a left-merge leaves behind.

    A merge against the Hub fills an unmatched row's object column with
    ``float("nan")``, not ``None``: a subset that gains items on Omeka would
    otherwise trip the embedding contract on the first new row (and fail the
    Arrow conversion later) purely because it has nothing computed yet.
    ``pd.isna`` on a list returns an array, so guard on the scalar case.
    """
    if value is None:
        return True
    return isinstance(value, float) and pd.isna(value)


def normalize_embedding_nulls(df: pd.DataFrame, config_name: str) -> pd.DataFrame:
    """Replace merge-introduced ``NaN`` with ``None`` in embedding columns.

    Arrow cannot put a float into a list column, so the ``NaN`` a left-merge
    leaves on brand-new rows has to become a real null before the push.
    """
    for column in (_spec(config_name).embedding_columns or {}):
        if column not in df.columns:
            continue
        df[column] = [None if _is_missing(v) else v for v in df[column]]
    return df


def _check_vector(config_name: str, column: str, row_id, value, expected: int,
                  allow_empty: bool) -> None:
    """One embedding value must be a flat, finite, numeric vector of ``expected``
    length (or empty, when allowed). A 768-character string, a nested list of
    outer length 768, or a vector holding NaN/inf all used to pass a bare
    ``len()`` check."""
    import numpy as np

    if isinstance(value, (str, bytes)):
        raise DataContractError(
            f"{config_name}.{column} for o:id={row_id} is not a vector"
        )
    try:
        size = len(value)
    except TypeError as exc:
        raise DataContractError(
            f"{config_name}.{column} for o:id={row_id} is not a vector"
        ) from exc
    if size == 0 and allow_empty:
        return
    if size != expected:
        raise DataContractError(
            f"{config_name}.{column} for o:id={row_id} has dimension "
            f"{size}, expected {expected}"
        )
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataContractError(
            f"{config_name}.{column} for o:id={row_id} is not a numeric vector"
        ) from exc
    if array.ndim != 1:
        raise DataContractError(
            f"{config_name}.{column} for o:id={row_id} is not one-dimensional"
        )
    if not np.isfinite(array).all():
        raise DataContractError(
            f"{config_name}.{column} for o:id={row_id} contains NaN or infinity"
        )


def validate_embedding_dimensions(
    df: pd.DataFrame, config_name: str, *, allow_empty: bool = True
) -> None:
    """Validate non-empty embedding values against the subset contract."""
    for column, expected in (_spec(config_name).embedding_columns or {}).items():
        if column not in df.columns:
            continue
        for row_id, value in zip(df["o:id"], df[column]):
            if _is_missing(value):
                continue
            _check_vector(config_name, column, row_id, value, expected, allow_empty)


def validate_frame(df: pd.DataFrame, config_name: str) -> None:
    """Run the inexpensive contracts shared by every subset write."""
    _spec(config_name)
    validate_ids(df, label=f"'{config_name}' dataframe")
    validate_embedding_dimensions(df, config_name)


# ---------------------------------------------------------------------------
# Arrow-level contracts (datasets.Dataset)
# ---------------------------------------------------------------------------


def _is_arrow_dataset(ds) -> bool:
    try:
        from datasets import Dataset
    except ImportError:  # pragma: no cover - datasets is a hard dependency
        return False
    return isinstance(ds, Dataset)


def _arrow_column(ds, column: str):
    import pyarrow as pa

    values = ds.with_format("arrow")[column]
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    return values


def _validate_vector_column_arrow(ds, config_name: str, column: str, expected: int,
                                  ids: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc

    values = _arrow_column(ds, column)
    kind = values.type
    if pa.types.is_null(kind):
        return
    if not (pa.types.is_list(kind) or pa.types.is_large_list(kind)
            or pa.types.is_fixed_size_list(kind)):
        raise DataContractError(
            f"{config_name}.{column} is {kind}, not a list of numbers"
        )
    element = kind.value_type
    if not (pa.types.is_floating(element) or pa.types.is_integer(element)):
        raise DataContractError(
            f"{config_name}.{column} holds {element} elements, not numbers "
            "(nested or non-numeric vectors)"
        )
    lengths = pc.list_value_length(values).to_pylist()
    for row_id, size in zip(ids, lengths):
        if size is None or size == 0:
            continue
        if size != expected:
            raise DataContractError(
                f"{config_name}.{column} for o:id={row_id} has dimension "
                f"{size}, expected {expected}"
            )
    flat = pc.list_flatten(values)
    if flat.null_count:
        raise DataContractError(f"{config_name}.{column} contains null vector elements")
    if pa.types.is_floating(element) and len(flat):
        finite = pc.is_finite(flat)
        if not pc.all(finite).as_py():
            bad = pc.list_parent_indices(values).filter(pc.invert(finite))
            row_id = ids[bad[0].as_py()]
            raise DataContractError(
                f"{config_name}.{column} for o:id={row_id} contains NaN or infinity"
            )


def validate_dataset(ds, config_name: str) -> None:
    """Validate a ``datasets.Dataset`` without materializing all columns.

    Arrow-backed datasets are checked column-wise in Arrow (vectorized finite
    and dimension checks); any other object exposing ``column_names`` and
    ``__getitem__`` goes through the per-row path.
    """
    spec = _spec(config_name)
    if "o:id" not in ds.column_names:
        raise DataContractError(f"'{config_name}' dataset is missing 'o:id'")
    raw_ids = list(ds["o:id"])
    if any(value is None for value in raw_ids):
        raise DataContractError(f"'{config_name}' dataset contains null 'o:id' values")
    ids = [str(value) for value in raw_ids]
    if any(not value.strip() for value in ids):
        raise DataContractError(f"'{config_name}' dataset contains blank 'o:id' values")
    if len(ids) != len(set(ids)):
        raise DataContractError(f"'{config_name}' dataset contains duplicate 'o:id' values")
    arrow = _is_arrow_dataset(ds)
    for column, expected in (spec.embedding_columns or {}).items():
        if column not in ds.column_names:
            continue
        if arrow:
            _validate_vector_column_arrow(ds, config_name, column, expected, ids)
            continue
        for row_id, value in zip(ids, ds[column]):
            if _is_missing(value):
                continue
            _check_vector(config_name, column, row_id, value, expected, True)


def _to_int64(values, config_name: str, column: str):
    import pyarrow as pa
    import pyarrow.compute as pc

    kind = values.type
    if pa.types.is_int64(kind):
        return values
    if pa.types.is_null(kind) or pa.types.is_integer(kind):
        return values.cast(pa.int64())
    if pa.types.is_floating(kind):
        values = pc.if_else(pc.is_nan(values), pa.scalar(None, kind), values)
        fractional = pc.not_equal(values, pc.floor(values))
        if pc.any(fractional).as_py():
            sample = values.filter(fractional)[0].as_py()
            raise DataContractError(
                f"{config_name}.{column} is declared integer but holds a "
                f"fractional value ({sample}); refusing to truncate it"
            )
        return values.cast(pa.int64())
    raise DataContractError(
        f"{config_name}.{column} is declared integer but has Arrow type {kind}"
    )


def _to_embedding_list(values, config_name: str, column: str):
    import pyarrow as pa

    target = pa.list_(getattr(pa, EMBEDDING_VALUE_TYPE)())
    kind = values.type
    if kind == target:
        return values
    if pa.types.is_null(kind):
        return values.cast(target)
    if (pa.types.is_list(kind) or pa.types.is_large_list(kind)
            or pa.types.is_fixed_size_list(kind)):
        element = kind.value_type
        if pa.types.is_floating(element) or pa.types.is_integer(element):
            return values.cast(target)
    raise DataContractError(
        f"{config_name}.{column} has Arrow type {kind}; expected a list of numbers"
    )


def _to_metadata_type(values, config_name: str, column: str):
    """Stabilize privacy/provenance types even when all values are empty."""
    import pyarrow as pa

    target = (pa.bool_() if column == "item_is_public" else
              pa.list_(pa.string()) if column == "private_fields" else pa.string())
    if values.type == target:
        return values
    kind = values.type
    if column == "item_is_public":
        compatible = pa.types.is_null(kind) or pa.types.is_boolean(kind)
    elif column == "private_fields":
        compatible = pa.types.is_null(kind) or (
            (pa.types.is_list(kind) or pa.types.is_large_list(kind)
             or pa.types.is_fixed_size_list(kind))
            and (pa.types.is_string(kind.value_type) or pa.types.is_large_string(kind.value_type)
                 or pa.types.is_null(kind.value_type))
        )
    else:
        compatible = (pa.types.is_null(kind) or pa.types.is_string(kind)
                      or pa.types.is_large_string(kind))
    if not compatible:
        raise DataContractError(
            f"{config_name}.{column} has incompatible privacy/provenance type {kind}; "
            "refusing to coerce visibility evidence"
        )
    try:
        return values.cast(target)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
        raise DataContractError(
            f"{config_name}.{column} has incompatible privacy/provenance type {values.type}"
        ) from exc


def conform_dataset(ds, config_name: str):
    """Return ``ds`` with the subset's canonical column types applied.

    - declared integer columns → ``int64`` (nulls kept; ``NaN`` → null; a
      fractional value raises :class:`DataContractError`);
    - embedding columns → ``list<float32>``.

    Columns the subset does not declare are left untouched, as is any object
    that is not an Arrow-backed ``datasets.Dataset`` (test doubles). Returns
    the same object when nothing needed casting.
    """
    spec = _spec(config_name)
    if not _is_arrow_dataset(ds):
        return ds
    plan = [(c, _to_int64) for c in spec.int_columns if c in ds.column_names]
    plan += [
        (c, _to_embedding_list)
        for c in (spec.embedding_columns or {})
        if c in ds.column_names
    ]
    plan += [
        (c, _to_metadata_type) for c in ds.column_names
        if c in ("item_is_public", "private_fields")
        or c.endswith(("_input_hash", "_config_hash", "_config_json"))
    ]
    replacements = {}
    for column, convert in plan:
        values = _arrow_column(ds, column)
        converted = convert(values, config_name, column)
        if converted is not values:
            replacements[column] = converted
    if not replacements:
        return ds

    from datasets import Dataset

    table = ds.with_format("arrow")[:]
    for column, values in replacements.items():
        index = table.column_names.index(column)
        table = table.set_column(index, column, values)
    conformed = Dataset(table)
    revision = getattr(ds, "_iwac_source_revision", None)
    if revision is not None:
        conformed._iwac_source_revision = revision
    return conformed


def arrow_to_pandas(table) -> pd.DataFrame:
    """``pyarrow.Table`` → pandas, keeping nullable integers and booleans.

    The default conversion turns an ``int64`` column with a null into
    ``float64`` — the round trip that published ``lda_topic_id = 12.0``.
    """
    import pyarrow as pa

    mapping = {
        pa.int8(): pd.Int8Dtype(), pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(), pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(), pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(), pa.uint64(): pd.UInt64Dtype(),
        pa.bool_(): pd.BooleanDtype(),
    }
    return table.to_pandas(types_mapper=mapping.get)


def dataset_to_pandas(ds) -> pd.DataFrame:
    """``datasets.Dataset`` → pandas via :func:`arrow_to_pandas`."""
    if not _is_arrow_dataset(ds):
        return ds.to_pandas()
    return arrow_to_pandas(ds.with_format("arrow")[:])


__all__ = [
    "SubsetDefinition",
    "SUBSETS",
    "ALL_CONFIGS",
    "CONTENT_COLUMNS",
    "COUNTRY_ITEM_SETS",
    "DERIVED_FROM",
    "SOURCE_FIELD_COLUMNS",
    "EMBEDDING_VALUE_TYPE",
    "DataContractError",
    "validate_ids",
    "normalize_embedding_nulls",
    "validate_embedding_dimensions",
    "validate_frame",
    "validate_dataset",
    "conform_dataset",
    "arrow_to_pandas",
    "dataset_to_pandas",
]
