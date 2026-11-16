#!/usr/bin/env python3
"""
upload_newspaper_hf.py
=====================

Extrait les articles de journaux (resource_class_id = 36) depuis l'API Omeka S
d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le Hugging Face
Hub.

Usage
-----
    python articles/upload_newspaper_hf.py \
        --repo fmadore/islam-west-africa-collection-full \
        --max-shard-size 1GB

Variables d'environnement
------------------------
  OMEKA_BASE_URL        Base URL de l'API, ex. https://islam.zmo.de/api
  OMEKA_KEY_IDENTITY    Identité de la clé Omeka
  OMEKA_KEY_CREDENTIAL  Credential de la clé Omeka
  HF_TOKEN              Jeton d'accès personnel Hugging Face (facultatif si
                        vous appelez login() de manière interactive)
"""

import os
import sys
from typing import Dict, Any

# Add parent directory to path for country_mapper / iwac_common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from country_mapper import get_country_from_outlets
from iwac_common.omeka_client import (
    OmekaApiClient,
    fetch_media_pointers,
    item_page_url,
)
from iwac_common.field_mappers import (
    visibility_metadata,
    extract_added_date,
    get_resource_ids,
    get_uri_value,
    get_value,
    get_value_by_language,
    is_content_public,
    parse_pub_date,
    to_int_or_none,
)
from iwac_common.sentiment_panel import (
    DIMENSION_FIELDS,
    LEGACY_VENDOR_COLUMNS,
    SUBJECTIVITE_SUFFIX,
    active_models,
    all_columns,
    prefixes,
)
from iwac_common.upload_runner import (
    COUNTRY_MAPPER_HINT,
    UploadSpec,
    report_unmapped_values,
    run_upload,
)



# Orchestration (fetch → map loop → merge → validate → push), the CLI and the
# Rich console/logging setup live in iwac_common.upload_runner. For the flags,
# run this script with --help (parser: iwac_common.upload_runner.build_parser).


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------


def _get_subjectivity_label(item: Dict[str, Any], field: str) -> str:
    """Subjectivité as the label Omeka stores, or ``""`` when unscored.

    Generation 2 asks the model for a label rather than the integer 1-5, so this
    is the value as given. ``""`` (not None) for the unscored rows, matching the
    convention every other string column in this subset follows — a model that
    called an article ``Non abordé`` legitimately declines to score it, and so
    does one that simply omitted the field.
    """
    return get_value(item, field).strip()


async def map_newspaper_article(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka en dict plat pour HF datasets."""

    # Thumbnail and IIIF manifest only when there is a PDF behind them.
    primary_url, thumbnail_url, manifest_url = await fetch_media_pointers(
        item, api, file_field="PDF"
    )

    newspaper_name = get_value(item, "dcterms:publisher")
    country = get_country_from_outlets(newspaper_name)
    pub_date = get_value(item, "dcterms:date")
    pub_year, pub_date_precision = parse_pub_date(pub_date)

    return {
        "o:id": item["o:id"],
        **visibility_metadata(item, "articles"),
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": extract_added_date(item),  # Date when item was added to Omeka
        "iwac_url": item_page_url(item["o:id"]),
        "iiif_manifest": manifest_url,
        "PDF": primary_url,
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "dcterms:creator"),
        "author_ids": get_resource_ids(item, "dcterms:creator"),
        "newspaper": newspaper_name,
        "newspaper_ids": get_resource_ids(item, "dcterms:publisher"),
        "country": country,
        "pub_date": pub_date,
        "pub_year": pub_year,
        "pub_date_precision": pub_date_precision,
        # One column per language, NOT get_value(): the summariser writes an
        # `fr` and an `en` literal on the same property, and pipe-joining them
        # yields a string splittable only on a delimiter the prose may contain,
        # in an order Omeka does not guarantee. Untagged legacy values are
        # French. See iwac_common/field_mappers.get_value_by_language.
        "descriptionAI": get_value_by_language(
            item, "bibo:shortDescription", "fr", untagged_matches=True
        ),
        "descriptionAI_en": get_value_by_language(
            item, "bibo:shortDescription", "en"
        ),
        "subject": get_value(item, "dcterms:subject"),
        "subject_ids": get_resource_ids(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "spatial_ids": get_resource_ids(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "nb_pages": to_int_or_none(get_value(item, "bibo:numPages")),
        "URL": get_uri_value(item, "fabio:hasURL"),
        "source": get_value(item, "dcterms:source"),
        "OCR": get_value(item, "bibo:content"),
        # Whether the full text is publicly visible on Omeka; drives
        # per-row OCR masking in publish_public.py.
        "OCR_is_public": is_content_public(item),
        **_sentiment_columns(item),
    }


def _sentiment_columns(item: Dict[str, Any]) -> Dict[str, Any]:
    """Read the panel's sentiment values off an Omeka item.

    Only ``active_models()`` are read: a frozen model's Omeka properties no
    longer exist, and its Hub columns survive because ``hub_merge`` preserves
    columns this mapper does not produce. That is the whole mechanism keeping
    generation 1 on the Hub after its deletion from the archive — so this loop
    growing quiet about a model is intentional, not a regression.
    """
    cols: Dict[str, Any] = {}
    for model in active_models():
        for suffix, omeka_suffix in DIMENSION_FIELDS:
            prop = model.omeka_property(omeka_suffix)
            if suffix == SUBJECTIVITE_SUFFIX:
                # Generation 1 stored the 1-5 rank in an int column; its Omeka
                # properties are retired (frozen models raise in
                # omeka_property), so no live model may emit that shape.
                if not model.subjectivite_is_label:
                    raise ValueError(
                        f"{model.prefix} is active but asks for a numeric "
                        "subjectivité score; only label-based (generation >= 2) "
                        "models can be ingested"
                    )
                cols[model.column(suffix)] = _get_subjectivity_label(item, prop)
            else:
                cols[model.column(suffix)] = get_value(item, prop)
    return cols


# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

def _sentiment_columns_last(final_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the AI sentiment block as the trailing columns, in panel order.

    Uses the full panel, not just the active models, so frozen columns stay
    grouped with the live ones instead of drifting into the metadata block.
    Ordering follows ``PANEL`` rather than whatever order the merge happened to
    produce, which puts the **live** generation first and the frozen one after
    it: a reader meets the current panel before its history, and the order stays
    stable across runs instead of tracking which columns the merge preserved.
    """
    sentiment_prefixes = tuple(f"{p}_" for p in prefixes())
    present = set(final_df.columns)
    sentiment_cols = [c for c in all_columns() if c in present]
    # Anything sentiment-prefixed but off-panel (a member retired mid-rotation)
    # trails the block rather than vanishing from the frame.
    sentiment_cols += [
        c for c in final_df.columns
        if c.startswith(sentiment_prefixes) and c not in sentiment_cols
    ]
    other_cols = [c for c in final_df.columns if c not in set(sentiment_cols)]
    return final_df[other_cols + sentiment_cols]


SPEC = UploadSpec(
    config_name="articles",
    map_item=map_newspaper_article,
    title="📰 IWAC Newspaper Upload",
    cache_dir=".cache_omk",  # shared with islamic-publications (keys include class id)
    description="Publie les articles de journaux IWAC sur le Hub HF",
    columns_to_exclude=LEGACY_VENDOR_COLUMNS,
    post_map=report_unmapped_values("newspaper", "country", hint=COUNTRY_MAPPER_HINT),
    post_merge=_sentiment_columns_last,
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))


