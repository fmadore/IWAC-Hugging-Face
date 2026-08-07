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
        --repo fmadore/islam-west-africa-collection \
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
from typing import Dict, Any, Optional

# Add parent directory to path for country_mapper / iwac_common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
from country_mapper import get_country_from_newspaper
from iwac_common.omeka_client import OmekaApiClient, conn_manager, fetch_iiif_thumbnail_url
from iwac_common.field_mappers import (
    extract_added_date,
    get_value,
    get_value_by_language,
    is_content_public,
    to_int_or_none,
)
from iwac_common.sentiment_panel import (
    DIMENSION_FIELDS,
    LEGACY_VENDOR_COLUMNS,
    SUBJECTIVITE_ORDER,
    SUBJECTIVITE_SUFFIX,
    active_models,
    all_columns,
    numeric_subjectivite_columns,
    prefixes,
)
from iwac_common.upload_runner import UploadSpec, run_upload

load_dotenv()


# Orchestration (fetch → map loop → merge → validate → push), the CLI
# (--repo, --max-shard-size, --no-cache, --dry-run, --force-shrink) and the
# Rich console/logging setup live in iwac_common.upload_runner.


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


def _get_subjectivity_score(item: Dict[str, Any], field: str) -> Optional[int]:
    """Subjectivité as the 1-5 ordinal, for models asked for a number.

    Both generations store the value as a ``resource:item`` link to one of the
    five labelled items (78043-78047), so the label is read first and ranked; the
    ``int()`` fallback covers a numeric literal, which no live property uses.
    Unreachable while every active model is generation 2 — kept because it is the
    other half of ``SentimentModel.subjectivite_is_label`` and the reader for the
    frozen generation-1 columns' semantics.
    """
    label = _get_subjectivity_label(item, field)
    if label in SUBJECTIVITE_ORDER:
        return SUBJECTIVITE_ORDER[label]
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


async def map_newspaper_article(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka en dict plat pour HF datasets."""

    primary_url = ""
    if item.get("o:primary_media"):
        mid = item["o:primary_media"]["@id"].split("/")[-1]
        mdata = await api.fetch_media_data(mid)
        primary_url = mdata.get("o:original_url", "")

    newspaper_name = get_value(item, "dcterms:publisher")
    country = get_country_from_newspaper(newspaper_name)

    # Custom logic to extract URL from fabio:hasURL, prioritizing @id
    fabio_has_url_data = item.get("fabio:hasURL")
    extracted_fabio_url = ""
    if isinstance(fabio_has_url_data, list):
        urls = []
        for v_item in fabio_has_url_data:
            if isinstance(v_item, dict):
                id_val = v_item.get("@id")
                if id_val and isinstance(id_val, str): # Ensure it's a non-empty string
                    urls.append(id_val)
        if urls:
            extracted_fabio_url = "|".join(urls)
    elif isinstance(fabio_has_url_data, dict):
        id_val = fabio_has_url_data.get("@id")
        if id_val and isinstance(id_val, str): # Ensure it's a non-empty string
            extracted_fabio_url = id_val
    elif isinstance(fabio_has_url_data, str) and fabio_has_url_data: # If it's already a non-empty string
        extracted_fabio_url = fabio_has_url_data
    nb_pages_int = to_int_or_none(get_value(item, "bibo:numPages"))
    added_date = extract_added_date(item)

    # Fetch thumbnail URL and set IIIF manifest URL only if PDF exists
    session = await conn_manager.get()
    thumbnail_url = ""
    iiif_manifest_url = ""
    
    if primary_url:  # Only fetch IIIF data if there's a PDF
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"

    return {
        "o:id": item["o:id"],
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": added_date, # Date when item was added to Omeka
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "iiif_manifest": iiif_manifest_url,
        "PDF": primary_url,
        "thumbnail": thumbnail_url, # Added thumbnail field
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "dcterms:creator"),
        "newspaper": newspaper_name,
        "country": country, # Added country field
        "pub_date": get_value(item, "dcterms:date"),
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
        "spatial": get_value(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "nb_pages": nb_pages_int, # Use converted integer value
        "URL": extracted_fabio_url, # Use the specifically extracted URL
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
                # Same Omeka storage (a link to a labelled item) either way;
                # the generation decides whether the column keeps the label or
                # its 1-5 rank.
                cols[model.column(suffix)] = (
                    _get_subjectivity_label(item, prop)
                    if model.subjectivite_is_label
                    else _get_subjectivity_score(item, prop)
                )
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
    resource_class_ids=(36,),  # bibo:Article — newspaper articles only
    map_item=map_newspaper_article,
    title="📰 IWAC Newspaper Upload",
    cache_dir=".cache_omk",  # shared with islamic-publications (keys include class id)
    description="Publie les articles de journaux IWAC sur le Hub HF",
    int_columns=(
        "nb_pages",
        # Generation-1 subjectivité only: those columns hold the 1-5 integer.
        # Casting a generation-2 label column to Int64 would blank it entirely.
        *numeric_subjectivite_columns(),
    ),
    columns_to_exclude=LEGACY_VENDOR_COLUMNS,
    post_merge=_sentiment_columns_last,
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))


