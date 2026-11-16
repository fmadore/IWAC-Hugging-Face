#!/usr/bin/env python3
"""
upload_documents_hf.py
======================

Extracts documents (resource_class_id = 49) from the IWAC Omeka S API, converts
them to Arrow/Parquet dataset, and pushes to Hugging Face Hub as the 'documents'
subset of fmadore/islam-west-africa-collection-full.

Usage
-----
    python document/upload_documents_hf.py --max-shard-size 1GB

CLI options
-----------
Shared by every upload script: run with ``--help`` for the full list
(parser: ``iwac_common.upload_runner.build_parser``).

Environment Variables
--------------------
  OMEKA_BASE_URL        API base URL, e.g., https://islam.zmo.de/api
  OMEKA_KEY_IDENTITY    Omeka key identity
  OMEKA_KEY_CREDENTIAL  Omeka key credential
  HF_TOKEN              Hugging Face access token (optional if using interactive login)
"""

import os
import sys
from typing import Dict, Any

# Add parent directory to path to import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iwac_common.omeka_client import (
    OmekaApiClient,
    fetch_media_pointers,
    item_page_url,
)
from iwac_common.field_mappers import (
    visibility_metadata,
    countries_from_item_sets,
    extract_added_date,
    get_resource_ids,
    get_rights_label,
    get_value,
    get_value_by_language,
    is_content_public,
    parse_pub_date,
    to_int_or_none,
)
from iwac_common.upload_runner import UploadSpec, run_upload
from iwac_common.schema import COUNTRY_ITEM_SETS



# Orchestration (fetch → map loop → merge → validate → push), the CLI (see
# --help) and the Rich console/logging setup live in iwac_common.upload_runner.


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

async def map_document(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka en dict plat pour HF datasets."""

    # Thumbnail and IIIF manifest only when there is a PDF behind them.
    primary_url, thumbnail_url, manifest_url = await fetch_media_pointers(
        item, api, file_field="PDF"
    )
    pub_date = get_value(item, "dcterms:date")
    pub_year, pub_date_precision = parse_pub_date(pub_date)

    return {
        "o:id": item["o:id"],
        **visibility_metadata(item, "documents"),
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": extract_added_date(item),  # Date when item was added to Omeka
        "iwac_url": item_page_url(item["o:id"]),
        "iiif_manifest": manifest_url,
        "PDF": primary_url,
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "dcterms:creator"),
        "author_ids": get_resource_ids(item, "dcterms:creator"),
        # get_value (literals included), unlike audiovisual's linked-only read.
        "contributor": get_value(item, "dcterms:contributor"),
        # Country from the item's country-specific collection.
        "country": countries_from_item_sets(item, COUNTRY_ITEM_SETS["documents"]),
        "pub_date": pub_date,
        "pub_year": pub_year,
        "pub_date_precision": pub_date_precision,
        # Like articles: documents have no table of contents, and their
        # abstract is the AI-generated one in bibo:shortDescription. The
        # dcterms:abstract field is being retired on the Omeka side.
        # One column per language — see the note in upload_newspaper_hf.py.
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
        # get_value keeps literal and URI values too; images/audiovisual's
        # linked-only get_display_titles would drop them, changing values.
        "type": get_value(item, "dcterms:type"),
        "nb_pages": to_int_or_none(get_value(item, "bibo:numPages")),
        "source": get_value(item, "dcterms:source"),
        # Label, falling back to the statement URI (same rule as audiovisual
        # and images; the local helper this replaced returned "" instead).
        "rights": get_rights_label(item),
        "OCR": get_value(item, "bibo:content"),
        "OCR_is_public": is_content_public(item),
    }


# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

SPEC = UploadSpec(
    config_name="documents",
    map_item=map_document,
    title="📄 IWAC Documents Upload",
    cache_dir=".cache_omk_documents",
    description="Upload IWAC documents to Hugging Face Hub",
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
