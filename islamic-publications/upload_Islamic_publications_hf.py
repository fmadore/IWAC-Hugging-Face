#!/usr/bin/env python3
"""
upload_Islamic_publications_hf.py
=================================

Extracts Islamic publications (resource_class_id = 60) from the IWAC Omeka S API,
converts them to Arrow/Parquet dataset format, and pushes to Hugging Face Hub.

Usage
-----
    python islamic-publications/upload_Islamic_publications_hf.py \
        --max-shard-size 1GB

CLI options (shared, see iwac_common.upload_runner)
---------------------------------------------------
  --repo            Target Hugging Face repository (default: private full mirror)
  --max-shard-size  Maximum Parquet shard size (e.g. 500MB, 1GB)
  --no-cache        Bypass the local Omeka response cache (24h TTL)
  --dry-run         Fetch, map and merge, but push nothing
  --force-shrink    Allow pushing a dataset markedly smaller than the Hub's

Environment Variables
--------------------
  OMEKA_BASE_URL        Base URL of the API, e.g., https://islam.zmo.de/api
  OMEKA_KEY_IDENTITY    Omeka API key identity
  OMEKA_KEY_CREDENTIAL  Omeka API key credential
  HF_TOKEN              Hugging Face personal access token (optional if
                        calling login() interactively)
"""

import os
import sys
from typing import Dict, Any

# Adjust sys.path to include the parent directory for country_mapper / iwac_common imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from country_mapper import get_country_from_newspaper
from iwac_common.omeka_client import (
    OmekaApiClient,
    conn_manager,
    fetch_iiif_thumbnail_url,
    fetch_primary_media_url,
    iiif_manifest_url,
    item_page_url,
)
from iwac_common.field_mappers import (
    visibility_metadata,
    extract_added_date,
    get_resource_ids,
    get_uri_value,
    get_value,
    is_content_public,
    parse_pub_date,
    to_int_or_none,
)
from iwac_common.upload_runner import UploadSpec, report_unmapped_values, run_upload
from iwac_common.schema import SUBSETS



# Orchestration (fetch → map loop → merge → validate → push), the CLI
# (--repo, --max-shard-size, --no-cache, --dry-run, --force-shrink) and the
# Rich console/logging setup live in iwac_common.upload_runner.
# islamic-publications and articles share the default ``.cache_omk`` cache
# directory (cache keys include the resource class id).


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

async def map_islamic_publication_item(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka (publication islamique) en dict plat pour HF datasets."""

    primary_url = await fetch_primary_media_url(
        item, api, affected_fields=("PDF", "thumbnail", "iiif_manifest")
    )


    publisher_name = get_value(item, "dcterms:publisher")
    country = get_country_from_newspaper(publisher_name)  # same outlet index as articles
    pub_date = get_value(item, "dcterms:date")
    pub_year, pub_date_precision = parse_pub_date(pub_date)

    # Thumbnail and IIIF manifest only when there is a PDF behind them.
    thumbnail_url = ""
    manifest_url = ""
    if primary_url:
        session = await conn_manager.get()
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        manifest_url = iiif_manifest_url(item["o:id"])

    return {
        "o:id": item["o:id"],
        **visibility_metadata(item, "publications"),
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": extract_added_date(item),  # Date when item was added to Omeka
        "iwac_url": item_page_url(item["o:id"]),
        "iiif_manifest": manifest_url,
        "PDF": primary_url,
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "dcterms:creator"),
        "author_ids": get_resource_ids(item, "dcterms:creator"),
        # Historical name shared with articles: the periodical's publisher.
        "newspaper": publisher_name,
        "newspaper_ids": get_resource_ids(item, "dcterms:publisher"),
        "country": country,
        "pub_date": pub_date,
        "pub_year": pub_year,
        "pub_date_precision": pub_date_precision,
        "issue": get_value(item, "bibo:issue"),
        "tableOfContents": get_value(item, "dcterms:tableOfContents"),
        "subject": get_value(item, "dcterms:subject"),
        "subject_ids": get_resource_ids(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "spatial_ids": get_resource_ids(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "nb_pages": to_int_or_none(get_value(item, "bibo:numPages")),
        "URL": get_uri_value(item, "fabio:hasURL"),
        "source": get_value(item, "dcterms:source"),
        "OCR": get_value(item, "bibo:content"),
        "OCR_is_public": is_content_public(item),
    }


# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

SPEC = UploadSpec(
    config_name="publications",
    resource_class_ids=SUBSETS["publications"].resource_class_ids,
    map_item=map_islamic_publication_item,
    title="📚 IWAC Islamic Publications Upload",
    cache_dir=".cache_omk",  # intentionally shared with articles (cache keys include class id)
    description="Upload IWAC Islamic Publications to Hugging Face Hub",
    int_columns=("nb_pages",),
    post_map=report_unmapped_values("newspaper", "country"),
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
