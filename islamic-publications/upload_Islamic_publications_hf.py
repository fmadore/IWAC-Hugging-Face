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

import logging
import os
import sys
from typing import Dict, Any

# Adjust sys.path to include the parent directory for country_mapper / iwac_common imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from dotenv import load_dotenv
from country_mapper import get_country_from_newspaper
from iwac_common.omeka_client import (
    OmekaApiClient,
    conn_manager,
    fetch_iiif_thumbnail_url,
    fetch_primary_media_url,
)
from iwac_common.field_mappers import (
    extract_added_date,
    get_value,
    is_content_public,
    to_int_or_none,
)
from iwac_common.upload_runner import UploadSpec, run_upload
from iwac_common.schema import SUBSETS

logger = logging.getLogger(__name__)

# Specify the path to .env in the parent directory
dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)


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
        item, api, affected_fields=("PDF",)
    )


    publisher_name = get_value(item, "dcterms:publisher") # Changed newspaper_name to publisher_name for clarity
    country = get_country_from_newspaper(publisher_name) # Assumes country_mapper is generic enough

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
    # If none of the above, extracted_fabio_url remains ""

    added_date = extract_added_date(item)

    # Fetch thumbnail URL and set IIIF manifest URL only if PDF exists
    session = await conn_manager.get()
    thumbnail_url = ""
    iiif_manifest_url = ""

    if primary_url:  # Only fetch IIIF data if there's a PDF
        try:
            thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
            iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"
        except Exception as e:
            logger.error(f"Error fetching IIIF data for item {item['o:id']}: {e}")
            thumbnail_url = ""
            iiif_manifest_url = ""

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
        "newspaper": publisher_name, # This was 'newspaper', for publications might be 'journal' or 'publisher'
        "country": country,
        "pub_date": get_value(item, "dcterms:date"),
        "issue": get_value(item, "bibo:issue"), # Added issue field
        "tableOfContents": get_value(item, "dcterms:tableOfContents"),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "nb_pages": to_int_or_none(get_value(item, "bibo:numPages")),
        "URL": extracted_fabio_url,
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
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
