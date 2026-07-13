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

CLI options (shared, see iwac_common.upload_runner)
---------------------------------------------------
  --repo            Target Hugging Face repository (default: private full mirror)
  --max-shard-size  Maximum Parquet shard size (e.g. 500MB, 1GB)
  --no-cache        Bypass the local Omeka response cache (24h TTL)
  --dry-run         Fetch, map and merge, but push nothing
  --force-shrink    Allow pushing a dataset markedly smaller than the Hub's

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from iwac_common.omeka_client import (
    OmekaApiClient,
    conn_manager,
    fetch_iiif_thumbnail_url,
)
from iwac_common.field_mappers import (
    extract_added_date,
    get_value,
    is_content_public,
    to_int_or_none,
)
from iwac_common.upload_runner import UploadSpec, run_upload

load_dotenv()


# Orchestration (fetch → map loop → merge → validate → push), the CLI
# (--repo, --max-shard-size, --no-cache, --dry-run, --force-shrink) and the
# Rich console/logging setup live in iwac_common.upload_runner.


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

def _get_label(item: Dict[str, Any], field: str) -> str:
    """Extract o:label from a field that contains an array of objects with o:label."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        labels = []
        for v in val:
            if isinstance(v, dict) and "o:label" in v:
                labels.append(str(v["o:label"]))
        return "|".join(filter(None, labels))
    elif isinstance(val, dict) and "o:label" in val:
        return str(val["o:label"])
    return ""


async def map_document(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka en dict plat pour HF datasets."""

    primary_url = ""
    if item.get("o:primary_media"):
        mid = item["o:primary_media"]["@id"].split("/")[-1]
        mdata = await api.fetch_media_data(mid)
        primary_url = mdata.get("o:original_url", "")

    # Map country based on item set IDs
    country = ""
    if "o:item_set" in item and isinstance(item["o:item_set"], list):
        for item_set in item["o:item_set"]:
            if isinstance(item_set, dict) and "o:id" in item_set:
                item_set_id = item_set["o:id"]
                if item_set_id == 23452:
                    country = "Benin"
                    break
                elif item_set_id == 23453:
                    country = "Burkina Faso"
                    break
                elif item_set_id == 26327:
                    country = "Togo"
                    break

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
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "dcterms:creator"),
        "contributor": get_value(item, "dcterms:contributor"),
        "country": country,
        "pub_date": get_value(item, "dcterms:date"),
        "descriptionAI": get_value(item, "bibo:shortDescription"),
        "abstract": get_value(item, "dcterms:abstract"),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "type": get_value(item, "dcterms:type"),
        "nb_pages": nb_pages_int,
        "source": get_value(item, "dcterms:source"),
        "rights": _get_label(item, "dcterms:rights"),
        "OCR": get_value(item, "bibo:content"),
        "OCR_is_public": is_content_public(item),
    }


# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

SPEC = UploadSpec(
    config_name="documents",
    resource_class_ids=(49,),  # bibo:Document — general documents
    map_item=map_document,
    title="📄 IWAC Documents Upload",
    cache_dir=".cache_omk_documents",
    description="Upload IWAC documents to Hugging Face Hub",
    int_columns=("nb_pages",),
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
