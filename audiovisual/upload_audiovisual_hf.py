#!/usr/bin/env python3
"""
upload_audiovisual_hf.py
========================

Extrait les documents audiovisuels (resource_class_id = 38) depuis l'API Omeka S
d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le Hugging Face
Hub comme subset 'audiovisual' du repository fmadore/islam-west-africa-collection.

Usage
-----
    python upload_audiovisual_hf.py \
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
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Add parent directory to path for iwac_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from huggingface_hub import login, get_token, utils as hf_utils
import huggingface_hub
from iwac_common.omeka_client import (
    Config,
    OmekaApiClient,
    conn_manager,
    fetch_iiif_thumbnail_url,
)
from iwac_common.field_mappers import (
    extract_added_date,
    get_media_ids,
    get_value,
)
from iwac_common.hub_merge import merge_with_hub_dataset, resolve_hf_token
from iwac_common.repos import PRIVATE_REPO_ID

# Disable symlinks warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------------------------
# Configuration & journalisation
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


# Config, Cache, ConnectionManager, async_retry and OmekaApiClient now live
# in iwac_common.omeka_client. The shared client uses a Rich progress bar
# while paginating; the audiovisual subset previously used tqdm here.


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

def _get_display_title(item: Dict[str, Any], field: str) -> str:
    """Extract display_title from a field."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        titles = []
        for v in val:
            if isinstance(v, dict) and "display_title" in v:
                titles.append(str(v["display_title"]))
        return "|".join(filter(None, titles))
    elif isinstance(val, dict) and "display_title" in val:
        return str(val["display_title"])
    return ""


def _get_at_value(item: Dict[str, Any], field: str) -> str:
    """Extract @value from a field."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        values = []
        for v in val:
            if isinstance(v, dict) and "@value" in v:
                values.append(str(v["@value"]))
        return "|".join(filter(None, values))
    elif isinstance(val, dict) and "@value" in val:
        return str(val["@value"])
    return ""


async def map_audiovisual_document(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka audiovisuel en dict plat pour HF datasets."""

    primary_url = ""
    if item.get("o:primary_media"):
        mid = item["o:primary_media"]["@id"].split("/")[-1]
        mdata = await api.fetch_media_data(mid)
        primary_url = mdata.get("o:original_url", "")

    # Extract publisher and determine country
    publisher = _get_display_title(item, "dcterms:publisher")
    country = "Nigeria"  # Fixed country for all audiovisual documents

    added_date = extract_added_date(item)

    # Fetch thumbnail URL and set IIIF manifest URL only if media exists
    session = await conn_manager.get()
    thumbnail_url = ""
    iiif_manifest_url = ""
    
    if primary_url:
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"

    return {
        "o:id": item["o:id"],
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": added_date,
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "iiif_manifest": iiif_manifest_url,
        "PDF": primary_url,  # Keeping as PDF for consistency, though it might be video/audio
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "creator": get_value(item, "dcterms:creator"),
        "publisher": publisher,
        "country": country,
        "pub_date": get_value(item, "dcterms:date"),
        "descriptionAI": get_value(item, "bibo:shortDescription"),
        "volume": _get_at_value(item, "bibo:volume"),
        "issue": _get_at_value(item, "bibo:issue"),
        "is_part_of": _get_at_value(item, "dcterms:isPartOf"),
        "extent": _get_at_value(item, "dcterms:extent"),
        "medium": _get_display_title(item, "dcterms:medium"),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "source": get_value(item, "dcterms:source"),
    }


# ---------------------------------------------------------------------------
# Pipeline principale : fetch → map → dataset → push
# ---------------------------------------------------------------------------

async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB"):
    api = OmekaApiClient(cfg, use_cache=True)

    # 1. Fetch current Omeka items and map them
    logger.info("Fetching audiovisual items from Omeka API...")
    omeka_items_raw = await api.fetch_items(38)  # Audiovisual documents (resource_class_id = 38)

    if not omeka_items_raw:
        logger.warning("No items returned from Omeka API. Exiting.")
        return

    logger.info(f"Fetched {len(omeka_items_raw)} items from Omeka.")
    omeka_records_list = []
    
    for it in tqdm(omeka_items_raw, desc="Mapping Omeka audiovisual documents"):
        try:
            record = await map_audiovisual_document(it, api)
            omeka_records_list.append(record)
        except Exception as e:
            logger.error(f"Error mapping item {it.get('o:id', 'Unknown ID')}: {e}", exc_info=True)
    
    if not omeka_records_list:
        logger.error("No records were successfully mapped. Exiting.")
        return
        
    new_omeka_df = pd.DataFrame(omeka_records_list)
    if 'o:id' not in new_omeka_df.columns or new_omeka_df['o:id'].isnull().any():
        logger.error("Critical: 'o:id' column is missing or contains null values in new Omeka data after mapping. Cannot proceed.")
        return
    new_omeka_df['o:id'] = new_omeka_df['o:id'].astype(str)

    # 2-3. Load existing Hub dataset and merge to preserve computed columns.
    token_to_use = resolve_hf_token()
    final_df = merge_with_hub_dataset(
        new_omeka_df,
        repo,
        config_name="audiovisual",
        token=token_to_use,
    )

    # 4. Conversion to Dataset and Push
    if not final_df.empty:
        logger.info(f"Preparing to push {len(final_df)} records to the Hub. Columns: {final_df.columns.tolist()}")
        
        if 'o:id' not in final_df.columns or final_df['o:id'].isnull().any():
            logger.error("Critical error: 'o:id' is missing or null in the final DataFrame before push. Aborting push.")
            await conn_manager.close()
            return

        ds = Dataset.from_pandas(final_df, preserve_index=False)
        logger.info("Dataset preview (first 5 rows):")
        logger.info(ds.to_pandas().head())

        if os.getenv("HF_TOKEN") is None and not hf_utils.is_notebook():
            login()
        
        try:
            logger.info(f"Pushing dataset to {repo} with config 'audiovisual'...")
            ds.push_to_hub(repo, max_shard_size=shard_size, config_name="audiovisual", token=token_to_use)
            logger.info(f"Dataset published/updated on {repo} with config 'audiovisual'")
        except Exception as e:
            logger.error(f"Failed to push dataset to Hub: {e}")
            logger.error("Details of the exception:", exc_info=True)

    else:
        logger.info("Final dataset is empty. No push operation will be performed.")

    await conn_manager.close()


# ---------------------------------------------------------------------------
# Exécution CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publie les documents audiovisuels IWAC sur le Hub HF")
    parser.add_argument("--repo", default=PRIVATE_REPO_ID, help="Repository Hugging Face où publier (défaut: miroir privé complet)")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille max d'un shard Parquet (ex. 500MB, 1GB)")
    args = parser.parse_args()

    asyncio.run(build_and_push(Config(CACHE_DIR=".cache_omk_audiovisual"), repo=args.repo, shard_size=args.max_shard_size))