#!/usr/bin/env python3
"""
upload_image_hf.py
==================

Extrait les photographies (resource_class_id = 58, ``bibo:Image``) depuis l'API
Omeka S d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le
Hugging Face Hub comme subset ``images`` du miroir privé complet
``fmadore/islam-west-africa-collection-full``.

Les photographies sont des clichés de terrain (mosquées, radios islamiques, …)
pris par le curateur dans les cinq pays du corpus. Elles ne portent quasiment
pas de texte libre (2 descriptions sur 30) ; leur contenu visuel est capté en
aval par ``embedding_image`` (voir ``post-processing/semantic_embedding_images.py``).

L'image elle-même n'est PAS stockée dans le dataset : on garde seulement un
pointeur d'URL (``image_url``), conformément à la convention « pas de média
binaire dans HF ».

Usage
-----
    python images/upload_image_hf.py \
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
from typing import Dict, Any

# Add parent directory to path for iwac_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import login, utils as hf_utils
from iwac_common.omeka_client import (
    Config,
    OmekaApiClient,
    conn_manager,
    fetch_iiif_thumbnail_url,
)
from iwac_common.field_mappers import (
    extract_added_date,
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

# Photographs (bibo:Image). The "Photograph" resource template's default class
# (33) is unused; every photograph item is filed under class 58.
IMAGE_RESOURCE_CLASS_ID = 58

# Each photograph belongs to exactly one country-specific item set. Map the
# item-set id to the canonical country label used across the dataset (matches
# country_mapper.py, incl. the un-accented raw ``Benin``).
ITEM_SET_COUNTRY = {
    2192: "Benin",           # Photographies (Bénin)
    2211: "Burkina Faso",    # Photographies (Burkina Faso)
    2216: "Côte d'Ivoire",   # Photographies (Côte d'Ivoire)
    2220: "Niger",           # Photographies (Niger)
    2227: "Togo",            # Photographies (Togo)
}


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

def _get_display_title(item: Dict[str, Any], field: str) -> str:
    """Extract display_title from a field (pipe-joined for lists)."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        titles = [str(v["display_title"]) for v in val
                  if isinstance(v, dict) and "display_title" in v]
        return "|".join(filter(None, titles))
    if isinstance(val, dict) and "display_title" in val:
        return str(val["display_title"])
    return ""


def _get_at_value(item: Dict[str, Any], field: str) -> str:
    """Extract @value from a field (pipe-joined for lists)."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        values = [str(v["@value"]) for v in val
                  if isinstance(v, dict) and "@value" in v]
        return "|".join(filter(None, values))
    if isinstance(val, dict) and "@value" in val:
        return str(val["@value"])
    return ""


def _get_rights_label(item: Dict[str, Any], field: str = "dcterms:rights") -> str:
    """Rights statements carry a human ``o:label`` alongside the ``@id`` URI;
    prefer the label, fall back to the URI, then to ``@value``."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, dict):
        val = [val]
    if not isinstance(val, list):
        return ""
    parts = [str(v.get("o:label") or v.get("@id") or v.get("@value") or "")
             for v in val if isinstance(v, dict)]
    return "|".join(filter(None, parts))


def _get_country_from_item_sets(item: Dict[str, Any]) -> str:
    """Map the item's country-specific photo collection to a country label."""
    for s in item.get("o:item_set", []) or []:
        country = ITEM_SET_COUNTRY.get(s.get("o:id"))
        if country:
            return country
    return ""


async def map_image_item(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka photographie (bibo:Image) en dict plat pour HF."""

    # Original image URL from the primary media (the actual JPEG).
    image_url = ""
    if item.get("o:primary_media"):
        mid = item["o:primary_media"]["@id"].split("/")[-1]
        mdata = await api.fetch_media_data(mid)
        image_url = mdata.get("o:original_url", "")

    added_date = extract_added_date(item)

    # IIIF thumbnail + manifest (only meaningful when media exists). Fall back
    # to the item's baked-in ``large`` derivative if the IIIF manifest is
    # unavailable.
    session = await conn_manager.get()
    thumbnail_url = ""
    iiif_manifest_url = ""
    if image_url:
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        if not thumbnail_url:
            thumbnail_url = (item.get("thumbnail_display_urls") or {}).get("large", "")
        iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"

    return {
        "o:id": item["o:id"],
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": added_date,
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "iiif_manifest": iiif_manifest_url,
        "image_url": image_url,
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "type": _get_display_title(item, "dcterms:type"),
        "creator": get_value(item, "dcterms:creator"),
        "pub_date": get_value(item, "dcterms:date"),
        "description": get_value(item, "dcterms:description"),
        "rights": _get_rights_label(item),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "coordinates": _get_at_value(item, "curation:coordinates"),
        "country": _get_country_from_item_sets(item),
    }


# ---------------------------------------------------------------------------
# Pipeline principale : fetch → map → dataset → push
# ---------------------------------------------------------------------------

async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB"):
    api = OmekaApiClient(cfg, use_cache=True)

    # 1. Fetch current Omeka items and map them
    logger.info("Fetching photograph items from Omeka API...")
    omeka_items_raw = await api.fetch_items(IMAGE_RESOURCE_CLASS_ID)

    if not omeka_items_raw:
        logger.warning("No items returned from Omeka API. Exiting.")
        return

    logger.info(f"Fetched {len(omeka_items_raw)} items from Omeka.")
    omeka_records_list = []

    for it in tqdm(omeka_items_raw, desc="Mapping Omeka photographs"):
        try:
            record = await map_image_item(it, api)
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

    # 2-3. Load existing Hub dataset and merge to preserve computed columns
    #      (notably embedding_image from semantic_embedding_images.py).
    token_to_use = resolve_hf_token()
    final_df = merge_with_hub_dataset(
        new_omeka_df,
        repo,
        config_name="images",
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
            logger.info(f"Pushing dataset to {repo} with config 'images'...")
            ds.push_to_hub(repo, max_shard_size=shard_size, config_name="images", token=token_to_use)
            logger.info(f"Dataset published/updated on {repo} with config 'images'")
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

    parser = argparse.ArgumentParser(description="Publie les photographies IWAC sur le Hub HF")
    parser.add_argument("--repo", default=PRIVATE_REPO_ID, help="Repository Hugging Face où publier (défaut: miroir privé complet)")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille max d'un shard Parquet (ex. 500MB, 1GB)")
    args = parser.parse_args()

    asyncio.run(build_and_push(Config(CACHE_DIR=".cache_omk_images"), repo=args.repo, shard_size=args.max_shard_size))
