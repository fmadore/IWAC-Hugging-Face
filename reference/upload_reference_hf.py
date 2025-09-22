#!/usr/bin/env python3
"""
upload_reference_hf.py
=====================

Extrait les références bibliographiques (resource_class_id = [35, 43, 88, 40, 82, 178, 52, 77, 305]) 
depuis l'API Omeka S d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le Hugging Face
Hub comme subset 'references' du repository fmadore/islam-west-africa-collection.

Usage
-----
    python upload_reference_hf.py \
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
import json
import io
import gzip
import hashlib
import asyncio
import logging
import argparse
import pandas as pd
import aiohttp
import aiofiles
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Type

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from huggingface_hub import login, HfFolder, utils as hf_utils

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

@dataclass
class Config:
    """Paramètres globaux chargés depuis .env ou variables d'environnement"""

    API_URL: str = os.getenv("OMEKA_BASE_URL", "https://islam.zmo.de/api")
    API_KEY_IDENTITY: str = os.getenv("OMEKA_KEY_IDENTITY", "")
    API_KEY_CREDENTIAL: str = os.getenv("OMEKA_KEY_CREDENTIAL", "")
    CACHE_DIR: str = ".cache_omk_references"
    CACHE_HOURS: int = 24


# Reference resource classes
RESOURCE_CLASSES = [35, 43, 88, 40, 82, 178, 52, 77, 305]

# Resource class mapping
RESOURCE_CLASS_MAPPING = {
    35: 'Article de revue',
    43: 'Chapitre',
    88: 'Thèse',
    40: 'Livre',
    82: 'Rapport',
    178: 'Compte rendu',
    52: 'Ouvrage collectif',
    77: 'Communication',
    305: 'Article de blog'
}

# Country mapping based on item sets
COUNTRY_ITEM_SETS = {
    2193: 'Bénin',
    2212: 'Burkina Faso',
    2217: 'Côte d\'Ivoire',
    2222: 'Niger',
    2225: 'Nigeria',
    2228: 'Togo'
}


# ---------------------------------------------------------------------------
# Cache disque (JSON Gzip) pour économiser l'API
# ---------------------------------------------------------------------------

class Cache:
    def __init__(self, directory: str, hours: int = 24):
        self.dir = directory
        self.duration = timedelta(hours=hours)
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        name = hashlib.md5(key.encode()).hexdigest() + ".json.gz"
        return os.path.join(self.dir, name)

    async def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        if datetime.now() - mtime > self.duration:
            return None
        async with aiofiles.open(path, "rb") as f:
            data = await f.read()
        with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as gz:
            return json.load(gz)

    async def set(self, key: str, value: Any):
        path = self._path(key)
        buf = io.BytesIO()
        with gzip.open(buf, "wt", encoding="utf-8") as gz:
            json.dump(value, gz)
        async with aiofiles.open(path, "wb") as f:
            await f.write(buf.getvalue())


# ---------------------------------------------------------------------------
# Gestion de la connexion HTTP
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=20, ssl=False),
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


conn_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Decorateur retry asynchrone simple
# ---------------------------------------------------------------------------

def async_retry(max_tries: int = 5, exceptions: Union[Type[Exception], tuple] = (aiohttp.ClientError, asyncio.TimeoutError)):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_tries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_tries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
            raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Client API Omeka minimum viable
# ---------------------------------------------------------------------------

class OmekaApiClient:
    def __init__(self, cfg: Config, use_cache: bool = True):
        self.cfg = cfg
        self.cache = Cache(cfg.CACHE_DIR, cfg.CACHE_HOURS) if use_cache else None

    @async_retry()
    async def _get(self, endpoint: str, params: Dict[str, Any]) -> Any:
        params.update(
            {
                "key_identity": self.cfg.API_KEY_IDENTITY,
                "key_credential": self.cfg.API_KEY_CREDENTIAL,
            }
        )
        url = f"{self.cfg.API_URL}/{endpoint}"
        sess = await conn_manager.get()
        async with sess.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def request(self, endpoint: str, params: Dict[str, Any]):
        key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        if self.cache:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached
        data = await self._get(endpoint, params)
        if self.cache:
            await self.cache.set(key, data)
        return data

    async def fetch_items_page(self, rcid: int, page: int, per: int = 100):
        return await self.request("items", {"resource_class_id": rcid, "page": page, "per_page": per})

    async def fetch_items(self, rcid: int) -> List[Dict[str, Any]]:
        first = await self.fetch_items_page(rcid, 1)
        items = list(first)
        per = 100
        if len(first) == per:
            # Estimate total pages if possible, or use an indefinite progress bar
            # This part requires knowing the total number of items or pages,
            # which might not be available directly from the first call.
            # For now, let's assume we don't know the total and use a simple counter.
            page = 2
            with tqdm(desc=f"Fetching items pages for class {rcid}", unit="page") as pbar:
                while True:
                    batch = await self.fetch_items_page(rcid, page)
                    if not batch:
                        break
                    items.extend(batch)
                    pbar.update(1)
                    if len(batch) < per:
                        break
                    page += 1
        logger.info("%d items récupérés pour la classe %d", len(items), rcid)
        return items

    async def fetch_all_reference_items(self) -> List[Dict[str, Any]]:
        """Fetch all items from all reference resource classes"""
        all_items = []
        for rcid in RESOURCE_CLASSES:
            try:
                items = await self.fetch_items(rcid)
                all_items.extend(items)
            except Exception as e:
                logger.error(f"Error fetching items for resource class {rcid}: {e}")
                continue
        return all_items

    async def fetch_media_data(self, media_id: str):
        return await self.request(f"media/{media_id}", {})


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


def _get_value(item: Dict[str, Any], field: str) -> str:
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        parts = [str(v.get("display_title") or v.get("@value") or v.get("@id", "")) for v in val]
        return "|".join(filter(None, parts))
    if isinstance(val, dict):
        return val.get("display_title", "") or val.get("@value", "")
    return str(val)


def _get_iwac_identifier(item: Dict[str, Any], field: str) -> str:
    """Extract identifier values that start with 'iwac-reference'"""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        for v in val:
            identifier = str(v.get("display_title") or v.get("@value") or v.get("@id", ""))
            if identifier.startswith("iwac-reference"):
                return identifier
    elif isinstance(val, dict):
        identifier = val.get("display_title", "") or val.get("@value", "")
        if identifier.startswith("iwac-reference"):
            return identifier
    else:
        identifier = str(val)
        if identifier.startswith("iwac-reference"):
            return identifier
    return ""


def _join(item: Dict[str, Any], field: str) -> str:
    return _get_value(item, field)


def _get_media_ids(item: Dict[str, Any]) -> str:
    if "o:media" in item and isinstance(item["o:media"], list):
        return "|".join(str(m["o:id"]) for m in item["o:media"])
    return ""


def _get_resource_class(item: Dict[str, Any]) -> str:
    """Extract resource class information and map to human-readable name"""
    if "o:resource_class" in item and isinstance(item["o:resource_class"], dict):
        class_id = item["o:resource_class"].get("o:id")
        if class_id and class_id in RESOURCE_CLASS_MAPPING:
            return RESOURCE_CLASS_MAPPING[class_id]
        elif class_id:
            return str(class_id)  # Return ID as string if not in mapping
    return ""


def _get_item_set_ids(item: Dict[str, Any]) -> str:
    """Extract item set IDs"""
    if "o:item_set" in item and isinstance(item["o:item_set"], list):
        return "|".join(str(item_set["o:id"]) for item_set in item["o:item_set"] if isinstance(item_set, dict) and "o:id" in item_set)
    return ""


def _get_countries_from_item_sets(item: Dict[str, Any]) -> str:
    """Map country based on item set IDs"""
    countries = []
    if "o:item_set" in item and isinstance(item["o:item_set"], list):
        for item_set in item["o:item_set"]:
            if isinstance(item_set, dict) and "o:id" in item_set:
                item_set_id = item_set["o:id"]
                if item_set_id in COUNTRY_ITEM_SETS:
                    countries.append(COUNTRY_ITEM_SETS[item_set_id])
    return "|".join(countries) if countries else ""


@async_retry(max_tries=3, exceptions=(aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError))
async def fetch_iiif_thumbnail_url(omeka_id: Union[str, int], session: aiohttp.ClientSession) -> str:
    """Fetches and extracts the thumbnail URL from an IIIF manifest."""
    manifest_url = f"https://islam.zmo.de/iiif/3/{omeka_id}/manifest"
    thumbnail_url = ""
    try:
        # Use a shorter timeout for this specific, potentially numerous, request type
        async with session.get(manifest_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                manifest = await resp.json()
                if "thumbnail" in manifest and isinstance(manifest["thumbnail"], list) and len(manifest["thumbnail"]) > 0:
                    thumbnail_url = manifest["thumbnail"][0].get("id", "")
    except asyncio.TimeoutError: # Specifically catch timeout
        logger.warning(f"Timeout fetching IIIF manifest for {omeka_id}. URL: {manifest_url}")
    except aiohttp.ClientError as e_client: # Specifically catch client errors
        logger.warning(f"Client error fetching IIIF manifest for {omeka_id}: {e_client}. URL: {manifest_url}")
    # Catching general Exception for unexpected issues, though specific ones are better
    except Exception as e_general:
        logger.error(f"Unexpected error fetching IIIF manifest for {omeka_id}: {e_general}. URL: {manifest_url}")
    return thumbnail_url


async def map_reference(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka de référence en dict plat pour HF datasets."""

    # Map country based on item set IDs
    country = _get_countries_from_item_sets(item)

    # Custom logic to extract URL from fabio:hasURL, prioritizing @id
    fabio_has_url_data = item.get("fabio:hasURL")
    extracted_fabio_url = ""
    if isinstance(fabio_has_url_data, list):
        urls = []
        for v_item in fabio_has_url_data:
            if isinstance(v_item, dict):
                id_val = v_item.get("@id")
                if id_val and isinstance(id_val, str):
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

    # Keep volume as string (can contain multiple values like "1|2")
    volume_str = _get_value(item, "bibo:volume")

    # Keep issue as string (can contain multiple values like "3|4")
    issue_str = _get_value(item, "bibo:issue")

    # Convert edition to int
    edition_str = _get_value(item, "bibo:edition")
    edition_int = ""
    if edition_str:
        try:
            edition_int = int(edition_str)
        except ValueError:
            logger.warning(
                f"Could not convert edition '{edition_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Convert chapter to int
    chapter_str = _get_value(item, "bibo:chapter")
    chapter_int = ""
    if chapter_str:
        try:
            chapter_int = int(chapter_str)
        except ValueError:
            logger.warning(
                f"Could not convert chapter '{chapter_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Convert nb_pages to int
    nb_pages_str = _get_value(item, "bibo:numPages")
    nb_pages_int = ""
    if nb_pages_str:
        try:
            nb_pages_int = int(nb_pages_str)
        except ValueError:
            logger.warning(
                f"Could not convert nb_pages '{nb_pages_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Convert page start/end to int
    page_start_str = _get_value(item, "bibo:pageStart")
    page_start_int = ""
    if page_start_str:
        try:
            page_start_int = int(page_start_str)
        except ValueError:
            logger.warning(
                f"Could not convert pageStart '{page_start_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    page_end_str = _get_value(item, "bibo:pageEnd")
    page_end_int = ""
    if page_end_str:
        try:
            page_end_int = int(page_end_str)
        except ValueError:
            logger.warning(
                f"Could not convert pageEnd '{page_end_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Extract date when item was added to Omeka (YYYY-MM-DD format)
    added_date = ""
    if "o:created" in item and isinstance(item["o:created"], dict):
        created_value = item["o:created"].get("@value", "")
        if created_value:
            try:
                added_date = created_value.split("T")[0]
            except Exception:
                logger.warning(f"Could not parse added date '{created_value}' for item {item['o:id']}")

    return {
        "o:id": item["o:id"],
        "url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "identifier": _get_iwac_identifier(item, "dcterms:identifier"),
        "added_date": added_date,
        "o:resource_class": _get_resource_class(item),
        "title": _get_value(item, "dcterms:title"),
        "author": _join(item, "bibo:authorList"),
        "editor": _join(item, "bibo:editorList"),
        "review_of": _get_value(item, "bibo:reviewOf"),
        "publisher": _get_value(item, "dcterms:publisher"),
        "pub_date": _get_value(item, "dcterms:date"),
        "type": _get_value(item, "dcterms:type"),
        "book_title": _get_value(item, "dcterms:alternative"),
        "chapter": chapter_int,
        "volume": volume_str,
        "issue": issue_str,
        "abstract": _get_value(item, "dcterms:abstract"),
        "edition": edition_int,
        "nb_pages": nb_pages_int,
        "page_start": page_start_int,
        "page_end": page_end_int,
        "extent": _get_value(item, "dcterms:extent"),
        "is_part_of": _get_value(item, "dcterms:isPartOf"),
        "provenance": _get_value(item, "dcterms:provenance"),
        "subject": _join(item, "dcterms:subject"),
        "spatial": _get_value(item, "dcterms:spatial"),
        "language": _get_value(item, "dcterms:language"),
        "doi": _get_value(item, "bibo:doi"),
        "URL": extracted_fabio_url,
        "country": country,
    }


# ---------------------------------------------------------------------------
# Pipeline principale : fetch → map → dataset → push
# ---------------------------------------------------------------------------

async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB"):
    api = OmekaApiClient(cfg, use_cache=True)

    # 1. Fetch current Omeka items and map them
    logger.info("Fetching reference items from Omeka API...")
    omeka_items_raw = await api.fetch_all_reference_items()

    if not omeka_items_raw:
        logger.warning("No items returned from Omeka API. Exiting.")
        return

    logger.info(f"Fetched {len(omeka_items_raw)} reference items from Omeka.")
    omeka_records_list = []
    # Use a standard tqdm here if the inner operations are not heavily async and blocking
    for it in tqdm(omeka_items_raw, desc="Mapping Omeka references"):
        try:
            record = await map_reference(it, api)
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
    new_omeka_df['o:id'] = new_omeka_df['o:id'].astype(str) # Ensure consistent type for merging

    # 2. Load existing dataset from Hugging Face Hub
    existing_df = pd.DataFrame()
    hf_token_env = os.getenv("HF_TOKEN")
    hf_token_stored = HfFolder.get_token()
    token_to_use = hf_token_env if hf_token_env else hf_token_stored

    try:
        logger.info(f"Attempting to load existing dataset from Hugging Face Hub: {repo}")
        # Corrected call to load_dataset
        existing_ds = load_dataset(repo, name="references", split="train", token=token_to_use, download_mode="force_redownload", verification_mode="no_checks")
        existing_df = existing_ds.to_pandas()
        
        if 'o:id' not in existing_df.columns or existing_df['o:id'].isnull().all():
            logger.warning("'o:id' column not found in existing Hub dataset or all values are null. Treating as empty.")
            existing_df = pd.DataFrame()
        else:
            existing_df['o:id'] = existing_df['o:id'].astype(str)
            logger.info(f"Successfully loaded existing dataset with {len(existing_df)} records.")
            
    except Exception as e:
        logger.warning(f"Could not load existing dataset from {repo} (may be first run or other issue, error: {e}). Proceeding as if Hub dataset is empty.")
        existing_df = pd.DataFrame() # Ensure it's an empty DataFrame on error

    # 3. Merge logic
    if existing_df.empty:
        logger.info("No existing data on Hub, 'o:id' missing in existing data, or error loading existing data; using new Omeka data directly.")
        final_df = new_omeka_df
    else:
        logger.info(f"Merging new Omeka data ({len(new_omeka_df)} records) with existing Hub data ({len(existing_df)} records).")
        
        # Define columns to exclude from existing data (old columns we want to remove)
        columns_to_exclude = ['o:item_set', 'o:media/file', 'iiif_manifest', 'thumbnail']
        
        # Identify columns in existing_df that are NOT in new_omeka_df and NOT in the exclusion list
        extra_cols_to_preserve = [col for col in existing_df.columns 
                                 if col not in new_omeka_df.columns and col not in columns_to_exclude]
        
        if extra_cols_to_preserve:
            logger.info(f"Extra columns to preserve from existing data: {extra_cols_to_preserve}")
            # Use outer merge to keep all records and preserve extra columns (excluding unwanted ones)
            final_df = pd.merge(new_omeka_df, existing_df[['o:id'] + extra_cols_to_preserve], on='o:id', how='outer', suffixes=('', '_old'))
            # Fill NaN values in new columns with data from existing columns where available
            final_df = final_df.ffill(axis=1).bfill(axis=1)
        else:
            logger.info("No extra columns to preserve from existing data (excluding unwanted columns).")
            final_df = new_omeka_df
            
        logger.info(f"Excluded columns from merge: {columns_to_exclude}")

        logger.info(f"Merge complete. Resulting dataset has {len(final_df)} records. Final columns: {final_df.columns.tolist()}")
        if extra_cols_to_preserve:
            logger.info(f"Preserved extra columns: {extra_cols_to_preserve}")

    # 4. Conversion to Dataset and Push
    if not final_df.empty:
        logger.info(f"Preparing to push {len(final_df)} records to the Hub. Columns: {final_df.columns.tolist()}")
        
        # Final check for o:id integrity
        if 'o:id' not in final_df.columns or final_df['o:id'].isnull().any():
            logger.error("Critical error: 'o:id' is missing or null in the final DataFrame before push. Aborting push.")
            await conn_manager.close()
            return

        # Ensure consistent data types for mixed columns
        # Convert numeric columns that might have mixed types to strings to avoid Arrow conversion issues
        mixed_type_columns = ['chapter', 'edition', 'nb_pages', 'page_start', 'page_end']
        for col in mixed_type_columns:
            if col in final_df.columns:
                # Convert all values to strings, replacing empty strings with empty strings (not 'nan')
                final_df[col] = final_df[col].astype(str).replace('nan', '')

        ds = Dataset.from_pandas(final_df, preserve_index=False)
        logger.info("Dataset preview (first 5 rows):")
        logger.info(ds.to_pandas().head())

        if os.getenv("HF_TOKEN") is None and not hf_utils.is_notebook():
            login()
        
        try:
            logger.info(f"Pushing dataset to {repo} with subset 'references'...")
            ds.push_to_hub(repo, config_name="references", max_shard_size=shard_size, token=token_to_use)
            logger.info(f"Successfully pushed {len(final_df)} reference records to {repo} (subset 'references')")
        except Exception as e:
            logger.error(f"Error pushing dataset to Hub: {e}", exc_info=True)

    else:
        logger.info("Final dataset is empty. No push operation will be performed (should have been handled by initial empty Omeka data check).")

    await conn_manager.close()


# ---------------------------------------------------------------------------
# Exécution CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publie les références bibliographiques IWAC sur le Hub HF")
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection", help="Repository Hugging Face où publier")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille max d'un shard Parquet (ex. 500MB, 1GB)")
    args = parser.parse_args()

    asyncio.run(build_and_push(Config(), repo=args.repo, shard_size=args.max_shard_size))
