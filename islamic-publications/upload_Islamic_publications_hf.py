#!/usr/bin/env python3
"""
upload_Islamic_publications_hf.py
=================================

Extrait les publications islamiques (resource_class_id = 60) depuis l'API Omeka S
d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le Hugging Face
Hub.

Usage
-----
    python upload_Islamic_publications_hf.py \
        --repo fmadore/iwac-newspaper-articles \
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
import sys # Add sys import
import json
import io
import gzip
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Type

import aiohttp
import aiofiles  # pour la mise en cache
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
from huggingface_hub import login, HfFolder, utils as hf_utils # Modified import
import huggingface_hub

# Adjust sys.path to include the parent directory for country_mapper import
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from country_mapper import get_country_from_newspaper # Added import

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

# Specify the path to .env in the parent directory
dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

@dataclass
class Config:
    """Paramètres globaux chargés depuis .env ou variables d'environnement"""

    API_URL: str = os.getenv("OMEKA_BASE_URL", "https://islam.zmo.de/api")
    API_KEY_IDENTITY: str = os.getenv("OMEKA_KEY_IDENTITY", "")
    API_KEY_CREDENTIAL: str = os.getenv("OMEKA_KEY_CREDENTIAL", "")
    CACHE_DIR: str = ".cache_omk"
    CACHE_HOURS: int = 24


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
                except exceptions as exc:
                    logger.warning(f"{func.__name__}: tentative {attempt + 1}/{max_tries} échouée ({exc})")
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
            with tqdm(desc="Fetching item pages", unit="page") as pbar:
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

    async def fetch_media_data(self, media_id: str):
        return await self.request(f"media/{media_id}", {})


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

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


def _join(item: Dict[str, Any], field: str) -> str:
    return _get_value(item, field)


def _get_media_ids(item: Dict[str, Any]) -> str:
    if "o:media" in item and isinstance(item["o:media"], list):
        return "|".join(str(m["o:id"]) for m in item["o:media"]) # Corrected indentation
    return ""


async def map_islamic_publication_item(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]: # Renamed function
    """Transforme un item Omeka (publication islamique) en dict plat pour HF datasets.""" # Updated docstring

    primary_url = ""
    if item.get("o:primary_media"):
        mid = item["o:primary_media"]["@id"].split("/")[-1]
        mdata = await api.fetch_media_data(mid)
        primary_url = mdata.get("o:original_url", "")

    publisher_name = _join(item, "dcterms:publisher") # Changed newspaper_name to publisher_name for clarity
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

    return {
        "o:id": item["o:id"],
        "identifier": _get_value(item, "dcterms:identifier"),
        "url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "PDF": primary_url,
        "title": _get_value(item, "dcterms:title"),
        "author": _join(item, "dcterms:creator"),
        "newspaper": publisher_name,
        "country": country,
        "pub_date": _get_value(item, "dcterms:date"),
        "issue": _get_value(item, "bibo:issue"),
        "subject": _join(item, "dcterms:subject"),
        "spatial": _get_value(item, "dcterms:spatial"),
        "language": _get_value(item, "dcterms:language"),
        "nb_pages": _get_value(item, "bibo:numPages"),
        "URL": extracted_fabio_url, 
        "source": _get_value(item, "dcterms:source"),
        "OCR": _get_value(item, "bibo:content"),
    }


# ---------------------------------------------------------------------------
# Pipeline principale : fetch → map → dataset → push
# ---------------------------------------------------------------------------

async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB"):
    api = OmekaApiClient(cfg, use_cache=True)

    # 1. Fetch current Omeka items and map them
    logger.info("Fetching Islamic publications from Omeka API (resource_class_id=60)...") # Updated log
    omeka_items_raw = await api.fetch_items(60)  # Publications islamiques seulement

    if not omeka_items_raw:
        logger.info("No items found from Omeka API. Attempting to clear the dataset on the Hub.")
        empty_df = pd.DataFrame(columns=['o:id']) 
        try:
            # Try to load with the target config name
            existing_meta_ds = Dataset.load_dataset(repo, name="publications", split="train", token=HfFolder.get_token(), trust_remote_code=True, download_mode="force_redownload", ignore_verifications=True)
            if existing_meta_ds.num_rows > 0: 
                 empty_df = pd.DataFrame(columns=existing_meta_ds.column_names)
            logger.info(f"Using schema from existing Hub dataset (config 'publications') for empty push: {empty_df.columns.tolist()}")
        except Exception as e_load_meta:
            logger.warning(f"Could not load existing dataset schema for config 'publications' (error: {e_load_meta}). Using minimal 'o:id' schema.")

        ds = Dataset.from_pandas(empty_df, preserve_index=False)
        logger.info("Pushing an empty or schema-based empty dataset to the Hub (config 'publications').")
        if os.getenv("HF_TOKEN") is None and not hf_utils.is_notebook():
            login()
        try:
            ds.push_to_hub(repo, max_shard_size=shard_size, config_name="publications") # Use "publications"
            logger.info(f"Empty dataset pushed to {repo} with config 'publications'.")
        except Exception as e_push_empty:
            logger.error(f"Failed to push empty dataset: {e_push_empty}")
        await conn_manager.close()
        return

    logger.info(f"Fetched {len(omeka_items_raw)} items from Omeka.")
    omeka_records_list = []
    for it in async_tqdm(omeka_items_raw, desc="Mapping Islamic publications"): # Updated description
        omeka_records_list.append(await map_islamic_publication_item(it, api)) # Call renamed function
    
    new_omeka_df = pd.DataFrame(omeka_records_list)
    if 'o:id' not in new_omeka_df.columns or new_omeka_df['o:id'].isnull().any():
        logger.error("'o:id' column is missing or contains null values in mapped Omeka data. Cannot proceed.")
        await conn_manager.close()
        return
    new_omeka_df['o:id'] = new_omeka_df['o:id'].astype(str)

    # 2. Load existing dataset from Hugging Face Hub
    existing_df = pd.DataFrame()
    hf_token_env = os.getenv("HF_TOKEN")
    hf_token_stored = HfFolder.get_token()
    token_to_use = hf_token_env if hf_token_env else hf_token_stored

    try:
        logger.info(f"Attempting to load existing dataset from Hugging Face Hub: {repo} (config 'publications')")
        existing_ds = Dataset.load_dataset(repo, name="publications", split="train", token=token_to_use, trust_remote_code=True, download_mode="force_redownload", ignore_verifications=True) # Use name="publications"
        existing_df = existing_ds.to_pandas()
        if 'o:id' not in existing_df.columns:
            logger.warning("'o:id' column not found in existing Hub dataset (config 'publications'). Treating as empty.")
            existing_df = pd.DataFrame() 
        else:
            existing_df['o:id'] = existing_df['o:id'].astype(str) 
            logger.info(f"Successfully loaded existing dataset (config 'publications') with {len(existing_df)} records.")
    except Exception as e:
        logger.warning(f"Could not load existing dataset from {repo} (config 'publications') (may be first run, error: {e}). Proceeding as if Hub dataset is empty.")
        existing_df = pd.DataFrame()

    # 3. Merge logic
    if existing_df.empty:
        final_df = new_omeka_df.copy()
        logger.info("No existing data on Hub (config 'publications') or 'o:id' missing; using new Omeka data directly.")
    else:
        # Identify Omeka-derived columns (all columns present in new_omeka_df)
        omeka_derived_cols = new_omeka_df.columns.tolist()
        
        # Identify Hub-specific columns (present in existing_df but not in new_omeka_df, excluding 'o:id' if it's already primary)
        hub_specific_cols = existing_df.columns.difference(new_omeka_df.columns).tolist()
        if 'o:id' in hub_specific_cols: # 'o:id' should be primary, not treated as purely hub-specific if also in omeka_df
            hub_specific_cols.remove('o:id')

        # Create a base for the merge from new_omeka_df (all current Omeka items)
        # This ensures all items from Omeka are included, and their Omeka-derived fields are up-to-date.
        merged_df = new_omeka_df.copy()

        if hub_specific_cols:
            # If there are hub-specific columns, merge them from the existing dataset
            # We only want to bring over the hub-specific columns for matching 'o:id's
            cols_to_preserve_from_existing = ['o:id'] + hub_specific_cols
            existing_data_to_preserve = existing_df[cols_to_preserve_from_existing]
            
            # Update merged_df with the hub_specific_columns from existing_data_to_preserve
            # This is like a left join but more of an update operation for specific columns.
            # We can do this by setting 'o:id' as index and using update, then reset_index.
            merged_df = merged_df.set_index('o:id')
            existing_data_to_preserve = existing_data_to_preserve.set_index('o:id')
            merged_df.update(existing_data_to_preserve[hub_specific_cols], join='left', overwrite=False) # overwrite=False to keep new Omeka data if col name overlaps
            merged_df = merged_df.reset_index()
            logger.info(f"Merged new Omeka data, preserving/updating {len(hub_specific_cols)} hub-specific columns from config 'publications': {hub_specific_cols}")
        else:
            logger.info("No unique hub-specific columns to preserve from existing dataset (config 'publications'). Using new Omeka data.")
        
        final_df = merged_df

        # Ensure all columns from both sources are present, filling NaNs where appropriate
        # This handles cases where new_omeka_df might have new Omeka columns not yet on the Hub,
        # or existing_df had columns that are no longer in Omeka (those would be dropped if not in hub_specific_cols).
        # The current logic should handle this by starting with new_omeka_df and adding hub_specific_cols.
        
        # Make sure final_df has all columns that were in existing_df (if they are hub-specific)
        # and all columns from new_omeka_df.
        # This can be complex if schemas diverge significantly. The current approach prioritizes new_omeka_df schema
        # and adds hub-specific columns.

    # 4. Conversion to Dataset and Push
    if not final_df.empty:
        logger.info(f"Preparing to push {len(final_df)} records to the Hub (config 'publications'). Columns: {final_df.columns.tolist()}")
        
        # Final check for o:id integrity
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
            logger.info(f"Pushing dataset to {repo} with config 'publications'...")
            ds.push_to_hub(repo, max_shard_size=shard_size, config_name="publications") # Use "publications"
            logger.info(f"Dataset published/updated on {repo} with config 'publications'")
        except Exception as e:
            logger.error(f"Failed to push dataset to Hub (config 'publications'): {e}")
            logger.error("Details of the exception:", exc_info=True)

    else:
        logger.info("Final dataset is empty. No push operation will be performed for config 'publications'.")

    await conn_manager.close()


# ---------------------------------------------------------------------------
# Exécution CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publie les publications islamiques IWAC sur le Hub HF") # Updated description
    parser.add_argument("--repo", default="fmadore/iwac-newspaper-articles", help="Dataset repo on the Hugging Face Hub (e.g. fmadore/iwac-islamic-publications or fmadore/iwac-newspaper-articles)") # Kept default as per user, but updated help text
    parser.add_argument("--max-shard-size", default="1GB", help="Taille max d'un shard Parquet (ex. 500MB, 1GB)")
    args = parser.parse_args()

    # The repo argument from CLI will be used. If user wants fmadore/iwac-islamic-publications, they should pass it.
    asyncio.run(build_and_push(Config(), repo=args.repo, shard_size=args.max_shard_size))
