#!/usr/bin/env python3
"""
upload_index_hf.py
==================

Extrait les données d'index depuis l'API Omeka S d'IWAC, calcule les statistiques
de fréquence depuis les datasets articles et publications, et pousse le tout
sur le Hugging Face Hub.

Usage
-----
    python upload_index_hf.py \
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
import json
import io
import gzip
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Type
from collections import defaultdict, Counter

import pandas as pd
import aiohttp
import aiofiles
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from huggingface_hub import login, HfFolder, utils as hf_utils
import huggingface_hub

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
    CACHE_DIR: str = ".cache_omk_index"
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


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

def _get_value(item: Dict[str, Any], field: str) -> str:
    """Extrait une valeur simple d'un champ Omeka"""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        parts = [str(v.get("display_title") or v.get("@value") or v.get("@id", "")) for v in val]
        return "|".join(filter(None, parts))
    if isinstance(val, dict):
        return val.get("display_title", "") or val.get("@value", "")
    return str(val)


def _get_value_only(item: Dict[str, Any], field: str) -> str:
    """Extrait seulement les valeurs @value d'un champ Omeka (pour dcterms:identifier)"""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        # Ne prendre que les @value, ignorer les autres types
        parts = [str(v.get("@value", "")) for v in val if isinstance(v, dict) and v.get("@value")]
        return "|".join(filter(None, parts))
    if isinstance(val, dict):
        return val.get("@value", "")
    return str(val)


def _get_value_with_lang(item: Dict[str, Any], field: str, preferred_lang: str = "fr") -> str:
    """Extrait une valeur en privilégiant une langue spécifique"""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        # Chercher d'abord la langue préférée
        for v in val:
            if isinstance(v, dict) and v.get("@language") == preferred_lang:
                return v.get("@value", "")
        # Sinon prendre la première valeur disponible
        for v in val:
            if isinstance(v, dict) and v.get("@value"):
                return v.get("@value", "")
    elif isinstance(val, dict):
        return val.get("@value", "")
    return str(val)


def _get_display_title(item: Dict[str, Any], field: str) -> str:
    """Extrait les display_title d'un champ"""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        parts = [str(v.get("display_title", "")) for v in val if v.get("display_title")]
        return "|".join(filter(None, parts))
    elif isinstance(val, dict):
        return val.get("display_title", "")
    return ""


def _get_resource_class_type(item: Dict[str, Any]) -> str:
    """Mappe le resource_class_id vers le type correspondant"""
    resource_class_mapping = {
        9: "Lieux",
        94: "Personnes", 
        96: "Organisations",
        54: "Événements",
        244: "Sujets"  # Par défaut, sera affiné selon l'item_set
    }
    
    resource_class = item.get("o:resource_class")
    if not resource_class or not isinstance(resource_class, dict):
        return ""
    
    class_id = resource_class.get("o:id")
    if not class_id:
        return ""
    
    # Cas spécial pour la classe 244 (Sujets/Notices d'autorité)
    if class_id == 244:
        # Vérifier l'item_set pour distinguer Sujets vs Notices d'autorité
        item_sets = item.get("o:item_set", [])
        if isinstance(item_sets, list):
            for item_set in item_sets:
                if isinstance(item_set, dict):
                    item_set_id = item_set.get("o:id")
                    if item_set_id == 1:
                        return "Sujets"
                    elif item_set_id == 267:
                        return "Notices d'autorité"
        # Si pas d'item_set spécifique trouvé, retourner "Sujets" par défaut
        return "Sujets"
    
    return resource_class_mapping.get(class_id, "")


def _get_item_set_ids(item: Dict[str, Any]) -> str:
    """Extrait les IDs des item sets (gardé pour compatibilité si nécessaire ailleurs)"""
    if "o:item_set" not in item or item["o:item_set"] is None:
        return ""
    item_sets = item["o:item_set"]
    if isinstance(item_sets, list):
        ids = []
        for item_set in item_sets:
            if isinstance(item_set, dict) and "o:id" in item_set:
                ids.append(str(item_set["o:id"]))
        return "|".join(ids)
    return ""


@async_retry(max_tries=3, exceptions=(aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError))
async def fetch_iiif_thumbnail_url(omeka_id: Union[str, int], session: aiohttp.ClientSession) -> str:
    """Fetches and extracts the thumbnail URL from an IIIF manifest."""
    manifest_url = f"https://islam.zmo.de/iiif/3/{omeka_id}/manifest"
    thumbnail_url = ""
    try:
        async with session.get(manifest_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                try:
                    manifest = await resp.json()
                    thumbnails = manifest.get("thumbnail")
                    if isinstance(thumbnails, list) and thumbnails:
                        thumbnail_info = thumbnails[0]
                        if isinstance(thumbnail_info, dict):
                            thumbnail_url = thumbnail_info.get("id", "")
                except json.JSONDecodeError as e_json:
                    logger.warning(f"JSON decoding error for IIIF manifest {omeka_id}: {e_json}")
            elif resp.status not in [408, 429, 500, 502, 503, 504]:
                logger.warning(f"IIIF manifest request for {omeka_id} returned status {resp.status}")
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching IIIF manifest for {omeka_id}")
    except aiohttp.ClientError as e_client:
        logger.warning(f"Client error fetching IIIF manifest for {omeka_id}: {e_client}")
    except Exception as e_general:
        logger.error(f"Unexpected error fetching IIIF manifest for {omeka_id}: {e_general}")
    return thumbnail_url


async def map_index_item(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item d'index Omeka en dict plat pour HF datasets."""
    
    # Extract date when item was added to Omeka (YYYY-MM-DD format)
    added_date = ""
    if "o:created" in item and isinstance(item["o:created"], dict):
        created_value = item["o:created"].get("@value", "")
        if created_value:
            try:
                # Extract date part from ISO format (e.g., "2025-07-09T14:02:51+00:00" -> "2025-07-09")
                added_date = created_value.split("T")[0]
            except Exception:
                logger.warning(f"Could not parse added date '{created_value}' for item {item['o:id']}")
    
    # Fetch thumbnail URL
    session = await conn_manager.get()
    thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)

    return {
        "o:id": item["o:id"],
        "identifier": _get_value_only(item, "dcterms:identifier"),
        "added_date": added_date, # Date when item was added to Omeka
        "url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "thumbnail": thumbnail_url,
        "Titre": item.get("o:title", ""),
        "Titre alternatif": _get_value(item, "dcterms:alternative"),
        "Type": _get_resource_class_type(item),
        "Description": _get_value_with_lang(item, "dcterms:description", "fr"),
        "Date création": _get_value(item, "dcterms:created"),
        "date": _get_value(item, "dcterms:date"),
        "Relation": _get_display_title(item, "dcterms:relation"),
        "Remplacé par": _get_value(item, "dcterms:isReplacedBy"),
        "Partie de": _get_display_title(item, "dcterms:isPartOf"),
        "spatial": _get_value(item, "dcterms:spatial"),
        "A une partie": _get_display_title(item, "dcterms:hasPart"),
        "Prénom": _get_value(item, "foaf:firstName"),
        "Nom": _get_value(item, "foaf:lastName"),
        "Genre": _get_display_title(item, "foaf:gender"),
        "Naissance": _get_value(item, "foaf:birthday"),
        "Coordonnées": _get_value(item, "curation:coordinates"),
    }


# ---------------------------------------------------------------------------
# Calcul des statistiques de fréquence
# ---------------------------------------------------------------------------

def extract_terms_from_field(field_value: str) -> List[str]:
    """Extrait les termes d'un champ qui peut contenir des valeurs multiples séparées par |"""
    if not field_value or pd.isna(field_value):
        return []
    return [term.strip() for term in str(field_value).split("|") if term.strip()]


def calculate_frequency_stats(articles_df: pd.DataFrame, publications_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calcule les statistiques de fréquence pour tous les termes des colonnes subject et spatial"""
    term_stats = defaultdict(lambda: {
        'frequency': 0,
        'first_occurrence': None,
        'last_occurrence': None,
        'countries': set()
    })
    
    # Traiter les articles
    logger.info("Calculating frequency stats from articles dataset...")
    if not articles_df.empty:
        for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Processing articles"):
            # Extraire la date
            date_str = row.get('pub_date', '') or row.get('date', '')
            country = row.get('country', '')
            
            # Traiter les sujets
            subjects = extract_terms_from_field(row.get('subject', ''))
            for subject in subjects:
                if subject:
                    term_stats[subject]['frequency'] += 1
                    if country:
                        term_stats[subject]['countries'].add(country)
                    if date_str:
                        if not term_stats[subject]['first_occurrence'] or date_str < term_stats[subject]['first_occurrence']:
                            term_stats[subject]['first_occurrence'] = date_str
                        if not term_stats[subject]['last_occurrence'] or date_str > term_stats[subject]['last_occurrence']:
                            term_stats[subject]['last_occurrence'] = date_str
            
            # Traiter les données spatiales
            spatials = extract_terms_from_field(row.get('spatial', ''))
            for spatial in spatials:
                if spatial:
                    term_stats[spatial]['frequency'] += 1
                    if country:
                        term_stats[spatial]['countries'].add(country)
                    if date_str:
                        if not term_stats[spatial]['first_occurrence'] or date_str < term_stats[spatial]['first_occurrence']:
                            term_stats[spatial]['first_occurrence'] = date_str
                        if not term_stats[spatial]['last_occurrence'] or date_str > term_stats[spatial]['last_occurrence']:
                            term_stats[spatial]['last_occurrence'] = date_str
    
    # Traiter les publications
    logger.info("Calculating frequency stats from publications dataset...")
    if not publications_df.empty:
        for _, row in tqdm(publications_df.iterrows(), total=len(publications_df), desc="Processing publications"):
            # Extraire la date
            date_str = row.get('pub_date', '') or row.get('date', '')
            country = row.get('country', '')
            
            # Traiter les sujets
            subjects = extract_terms_from_field(row.get('subject', ''))
            for subject in subjects:
                if subject:
                    term_stats[subject]['frequency'] += 1
                    if country:
                        term_stats[subject]['countries'].add(country)
                    if date_str:
                        if not term_stats[subject]['first_occurrence'] or date_str < term_stats[subject]['first_occurrence']:
                            term_stats[subject]['first_occurrence'] = date_str
                        if not term_stats[subject]['last_occurrence'] or date_str > term_stats[subject]['last_occurrence']:
                            term_stats[subject]['last_occurrence'] = date_str
            
            # Traiter les données spatiales
            spatials = extract_terms_from_field(row.get('spatial', ''))
            for spatial in spatials:
                if spatial:
                    term_stats[spatial]['frequency'] += 1
                    if country:
                        term_stats[spatial]['countries'].add(country)
                    if date_str:
                        if not term_stats[spatial]['first_occurrence'] or date_str < term_stats[spatial]['first_occurrence']:
                            term_stats[spatial]['first_occurrence'] = date_str
                        if not term_stats[spatial]['last_occurrence'] or date_str > term_stats[spatial]['last_occurrence']:
                            term_stats[spatial]['last_occurrence'] = date_str
    
    # Convertir les sets en chaînes séparées par |
    result = {}
    for term, stats in term_stats.items():
        result[term] = {
            'frequency': stats['frequency'],
            'first_occurrence': stats['first_occurrence'] or '',
            'last_occurrence': stats['last_occurrence'] or '',
            'countries': "|".join(sorted(stats['countries'])) if stats['countries'] else ''
        }
    
    logger.info(f"Calculated frequency stats for {len(result)} unique terms")
    return result


async def load_reference_datasets(token: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les datasets articles et publications depuis Hugging Face Hub"""
    articles_df = pd.DataFrame()
    publications_df = pd.DataFrame()
    
    try:
        logger.info("Loading articles dataset from Hugging Face Hub...")
        articles_ds = load_dataset("fmadore/islam-west-africa-collection", name="articles", split="train", token=token, download_mode="force_redownload", verification_mode="no_checks")
        articles_df = articles_ds.to_pandas()
        logger.info(f"Loaded {len(articles_df)} articles")
    except Exception as e:
        logger.warning(f"Could not load articles dataset: {e}")
    
    try:
        logger.info("Loading publications dataset from Hugging Face Hub...")
        publications_ds = load_dataset("fmadore/islam-west-africa-collection", name="publications", split="train", token=token, download_mode="force_redownload", verification_mode="no_checks")
        publications_df = publications_ds.to_pandas()
        logger.info(f"Loaded {len(publications_df)} publications")
    except Exception as e:
        logger.warning(f"Could not load publications dataset: {e}")
    
    return articles_df, publications_df


# ---------------------------------------------------------------------------
# Pipeline principale : fetch → map → calculate stats → dataset → push
# ---------------------------------------------------------------------------

async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB"):
    api = OmekaApiClient(cfg, use_cache=True)

    # 1. Charger les datasets de référence pour calculer les statistiques
    hf_token_env = os.getenv("HF_TOKEN")
    hf_token_stored = HfFolder.get_token()
    token_to_use = hf_token_env if hf_token_env else hf_token_stored
    
    articles_df, publications_df = await load_reference_datasets(token_to_use)
    
    # 2. Calculer les statistiques de fréquence
    frequency_stats = calculate_frequency_stats(articles_df, publications_df)

    # 3. Fetch current Omeka index items and map them
    logger.info("Fetching index items from Omeka API...")
    
    # Récupérer tous les types d'items d'index
    omeka_items_raw = []
    resource_class_ids = [9, 94, 96, 54, 244]  # Lieux, Personnes, Organisations, Événements, Sujets/Notices
    
    for rcid in resource_class_ids:
        logger.info(f"Fetching items for resource class {rcid}...")
        try:
            items = await api.fetch_items(rcid)
            omeka_items_raw.extend(items)
            logger.info(f"Fetched {len(items)} items for resource class {rcid}")
        except Exception as e:
            logger.error(f"Error fetching items for resource class {rcid}: {e}")
            continue

    if not omeka_items_raw:
        logger.warning("No index items returned from Omeka API. Exiting.")
        return

    logger.info(f"Fetched {len(omeka_items_raw)} index items from Omeka.")
    omeka_records_list = []
    
    for it in tqdm(omeka_items_raw, desc="Mapping Omeka index items"):
        try:
            record = await map_index_item(it, api)
            omeka_records_list.append(record)
        except Exception as e:
            logger.error(f"Error mapping index item {it.get('o:id', 'Unknown ID')}: {e}", exc_info=True)
    
    if not omeka_records_list:
        logger.error("No index records were successfully mapped. Exiting.")
        return
        
    new_omeka_df = pd.DataFrame(omeka_records_list)
    
    # 4. Ajouter les statistiques de fréquence
    logger.info("Adding frequency statistics to index records...")
    
    # Créer des colonnes pour les statistiques
    new_omeka_df['frequency'] = 0
    new_omeka_df['first_occurrence'] = ''
    new_omeka_df['last_occurrence'] = ''
    new_omeka_df['countries'] = None
    
    # Mapper les statistiques basées sur le titre
    for idx, row in new_omeka_df.iterrows():
        titre = row.get('Titre', '')
        if titre in frequency_stats:
            stats = frequency_stats[titre]
            new_omeka_df.at[idx, 'frequency'] = stats['frequency']
            new_omeka_df.at[idx, 'first_occurrence'] = stats['first_occurrence']
            new_omeka_df.at[idx, 'last_occurrence'] = stats['last_occurrence']
            new_omeka_df.at[idx, 'countries'] = stats['countries']
        else:
            new_omeka_df.at[idx, 'countries'] = ''

    # 5. Load existing dataset from Hugging Face Hub
    existing_df = pd.DataFrame()
    
    try:
        logger.info(f"Attempting to load existing index dataset from Hugging Face Hub: {repo}")
        existing_ds = load_dataset(repo, name="index", split="train", token=token_to_use, download_mode="force_redownload", verification_mode="no_checks")
        existing_df = existing_ds.to_pandas()
        
        if 'o:id' not in existing_df.columns or existing_df['o:id'].isnull().all():
            logger.warning("'o:id' column missing or all null in existing Hub dataset. Treating as empty.")
            existing_df = pd.DataFrame() 
        else:
            existing_df['o:id'] = existing_df['o:id'].astype(str)
            logger.info(f"Successfully loaded {len(existing_df)} records from {repo}")
            
    except Exception as e:
        logger.warning(f"Could not load existing dataset from {repo}: {e}. Proceeding as if Hub dataset is empty.")
        existing_df = pd.DataFrame()

    # 6. Merge logic
    if existing_df.empty:
        logger.info("No existing data on Hub; using new Omeka data directly.")
        final_df = new_omeka_df
    else:
        logger.info(f"Merging new Omeka data ({len(new_omeka_df)} records) with existing Hub data ({len(existing_df)} records).")
        
        # Identifier les colonnes à préserver
        extra_cols_to_preserve = [col for col in existing_df.columns if col not in new_omeka_df.columns]
        
        if extra_cols_to_preserve:
            logger.info(f"Preserving these columns from existing dataset: {extra_cols_to_preserve}")
            cols_from_existing_for_merge = ['o:id'] + extra_cols_to_preserve
            final_df = pd.merge(new_omeka_df, existing_df[cols_from_existing_for_merge], on='o:id', how='left')
        else:
            logger.info("No unique columns to preserve from existing dataset.")
            final_df = new_omeka_df

        logger.info(f"Merge complete. Resulting dataset has {len(final_df)} records.")

    # 7. Conversion to Dataset and Push
    if not final_df.empty:
        logger.info(f"Preparing to push {len(final_df)} index records to the Hub.")
        
        # Final check for o:id integrity
        if 'o:id' not in final_df.columns or final_df['o:id'].isnull().any():
            logger.error("Critical error: 'o:id' is missing or null in the final DataFrame before push. Aborting push.")
            await conn_manager.close()
            return

        ds = Dataset.from_pandas(final_df, preserve_index=False)
        logger.info("Index dataset preview (first 5 rows):")
        logger.info(ds.to_pandas().head())

        if os.getenv("HF_TOKEN") is None and not hf_utils.is_notebook():
            login()
        
        try:
            logger.info(f"Pushing index dataset to {repo} with config 'index'...")
            ds.push_to_hub(repo, max_shard_size=shard_size, config_name="index")
            logger.info(f"Index dataset published/updated on {repo} with config 'index'")
        except Exception as e:
            logger.error(f"Failed to push index dataset to Hub: {e}")
            logger.error("Details of the exception:", exc_info=True)

    else:
        logger.info("Final index dataset is empty. No push operation will be performed.")

    await conn_manager.close()


# ---------------------------------------------------------------------------
# Exécution CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publie l'index IWAC sur le Hub HF")
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection", help="Nom du repo HF (ex. fmadore/islam-west-africa-collection)")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille max d'un shard Parquet (ex. 500MB, 1GB)")
    args = parser.parse_args()

    asyncio.run(build_and_push(Config(), repo=args.repo, shard_size=args.max_shard_size))
