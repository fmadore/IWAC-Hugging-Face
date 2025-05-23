#!/usr/bin/env python3
"""
upload_newspaper_hf.py
=====================

Extrait les articles de journaux (resource_class_id = 36) depuis l'API Omeka‑S
d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le Hugging Face
Hub.

Usage
-----
    python upload_newspaper_hf.py \
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
from huggingface_hub import login

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
            page = 2
            while True:
                batch = await self.fetch_items_page(rcid, page)
                if not batch:
                    break
                items.extend(batch)
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
        return "|".join(str(m["o:id"]) for m in item["o:media"])
    return ""


async def map_newspaper_article(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka en dict plat pour HF datasets."""

    primary_url = ""
    if item.get("o:primary_media"):
        mid = item["o:primary_media"]["@id"].split("/")[-1]
        mdata = await api.fetch_media_data(mid)
        primary_url = mdata.get("o:original_url", "")

    return {
        "o:id": item["o:id"],
        "url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "o:primary_media": primary_url,
        "o:media/file": _get_media_ids(item),
        "dcterms:title": _get_value(item, "dcterms:title"),
        "dcterms:creator": _join(item, "dcterms:creator"),
        "dcterms:publisher": _join(item, "dcterms:publisher"),
        "dcterms:date": _get_value(item, "dcterms:date"),
        "dcterms:subject": _join(item, "dcterms:subject"),
        "dcterms:language": _get_value(item, "dcterms:language"),
        "fabio:hasURL": _get_value(item, "fabio:hasURL"),
        "bibo:content": _get_value(item, "bibo:content"),
    }


# ---------------------------------------------------------------------------
# Pipeline principale : fetch → map → dataset → push
# ---------------------------------------------------------------------------

async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB"):
    api = OmekaApiClient(cfg, use_cache=True)
    items = await api.fetch_items(36)  # Newspaper articles seulement

    records: List[Dict[str, Any]] = []
    for it in async_tqdm(items, desc="Mapping des articles"):
        records.append(await map_newspaper_article(it, api))

    # Conversion en Dataset (pyarrow/Parquet sous le capot)
    ds = Dataset.from_pandas(pd.DataFrame(records), preserve_index=False)
    logger.info(ds)

    # Authentification HF si nécessaire
    if os.getenv("HF_TOKEN") is None:
        login()

    ds.push_to_hub(repo, max_shard_size=shard_size)
    await conn_manager.close()
    logger.info("Dataset publié sur %s", repo)


# ---------------------------------------------------------------------------
# Exécution CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publie les articles de journaux IWAC sur le Hub HF")
    parser.add_argument("--repo", required=True, help="Nom du repo Hugging Face, ex. fmadore/iwac-newspaper-articles")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille max d'un shard Parquet (ex. 500MB, 1GB)")
    args = parser.parse_args()

    asyncio.run(build_and_push(Config(), repo=args.repo, shard_size=args.max_shard_size))
