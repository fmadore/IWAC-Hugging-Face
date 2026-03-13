#!/usr/bin/env python3
"""
upload_documents_hf.py
======================

Extracts documents (resource_class_id = 49) from the IWAC Omeka S API, converts
them to Arrow/Parquet dataset, and pushes to Hugging Face Hub as the 'documents'
subset of fmadore/islam-west-africa-collection.

Usage
-----
    python upload_documents_hf.py --max-shard-size 1GB

Environment Variables
--------------------
  OMEKA_BASE_URL        API base URL, e.g., https://islam.zmo.de/api
  OMEKA_KEY_IDENTITY    Omeka key identity
  OMEKA_KEY_CREDENTIAL  Omeka key credential
  HF_TOKEN              Hugging Face access token (optional if using interactive login)
"""

import os
import sys
import json
import io
import gzip
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Type

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import aiohttp
import aiofiles
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from huggingface_hub import login, get_token, utils as hf_utils

# Rich console imports for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich import box

# Disable symlinks warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------------------------
# Configuration & logging
# ---------------------------------------------------------------------------

# Initialize Rich console
console = Console()

# Configure logging with Rich handler for beautiful output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class Config:
    """Paramètres globaux chargés depuis .env ou variables d'environnement"""

    API_URL: str = os.getenv("OMEKA_BASE_URL", "https://islam.zmo.de/api")
    API_KEY_IDENTITY: str = os.getenv("OMEKA_KEY_IDENTITY", "")
    API_KEY_CREDENTIAL: str = os.getenv("OMEKA_KEY_CREDENTIAL", "")
    CACHE_DIR: str = ".cache_omk_documents"
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
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Fetching pages..."),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Fetching item pages", total=None)
                while True:
                    batch = await self.fetch_items_page(rcid, page)
                    if not batch:
                        break
                    items.extend(batch)
                    progress.update(task, advance=1, description=f"[cyan]Page {page} fetched")
                    if len(batch) < per:
                        break
                    page += 1
        console.print(f"[green]✓[/green] {len(items)} items retrieved for class {rcid}")
        return items

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


def _join(item: Dict[str, Any], field: str) -> str:
    return _get_value(item, field)


def _get_media_ids(item: Dict[str, Any]) -> str:
    if "o:media" in item and isinstance(item["o:media"], list):
        return "|".join(str(m["o:id"]) for m in item["o:media"])
    return ""


@async_retry(max_tries=3, exceptions=(aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError))
async def fetch_iiif_thumbnail_url(omeka_id: Union[str, int], session: aiohttp.ClientSession) -> str:
    """Fetches and extracts the thumbnail URL from an IIIF manifest."""
    manifest_url = f"https://islam.zmo.de/iiif/3/{omeka_id}/manifest"
    thumbnail_url = ""
    try:
        # Use a shorter timeout for this specific, potentially numerous, request type
        async with session.get(manifest_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                # It's crucial to handle potential JSONDecodeError here if the response is not valid JSON
                try:
                    manifest = await resp.json()
                    thumbnails = manifest.get("thumbnail")
                    if isinstance(thumbnails, list) and thumbnails:
                        thumbnail_info = thumbnails[0]
                        if isinstance(thumbnail_info, dict):
                            thumbnail_url = thumbnail_info.get("id", "")
                except json.JSONDecodeError as e_json:
                    logger.warning(f"JSON decoding error for IIIF manifest {omeka_id}: {e_json}. URL: {manifest_url}")
            # Log other non-200 responses that are not exceptions handled by async_retry
            elif resp.status not in [408, 429, 500, 502, 503, 504]: # Avoid redundant logs for retryable http errors
                logger.warning(f"IIIF manifest request for {omeka_id} returned status {resp.status}. URL: {manifest_url}")
    except asyncio.TimeoutError: # Specifically catch timeout
        logger.warning(f"Timeout fetching IIIF manifest for {omeka_id}. URL: {manifest_url}")
    except aiohttp.ClientError as e_client: # Specifically catch client errors
        logger.warning(f"Client error fetching IIIF manifest for {omeka_id}: {e_client}. URL: {manifest_url}")
    # Catching general Exception for unexpected issues, though specific ones are better
    except Exception as e_general:
        logger.error(f"Unexpected error fetching IIIF manifest for {omeka_id}: {e_general}. URL: {manifest_url}")
    return thumbnail_url


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

    # Convert nb_pages to int
    nb_pages_str = _get_value(item, "bibo:numPages")
    nb_pages_int = None
    if nb_pages_str:
        try:
            nb_pages_int = int(nb_pages_str)
        except ValueError:
            logger.warning(
                f"Could not convert nb_pages '{nb_pages_str}' to int for item {item['o:id']}. Defaulting to null."
            )

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

    # Fetch thumbnail URL and set IIIF manifest URL only if PDF exists
    session = await conn_manager.get()
    thumbnail_url = ""
    iiif_manifest_url = ""
    
    if primary_url:  # Only fetch IIIF data if there's a PDF
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"

    return {
        "o:id": item["o:id"],
        "identifier": _get_value(item, "dcterms:identifier"),
        "added_date": added_date, # Date when item was added to Omeka
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "iiif_manifest": iiif_manifest_url,
        "PDF": primary_url,
        "thumbnail": thumbnail_url,
        "title": _get_value(item, "dcterms:title"),
        "author": _join(item, "dcterms:creator"),
        "country": country,
        "pub_date": _get_value(item, "dcterms:date"),
        "descriptionAI": _get_value(item, "bibo:shortDescription"),
        "subject": _join(item, "dcterms:subject"),
        "spatial": _get_value(item, "dcterms:spatial"),
        "language": _get_value(item, "dcterms:language"),
        "type": _get_value(item, "dcterms:type"),
        "nb_pages": nb_pages_int,
        "source": _get_value(item, "dcterms:source"),
        "rights": _get_label(item, "dcterms:rights"),
        "OCR": _get_value(item, "bibo:content"),
    }


# ---------------------------------------------------------------------------
# Configuration display
# ---------------------------------------------------------------------------

def display_config_panel(cfg: Config, repo: str, shard_size: str):
    """Display configuration in a beautiful Rich panel."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("API URL", cfg.API_URL)
    table.add_row("Repository", repo)
    table.add_row("Config Name", "documents")
    table.add_row("Max Shard Size", shard_size)
    table.add_row("Cache Directory", cfg.CACHE_DIR)
    table.add_row("Cache Duration", f"{cfg.CACHE_HOURS} hours")
    
    console.print(Panel(table, title="[bold blue]📄 IWAC Documents Upload Configuration", border_style="blue"))


# ---------------------------------------------------------------------------
# Pipeline: fetch → map → dataset → push
# ---------------------------------------------------------------------------

async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB"):
    # Display configuration panel
    display_config_panel(cfg, repo, shard_size)
    
    api = OmekaApiClient(cfg, use_cache=True)

    # 1. Fetch current Omeka items and map them
    console.print("\n[bold cyan]Step 1:[/bold cyan] Fetching items from Omeka API...")
    omeka_items_raw = await api.fetch_items(49)  # Documents (resource_class_id = 49)

    if not omeka_items_raw:
        console.print("[bold yellow]⚠ Warning:[/bold yellow] No items returned from Omeka API. Exiting.")
        return

    console.print(f"[green]✓[/green] Fetched {len(omeka_items_raw)} items from Omeka.")
    omeka_records_list = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Mapping Omeka documents", total=len(omeka_items_raw))
        for it in omeka_items_raw:
            try:
                record = await map_document(it, api)
                omeka_records_list.append(record)
            except Exception as e:
                logger.error(f"Error mapping item {it.get('o:id', 'Unknown ID')}: {e}", exc_info=True)
            progress.update(task, advance=1)
    
    if not omeka_records_list:
        console.print("[bold red]✗ Error:[/bold red] No records were successfully mapped. Exiting.")
        return
        
    new_omeka_df = pd.DataFrame(omeka_records_list)
    if 'o:id' not in new_omeka_df.columns or new_omeka_df['o:id'].isnull().any():
        console.print("[bold red]✗ Critical:[/bold red] 'o:id' column is missing or contains null values. Cannot proceed.")
        return
    new_omeka_df['o:id'] = new_omeka_df['o:id'].astype(str)  # Ensure consistent type for merging

    # 2. Load existing dataset from Hugging Face Hub
    console.print("\n[bold cyan]Step 2:[/bold cyan] Loading existing dataset from Hub...")
    existing_df = pd.DataFrame()
    hf_token_env = os.getenv("HF_TOKEN")
    hf_token_stored = get_token()
    token_to_use = hf_token_env if hf_token_env else hf_token_stored

    try:
        with console.status("[bold green]Loading existing dataset from Hub...", spinner="dots"):
            existing_ds = load_dataset(repo, name="documents", split="train", token=token_to_use, download_mode="force_redownload", verification_mode="no_checks")
            existing_df = existing_ds.to_pandas()
        
        if 'o:id' not in existing_df.columns or existing_df['o:id'].isnull().all():
            console.print("[yellow]⚠[/yellow] 'o:id' column missing or all null in existing Hub dataset. Treating as empty.")
            existing_df = pd.DataFrame() 
        else:
            # Ensure 'o:id' is string type for consistent merging
            existing_df['o:id'] = existing_df['o:id'].astype(str)
            console.print(f"[green]✓[/green] Loaded {len(existing_df)} records from {repo}")
            
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not load existing dataset (may be first run): {e}")
        existing_df = pd.DataFrame()  # Ensure it's an empty DataFrame on error

    # 3. Merge logic
    console.print("\n[bold cyan]Step 3:[/bold cyan] Merging datasets...")
    if existing_df.empty:
        console.print("[yellow]ℹ[/yellow] No existing data on Hub; using new Omeka data directly.")
        final_df = new_omeka_df
    else:
        console.print(f"[blue]→[/blue] Merging new Omeka data ({len(new_omeka_df)} records) with existing Hub data ({len(existing_df)} records).")
        
        # Identify columns in existing_df that are NOT in new_omeka_df. These are "extra" columns to preserve.
        extra_cols_to_preserve = [col for col in existing_df.columns if col not in new_omeka_df.columns]
        
        if extra_cols_to_preserve:
            console.print(f"[green]✓[/green] Preserving columns: {', '.join(extra_cols_to_preserve)}")
            # Select these columns plus 'o:id' from existing_df for the merge
            cols_from_existing_for_merge = ['o:id'] + extra_cols_to_preserve
            final_df = pd.merge(new_omeka_df, existing_df[cols_from_existing_for_merge], on='o:id', how='left')
        else:
            console.print("[yellow]ℹ[/yellow] No unique columns to preserve from existing dataset.")
            final_df = new_omeka_df

        console.print(f"[green]✓[/green] Merge complete: {len(final_df)} records, {len(final_df.columns)} columns")
        if extra_cols_to_preserve:
            for col_name in extra_cols_to_preserve:
                if col_name in final_df.columns:
                    nan_count = final_df[col_name].isnull().sum()
                    if nan_count > 0:
                        console.print(f"[yellow]ℹ[/yellow] Column '{col_name}' has {nan_count} null values (new items needing processing)")

    # 4. Conversion to Dataset and Push
    console.print("\n[bold cyan]Step 4:[/bold cyan] Preparing and pushing to Hub...")
    if not final_df.empty:
        # Final check for o:id integrity
        if 'o:id' not in final_df.columns or final_df['o:id'].isnull().any():
            console.print("[bold red]✗ Critical error:[/bold red] 'o:id' is missing or null. Aborting push.")
            await conn_manager.close()
            return

        # Convert nb_pages to nullable integer type to preserve integer dtype with null values
        if 'nb_pages' in final_df.columns:
            final_df['nb_pages'] = final_df['nb_pages'].astype('Int64')  # Nullable integer type

        ds = Dataset.from_pandas(final_df, preserve_index=False)
        
        # Display dataset summary table
        summary_table = Table(title="Dataset Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total Records", str(len(final_df)))
        summary_table.add_row("Total Columns", str(len(final_df.columns)))
        summary_table.add_row("Columns", ", ".join(final_df.columns[:5]) + ("..." if len(final_df.columns) > 5 else ""))
        console.print(summary_table)

        if os.getenv("HF_TOKEN") is None and not hf_utils.is_notebook():
            login()
        
        try:
            with console.status("[bold green]Pushing dataset to Hugging Face Hub...", spinner="dots"):
                ds.push_to_hub(repo, max_shard_size=shard_size, config_name="documents")
            
            # Success panel
            console.print(Panel(
                f"[bold green]✓ Dataset successfully published![/bold green]\n\n"
                f"Repository: [cyan]{repo}[/cyan]\n"
                f"Config: [cyan]documents[/cyan]\n"
                f"Records: [cyan]{len(final_df)}[/cyan]",
                title="🎉 Upload Complete",
                border_style="green"
            ))
        except Exception as e:
            console.print(Panel(
                f"[bold red]✗ Failed to push dataset[/bold red]\n\n{e}",
                title="Error",
                border_style="red"
            ))
            logger.error("Details of the exception:", exc_info=True)

    else:
        console.print("[yellow]ℹ[/yellow] Final dataset is empty. No push operation performed.")

    await conn_manager.close()


# ---------------------------------------------------------------------------
# Exécution CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload IWAC documents to Hugging Face Hub")
    parser.add_argument("--max-shard-size", default="1GB", help="Max shard size for Parquet (e.g., 500MB, 1GB)")
    args = parser.parse_args()

    asyncio.run(build_and_push(Config(), repo="fmadore/islam-west-africa-collection", shard_size=args.max_shard_size))
