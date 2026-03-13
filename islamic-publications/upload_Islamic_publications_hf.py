#!/usr/bin/env python3
"""
upload_Islamic_publications_hf.py
=================================

Extracts Islamic publications (resource_class_id = 60) from the IWAC Omeka S API,
converts them to Arrow/Parquet dataset format, and pushes to Hugging Face Hub.

Usage
-----
    python upload_Islamic_publications_hf.py \
        --repo fmadore/islam-west-africa-collection \
        --max-shard-size 1GB

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
import json
import io
import gzip
import hashlib
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Type

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

# Adjust sys.path to include the parent directory for country_mapper import
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from country_mapper import get_country_from_newspaper

# Disable symlinks warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------------------------
# Configuration & Logging
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


@async_retry(max_tries=3, exceptions=(aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError))
async def fetch_iiif_thumbnail_url(omeka_id: Union[str, int], session: aiohttp.ClientSession) -> str:
    """Fetches and extracts the thumbnail URL from an IIIF manifest."""
    # Updated to match newspaper articles structure
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

def _to_int(value: str) -> Optional[int]:
    """Safely convert a string to an integer, returning None if conversion fails."""
    if not value or not value.strip():
        return None
    try:
        return int(value.strip())
    except (ValueError, TypeError):
        return None


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
        try:
            media_id_url = item["o:primary_media"]["@id"]
            media_id = media_id_url.split("/")[-1]
            media_data = await api.fetch_media_data(media_id) # Ensure api object is passed and used
            primary_url = media_data.get("o:original_url", "")
        except Exception as e:
            logger.error(f"Error fetching primary media for item {item.get('o:id')}: {e}")
            primary_url = ""


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
        try:
            thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
            iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"
        except Exception as e:
            logger.error(f"Error fetching IIIF data for item {item['o:id']}: {e}")
            thumbnail_url = ""
            iiif_manifest_url = ""

    return {
        "o:id": item["o:id"],
        "identifier": _get_value(item, "dcterms:identifier"),
        "added_date": added_date, # Date when item was added to Omeka
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "iiif_manifest": iiif_manifest_url,
        "PDF": primary_url,
        "thumbnail": thumbnail_url, # Added thumbnail field
        "title": _get_value(item, "dcterms:title"),
        "author": _join(item, "dcterms:creator"),
        "newspaper": publisher_name, # This was 'newspaper', for publications might be 'journal' or 'publisher'
        "country": country,
        "pub_date": _get_value(item, "dcterms:date"),
        "issue": _get_value(item, "bibo:issue"), # Added issue field
        "tableOfContents": _get_value(item, "dcterms:tableOfContents"),
        "subject": _join(item, "dcterms:subject"),
        "spatial": _get_value(item, "dcterms:spatial"),
        "language": _get_value(item, "dcterms:language"),
        "nb_pages": _to_int(_get_value(item, "bibo:numPages")),
        "URL": extracted_fabio_url,
        "source": _get_value(item, "dcterms:source"),
        "OCR": _get_value(item, "bibo:content"),
    }


# ---------------------------------------------------------------------------
# Pipeline: fetch → map → dataset → push
# ---------------------------------------------------------------------------

def display_config_panel(cfg: Config, repo: str, shard_size: str):
    """Display configuration in a beautiful Rich panel."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("API URL", cfg.API_URL)
    table.add_row("Repository", repo)
    table.add_row("Config Name", "publications")
    table.add_row("Max Shard Size", shard_size)
    table.add_row("Cache Directory", cfg.CACHE_DIR)
    table.add_row("Cache Duration", f"{cfg.CACHE_HOURS} hours")
    
    console.print(Panel(table, title="[bold blue]📚 IWAC Islamic Publications Upload Configuration", border_style="blue"))


async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB"):
    # Display configuration panel
    display_config_panel(cfg, repo, shard_size)
    
    api = OmekaApiClient(cfg, use_cache=True)

    # 1. Fetch current Omeka items and map them
    console.print("\n[bold cyan]Step 1:[/bold cyan] Fetching items from Omeka API...")
    omeka_items_raw = await api.fetch_items(60)  # Islamic publications only

    if not omeka_items_raw:
        console.print("[bold yellow]⚠ Warning:[/bold yellow] No items returned from Omeka API.")
        # Try to get schema from existing dataset or define a default one
        final_df = pd.DataFrame()
        try:
            with console.status("[bold green]Loading existing dataset schema...", spinner="dots"):
                existing_ds = load_dataset(repo, name="publications", split="train", token=token_to_use, download_mode="force_redownload", verification_mode="no_checks")
            if existing_ds.num_rows > 0:
                final_df = pd.DataFrame(columns=existing_ds.column_names)
                console.print(f"[green]✓[/green] Using existing dataset schema with {len(existing_ds.column_names)} columns")
            else:
                console.print("[yellow]ℹ[/yellow] Existing dataset on Hub is empty. Creating empty dataset with default schema.")
                expected_cols = [
                    "o:id", "identifier", "added_date", "iwac_url", "iiif_manifest", "PDF", "thumbnail", "title", "author",
                    "newspaper", "country", "pub_date", "issue", "tableOfContents", "subject", "spatial",
                    "language", "nb_pages", "URL", "source", "OCR"
                ]
                final_df = pd.DataFrame(columns=expected_cols)

        except Exception as e_load_meta:
            console.print(f"[yellow]⚠[/yellow] Could not load existing dataset schema: {e_load_meta}")
            expected_cols = [
                "o:id", "identifier", "added_date", "iwac_url", "iiif_manifest", "PDF", "thumbnail", "title", "author",
                "newspaper", "country", "pub_date", "issue", "subject", "spatial",
                "language", "nb_pages", "URL", "source", "OCR"
            ]
            final_df = pd.DataFrame(columns=expected_cols)
        
        if 'o:id' in final_df.columns:
            final_df['o:id'] = final_df['o:id'].astype(str)

        if not final_df.empty or (final_df.empty and not omeka_items_raw):
            console.print("[blue]→[/blue] Preparing to push empty dataset to the Hub...")
            dataset_to_push = Dataset.from_pandas(final_df, preserve_index=False)
            
            hf_token_env = os.getenv("HF_TOKEN")
            hf_token_stored = get_token()
            token_to_use = hf_token_env if hf_token_env else hf_token_stored
            if not token_to_use and not hf_utils.is_notebook():
                login()
                token_to_use = get_token()

            try:
                with console.status("[bold green]Pushing empty dataset...", spinner="dots"):
                    dataset_to_push.push_to_hub(
                        repo,
                        config_name="publications",
                        token=token_to_use,
                        max_shard_size=shard_size,
                        commit_message="Dataset cleared or initialized as empty (no Omeka items found)"
                    )
                console.print("[green]✓[/green] Successfully pushed empty dataset to the Hub.")
            except Exception as e_push_empty:
                console.print(Panel(
                    f"[bold red]✗ Failed to push empty dataset[/bold red]\n\n{e_push_empty}",
                    title="Error",
                    border_style="red"
                ))
        await conn_manager.close()
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
        task = progress.add_task("[cyan]Mapping Islamic publications", total=len(omeka_items_raw))
        for it in omeka_items_raw:
            try:
                record = await map_islamic_publication_item(it, api)
                omeka_records_list.append(record)
            except Exception as e:
                logger.error(f"Error mapping item {it.get('o:id', 'Unknown ID')}: {e}", exc_info=True)
            progress.update(task, advance=1)
    
    if not omeka_records_list:
        console.print("[bold red]✗ Error:[/bold red] No records were successfully mapped. Exiting.")
        await conn_manager.close()
        return
        
    new_omeka_df = pd.DataFrame(omeka_records_list)
    if 'o:id' not in new_omeka_df.columns or new_omeka_df['o:id'].isnull().any():
        console.print("[bold red]✗ Critical:[/bold red] 'o:id' column is missing or contains null values. Cannot proceed.")
        await conn_manager.close()
        return
    new_omeka_df['o:id'] = new_omeka_df['o:id'].astype(str)

    # 2. Load existing dataset from Hugging Face Hub
    console.print("\n[bold cyan]Step 2:[/bold cyan] Loading existing dataset from Hub...")
    existing_df = pd.DataFrame()
    hf_token_env = os.getenv("HF_TOKEN")
    hf_token_stored = get_token()
    token_to_use = hf_token_env if hf_token_env else hf_token_stored

    try:
        with console.status("[bold green]Loading existing dataset from Hub...", spinner="dots"):
            existing_ds = load_dataset(repo, name="publications", split="train", token=token_to_use, download_mode="force_redownload", verification_mode="no_checks")
            existing_df = existing_ds.to_pandas()
        
        if 'o:id' not in existing_df.columns or existing_df['o:id'].isnull().all():
            console.print("[yellow]⚠[/yellow] 'o:id' column missing or all null in existing Hub dataset. Treating as empty.")
            existing_df = pd.DataFrame() 
        else:
            existing_df['o:id'] = existing_df['o:id'].astype(str)
            console.print(f"[green]✓[/green] Loaded {len(existing_df)} records from {repo}")
            
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not load existing dataset (may be first run): {e}")
        existing_df = pd.DataFrame()

    # 3. Merge logic
    console.print("\n[bold cyan]Step 3:[/bold cyan] Merging datasets...")
    if existing_df.empty:
        console.print("[yellow]ℹ[/yellow] No existing data on Hub; using new Omeka data directly.")
        final_df = new_omeka_df.copy()
    else:
        console.print(f"[blue]→[/blue] Merging new Omeka data ({len(new_omeka_df)} records) with existing Hub data ({len(existing_df)} records).")
        
        # Identify columns in existing_df that are NOT in new_omeka_df
        extra_cols_to_preserve = [col for col in existing_df.columns if col not in new_omeka_df.columns]
        
        if extra_cols_to_preserve:
            console.print(f"[green]✓[/green] Preserving columns: {', '.join(extra_cols_to_preserve)}")
            cols_from_existing_for_merge = ['o:id'] + extra_cols_to_preserve
            final_df = pd.merge(new_omeka_df, existing_df[cols_from_existing_for_merge], on='o:id', how='left')
        else:
            console.print("[yellow]ℹ[/yellow] No unique columns to preserve from existing dataset.")
            final_df = new_omeka_df.copy()

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
            final_df['nb_pages'] = final_df['nb_pages'].astype('Int64')

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
                ds.push_to_hub(repo, max_shard_size=shard_size, config_name="publications")
            
            # Success panel
            console.print(Panel(
                f"[bold green]✓ Dataset successfully published![/bold green]\n\n"
                f"Repository: [cyan]{repo}[/cyan]\n"
                f"Config: [cyan]publications[/cyan]\n"
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
# CLI Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload IWAC Islamic Publications to Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--repo",
        default="fmadore/islam-west-africa-collection",
        help="Dataset repository on the Hugging Face Hub"
    )
    parser.add_argument(
        "--max-shard-size",
        default="1GB",
        help="Maximum shard size for Parquet files (e.g., 500MB, 1GB)"
    )
    args = parser.parse_args()

    try:
        asyncio.run(build_and_push(Config(), repo=args.repo, shard_size=args.max_shard_size))
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Operation cancelled by user.")
    except Exception as e:
        console.print(Panel(
            f"[bold red]✗ Unexpected error[/bold red]\n\n{e}",
            title="Fatal Error",
            border_style="red"
        ))
        raise
