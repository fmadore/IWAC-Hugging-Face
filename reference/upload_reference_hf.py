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
import re
import time
import pandas as pd
import aiohttp
import aiofiles
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Type

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# Configuration & journalisation
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
    2193: 'Benin',
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
            page = 2
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Fetching pages..."),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"[cyan]Fetching pages for class {rcid}", total=None)
                while True:
                    batch = await self.fetch_items_page(rcid, page)
                    if not batch:
                        break
                    items.extend(batch)
                    progress.update(task, advance=1, description=f"[cyan]Class {rcid} - Page {page} fetched")
                    if len(batch) < per:
                        break
                    page += 1
        console.print(f"[green]✓[/green] {len(items)} items retrieved for class {rcid} ({RESOURCE_CLASS_MAPPING.get(rcid, 'Unknown')})")
        return items

    async def fetch_all_reference_items(self) -> List[Dict[str, Any]]:
        """Fetch all items from all reference resource classes"""
        all_items = []
        failed_classes = []
        for rcid in RESOURCE_CLASSES:
            try:
                items = await self.fetch_items(rcid)
                all_items.extend(items)
            except Exception as e:
                console.print(f"[red]✗[/red] Error fetching items for resource class {rcid} ({RESOURCE_CLASS_MAPPING.get(rcid, 'Unknown')}): {e}")
                failed_classes.append(rcid)
                continue
        
        if failed_classes:
            console.print(f"[yellow]⚠[/yellow] Failed to fetch {len(failed_classes)} resource class(es): {failed_classes}")
        
        return all_items

    async def fetch_media_data(self, media_id: str):
        return await self.request(f"media/{media_id}", {})


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """
    Compte le nombre de mots dans une chaîne de caractères.
    Les mots sont simplement séparés par des espaces.
    Retourne 0 si le texte est None ou vide.
    """
    if not text:
        return 0
    # Utilise une expression régulière pour mieux gérer les séparateurs multiples
    # et la ponctuation simple attachée aux mots.
    words = re.findall(r"\b\w+\b", str(text).lower())
    return len(words)


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


def _get_resource_class(item: Dict[str, Any]) -> str:
    """Extract resource class information and map to human-readable name"""
    if "o:resource_class" in item and isinstance(item["o:resource_class"], dict):
        class_id = item["o:resource_class"].get("o:id")
        if class_id and class_id in RESOURCE_CLASS_MAPPING:
            return RESOURCE_CLASS_MAPPING[class_id]
        elif class_id:
            return str(class_id)  # Return ID as string if not in mapping
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

    # Calculate word count from bibo:content (but don't include content in output)
    content_text = _get_value(item, "bibo:content")
    nb_mots = count_words(content_text)

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
        "nb_mots": nb_mots,
        "country": country,
    }


# ---------------------------------------------------------------------------
# Pipeline principale : fetch → map → dataset → push
# ---------------------------------------------------------------------------

def display_config_panel(cfg: Config, repo: str, shard_size: str, use_cache: bool = True):
    """Display configuration in a beautiful Rich panel."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("API URL", cfg.API_URL)
    table.add_row("Repository", repo)
    table.add_row("Config Name", "references")
    table.add_row("Max Shard Size", shard_size)
    table.add_row("Cache Directory", cfg.CACHE_DIR)
    table.add_row("Cache Duration", f"{cfg.CACHE_HOURS} hours")
    table.add_row("Cache Enabled", "[green]Yes[/green]" if use_cache else "[yellow]No (fresh fetch)[/yellow]")
    
    console.print(Panel(table, title="[bold blue]📚 IWAC References Upload Configuration", border_style="blue"))
    
    # Display resource class mapping table
    class_table = Table(title="Resource Classes to Fetch", box=box.SIMPLE)
    class_table.add_column("ID", style="cyan", justify="right")
    class_table.add_column("Type", style="green")
    for rc_id, rc_name in RESOURCE_CLASS_MAPPING.items():
        class_table.add_row(str(rc_id), rc_name)
    console.print(class_table)


async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB", use_cache: bool = True):
    start_time = time.time()
    
    # Display configuration panel
    display_config_panel(cfg, repo, shard_size, use_cache)
    
    api = OmekaApiClient(cfg, use_cache=use_cache)

    # 1. Fetch current Omeka items and map them
    console.print("\n[bold cyan]Step 1:[/bold cyan] Fetching reference items from Omeka API...")
    omeka_items_raw = await api.fetch_all_reference_items()

    if not omeka_items_raw:
        console.print("[bold yellow]⚠ Warning:[/bold yellow] No items returned from Omeka API. Exiting.")
        return

    console.print(f"[green]✓[/green] Fetched {len(omeka_items_raw)} reference items from Omeka.")
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
        task = progress.add_task("[cyan]Mapping Omeka references", total=len(omeka_items_raw))
        for it in omeka_items_raw:
            try:
                record = await map_reference(it, api)
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
    new_omeka_df['o:id'] = new_omeka_df['o:id'].astype(str) # Ensure consistent type for merging

    # 2. Load existing dataset from Hugging Face Hub
    console.print("\n[bold cyan]Step 2:[/bold cyan] Loading existing dataset from Hub...")
    existing_df = pd.DataFrame()
    hf_token_env = os.getenv("HF_TOKEN")
    hf_token_stored = get_token()
    token_to_use = hf_token_env if hf_token_env else hf_token_stored

    try:
        with console.status("[bold green]Loading existing dataset from Hub...", spinner="dots"):
            existing_ds = load_dataset(repo, name="references", split="train", token=token_to_use, download_mode="force_redownload", verification_mode="no_checks")
            existing_df = existing_ds.to_pandas()
        
        if 'o:id' not in existing_df.columns or existing_df['o:id'].isnull().all():
            console.print("[yellow]⚠[/yellow] 'o:id' column missing or all null in existing Hub dataset. Treating as empty.")
            existing_df = pd.DataFrame()
        else:
            existing_df['o:id'] = existing_df['o:id'].astype(str)
            console.print(f"[green]✓[/green] Loaded {len(existing_df)} records from {repo}")
            
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not load existing dataset (may be first run): {e}")
        existing_df = pd.DataFrame() # Ensure it's an empty DataFrame on error

    # 3. Merge logic
    console.print("\n[bold cyan]Step 3:[/bold cyan] Merging datasets...")
    if existing_df.empty:
        console.print("[yellow]ℹ[/yellow] No existing data on Hub; using new Omeka data directly.")
        final_df = new_omeka_df
    else:
        console.print(f"[blue]→[/blue] Merging new Omeka data ({len(new_omeka_df)} records) with existing Hub data ({len(existing_df)} records).")
        
        # Define columns to exclude from existing data (old columns we want to remove)
        columns_to_exclude = ['o:item_set', 'o:media/file', 'iiif_manifest', 'thumbnail']
        
        # Identify columns in existing_df that are NOT in new_omeka_df and NOT in the exclusion list
        extra_cols_to_preserve = [col for col in existing_df.columns 
                                 if col not in new_omeka_df.columns and col not in columns_to_exclude]
        
        if extra_cols_to_preserve:
            console.print(f"[green]✓[/green] Preserving columns: {', '.join(extra_cols_to_preserve)}")
            # Use outer merge to keep all records and preserve extra columns (excluding unwanted ones)
            final_df = pd.merge(new_omeka_df, existing_df[['o:id'] + extra_cols_to_preserve], on='o:id', how='outer', suffixes=('', '_old'))
            # Fill NaN values in new columns with data from existing columns where available
            final_df = final_df.ffill(axis=1).bfill(axis=1)
        else:
            console.print("[yellow]ℹ[/yellow] No unique columns to preserve from existing dataset.")
            final_df = new_omeka_df
            
        console.print(f"[dim]Excluded columns: {', '.join(columns_to_exclude)}[/dim]")
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

        # Ensure consistent data types for mixed columns
        # Convert numeric columns that might have mixed types to strings to avoid Arrow conversion issues
        mixed_type_columns = ['chapter', 'edition', 'nb_pages', 'page_start', 'page_end']
        for col in mixed_type_columns:
            if col in final_df.columns:
                # Convert all values to strings, replacing empty strings with empty strings (not 'nan')
                final_df[col] = final_df[col].astype(str).replace('nan', '')

        ds = Dataset.from_pandas(final_df, preserve_index=False)
        
        # Display dataset summary table
        summary_table = Table(title="Dataset Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total Records", str(len(final_df)))
        summary_table.add_row("Total Columns", str(len(final_df.columns)))
        summary_table.add_row("Columns", ", ".join(final_df.columns[:5]) + ("..." if len(final_df.columns) > 5 else ""))
        
        # Add resource class breakdown
        if 'o:resource_class' in final_df.columns:
            class_counts = final_df['o:resource_class'].value_counts()
            for class_name, count in class_counts.head(5).items():
                summary_table.add_row(f"  {class_name}", str(count))
        
        console.print(summary_table)

        if os.getenv("HF_TOKEN") is None and not hf_utils.is_notebook():
            login()
        
        try:
            with console.status("[bold green]Pushing dataset to Hugging Face Hub...", spinner="dots"):
                ds.push_to_hub(repo, config_name="references", max_shard_size=shard_size, token=token_to_use)
            
            # Success panel
            console.print(Panel(
                f"[bold green]✓ Dataset successfully published![/bold green]\n\n"
                f"Repository: [cyan]{repo}[/cyan]\n"
                f"Config: [cyan]references[/cyan]\n"
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
    
    # Display execution time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    console.print(f"\n[dim]Total execution time: {minutes}m {seconds}s[/dim]")


# ---------------------------------------------------------------------------
# Exécution CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publie les références bibliographiques IWAC sur le Hub HF")
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection", help="Repository Hugging Face où publier")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille max d'un shard Parquet (ex. 500MB, 1GB)")
    parser.add_argument("--no-cache", action="store_true", help="Disable API cache (force fresh fetch from Omeka)")
    args = parser.parse_args()

    asyncio.run(build_and_push(Config(), repo=args.repo, shard_size=args.max_shard_size, use_cache=not args.no_cache))
