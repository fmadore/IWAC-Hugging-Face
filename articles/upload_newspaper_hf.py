#!/usr/bin/env python3
"""
upload_newspaper_hf.py
=====================

Extrait les articles de journaux (resource_class_id = 36) depuis l'API Omeka S
d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le Hugging Face
Hub.

Usage
-----
    python articles/upload_newspaper_hf.py \
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
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union

# Add parent directory to path for country_mapper / iwac_common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import aiohttp
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from huggingface_hub import login, get_token, utils as hf_utils
import huggingface_hub
from country_mapper import get_country_from_newspaper
from iwac_common.omeka_client import (
    Config,
    OmekaApiClient,
    async_retry,
    conn_manager,
)

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


# Config, Cache, ConnectionManager, async_retry and OmekaApiClient live in
# iwac_common.omeka_client (imported above) and are used unchanged.


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

# Mapping for subjectivity score labels (Mistral uses resource:item instead of numeric)
SUBJECTIVITY_LABEL_TO_SCORE = {
    "Très objectif": 1,
    "Plutôt objectif": 2,
    "Mixte": 3,
    "Plutôt subjectif": 4,
    "Très subjectif": 5,
    "Non applicable": None,
}


def _get_subjectivity_score(item: Dict[str, Any], field: str) -> Optional[int]:
    """Extract subjectivity score, handling both numeric:integer and resource:item types."""
    if field not in item or item[field] is None:
        return None
    val = item[field]
    if isinstance(val, list) and val:
        val = val[0]
    if isinstance(val, dict):
        at_value = val.get("@value")
        if at_value is not None:
            try:
                return int(at_value)
            except (ValueError, TypeError):
                pass
        display_title = val.get("display_title", "")
        if display_title in SUBJECTIVITY_LABEL_TO_SCORE:
            return SUBJECTIVITY_LABEL_TO_SCORE[display_title]
    return None


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


async def map_newspaper_article(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka en dict plat pour HF datasets."""

    primary_url = ""
    if item.get("o:primary_media"):
        mid = item["o:primary_media"]["@id"].split("/")[-1]
        mdata = await api.fetch_media_data(mid)
        primary_url = mdata.get("o:original_url", "")

    newspaper_name = _join(item, "dcterms:publisher")
    country = get_country_from_newspaper(newspaper_name)

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
    # If none of the above, extracted_fabio_url remains ""    # Convert nb_pages to int
    nb_pages_str = _get_value(item, "bibo:numPages")
    nb_pages_int = None
    if nb_pages_str:
        try:
            nb_pages_int = int(nb_pages_str)
        except ValueError:
            logger.warning(
                f"Could not convert nb_pages '{nb_pages_str}' to int for item {item['o:id']}. Defaulting to null."
            )
            # nb_pages_int remains None

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
        "thumbnail": thumbnail_url, # Added thumbnail field
        "title": _get_value(item, "dcterms:title"),
        "author": _join(item, "dcterms:creator"),
        "newspaper": newspaper_name,
        "country": country, # Added country field
        "pub_date": _get_value(item, "dcterms:date"),
        "descriptionAI": _get_value(item, "bibo:shortDescription"),
        "subject": _join(item, "dcterms:subject"),
        "spatial": _get_value(item, "dcterms:spatial"),
        "language": _get_value(item, "dcterms:language"),
        "nb_pages": nb_pages_int, # Use converted integer value
        "URL": extracted_fabio_url, # Use the specifically extracted URL
        "source": _get_value(item, "dcterms:source"),
        "OCR": _get_value(item, "bibo:content"),
        # Gemini sentiment analysis
        "gemini_centralite_islam_musulmans": _get_value(item, "iwac:geminiCentralite"),
        "gemini_centralite_justification": _get_value(item, "iwac:geminiCentraliteJustification"),
        "gemini_polarite": _get_value(item, "iwac:geminiPolarite"),
        "gemini_polarite_justification": _get_value(item, "iwac:geminiPolariteJustification"),
        "gemini_subjectivite_score": _get_subjectivity_score(item, "iwac:geminiSubjectiviteScore"),
        "gemini_subjectivite_justification": _get_value(item, "iwac:geminiSubjectiviteJustification"),
        # ChatGPT sentiment analysis
        "chatgpt_centralite_islam_musulmans": _get_value(item, "iwac:chatgptCentralite"),
        "chatgpt_centralite_justification": _get_value(item, "iwac:chatgptCentraliteJustification"),
        "chatgpt_polarite": _get_value(item, "iwac:chatgptPolarite"),
        "chatgpt_polarite_justification": _get_value(item, "iwac:chatgptPolariteJustification"),
        "chatgpt_subjectivite_score": _get_subjectivity_score(item, "iwac:chatgptSubjectiviteScore"),
        "chatgpt_subjectivite_justification": _get_value(item, "iwac:chatgptSubjectiviteJustification"),
        # Mistral sentiment analysis
        "mistral_centralite_islam_musulmans": _get_value(item, "iwac:mistralCentralite"),
        "mistral_centralite_justification": _get_value(item, "iwac:mistralCentraliteJustification"),
        "mistral_polarite": _get_value(item, "iwac:mistralPolarite"),
        "mistral_polarite_justification": _get_value(item, "iwac:mistralPolariteJustification"),
        "mistral_subjectivite_score": _get_subjectivity_score(item, "iwac:mistralSubjectiviteScore"),
        "mistral_subjectivite_justification": _get_value(item, "iwac:mistralSubjectiviteJustification"),
    }


# ---------------------------------------------------------------------------
# Pipeline principale : fetch → map → dataset → push
# ---------------------------------------------------------------------------

def display_config_panel(cfg: Config, repo: str, shard_size: str):
    """Display configuration in a beautiful Rich panel."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("API URL", cfg.API_URL)
    table.add_row("Repository", repo)
    table.add_row("Max Shard Size", shard_size)
    table.add_row("Cache Directory", cfg.CACHE_DIR)
    table.add_row("Cache Duration", f"{cfg.CACHE_HOURS} hours")
    
    console.print(Panel(table, title="[bold blue]📰 IWAC Newspaper Upload Configuration", border_style="blue"))


async def build_and_push(cfg: Config, repo: str, shard_size: str = "1GB"):
    # Display configuration panel
    display_config_panel(cfg, repo, shard_size)
    
    api = OmekaApiClient(cfg, use_cache=True, console=console)

    # 1. Fetch current Omeka items and map them
    console.print("\n[bold cyan]Step 1:[/bold cyan] Fetching items from Omeka API...")
    omeka_items_raw = await api.fetch_items(36)  # Newspaper articles seulement

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
        task = progress.add_task("[cyan]Mapping Omeka articles", total=len(omeka_items_raw))
        for it in omeka_items_raw:
            try:
                record = await map_newspaper_article(it, api)
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
            # Corrected call to load_dataset
            existing_ds = load_dataset(repo, name="articles", split="train", token=token_to_use, download_mode="force_redownload", verification_mode="no_checks")
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
        existing_df = pd.DataFrame() # Ensure it's an empty DataFrame on error

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

    # Reorder columns: move AI sentiment columns to the very end
    sentiment_prefixes = ("gemini_", "chatgpt_", "mistral_")
    sentiment_cols = [c for c in final_df.columns if c.startswith(sentiment_prefixes)]
    other_cols = [c for c in final_df.columns if not c.startswith(sentiment_prefixes)]
    final_df = final_df[other_cols + sentiment_cols]

    # 4. Conversion to Dataset and Push
    console.print("\n[bold cyan]Step 4:[/bold cyan] Preparing and pushing to Hub...")
    if not final_df.empty:
        # Final check for o:id integrity
        if 'o:id' not in final_df.columns or final_df['o:id'].isnull().any():
            console.print("[bold red]✗ Critical error:[/bold red] 'o:id' is missing or null. Aborting push.")
            await conn_manager.close()
            return

        # Convert integer columns to nullable integer type to preserve dtype with null values
        for int_col in ['nb_pages', 'gemini_subjectivite_score', 'chatgpt_subjectivite_score', 'mistral_subjectivite_score']:
            if int_col in final_df.columns:
                final_df[int_col] = final_df[int_col].astype('Int64')

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
                ds.push_to_hub(repo, max_shard_size=shard_size, config_name="articles")
            
            # Success panel
            console.print(Panel(
                f"[bold green]✓ Dataset successfully published![/bold green]\n\n"
                f"Repository: [cyan]{repo}[/cyan]\n"
                f"Config: [cyan]articles[/cyan]\n"
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

    parser = argparse.ArgumentParser(description="Publie les articles de journaux IWAC sur le Hub HF")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille max d'un shard Parquet (ex. 500MB, 1GB)")
    args = parser.parse_args()

    asyncio.run(build_and_push(Config(), repo="fmadore/islam-west-africa-collection", shard_size=args.max_shard_size))
