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
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from huggingface_hub import login, get_token, utils as hf_utils
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
    is_content_public,
    to_int_or_none,
)
from iwac_common.hub_merge import merge_with_hub_dataset, resolve_hf_token
from iwac_common.repos import PRIVATE_REPO_ID

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


# Config, Cache, ConnectionManager, async_retry and OmekaApiClient now live
# in iwac_common.omeka_client.


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

    nb_pages_int = to_int_or_none(get_value(item, "bibo:numPages"))
    added_date = extract_added_date(item)

    # Fetch thumbnail URL and set IIIF manifest URL only if PDF exists
    session = await conn_manager.get()
    thumbnail_url = ""
    iiif_manifest_url = ""
    
    if primary_url:  # Only fetch IIIF data if there's a PDF
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"

    return {
        "o:id": item["o:id"],
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": added_date, # Date when item was added to Omeka
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "iiif_manifest": iiif_manifest_url,
        "PDF": primary_url,
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "dcterms:creator"),
        "contributor": get_value(item, "dcterms:contributor"),
        "country": country,
        "pub_date": get_value(item, "dcterms:date"),
        "descriptionAI": get_value(item, "bibo:shortDescription"),
        "abstract": get_value(item, "dcterms:abstract"),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "type": get_value(item, "dcterms:type"),
        "nb_pages": nb_pages_int,
        "source": get_value(item, "dcterms:source"),
        "rights": _get_label(item, "dcterms:rights"),
        "OCR": get_value(item, "bibo:content"),
        "OCR_is_public": is_content_public(item),
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
    
    api = OmekaApiClient(cfg, use_cache=True, console=console)

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

    # 2-3. Load existing Hub dataset and merge to preserve computed columns.
    console.print("\n[bold cyan]Steps 2-3:[/bold cyan] Loading and merging with existing Hub dataset...")
    token_to_use = resolve_hf_token()
    final_df = merge_with_hub_dataset(
        new_omeka_df,
        repo,
        config_name="documents",
        token=token_to_use,
        console=console,
    )

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
                ds.push_to_hub(repo, max_shard_size=shard_size, config_name="documents", token=token_to_use)
            
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
    parser.add_argument("--repo", default=PRIVATE_REPO_ID, help="Hugging Face repository to publish to (default: private full mirror)")
    parser.add_argument("--max-shard-size", default="1GB", help="Max shard size for Parquet (e.g., 500MB, 1GB)")
    args = parser.parse_args()

    asyncio.run(build_and_push(Config(CACHE_DIR=".cache_omk_documents"), repo=args.repo, shard_size=args.max_shard_size))
