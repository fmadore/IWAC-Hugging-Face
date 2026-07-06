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
import asyncio
import logging
import argparse
from typing import Dict, Any, List, Optional

import pandas as pd
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

# Adjust sys.path to include the parent directory for country_mapper / iwac_common imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from country_mapper import get_country_from_newspaper
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


# Config, Cache, ConnectionManager, async_retry and OmekaApiClient now live
# in iwac_common.omeka_client. islamic-publications and articles share the
# default ``.cache_omk`` cache directory.


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

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


    publisher_name = get_value(item, "dcterms:publisher") # Changed newspaper_name to publisher_name for clarity
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

    added_date = extract_added_date(item)

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
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": added_date, # Date when item was added to Omeka
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "iiif_manifest": iiif_manifest_url,
        "PDF": primary_url,
        "thumbnail": thumbnail_url, # Added thumbnail field
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "dcterms:creator"),
        "newspaper": publisher_name, # This was 'newspaper', for publications might be 'journal' or 'publisher'
        "country": country,
        "pub_date": get_value(item, "dcterms:date"),
        "issue": get_value(item, "bibo:issue"), # Added issue field
        "tableOfContents": get_value(item, "dcterms:tableOfContents"),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "nb_pages": to_int_or_none(get_value(item, "bibo:numPages")),
        "URL": extracted_fabio_url,
        "source": get_value(item, "dcterms:source"),
        "OCR": get_value(item, "bibo:content"),
        "OCR_is_public": is_content_public(item),
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
    
    api = OmekaApiClient(cfg, use_cache=True, console=console)

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

    # 2-3. Load existing Hub dataset and merge to preserve computed columns.
    console.print("\n[bold cyan]Steps 2-3:[/bold cyan] Loading and merging with existing Hub dataset...")
    token_to_use = resolve_hf_token()
    final_df = merge_with_hub_dataset(
        new_omeka_df,
        repo,
        config_name="publications",
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
                ds.push_to_hub(repo, max_shard_size=shard_size, config_name="publications", token=token_to_use)
            
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
        default=PRIVATE_REPO_ID,
        help="Dataset repository on the Hugging Face Hub (default: private full mirror)"
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
