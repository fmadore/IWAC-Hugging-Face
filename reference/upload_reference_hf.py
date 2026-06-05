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
import asyncio
import logging
import re
import time
import pandas as pd
import aiohttp
from typing import Dict, Any, List, Optional, Union

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

from iwac_common.omeka_client import (
    Config,
    OmekaApiClient as _BaseOmekaApiClient,
    async_retry,
    conn_manager,
)
from iwac_common.field_mappers import extract_added_date, get_value
from iwac_common.hub_merge import merge_with_hub_dataset, resolve_hf_token

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


# Config, Cache, ConnectionManager, async_retry and OmekaApiClient now live
# in iwac_common.omeka_client. We subclass OmekaApiClient below to preserve
# the resource-type label in the per-class success message and to add the
# ``fetch_all_reference_items`` helper.


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
# Reference-specific Omeka API client
# ---------------------------------------------------------------------------


class OmekaApiClient(_BaseOmekaApiClient):
    """Reference subset client: appends the resource-type label after the
    generic per-class confirmation, and adds ``fetch_all_reference_items``.
    """

    async def fetch_items(self, rcid: int) -> List[Dict[str, Any]]:
        items = await super().fetch_items(rcid)
        self.console.print(
            f"[green]✓[/green] (reference type: {RESOURCE_CLASS_MAPPING.get(rcid, 'Unknown')})"
        )
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
    volume_str = get_value(item, "bibo:volume")

    # Keep issue as string (can contain multiple values like "3|4")
    issue_str = get_value(item, "bibo:issue")

    # Convert edition to int
    edition_str = get_value(item, "bibo:edition")
    edition_int = ""
    if edition_str:
        try:
            edition_int = int(edition_str)
        except ValueError:
            logger.warning(
                f"Could not convert edition '{edition_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Convert chapter to int
    chapter_str = get_value(item, "bibo:chapter")
    chapter_int = ""
    if chapter_str:
        try:
            chapter_int = int(chapter_str)
        except ValueError:
            logger.warning(
                f"Could not convert chapter '{chapter_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Convert nb_pages to int
    nb_pages_str = get_value(item, "bibo:numPages")
    nb_pages_int = ""
    if nb_pages_str:
        try:
            nb_pages_int = int(nb_pages_str)
        except ValueError:
            logger.warning(
                f"Could not convert nb_pages '{nb_pages_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Convert page start/end to int
    page_start_str = get_value(item, "bibo:pageStart")
    page_start_int = ""
    if page_start_str:
        try:
            page_start_int = int(page_start_str)
        except ValueError:
            logger.warning(
                f"Could not convert pageStart '{page_start_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    page_end_str = get_value(item, "bibo:pageEnd")
    page_end_int = ""
    if page_end_str:
        try:
            page_end_int = int(page_end_str)
        except ValueError:
            logger.warning(
                f"Could not convert pageEnd '{page_end_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    added_date = extract_added_date(item)

    # Calculate word count from bibo:content (but don't include content in output)
    content_text = get_value(item, "bibo:content")
    nb_mots = count_words(content_text)

    return {
        "o:id": item["o:id"],
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "identifier": _get_iwac_identifier(item, "dcterms:identifier"),
        "added_date": added_date,
        "o:resource_class": _get_resource_class(item),
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "bibo:authorList"),
        "editor": get_value(item, "bibo:editorList"),
        "review_of": get_value(item, "bibo:reviewOf"),
        "publisher": get_value(item, "dcterms:publisher"),
        "pub_date": get_value(item, "dcterms:date"),
        "type": get_value(item, "dcterms:type"),
        "book_title": get_value(item, "dcterms:alternative"),
        "chapter": chapter_int,
        "volume": volume_str,
        "issue": issue_str,
        "abstract": get_value(item, "dcterms:abstract"),
        "edition": edition_int,
        "nb_pages": nb_pages_int,
        "page_start": page_start_int,
        "page_end": page_end_int,
        "extent": get_value(item, "dcterms:extent"),
        "is_part_of": get_value(item, "dcterms:isPartOf"),
        "provenance": get_value(item, "dcterms:provenance"),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "doi": get_value(item, "bibo:doi"),
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
    
    api = OmekaApiClient(cfg, use_cache=use_cache, console=console)

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

    # 2-3. Load existing Hub dataset and merge to preserve computed columns.
    # Reference uses an outer merge with explicit suffixes, drops a few legacy
    # columns, and runs an axis=1 ffill/bfill — encoded via shared helper params.
    console.print("\n[bold cyan]Steps 2-3:[/bold cyan] Loading and merging with existing Hub dataset...")
    token_to_use = resolve_hf_token()
    final_df = merge_with_hub_dataset(
        new_omeka_df,
        repo,
        config_name="references",
        token=token_to_use,
        how="outer",
        suffixes=("", "_old"),
        columns_to_exclude=("o:item_set", "o:media/file", "iiif_manifest", "thumbnail"),
        fill_after_merge=True,
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

    asyncio.run(build_and_push(Config(CACHE_DIR=".cache_omk_references"), repo=args.repo, shard_size=args.max_shard_size, use_cache=not args.no_cache))
