#!/usr/bin/env python3
"""
calculate_word_count.py
=======================

Ajoute une colonne avec le nombre de mots à un dataset Hugging Face existant.
Le script charge un dataset depuis le repository Hugging Face 'fmadore/islam-west-africa-collection',
compte les mots dans la colonne 'OCR', et ajoute ces comptes dans une nouvelle
colonne nommée 'nb_mots'. Le dataset mis à jour est ensuite poussé vers le Hub.

L'utilisateur est invité à choisir la configuration ('articles', 'publications', 'documents' ou 'references')
à traiter.

Pour la configuration 'references', le script récupère le contenu bibo:content depuis l'API Omeka
(incluant les valeurs privées) pour calculer le nombre de mots, sans stocker le contenu complet.

Usage
-----
    python post-processing/calculate_word_count.py [--config articles|publications|documents|references] [-y]

Exemple:
    python post-processing/calculate_word_count.py            # menu interactif
    python post-processing/calculate_word_count.py --config articles -y   # non interactif

Variables d'environnement
---------------------
HF_TOKEN              Jeton d'accès personnel pour le Hugging Face Hub (sinon, une
                      connexion interactive sera demandée).
OMEKA_BASE_URL        Base URL de l'API Omeka (pour references)
OMEKA_KEY_IDENTITY    Identité de la clé Omeka (pour references, accès aux valeurs privées)
OMEKA_KEY_CREDENTIAL  Credential de la clé Omeka (pour references, accès aux valeurs privées)
"""
import argparse
import asyncio
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich import box
from rich.prompt import Prompt, Confirm

# Make ``post-processing/_common.py`` and ``iwac_common`` importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from _common import ensure_hf_token  # noqa: E402
from iwac_common.omeka_client import Config, OmekaApiClient, conn_manager  # noqa: E402

load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# Omeka client — shared infra from iwac_common + reference-specific fetches
# ---------------------------------------------------------------------------

# Reference resource classes
REFERENCE_RESOURCE_CLASSES = [35, 43, 88, 40, 82, 178, 52, 77, 305]

WORD_COUNT_CACHE_DIR = ".cache_word_count"


class ReferenceContentClient(OmekaApiClient):
    """Omeka client with per-item ``bibo:content`` fetches for references.

    Cache, retry, connection pooling and auth come from
    ``iwac_common.omeka_client``; only the content extraction (which needs
    the API key to see private values) is specific to this script.
    """

    async def fetch_item(self, item_id: int) -> Dict[str, Any]:
        """Fetch a single item by ID to get its bibo:content including private values."""
        return await self.request(f"items/{item_id}", {})

    async def fetch_items_content(self, item_ids: List[int]) -> Dict[int, str]:
        """Fetch bibo:content for multiple items concurrently."""
        results = {}

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(10)

        async def fetch_one(item_id: int) -> tuple:
            async with semaphore:
                try:
                    item = await self.fetch_item(item_id)
                    content = self._extract_content(item)
                    return (item_id, content)
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Failed to fetch item {item_id}: {e}")
                    return (item_id, "")

        tasks = [fetch_one(item_id) for item_id in item_ids]

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Fetching content from Omeka API", total=len(tasks))

            for coro in asyncio.as_completed(tasks):
                item_id, content = await coro
                results[item_id] = content
                progress.update(task, advance=1)

        return results

    def _extract_content(self, item: Dict[str, Any]) -> str:
        """Extract bibo:content from an item."""
        if "bibo:content" not in item or item["bibo:content"] is None:
            return ""
        val = item["bibo:content"]
        if isinstance(val, list):
            parts = [str(v.get("@value", "")) for v in val]
            return " ".join(filter(None, parts))
        if isinstance(val, dict):
            return val.get("@value", "")
        return str(val)


def configure_logging() -> None:
    """Configure le logging avec Rich."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )

def count_words(text: str | None) -> int:
    """
    Compte le nombre de mots dans une chaîne de caractères.
    
    Utilise une expression régulière pour identifier les mots composés de
    caractères alphanumériques, en ignorant la ponctuation et les espaces multiples.
    
    Args:
        text: Texte à analyser (peut être None)
        
    Returns:
        Nombre de mots trouvés (0 si le texte est None ou vide)
    """
    if not text:
        return 0
    # Utilise une expression régulière pour mieux gérer les séparateurs multiples
    # et la ponctuation simple attachée aux mots.
    words = re.findall(r"\b\w+\b", str(text).lower())
    return len(words)

def add_word_count_batch(batch: dict[str, list], text_col: str, count_col: str) -> dict[str, list]:
    """
    Applique le comptage de mots à un batch d'exemples.
    
    Args:
        batch: Dictionnaire contenant les colonnes du batch
        text_col: Nom de la colonne contenant le texte à analyser
        count_col: Nom de la colonne où stocker les comptes de mots
        
    Returns:
        Le batch avec la colonne de comptage ajoutée ou mise à jour
    """
    if text_col not in batch:
        # Si la colonne de texte n'est pas dans ce batch (peut arriver avec des datasets hétérogènes)
        # ou si le batch est vide, retourner le batch tel quel ou avec une colonne de comptes vide.
        if count_col not in batch:
            batch[count_col] = [0] * len(batch.get(next(iter(batch)), []))  # Crée une colonne de zéros
        return batch

    texts_in_batch: list = batch[text_col]
    word_counts = [count_words(text) for text in texts_in_batch]
    batch[count_col] = word_counts
    return batch

def main() -> None:
    """
    Fonction principale pour ajouter une colonne de comptage de mots au dataset.
    
    Charge le dataset depuis Hugging Face Hub, compte les mots dans la colonne OCR
    (ou depuis l'API Omeka pour les références), et pousse le dataset mis à jour vers le Hub.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    # Display script header
    console.print(Panel.fit(
        "[bold cyan]Word Count Calculator[/bold cyan]\n"
        "[dim]Add word count column to Hugging Face dataset[/dim]",
        border_style="cyan"
    ))

    parser = argparse.ArgumentParser(
        description="Ajoute/actualise la colonne 'nb_mots' d'un subset du dataset IWAC."
    )
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection")
    parser.add_argument(
        "--config",
        choices=["articles", "publications", "documents", "references"],
        default=None,
        help="Subset à traiter (évite le menu interactif)",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Recalculer sans confirmation quand 'nb_mots' existe déjà",
    )
    args = parser.parse_args()

    repo_id = args.repo
    count_column_name = "nb_mots"
    max_shard_size = "1GB"
    batch_size = 1000

    # --- Choix de la configuration (CLI ou menu interactif) ---
    valid_configs = ["articles", "publications", "documents", "references"]
    if args.config:
        config_name_choice = args.config
    else:
        try:
            config_name_choice = Prompt.ask(
                "[cyan]Quelle configuration traiter?[/cyan]",
                choices=valid_configs,
                default="articles"
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Opération annulée par l'utilisateur.")
            return
    
    console.print(f"[green]→[/green] Configuration sélectionnée: [bold]{config_name_choice}[/bold]")
    
    # For references, we fetch content from API; for others, we use the OCR column
    is_references = config_name_choice == "references"
    text_column_fixed = None if is_references else "OCR"

    # --- Authentification avec le Hub ---
    token = ensure_hf_token(console=console)
    console.print("[green]✓[/green] Authentification Hugging Face réussie")

    # --- For references, check Omeka API credentials ---
    if is_references:
        omeka_cfg = Config(CACHE_DIR=WORD_COUNT_CACHE_DIR)
        if not omeka_cfg.API_KEY_IDENTITY or not omeka_cfg.API_KEY_CREDENTIAL:
            console.print("[red]✗[/red] Les credentials Omeka (OMEKA_KEY_IDENTITY et OMEKA_KEY_CREDENTIAL) sont requis pour les références.")
            console.print("[yellow]ℹ[/yellow] Ces credentials sont nécessaires pour accéder aux valeurs privées de bibo:content.")
            return
        console.print("[green]✓[/green] Credentials Omeka configurés")

    # --- Chargement du dataset ---
    console.print(f"\n[blue]→[/blue] Chargement du dataset [bold]{repo_id}[/bold], configuration [bold]{config_name_choice}[/bold]...")
    try:
        with console.status("[bold green]Chargement en cours...", spinner="dots"):
            ds = load_dataset(repo_id, name=config_name_choice, split="train", token=token)
        console.print(f"[green]✓[/green] Dataset chargé: [bold]{len(ds):,}[/bold] lignes")
    except Exception as e:
        console.print(f"[red]✗[/red] Erreur lors du chargement du dataset: {e}")
        return

    # Display dataset info
    info_table = Table(title="Dataset Information", box=box.ROUNDED, show_header=False)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Nombre de lignes", f"{len(ds):,}")
    info_table.add_row("Nombre de colonnes", str(len(ds.column_names)))
    info_table.add_row("Colonnes", ", ".join(ds.column_names[:5]) + ("..." if len(ds.column_names) > 5 else ""))
    console.print(info_table)

    # For non-references, check that OCR column exists
    if not is_references and text_column_fixed not in ds.column_names:
        console.print(f"[red]✗[/red] La colonne de texte [bold]{text_column_fixed}[/bold] n'existe pas dans le dataset.")
        console.print(f"[yellow]ℹ[/yellow] Colonnes disponibles: {', '.join(ds.column_names)}")
        return

    # Check if o:id column exists (required for references)
    if is_references and "o:id" not in ds.column_names:
        console.print("[red]✗[/red] La colonne [bold]o:id[/bold] est requise pour les références mais n'existe pas.")
        return

    if count_column_name in ds.column_names:
        # Ask user if they want to recalculate existing word counts
        console.print(f"\n[yellow]⚠[/yellow] La colonne [bold]{count_column_name}[/bold] existe déjà.")
        if args.yes:
            console.print("[green]→[/green] Recalcul confirmé via --yes.")
        else:
            try:
                recalculate = Confirm.ask("Voulez-vous recalculer les comptes de mots existants?", default=False)
                if not recalculate:
                    console.print("[yellow]ℹ[/yellow] Opération annulée. Les comptes de mots existants sont conservés.")
                    return
                else:
                    console.print("[green]→[/green] Recalcul des comptes de mots confirmé.")
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠[/yellow] Opération annulée par l'utilisateur.")
                return

    # --- Process based on configuration type ---
    if is_references:
        # For references: fetch content from Omeka API and calculate word counts
        ds_processed = asyncio.run(process_references_word_count(ds, omeka_cfg, count_column_name))
        if ds_processed is None:
            return
    else:
        # For other configs: use the OCR column directly
        console.print(f"\n[blue]→[/blue] Calcul du nombre de mots pour la colonne [bold]{text_column_fixed}[/bold]...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Comptage des mots dans '{text_column_fixed}'", total=None)
            ds_processed = ds.map(
                add_word_count_batch,
                batched=True,
                batch_size=batch_size,
                fn_kwargs={
                    "text_col": text_column_fixed,
                    "count_col": count_column_name,
                },
            )
        
        console.print(f"[green]✓[/green] Comptage des mots terminé")
        sample_counts = ds_processed[count_column_name][:5]
        console.print(f"[dim]Aperçu (premiers 5): {sample_counts}[/dim]")

        # Convert to pandas to ensure proper integer typing, then back to Dataset
        with console.status("[bold blue]Conversion en format entier...", spinner="dots"):
            df = ds_processed.to_pandas()
            if count_column_name in df.columns:
                df[count_column_name] = df[count_column_name].astype('Int64')  # Nullable integer type
            ds_processed = Dataset.from_pandas(df, preserve_index=False)
        console.print(f"[green]✓[/green] Colonne [bold]{count_column_name}[/bold] convertie en type entier (Int64)")

        # --- Réorganisation des colonnes ---
        console.print(f"\n[blue]→[/blue] Réorganisation des colonnes pour placer [bold]{count_column_name}[/bold] après [bold]{text_column_fixed}[/bold]")
        current_columns = ds_processed.column_names
        
        if count_column_name in current_columns:
            current_columns.remove(count_column_name)
        
        try:
            ocr_index = current_columns.index(text_column_fixed)
            new_column_order = current_columns[:ocr_index+1] + [count_column_name] + current_columns[ocr_index+1:]
            ds_processed = ds_processed.select_columns(new_column_order)
            console.print(f"[green]✓[/green] Colonnes réorganisées")
            console.print(f"[dim]Nouvel ordre: {', '.join(ds_processed.column_names[:5])}{'...' if len(ds_processed.column_names) > 5 else ''}[/dim]")
        except ValueError:
            console.print(f"[yellow]⚠[/yellow] La colonne de référence [bold]{text_column_fixed}[/bold] n'a pas été trouvée. Le dataset sera poussé sans réorganisation.")

    # --- Push du dataset mis à jour vers le Hub ---
    console.print(f"\n[blue]→[/blue] Push du dataset mis à jour vers [bold]{repo_id}[/bold] (config: [bold]{config_name_choice}[/bold])...")
    try:
        with console.status("[bold green]Upload en cours...", spinner="dots"):
            commit_msg = f"Ajout/mise à jour de la colonne '{count_column_name}'"
            if is_references:
                commit_msg += " (calculée depuis bibo:content via API Omeka)"
            else:
                commit_msg += f" basée sur '{text_column_fixed}'"
            commit_msg += f" (config: {config_name_choice})"
            
            ds_processed.push_to_hub(
                repo_id,
                config_name=config_name_choice,
                token=token,
                max_shard_size=max_shard_size,
                commit_message=commit_msg,
            )
        console.print("[green]✓[/green] Dataset poussé avec succès vers le Hub")
        
        # Final summary
        source_info = "bibo:content (API Omeka)" if is_references else f"colonne '{text_column_fixed}'"
        summary_panel = Panel(
            f"[green]✓[/green] Colonne [bold]{count_column_name}[/bold] ajoutée avec succès\n"
            f"[dim]Configuration: {config_name_choice}\n"
            f"Source: {source_info}\n"
            f"Lignes traitées: {len(ds_processed):,}\n"
            f"Repository: {repo_id}[/dim]",
            title="[bold green]Opération terminée[/bold green]",
            border_style="green"
        )
        console.print(summary_panel)
    except Exception as e:
        console.print(f"[red]✗[/red] Erreur lors du push du dataset vers le Hub: {e}")


async def process_references_word_count(ds: Dataset, omeka_cfg: Config, count_column_name: str) -> Optional[Dataset]:
    """
    Process word count for references by fetching bibo:content from Omeka API.
    
    This function fetches the content from the Omeka API (including private values)
    for each reference item, calculates the word count, and adds it to the dataset
    without storing the actual content.
    
    Args:
        ds: The references dataset from Hugging Face Hub
        omeka_cfg: Configuration for Omeka API access
        count_column_name: Name of the column to store word counts
        
    Returns:
        The updated dataset with word counts, or None on failure
    """
    console.print("\n[blue]→[/blue] Récupération du contenu depuis l'API Omeka (incluant les valeurs privées)...")
    console.print("[dim]Note: Le contenu bibo:content n'est pas stocké, seul le nombre de mots est conservé.[/dim]")
    
    # Get all item IDs from the dataset
    df = ds.to_pandas()
    item_ids = df["o:id"].tolist()
    
    # Convert to integers (they might be strings)
    try:
        item_ids_int = [int(item_id) for item_id in item_ids]
    except (ValueError, TypeError) as e:
        console.print(f"[red]✗[/red] Erreur lors de la conversion des IDs: {e}")
        return None
    
    console.print(f"[blue]→[/blue] {len(item_ids_int)} références à traiter...")
    
    # Fetch content for all items
    api = ReferenceContentClient(omeka_cfg, use_cache=True, console=console)
    try:
        content_map = await api.fetch_items_content(item_ids_int)
    finally:
        await conn_manager.close()
    
    # Calculate word counts
    console.print("\n[blue]→[/blue] Calcul du nombre de mots...")
    word_counts = []
    items_with_content = 0
    
    for item_id in item_ids:
        item_id_int = int(item_id)
        content = content_map.get(item_id_int, "")
        wc = count_words(content)
        word_counts.append(wc)
        if content:
            items_with_content += 1
    
    console.print(f"[green]✓[/green] Comptage terminé: {items_with_content}/{len(item_ids)} références avec contenu")
    
    # Add word counts to dataframe
    df[count_column_name] = word_counts
    df[count_column_name] = df[count_column_name].astype('Int64')
    
    # Show sample
    sample_counts = df[count_column_name].head(5).tolist()
    console.print(f"[dim]Aperçu (premiers 5): {sample_counts}[/dim]")
    
    # Show statistics
    stats_table = Table(title="Word Count Statistics", box=box.ROUNDED)
    stats_table.add_column("Statistic", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Total references", f"{len(df):,}")
    stats_table.add_row("References with content", f"{items_with_content:,}")
    stats_table.add_row("References without content", f"{len(df) - items_with_content:,}")
    stats_table.add_row("Total words", f"{df[count_column_name].sum():,}")
    stats_table.add_row("Average words/reference", f"{df[count_column_name].mean():.1f}")
    stats_table.add_row("Max words", f"{df[count_column_name].max():,}")
    console.print(stats_table)
    
    # Convert back to Dataset
    ds_processed = Dataset.from_pandas(df, preserve_index=False)
    
    return ds_processed

if __name__ == "__main__":
    main()
