#!/usr/bin/env python3
"""
semantic_embedding.py
=====================

Ajoute des colonnes avec les embeddings sémantiques à un dataset Hugging Face 
existant, basées sur la colonne 'descriptionAI' (résumés français générés par Gemini).
Le script charge un dataset, calcule les embeddings à l'aide d'un modèle de 
sentence-transformers, et ajoute les résultats dans une nouvelle colonne.

L'utilisateur est invité à choisir la configuration ('articles', 'publications' ou 'documents').
Le nom de la nouvelle colonne est : "embedding_descriptionAI".

Usage
-----
    python post-processing/semantic_embedding.py [--repo MON_USER/MON_DATASET]

Exemple:
    python post-processing/semantic_embedding.py --repo fmadore/islam-west-africa-collection

Variables d'environnement
---------------------
HF_TOKEN   Jeton d'accès personnel pour le Hugging Face Hub (sinon, une
           connexion interactive sera demandée).

Dépendances supplémentaires
-------------------------
    pip install sentence-transformers torch datasets huggingface_hub rich
"""
import argparse
import logging
import os
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from huggingface_hub import get_token, login, dataset_info
from sentence_transformers import SentenceTransformer
import torch
import random
import numpy as np
import uuid

# Disable symlinks warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Rich console imports for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.prompt import Prompt, IntPrompt
from rich import box

# Initialize Rich console
console = Console()

# Configure logging with Rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger(__name__)

# Modèle d'embedding global (sera initialisé dans main())
embedding_model: Optional[SentenceTransformer] = None


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_available_configs(repo_id: str, token: str) -> List[str]:
    """
    Récupère la liste des configurations disponibles pour un dataset.
    
    Args:
        repo_id (str): ID du repository Hugging Face.
        token (str): Token d'authentification.
    
    Returns:
        List[str]: Liste des noms de configurations disponibles.
    """
    try:
        info = dataset_info(repo_id, token=token)
        if hasattr(info, 'config_names') and info.config_names:
            return info.config_names
        else:
            return ['articles', 'publications', 'documents']
    except Exception:
        return ['articles', 'publications', 'documents']


def choose_config(available_configs: List[str]) -> str:
    """
    Demande à l'utilisateur de choisir une configuration parmi celles disponibles.
    
    Args:
        available_configs (List[str]): Liste des configurations disponibles.
    
    Returns:
        str: Nom de la configuration choisie.
    """
    if len(available_configs) == 1:
        console.print(f"[yellow]ℹ[/yellow] Single configuration available: [cyan]{available_configs[0]}[/cyan]")
        return available_configs[0]
    
    # Display available configurations in a table
    table = Table(title="Available Configurations", box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Configuration", style="green")
    
    for i, config in enumerate(available_configs, 1):
        table.add_row(str(i), config)
    
    console.print(table)
    
    while True:
        try:
            choice = IntPrompt.ask(
                f"Choose a configuration",
                choices=[str(i) for i in range(1, len(available_configs) + 1)],
                show_choices=False
            )
            return available_configs[choice - 1]
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.[/yellow]")
            raise SystemExit(0)


def choose_update_mode() -> str:
    """
    Demande à l'utilisateur de choisir le mode de mise à jour des embeddings.
    
    Returns:
        str: Mode choisi ('all' pour tout recalculer, 'missing' pour seulement les valeurs manquantes)
    """
    console.print("\n[bold]Update Mode:[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Mode", style="green")
    table.add_column("Description", style="white")
    
    table.add_row("1", "missing", "Update only rows without embeddings (recommended)")
    table.add_row("2", "all", "Recalculate all embeddings (may take longer)")
    
    console.print(table)
    
    while True:
        try:
            choice = Prompt.ask("Choose update mode", choices=["1", "2"], default="1")
            return "missing" if choice == "1" else "all"
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.[/yellow]")
            raise SystemExit(0)

def compute_embeddings_batch(
    batch: Dict[str, List[Any]], 
    text_col: str, 
    embedding_col: str, 
    update_mode: str = "all"
) -> Dict[str, List[Any]]:
    """
    Calcule les embeddings pour un batch de textes.
    
    Args:
        batch (Dict[str, List[Any]]): Batch de données du dataset.
        text_col (str): Nom de la colonne contenant le texte source.
        embedding_col (str): Nom de la colonne où stocker les embeddings.
        update_mode (str): Mode de mise à jour ('all' ou 'missing').
    
    Returns:
        Dict[str, List[Any]]: Batch avec les embeddings ajoutés.
    """
    global embedding_model
    
    if embedding_model is None:
        raise RuntimeError("Embedding model not initialized")
    
    texts = batch[text_col]
    existing_embeddings = batch.get(embedding_col, [None] * len(texts))
    
    # Déterminer quels textes ont besoin d'embeddings
    indices_to_update: List[int] = []
    processed_texts: List[str] = []
    
    for i, (text, existing_emb) in enumerate(zip(texts, existing_embeddings)):
        should_process = False
        
        if update_mode == "all":
            should_process = True
        elif update_mode == "missing":
            # Traiter seulement si l'embedding n'existe pas ou est vide/invalide
            if (existing_emb is None or 
                existing_emb == [] or 
                (isinstance(existing_emb, list) and len(existing_emb) > 0 and all(x == 0.0 for x in existing_emb))):
                should_process = True
        
        if should_process:
            processed_texts.append(str(text) if text else "")
            indices_to_update.append(i)
    
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    
    # Si aucun texte à traiter, retourner le batch tel quel
    if not processed_texts:
        if embedding_col not in batch:
            batch[embedding_col] = [
                existing_emb if existing_emb is not None else [0.0] * embedding_dim 
                for existing_emb in existing_embeddings
            ]
        return batch
    
    # Calculer les embeddings pour les textes sélectionnés
    try:
        embeddings = embedding_model.encode(
            processed_texts, 
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        new_embeddings_list = [emb.tolist() for emb in embeddings]
        
    except Exception as e:
        logger.error(f"Error computing embeddings: {e}")
        new_embeddings_list = [[0.0] * embedding_dim for _ in processed_texts]
    
    # Construire la liste finale des embeddings
    if embedding_col not in batch:
        final_embeddings = [
            existing_emb if existing_emb is not None else [0.0] * embedding_dim 
            for existing_emb in existing_embeddings
        ]
    else:
        final_embeddings = list(batch[embedding_col])
    
    # Mettre à jour seulement les indices sélectionnés
    for idx, new_emb in zip(indices_to_update, new_embeddings_list):
        final_embeddings[idx] = new_emb
    
    batch[embedding_col] = final_embeddings
    return batch


def display_config_panel(repo_id: str, config_name: str, model_name: str, update_mode: str, batch_size: int):
    """Display configuration in a beautiful Rich panel."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Repository", repo_id)
    table.add_row("Configuration", config_name)
    table.add_row("Model", model_name)
    table.add_row("Update Mode", update_mode)
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Device", "CUDA" if torch.cuda.is_available() else "CPU")
    
    console.print(Panel(table, title="[bold blue]🔢 Semantic Embedding Configuration", border_style="blue"))


def display_text_stats(texts: List[Any], column_name: str) -> int:
    """Display statistics about the text column and return count of non-empty texts."""
    non_empty_texts = [t for t in texts if t is not None and str(t).strip() != ""]
    empty_count = len(texts) - len(non_empty_texts)
    
    table = Table(title=f"Source Column Statistics: '{column_name}'", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total entries", str(len(texts)))
    table.add_row("Non-empty entries", str(len(non_empty_texts)))
    table.add_row("Empty/None entries", str(empty_count))
    
    if non_empty_texts:
        avg_length = sum(len(str(t)) for t in non_empty_texts) / len(non_empty_texts)
        table.add_row("Avg. text length", f"{avg_length:.1f} characters")
    
    console.print(table)
    return len(non_empty_texts)


def display_embedding_stats(existing_embeddings: List[Any]) -> tuple[int, int]:
    """Display statistics about existing embeddings and return (valid, missing) counts."""
    valid_embeddings = 0
    empty_embeddings = 0
    
    for emb in existing_embeddings:
        if emb is None or emb == [] or (isinstance(emb, list) and len(emb) > 0 and all(x == 0.0 for x in emb)):
            empty_embeddings += 1
        else:
            valid_embeddings += 1
    
    table = Table(title="Existing Embeddings Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Valid embeddings", str(valid_embeddings))
    table.add_row("Missing/empty embeddings", str(empty_embeddings))
    percentage = (empty_embeddings / len(existing_embeddings) * 100) if existing_embeddings else 0
    table.add_row("To be processed", f"{percentage:.1f}%")
    
    console.print(table)
    return valid_embeddings, empty_embeddings

def main():
    global embedding_model

    parser = argparse.ArgumentParser(
        description="Add semantic embedding column ('embedding_descriptionAI') "
                   "to a Hugging Face dataset, based on the 'descriptionAI' column."
    )
    parser.add_argument(
        "--repo", 
        default="fmadore/islam-west-africa-collection", 
        help="Repository ID on Hugging Face Hub (e.g., user/dataset_name)."
    )
    parser.add_argument(
        "--model", 
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model (default: MiniLM-L12-v2, optimized for CPU)."
    )
    parser.add_argument(
        "--max-shard-size", 
        default="1GB", 
        help="Maximum Parquet shard size when pushing to Hub."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=50, 
        help="Batch size for the .map() processing."
    )
    
    args = parser.parse_args()

    repo_id = args.repo
    model_name = args.model
    text_column_name = "descriptionAI"
    embedding_column_name = "embedding_descriptionAI"
    max_shard_size = args.max_shard_size
    batch_size = args.batch_size

    # --- Configuration de la reproductibilité ---
    set_seed(42)

    # --- Authentification avec le Hub ---
    console.print("\n[bold cyan]Step 1:[/bold cyan] Authenticating with Hugging Face Hub...")
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        console.print("[yellow]ℹ[/yellow] HF token not found. Attempting interactive login...")
        try:
            login()
            token = get_token()
            if not token:
                console.print("[red]✗[/red] Interactive login failed. Please set HF_TOKEN or login manually.")
                return
        except Exception as e:
            console.print(f"[red]✗[/red] Login error: {e}")
            return
    console.print("[green]✓[/green] Authenticated successfully.")

    # --- Choix de la configuration ---
    console.print("\n[bold cyan]Step 2:[/bold cyan] Selecting configuration...")
    with console.status("[bold green]Fetching available configurations...", spinner="dots"):
        available_configs = get_available_configs(repo_id, token)
    
    config_name_choice = choose_config(available_configs)
    console.print(f"[green]✓[/green] Selected configuration: [cyan]{config_name_choice}[/cyan]")
    
    # --- Choix du mode de mise à jour ---
    update_mode = choose_update_mode()
    console.print(f"[green]✓[/green] Update mode: [cyan]{update_mode}[/cyan]")

    # --- Initialisation du modèle d'embedding ---
    console.print(f"\n[bold cyan]Step 3:[/bold cyan] Loading embedding model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[blue]→[/blue] Using device: [cyan]{device}[/cyan]")
        
        with console.status(f"[bold green]Loading model: {model_name}...", spinner="dots"):
            embedding_model = SentenceTransformer(model_name, device=device)
        
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        console.print(f"[green]✓[/green] Model loaded. Embedding dimension: [cyan]{embedding_dim}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load embedding model: {e}")
        return

    # --- Display configuration panel ---
    console.print()
    display_config_panel(repo_id, config_name_choice, model_name, update_mode, batch_size)

    # --- Chargement du dataset ---
    console.print(f"\n[bold cyan]Step 4:[/bold cyan] Loading dataset...")
    try:
        with console.status(f"[bold green]Loading '{repo_id}' (config: {config_name_choice})...", spinner="dots"):
            ds = load_dataset(repo_id, name=config_name_choice, split="train", token=token)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load dataset: {e}")
        return

    console.print(f"[green]✓[/green] Dataset loaded: [cyan]{len(ds)}[/cyan] rows")
    logger.debug(f"Available columns: {ds.column_names}")

    # --- Vérifications des colonnes ---
    if text_column_name not in ds.column_names:
        console.print(f"[red]✗[/red] Source column '{text_column_name}' not found in dataset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        return

    if embedding_column_name in ds.column_names:
        if update_mode == "all":
            console.print(f"[yellow]⚠[/yellow] Embedding column '{embedding_column_name}' exists and will be overwritten.")
        else:
            console.print(f"[yellow]ℹ[/yellow] Embedding column '{embedding_column_name}' exists. Only missing values will be computed.")
    else:
        console.print(f"[blue]→[/blue] Embedding column '{embedding_column_name}' will be created.")

    # --- Statistiques sur la colonne source ---
    console.print(f"\n[bold cyan]Step 5:[/bold cyan] Analyzing source data...")
    texts = ds[text_column_name]
    display_text_stats(texts, text_column_name)
    
    # --- Statistiques sur les embeddings existants (si mode 'missing') ---
    if update_mode == "missing" and embedding_column_name in ds.column_names:
        existing_embeddings = ds[embedding_column_name]
        valid_count, missing_count = display_embedding_stats(existing_embeddings)
        
        if missing_count == 0:
            console.print(Panel(
                "[green]All embeddings are already computed![/green]\n\n"
                "No processing needed.",
                title="ℹ Nothing to do",
                border_style="green"
            ))
            return

    # --- Application du calcul des embeddings ---
    console.print(f"\n[bold cyan]Step 6:[/bold cyan] Computing embeddings...")
    
    mode_desc = "all rows" if update_mode == "all" else "missing rows only"
    console.print(f"[blue]→[/blue] Processing {mode_desc}...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Computing embeddings", total=len(ds))
        
        def compute_with_progress(batch):
            result = compute_embeddings_batch(
                batch, 
                text_column_name, 
                embedding_column_name,
                update_mode
            )
            progress.update(task, advance=len(batch[text_column_name]))
            return result
        
        ds_processed = ds.map(
            compute_with_progress,
            batched=True,
            batch_size=batch_size,
            desc=None,  # Disable tqdm since we use Rich
            load_from_cache_file=False,  # Disable caching
            new_fingerprint=str(uuid.uuid4()),  # Bypass hashing to avoid warnings
        )

    console.print("[green]✓[/green] Embedding computation complete.")
    
    # --- Vérification des résultats ---
    console.print("\n[bold]Sample embeddings (first 3):[/bold]")
    embeddings_sample = ds_processed[embedding_column_name][:3]
    for i, emb in enumerate(embeddings_sample):
        if emb and len(emb) > 0 and not all(x == 0.0 for x in emb):
            console.print(f"  [cyan]#{i+1}[/cyan]: dim={len(emb)}, values=[{emb[0]:.4f}, {emb[1]:.4f}, ...]")
        else:
            console.print(f"  [cyan]#{i+1}[/cyan]: [dim]empty/zero[/dim]")

    # --- Réorganisation des colonnes ---
    insert_after_col = "descriptionAI"
    if insert_after_col in ds_processed.column_names:
        existing_columns = list(ds_processed.column_names)
        insert_index = existing_columns.index(insert_after_col) + 1
        
        new_columns = existing_columns[:insert_index]
        if embedding_column_name in existing_columns and embedding_column_name not in new_columns:
            new_columns.append(embedding_column_name)
        for col in existing_columns[insert_index:]:
            if col not in new_columns:
                new_columns.append(col)
        
        ds_processed = ds_processed.select_columns(new_columns)
        console.print(f"[blue]→[/blue] Columns reordered ('{embedding_column_name}' after '{insert_after_col}')")

    # --- Sauvegarde du dataset traité ---
    console.print(f"\n[bold cyan]Step 7:[/bold cyan] Pushing to Hugging Face Hub...")
    
    try:
        commit_message = (
            f"Add/update '{embedding_column_name}' embeddings using {model_name} "
            f"(config: {config_name_choice}, mode: {update_mode})"
        )
        
        with console.status("[bold green]Pushing dataset to Hub...", spinner="dots"):
            ds_processed.push_to_hub(
                repo_id=repo_id,
                config_name=config_name_choice,
                commit_message=commit_message,
                token=token,
                max_shard_size=max_shard_size,
            )
        
        # Success panel
        action = "updated" if embedding_column_name in ds.column_names else "created"
        console.print(Panel(
            f"[bold green]✓ Dataset successfully published![/bold green]\n\n"
            f"Repository: [cyan]{repo_id}[/cyan]\n"
            f"Configuration: [cyan]{config_name_choice}[/cyan]\n"
            f"Column: [cyan]{embedding_column_name}[/cyan] ({action})\n"
            f"Model: [cyan]{model_name}[/cyan]\n"
            f"Embedding dimension: [cyan]{embedding_dim}[/cyan]\n"
            f"Records: [cyan]{len(ds_processed)}[/cyan]",
            title="🎉 Upload Complete",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(Panel(
            f"[bold red]✗ Failed to push dataset[/bold red]\n\n{e}",
            title="Error",
            border_style="red"
        ))
        logger.error("Push error details:", exc_info=True)


if __name__ == "__main__":
    main()
