#!/usr/bin/env python3
"""
calculate_word_count.py
=======================

Ajoute une colonne avec le nombre de mots à un dataset Hugging Face existant.
Le script charge un dataset depuis le repository Hugging Face 'fmadore/islam-west-africa-collection',
compte les mots dans la colonne 'OCR', et ajoute ces comptes dans une nouvelle
colonne nommée 'nb_mots'. Le dataset mis à jour est ensuite poussé vers le Hub.

L'utilisateur est invité à choisir la configuration ('articles', 'publications' ou 'documents')
à traiter.

Usage
-----
    python post-processing/calculate_word_count.py

Exemple:
    python post-processing/calculate_word_count.py
    (Le script demandera ensuite la configuration)

Variables d'environnement
---------------------
HF_TOKEN   Jeton d'accès personnel pour le Hugging Face Hub (sinon, une
           connexion interactive sera demandée).
"""
import logging
import os
import re
from datasets import load_dataset, Dataset
from huggingface_hub import get_token, login
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich import box
from rich.prompt import Prompt, Confirm

console = Console()

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
    
    Charge le dataset depuis Hugging Face Hub, compte les mots dans la colonne OCR,
    et pousse le dataset mis à jour vers le Hub.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    # Display script header
    console.print(Panel.fit(
        "[bold cyan]Word Count Calculator[/bold cyan]\n"
        "[dim]Add word count column to Hugging Face dataset[/dim]",
        border_style="cyan"
    ))

    # Hardcoded values
    repo_id = "fmadore/islam-west-africa-collection"
    text_column_fixed = "OCR"
    count_column_name = "nb_mots"
    max_shard_size = "1GB"
    batch_size = 1000

    # --- Choix de la configuration par l'utilisateur ---
    valid_configs = ["articles", "publications", "documents"]
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
    # text_column_fixed = "OCR" # Déjà défini plus haut

    # --- Authentification avec le Hub ---
    with console.status("[bold blue]Authentification Hugging Face...", spinner="dots"):
        token = os.getenv("HF_TOKEN") or get_token()
        if not token:
            console.print("[yellow]ℹ[/yellow] Token Hugging Face non trouvé. Tentative de connexion interactive.")
            login()
            token = get_token()
            if not token:
                console.print("[red]✗[/red] Échec de la connexion au Hugging Face Hub.")
                return
    console.print("[green]✓[/green] Authentification réussie")

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

    if text_column_fixed not in ds.column_names:
        console.print(f"[red]✗[/red] La colonne de texte [bold]{text_column_fixed}[/bold] n'existe pas dans le dataset.")
        console.print(f"[yellow]ℹ[/yellow] Colonnes disponibles: {', '.join(ds.column_names)}")
        return

    if count_column_name in ds.column_names:
        # Ask user if they want to recalculate existing word counts
        console.print(f"\n[yellow]⚠[/yellow] La colonne [bold]{count_column_name}[/bold] existe déjà.")
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

    # --- Application du comptage de mots ---
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
    
    # Enlever la colonne de comptage de sa position actuelle (généralement à la fin)
    # pour la réinsérer au bon endroit. Si elle n'y est pas pour une raison quelconque, pas de souci.
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
            ds_processed.push_to_hub(
                repo_id,
                config_name=config_name_choice,
                token=token,
                max_shard_size=max_shard_size,
                commit_message=f"Ajout de la colonne de comptage de mots '{count_column_name}' basée sur '{text_column_fixed}' (config: {config_name_choice})",
            )
        console.print("[green]✓[/green] Dataset poussé avec succès vers le Hub")
        
        # Final summary
        summary_panel = Panel(
            f"[green]✓[/green] Colonne [bold]{count_column_name}[/bold] ajoutée avec succès\n"
            f"[dim]Configuration: {config_name_choice}\n"
            f"Lignes traitées: {len(ds_processed):,}\n"
            f"Repository: {repo_id}[/dim]",
            title="[bold green]Opération terminée[/bold green]",
            border_style="green"
        )
        console.print(summary_panel)
    except Exception as e:
        console.print(f"[red]✗[/red] Erreur lors du push du dataset vers le Hub: {e}")

if __name__ == "__main__":
    main()
