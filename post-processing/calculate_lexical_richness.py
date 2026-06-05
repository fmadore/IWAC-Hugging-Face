#!/usr/bin/env python3
"""
calculate_lexical_richness.py
=============================

Adds lexical richness (MATTR - Moving Average Type-Token Ratio) and readability
(Flesch Reading Ease) columns to an existing Hugging Face dataset, based on the
'OCR' column.

The user is prompted to choose the dataset configuration. Column names:
"Richesse_Lexicale_OCR" (MATTR) and "Lisibilite_OCR" (Flesch).

Usage
-----
    python post-processing/calculate_lexical_richness.py [--repo USER/DATASET]

Example:
    python post-processing/calculate_lexical_richness.py --repo fmadore/islam-west-africa-collection

Environment Variables
---------------------
HF_TOKEN   Personal access token for the Hugging Face Hub (otherwise,
           interactive login will be requested).

Dependencies
------------
    pip install textstat datasets huggingface_hub rich
"""
import argparse
import logging
import os
import re
import sys
import uuid
from collections import Counter
from typing import List, Dict, Any, Optional

from datasets import load_dataset
import textstat

# Make ``post-processing/_common.py`` importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import choose_config, ensure_hf_token, get_available_configs  # noqa: E402

# Disable symlinks warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.prompt import Prompt, IntPrompt
from rich import box

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger(__name__)


def calculate_mattr(text: str, window_size: int = 50) -> Optional[float]:
    """Compute Moving Average Type-Token Ratio (MATTR).

    Unlike raw TTR, MATTR is not biased by text length because it uses a
    fixed-size sliding window. Falls back to regular TTR when the text has
    fewer tokens than the window size.

    Returns None if the text is missing or has no tokens.
    """
    if not text or not isinstance(text, str):
        return None

    tokens = re.findall(r"\b\w+\b", text.lower())

    if not tokens:
        return None

    n = len(tokens)
    if n <= window_size:
        return len(set(tokens)) / n

    # Efficient sliding window using Counter
    window_counter = Counter(tokens[:window_size])
    ttr_sum = len(window_counter) / window_size
    num_windows = n - window_size + 1

    for i in range(1, num_windows):
        outgoing = tokens[i - 1]
        window_counter[outgoing] -= 1
        if window_counter[outgoing] == 0:
            del window_counter[outgoing]

        incoming = tokens[i + window_size - 1]
        window_counter[incoming] += 1

        ttr_sum += len(window_counter) / window_size

    return ttr_sum / num_windows


def calculate_readability(text: str) -> Optional[float]:
    """Compute the Flesch Reading Ease score.

    Returns None if the text is missing or computation fails.
    textstat.set_lang('fr') must be called beforehand.
    """
    if not text or not isinstance(text, str):
        return None
    try:
        return textstat.flesch_reading_ease(text)
    except Exception:
        return None


def choose_update_mode() -> str:
    """Prompt the user to choose the update mode."""
    console.print("\n[bold]Update Mode:[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Mode", style="green")
    table.add_column("Description", style="white")

    table.add_row("1", "missing", "Compute only rows without values (recommended)")
    table.add_row("2", "all", "Recalculate all values (may take longer)")

    console.print(table)

    while True:
        try:
            choice = Prompt.ask("Choose update mode", choices=["1", "2"], default="1")
            return "missing" if choice == "1" else "all"
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.[/yellow]")
            raise SystemExit(0)


def compute_metrics_batch(
    batch: Dict[str, List[Any]],
    *,
    text_col: str,
    richness_col: str,
    readability_col: str,
    update_mode: str,
    window_size: int,
    error_counter: Dict[str, int],
) -> Dict[str, List[Any]]:
    """Compute lexical richness and readability for a batch.

    Args:
        batch: Dataset batch.
        text_col: Name of the source text column.
        richness_col: Name of the MATTR column.
        readability_col: Name of the readability column.
        update_mode: 'all' or 'missing'.
        window_size: MATTR window size.
        error_counter: Mutable dict to accumulate error counts.

    Returns:
        Batch with metrics added.
    """
    texts = batch[text_col]
    existing_richness = batch.get(richness_col, [None] * len(texts))
    existing_readability = batch.get(readability_col, [None] * len(texts))

    richness_results = list(existing_richness)
    readability_results = list(existing_readability)

    for i, text in enumerate(texts):
        text_str = str(text) if text is not None else ""
        has_content = text is not None and text_str.strip()

        # Determine whether to process this row
        if update_mode == "missing":
            richness_needed = existing_richness[i] is None
            readability_needed = existing_readability[i] is None
        else:
            richness_needed = True
            readability_needed = True

        if richness_needed:
            result = calculate_mattr(text_str, window_size)
            if result is None and has_content:
                error_counter["richness_failed"] += 1
            richness_results[i] = result

        if readability_needed:
            result = calculate_readability(text_str)
            if result is None and has_content:
                error_counter["readability_failed"] += 1
            readability_results[i] = result

    batch[richness_col] = richness_results
    batch[readability_col] = readability_results
    return batch


def display_config_panel(
    repo_id: str, config_name: str, update_mode: str,
    batch_size: int, window_size: int, dry_run: bool
):
    """Display configuration in a Rich panel."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repository", repo_id)
    table.add_row("Configuration", config_name)
    table.add_row("Update Mode", update_mode)
    table.add_row("Batch Size", str(batch_size))
    table.add_row("MATTR Window Size", f"{window_size} tokens")
    if dry_run:
        table.add_row("Dry Run", "[yellow]YES — no changes will be pushed[/yellow]")

    console.print(Panel(table, title="[bold blue]Lexical Richness Configuration", border_style="blue"))


def display_text_stats(texts: List[Any], column_name: str) -> int:
    """Display statistics about the text column and return count of non-empty texts."""
    non_empty = [t for t in texts if t is not None and str(t).strip()]
    empty_count = len(texts) - len(non_empty)

    table = Table(title=f"Source Column Statistics: '{column_name}'", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total entries", str(len(texts)))
    table.add_row("Non-empty entries", str(len(non_empty)))
    table.add_row("Empty/None entries", str(empty_count))

    if non_empty:
        avg_length = sum(len(str(t)) for t in non_empty) / len(non_empty)
        table.add_row("Avg. text length", f"{avg_length:.1f} characters")

    console.print(table)
    return len(non_empty)


def display_existing_stats(
    ds, richness_col: str, readability_col: str
) -> tuple[int, int]:
    """Display stats on existing metrics. Returns (richness_missing, readability_missing)."""
    total = len(ds)

    if richness_col in ds.column_names:
        richness_vals = ds[richness_col]
        richness_missing = sum(1 for v in richness_vals if v is None)
    else:
        richness_missing = total

    if readability_col in ds.column_names:
        readability_vals = ds[readability_col]
        readability_missing = sum(1 for v in readability_vals if v is None)
    else:
        readability_missing = total

    table = Table(title="Existing Metrics Statistics", box=box.ROUNDED)
    table.add_column("Column", style="cyan")
    table.add_column("Valid", style="green")
    table.add_column("Missing", style="yellow")

    table.add_row(richness_col, str(total - richness_missing), str(richness_missing))
    table.add_row(readability_col, str(total - readability_missing), str(readability_missing))

    console.print(table)
    return richness_missing, readability_missing


def main():
    textstat.set_lang('fr')

    parser = argparse.ArgumentParser(
        description="Add lexical richness (MATTR) and readability (Flesch) columns "
                   "to a Hugging Face dataset, based on the 'OCR' column."
    )
    parser.add_argument(
        "--repo",
        default="fmadore/islam-west-africa-collection",
        help="Repository ID on Hugging Face Hub (e.g., user/dataset_name)."
    )
    parser.add_argument(
        "--max-shard-size",
        default="1GB",
        help="Maximum Parquet shard size when pushing to Hub."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for the .map() processing."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Window size for MATTR computation (default: 50 tokens)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics but do not push to Hub."
    )

    args = parser.parse_args()

    repo_id = args.repo
    text_column_name = "OCR"
    richness_column_name = "Richesse_Lexicale_OCR"
    readability_column_name = "Lisibilite_OCR"
    max_shard_size = args.max_shard_size
    batch_size = args.batch_size
    window_size = args.window_size
    dry_run = args.dry_run

    # --- Authentication ---
    console.print("\n[bold cyan]Step 1:[/bold cyan] Authenticating with Hugging Face Hub...")
    token = ensure_hf_token(console=console)
    console.print("[green]✓[/green] Authenticated successfully.")

    # --- Configuration selection ---
    console.print("\n[bold cyan]Step 2:[/bold cyan] Selecting configuration...")
    with console.status("[bold green]Fetching available configurations...", spinner="dots"):
        available_configs = get_available_configs(repo_id, token=token)

    config_name_choice = choose_config(available_configs, console=console)
    console.print(f"[green]✓[/green] Selected configuration: [cyan]{config_name_choice}[/cyan]")

    # --- Update mode selection ---
    update_mode = choose_update_mode()
    console.print(f"[green]✓[/green] Update mode: [cyan]{update_mode}[/cyan]")

    # --- Display configuration ---
    console.print()
    display_config_panel(repo_id, config_name_choice, update_mode, batch_size, window_size, dry_run)

    # --- Load dataset ---
    console.print(f"\n[bold cyan]Step 3:[/bold cyan] Loading dataset...")
    try:
        with console.status(f"[bold green]Loading '{repo_id}' (config: {config_name_choice})...", spinner="dots"):
            ds = load_dataset(repo_id, name=config_name_choice, split="train", token=token)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load dataset: {e}")
        return

    console.print(f"[green]✓[/green] Dataset loaded: [cyan]{len(ds)}[/cyan] rows")

    # --- Column checks ---
    if text_column_name not in ds.column_names:
        console.print(f"[red]✗[/red] Source column '{text_column_name}' not found in dataset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        return

    has_richness = richness_column_name in ds.column_names
    has_readability = readability_column_name in ds.column_names

    if has_richness or has_readability:
        existing = []
        if has_richness:
            existing.append(richness_column_name)
        if has_readability:
            existing.append(readability_column_name)
        if update_mode == "all":
            console.print(f"[yellow]⚠[/yellow] Existing columns will be overwritten: {', '.join(existing)}")
        else:
            console.print(f"[yellow]ℹ[/yellow] Existing columns found: {', '.join(existing)}. Only missing values will be computed.")
    else:
        console.print(f"[blue]→[/blue] Columns '{richness_column_name}' and '{readability_column_name}' will be created.")

    # --- Source data statistics ---
    console.print(f"\n[bold cyan]Step 4:[/bold cyan] Analyzing source data...")
    texts = ds[text_column_name]
    display_text_stats(texts, text_column_name)

    # --- Existing metrics statistics ---
    if update_mode == "missing" and (has_richness or has_readability):
        richness_missing, readability_missing = display_existing_stats(
            ds, richness_column_name, readability_column_name
        )

        if richness_missing == 0 and readability_missing == 0:
            console.print(Panel(
                "[green]All metrics are already computed![/green]\n\n"
                "No processing needed.",
                title="Nothing to do",
                border_style="green"
            ))
            return

    # --- Compute metrics ---
    console.print(f"\n[bold cyan]Step 5:[/bold cyan] Computing lexical metrics...")

    mode_desc = "all rows" if update_mode == "all" else "missing rows only"
    console.print(f"[blue]→[/blue] Processing {mode_desc}...")

    error_counter: Dict[str, int] = {"richness_failed": 0, "readability_failed": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Computing metrics", total=len(ds))

        def compute_with_progress(batch):
            result = compute_metrics_batch(
                batch,
                text_col=text_column_name,
                richness_col=richness_column_name,
                readability_col=readability_column_name,
                update_mode=update_mode,
                window_size=window_size,
                error_counter=error_counter,
            )
            progress.update(task, advance=len(batch[text_column_name]))
            return result

        ds_processed = ds.map(
            compute_with_progress,
            batched=True,
            batch_size=batch_size,
            desc=None,
            load_from_cache_file=False,
            new_fingerprint=str(uuid.uuid4()),
        )

    console.print("[green]✓[/green] Metrics computation complete.")

    total_failures = error_counter["richness_failed"] + error_counter["readability_failed"]
    if total_failures > 0:
        console.print(
            f"[yellow]⚠[/yellow] Failures: {error_counter['richness_failed']} richness, "
            f"{error_counter['readability_failed']} readability (stored as None)."
        )

    # --- Verify results ---
    console.print("\n[bold]Sample results (first 5):[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("MATTR", style="green", justify="right")
    table.add_column("Flesch", style="green", justify="right")

    for i in range(min(5, len(ds_processed))):
        r = ds_processed[richness_column_name][i]
        f = ds_processed[readability_column_name][i]
        table.add_row(
            str(i + 1),
            f"{r:.4f}" if r is not None else "[dim]None[/dim]",
            f"{f:.2f}" if f is not None else "[dim]None[/dim]",
        )

    console.print(table)

    # --- Reorder columns ---
    insert_after_col = "nb_mots"
    existing_columns = list(ds_processed.column_names)
    metric_cols = [richness_column_name, readability_column_name]

    if insert_after_col in existing_columns:
        ordered = []
        for col in existing_columns:
            if col in metric_cols:
                continue
            ordered.append(col)
            if col == insert_after_col:
                for mc in metric_cols:
                    if mc in existing_columns:
                        ordered.append(mc)

        if set(ordered) == set(existing_columns) and len(ordered) == len(existing_columns):
            ds_processed = ds_processed.select_columns(ordered)
            console.print(f"[blue]→[/blue] Columns reordered (metrics after '{insert_after_col}')")
        else:
            console.print(f"[yellow]⚠[/yellow] Column reordering skipped due to mismatch.")
    else:
        console.print(f"[yellow]ℹ[/yellow] Column '{insert_after_col}' not found; metrics appended at end.")

    # --- Push to Hub ---
    if dry_run:
        console.print(Panel(
            "[yellow]Dry run mode — no changes pushed to Hub.[/yellow]\n\n"
            f"Would have pushed [cyan]{len(ds_processed)}[/cyan] rows to "
            f"[cyan]{repo_id}[/cyan] (config: {config_name_choice}).",
            title="Dry Run Complete",
            border_style="yellow"
        ))
        return

    console.print(f"\n[bold cyan]Step 6:[/bold cyan] Pushing to Hugging Face Hub...")

    try:
        commit_message = (
            f"Add/update '{richness_column_name}' (MATTR, window={window_size}) and "
            f"'{readability_column_name}' (Flesch) from '{text_column_name}' "
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

        action = "updated" if (has_richness or has_readability) else "created"
        summary = (
            f"[bold green]Dataset successfully published![/bold green]\n\n"
            f"Repository: [cyan]{repo_id}[/cyan]\n"
            f"Configuration: [cyan]{config_name_choice}[/cyan]\n"
            f"Columns: [cyan]{richness_column_name}[/cyan], "
            f"[cyan]{readability_column_name}[/cyan] ({action})\n"
            f"MATTR window: [cyan]{window_size}[/cyan] tokens\n"
            f"Records: [cyan]{len(ds_processed)}[/cyan]"
        )
        if total_failures > 0:
            summary += f"\n[yellow]Total failures: {total_failures}[/yellow]"

        console.print(Panel(summary, title="Upload Complete", border_style="green"))

    except Exception as e:
        console.print(Panel(
            f"[bold red]Failed to push dataset[/bold red]\n\n{e}",
            title="Error",
            border_style="red"
        ))
        logger.error("Push error details:", exc_info=True)


if __name__ == "__main__":
    main()
