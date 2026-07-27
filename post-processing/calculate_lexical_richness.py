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
        [--config SUBSET] [--update-mode missing|all] [--dry-run] [-y]

Examples:
    python post-processing/calculate_lexical_richness.py          # fully interactive
    python post-processing/calculate_lexical_richness.py --config articles --update-mode missing   # headless
    python post-processing/calculate_lexical_richness.py --config articles -y --dry-run

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
import sys
import uuid
from collections import Counter
from typing import List, Dict, Any, Optional

import textstat

# Make ``post-processing/_common.py`` and ``iwac_common`` importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from _common import (  # noqa: E402
    PRIVATE_REPO_ID,
    choose_update_mode,
    ensure_hf_token,
    load_hub_dataset,
    map_with_progress,
    print_dry_run_panel,
    push_dataset,
    reorder_columns_after,
    resolve_config,
)
from iwac_common.text_utils import tokenize_words  # noqa: E402

# Disable symlinks warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich import box

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)

logger = logging.getLogger("lexical_richness")


def calculate_mattr(text: str, window_size: int = 50) -> Optional[float]:
    """Compute Moving Average Type-Token Ratio (MATTR).

    Unlike raw TTR, MATTR is not biased by text length because it uses a
    fixed-size sliding window. Tokenization is French-aware
    (``tokenize_words``): elided clitics are split off (``l'islam`` counts
    one type, not two), so type counts are not inflated by ``l``/``d``/``qu``
    fragments.

    Texts with fewer tokens than ``window_size`` return None — a plain TTR
    fallback would mix two incomparable metrics in the same column.

    Returns None if the text is missing, has no tokens, or is too short
    for MATTR.
    """
    if not text or not isinstance(text, str):
        return None

    tokens = tokenize_words(text)

    if not tokens:
        return None

    n = len(tokens)
    if n < window_size:
        return None  # too short for MATTR (no TTR fallback: incomparable metric)

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
        # With set_lang('fr'), textstat's flesch_reading_ease applies the
        # French-calibrated constants (Kandel–Moles adaptation) — verified in
        # textstat's source (get_lang_cfg(lang_root, "fre_base") etc.) — NOT
        # the English Flesch formula.
        return textstat.flesch_reading_ease(text)
    except Exception:
        return None


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
        error_counter: Mutable dict to accumulate counts. 'richness_too_short'
            counts non-empty texts shorter than the MATTR window (stored as
            None by design, not an error); 'readability_failed' counts
            readability computation failures.

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
                # Non-empty text without a MATTR value means it has fewer
                # tokens than the window — too short for MATTR, not an error.
                error_counter["richness_too_short"] += 1
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
        default=PRIVATE_REPO_ID,
        help="Repository ID on Hugging Face Hub (default: private full mirror)."
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
        "--config",
        default=None,
        help="Dataset configuration/subset to process (skips the interactive picker)."
    )
    parser.add_argument(
        "--update-mode",
        choices=["missing", "all"],
        default=None,
        help="'missing' computes only rows without values; 'all' recalculates "
             "everything (skips the interactive prompt)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics but do not push to Hub."
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmations/prompts by accepting defaults (update mode: "
             "missing). Combine with --config for a fully non-interactive run."
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

    # --- Configuration selection (CLI flag or interactive picker) ---
    console.print("\n[bold cyan]Step 2:[/bold cyan] Selecting configuration...")
    config_name_choice = resolve_config(repo_id, token=token, cli_config=args.config, console=console)
    console.print(f"[green]✓[/green] Selected configuration: [cyan]{config_name_choice}[/cyan]")

    # --- Update mode selection (CLI flag, --yes default, or interactive) ---
    if args.update_mode:
        update_mode = args.update_mode
    elif args.yes:
        update_mode = "missing"
        console.print("[green]→[/green] Update mode defaulted to 'missing' via --yes.")
    else:
        update_mode = choose_update_mode(console=console)
    console.print(f"[green]✓[/green] Update mode: [cyan]{update_mode}[/cyan]")

    # --- Display configuration ---
    console.print()
    display_config_panel(repo_id, config_name_choice, update_mode, batch_size, window_size, dry_run)

    # --- Load dataset ---
    console.print(f"\n[bold cyan]Step 3:[/bold cyan] Loading dataset...")
    ds = load_hub_dataset(repo_id, config_name_choice, token=token, console=console)
    if ds is None:
        return

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

    error_counter: Dict[str, int] = {"richness_too_short": 0, "readability_failed": 0}

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

    too_short = error_counter["richness_too_short"]
    total_failures = error_counter["readability_failed"]
    if too_short > 0:
        console.print(
            f"[yellow]ℹ[/yellow] {too_short} texts too short for MATTR "
            f"(< {window_size} tokens) — stored as None."
        )
    if total_failures > 0:
        console.print(
            f"[yellow]⚠[/yellow] Failures: {total_failures} readability (stored as None)."
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
        if too_short > 0:
            summary += f"\n[yellow]Too short for MATTR: {too_short}[/yellow]"
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
