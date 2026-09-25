#!/usr/bin/env python3
"""
calculate_word_count.py
=======================

Adds or refreshes the ``nb_mots`` column of an IWAC subset from its ``OCR``
column, and pushes the subset back to the private Hub repository.

``nb_mots`` has a single definition, :func:`iwac_common.text_utils.count_words`
(French-elision-aware: ``l'islam`` is one word). The references and
audiovisual upload mappers use the same function, so a value is the same
whichever of the two wrote it last — which was not the case while the
mappers counted ``\\b\\w+\\b`` matches.

``references`` is counted like every other subset. The private repository
holds its full text in ``OCR``, so the per-item Omeka fetch this script used to
make for references (to see private ``bibo:content``) is no longer needed.

Usage
-----
    python post-processing/calculate_word_count.py            # interactive
    python post-processing/calculate_word_count.py --config articles -y
    python post-processing/calculate_word_count.py --config references --update-mode missing

Environment
-----------
HF_TOKEN    Hugging Face token (otherwise an interactive login is requested).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Make ``post-processing/_common.py`` and ``iwac_common`` importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from _common import (  # noqa: E402
    PRIVATE_REPO_ID,
    ensure_hf_token,
    load_hub_dataset,
    map_with_progress,
    print_dry_run_panel,
    push_dataset,
    reorder_columns_after,
    resolve_config,
)
from iwac_common.text_utils import count_words as _count_words  # noqa: E402

load_dotenv()

console = Console()

TEXT_COLUMN = "OCR"
COUNT_COLUMN = "nb_mots"
WORD_COUNT_SUBSETS = ["articles", "publications", "documents", "references", "audiovisual"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def count_words(text: str | None) -> int:
    """Word count of ``text`` — see :func:`iwac_common.text_utils.count_words`."""
    return _count_words(text)


def add_word_count_batch(
    batch: Dict[str, List[Any]], text_col: str, count_col: str, update_mode: str = "all"
) -> Dict[str, List[Any]]:
    """Fill ``count_col`` for one ``.map(batched=True)`` batch.

    ``update_mode="all"`` recounts every row; ``"missing"`` only fills rows
    whose count is null and keeps the existing values.
    """
    if text_col not in batch:
        # An empty batch (or a subset without the text column): guard against
        # next(iter(batch)) raising StopIteration.
        if count_col not in batch:
            first_col = next(iter(batch), None)
            batch[count_col] = [0] * (len(batch[first_col]) if first_col is not None else 0)
        return batch

    texts = batch[text_col]
    existing = batch.get(count_col) if update_mode == "missing" else None
    if existing is not None:
        batch[count_col] = [
            existing[i] if existing[i] is not None else count_words(text)
            for i, text in enumerate(texts)
        ]
    else:
        batch[count_col] = [count_words(text) for text in texts]
    return batch


def print_summary(counts: List[Any], config_name: str) -> None:
    values = [v for v in counts if v is not None]
    table = Table(title=f"Word counts — {config_name}", box=box.ROUNDED)
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Rows", f"{len(counts):,}")
    table.add_row("Rows with text", f"{sum(1 for v in values if v):,}")
    table.add_row("Total words", f"{sum(values):,}")
    if values:
        table.add_row("Mean words/row", f"{sum(values) / len(values):.1f}")
        table.add_row("Max words", f"{max(values):,}")
    console.print(table)


def main() -> int:
    configure_logging()
    console.print(Panel.fit(
        "[bold cyan]Word Count Calculator[/bold cyan]\n"
        f"[dim]{COUNT_COLUMN} from {TEXT_COLUMN} (elision-aware)[/dim]",
        border_style="cyan",
    ))

    parser = argparse.ArgumentParser(
        description=f"Add/refresh the '{COUNT_COLUMN}' column of an IWAC subset."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", choices=WORD_COUNT_SUBSETS, default=None,
                        help="Subset to process (skips the interactive menu)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help=f"Recompute without confirmation when '{COUNT_COLUMN}' exists")
    parser.add_argument("--update-mode", choices=["missing", "all"], default="all",
                        help="'all' recounts every row (default); 'missing' fills only null counts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report, but push nothing")
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    token = ensure_hf_token(console=console)
    config_name = resolve_config(
        args.repo, token=token, cli_config=args.config,
        restrict_to=WORD_COUNT_SUBSETS, console=console,
    )
    console.print(f"[green]→[/green] Subset: [bold]{config_name}[/bold]")

    ds = load_hub_dataset(args.repo, config_name, token=token, console=console)
    source_revision = getattr(ds, "_iwac_source_revision", None)

    if TEXT_COLUMN not in ds.column_names:
        console.print(f"[red]✗[/red] '{TEXT_COLUMN}' is missing from this subset.")
        return 1

    if COUNT_COLUMN in ds.column_names and args.update_mode == "all" and not (
        args.yes or args.dry_run
    ):
        console.print(f"\n[yellow]⚠[/yellow] '{COUNT_COLUMN}' already exists.")
        try:
            if not Confirm.ask("Recompute every count?", default=False):
                console.print("[yellow]ℹ[/yellow] Cancelled; existing counts kept.")
                return 0
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Cancelled.")
            return 0

    from datasets import Value

    ds = map_with_progress(
        ds,
        lambda batch: add_word_count_batch(
            batch, text_col=TEXT_COLUMN, count_col=COUNT_COLUMN,
            update_mode=args.update_mode,
        ),
        batch_size=args.batch_size,
        description=f"[cyan]Counting words in '{TEXT_COLUMN}'",
        console=console,
        output_types={COUNT_COLUMN: Value("int64")},
    )
    ds = reorder_columns_after(ds, [COUNT_COLUMN], TEXT_COLUMN, console=console)
    print_summary(ds[COUNT_COLUMN][:], config_name)

    if args.dry_run:
        print_dry_run_panel(
            repo_id=args.repo, config_name=config_name, n_rows=len(ds), console=console,
        )
        return 0

    if push_dataset(
        ds,
        repo_id=args.repo,
        config_name=config_name,
        token=token,
        max_shard_size=args.max_shard_size,
        commit_message=(
            f"Add/update '{COUNT_COLUMN}' from '{TEXT_COLUMN}' "
            f"(elision-aware count; config: {config_name}, mode: {args.update_mode})"
        ),
        console=console,
        expected_revision=source_revision,
    ):
        console.print(Panel(
            f"[green]✓[/green] '{COUNT_COLUMN}' written for [bold]{config_name}[/bold]\n"
            f"[dim]Rows: {len(ds):,} · Repository: {args.repo}\n"
            "Remember: the public dataset only changes when "
            "post-processing/publish_public.py is re-run.[/dim]",
            title="[bold green]Done[/bold green]",
            border_style="green",
        ))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
