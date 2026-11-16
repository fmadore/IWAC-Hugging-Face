#!/usr/bin/env python3
"""
calculate_word_count.py
=======================

Adds or refreshes the ``nb_mots`` column of an IWAC subset from its ``OCR``
column, and pushes the subset back to the private Hub repository.

``nb_mots`` has a single definition, :func:`iwac_common.text_utils.count_words`,
called with the row's ``language``: for a row whose languages include French an
elided clitic is a word of its own (``l'église`` = 2, like ``la mosquée``);
in other languages an apostrophe inside a word keeps it whole (``da'wa`` = 1).
``calculate_lexical_richness.py`` tokenizes MATTR through the same function
with the same argument, so ``nb_mots`` and the MATTR token count agree.

Each count carries private provenance (``nb_mots_input_hash`` over text and
language labels, ``nb_mots_config_hash`` over ``TOKENIZER_VERSION`` and the
language rule), so ``--update-mode missing`` recomputes a count whose text,
language or tokenizer changed, not only a null one. Counts written before
provenance existed are recomputed once.

``references`` is counted like every other subset. The private repository
holds its full text in ``OCR``, so the per-item Omeka fetch this script used to
make for references (to see private ``bibo:content``) is no longer needed.

Usage
-----
    python post-processing/calculate_word_count.py            # interactive
    python post-processing/calculate_word_count.py --config articles -y
    python post-processing/calculate_word_count.py --config references --update-mode all -y

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
    add_update_mode_args,
    ensure_hf_token,
    load_hub_dataset,
    map_with_progress,
    print_dry_run_panel,
    push_dataset,
    reorder_columns_after,
    resolve_config,
    resolve_update_mode,
    workspace_root,
)
from iwac_common.enrichment import (  # noqa: E402
    changed_row_count, config_fingerprint, config_json, fingerprint, provenance_columns,
)
from iwac_common.text_utils import TOKENIZER_VERSION, count_words, language_labels  # noqa: E402

load_dotenv(workspace_root() / ".env")

console = Console()

TEXT_COLUMN = "OCR"
COUNT_COLUMN = "nb_mots"
LANGUAGE_COLUMN = "language"
WORD_COUNT_SUBSETS = ["articles", "publications", "documents", "references", "audiovisual"]

#: What a count was computed with; its hash is ``nb_mots_config_hash``.
WORD_COUNT_SETTINGS = {
    "metric": "nb_mots", "tokenizer": TOKENIZER_VERSION,
    "french_elision": "row languages include Français",
}
WORD_COUNT_CONFIG = config_fingerprint(WORD_COUNT_SETTINGS)
WORD_COUNT_CONFIG_JSON = config_json(WORD_COUNT_SETTINGS)
INPUT_HASH_COLUMN, CONFIG_HASH_COLUMN = provenance_columns(COUNT_COLUMN)
CONFIG_JSON_COLUMN = f"{COUNT_COLUMN}_config_json"
OUTPUT_COLUMNS = [COUNT_COLUMN, INPUT_HASH_COLUMN, CONFIG_HASH_COLUMN, CONFIG_JSON_COLUMN]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def add_word_count_batch(
    batch: Dict[str, List[Any]], text_col: str, count_col: str, update_mode: str = "all",
    language_col: str = LANGUAGE_COLUMN,
) -> Dict[str, List[Any]]:
    """Fill ``count_col`` and its provenance for one ``.map(batched=True)`` batch.

    ``update_mode="all"`` recounts every row; ``"missing"`` keeps a count only
    when its persisted provenance matches the current text, language labels
    and tokenizer configuration, and recounts every other row.
    """
    if text_col not in batch:
        # An empty batch (or a subset without the text column): guard against
        # next(iter(batch)) raising StopIteration.
        if count_col not in batch:
            first_col = next(iter(batch), None)
            batch[count_col] = [0] * (len(batch[first_col]) if first_col is not None else 0)
        return batch

    texts = batch[text_col]
    n = len(texts)
    languages = batch.get(language_col, [None] * n)
    input_col, config_col = provenance_columns(count_col)
    json_col = f"{count_col}_config_json"
    existing = batch.get(count_col, [None] * n)
    old_inputs = batch.get(input_col, [None] * n)
    old_configs = batch.get(config_col, [None] * n)

    counts, inputs = [], []
    for i, text in enumerate(texts):
        text_str = "" if text is None else str(text)
        input_hash = fingerprint(text_str, language_labels(languages[i]))
        inputs.append(input_hash)
        if (update_mode == "missing" and existing[i] is not None
                and old_inputs[i] == input_hash and old_configs[i] == WORD_COUNT_CONFIG):
            counts.append(existing[i])
        else:
            counts.append(count_words(text_str, language=languages[i]))
    batch[count_col] = counts
    batch[input_col] = inputs
    batch[config_col] = [WORD_COUNT_CONFIG] * n
    batch[json_col] = [WORD_COUNT_CONFIG_JSON] * n
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


def confirm_full_recompute(args, mode: str, what: str) -> bool:
    """Ask before ``all`` recomputes every row; ``-y``, a dry run or a headless
    run (whose explicit ``--update-mode all`` is the decision) skip it."""
    if mode != "all" or args.yes or args.dry_run or not sys.stdin.isatty():
        return True
    try:
        return Confirm.ask(f"Recompute {what} for every row?", default=False)
    except KeyboardInterrupt:
        return False


def word_count_output_types() -> Dict[str, Any]:
    """Declared features of the count and its provenance columns."""
    from datasets import Value

    types = {COUNT_COLUMN: Value("int64")}
    types.update({name: Value("string") for name in OUTPUT_COLUMNS[1:]})
    return types


def count_subset(ds, update_mode: str, batch_size: int):
    if LANGUAGE_COLUMN not in ds.column_names:
        console.print(f"[yellow]⚠[/yellow] No '{LANGUAGE_COLUMN}' column: no row gets the French elision split.")
    return map_with_progress(
        ds,
        lambda batch: add_word_count_batch(
            batch, text_col=TEXT_COLUMN, count_col=COUNT_COLUMN, update_mode=update_mode,
        ),
        batch_size=batch_size,
        description=f"[cyan]Counting words in '{TEXT_COLUMN}'",
        console=console,
        output_types=word_count_output_types(),
    )


def main() -> int:
    configure_logging()
    console.print(Panel.fit(
        "[bold cyan]Word Count Calculator[/bold cyan]\n"
        f"[dim]{COUNT_COLUMN} from {TEXT_COLUMN} (language-aware, tokenizer v{TOKENIZER_VERSION})[/dim]",
        border_style="cyan",
    ))

    parser = argparse.ArgumentParser(
        description=f"Add/refresh the '{COUNT_COLUMN}' column of an IWAC subset."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", choices=WORD_COUNT_SUBSETS, default=None,
                        help="Subset to process (skips the interactive menu)")
    add_update_mode_args(parser)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report, but push nothing")
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    token = ensure_hf_token(console=console)
    config_name = resolve_config(
        args.repo, token=token, cli_config=args.config,
        restrict_to=WORD_COUNT_SUBSETS, console=console,
    )
    console.print(f"[green]→[/green] Subset: [bold]{config_name}[/bold]")
    update_mode = resolve_update_mode(args, console=console)
    console.print(f"[green]✓[/green] Update mode: [cyan]{update_mode}[/cyan]")

    ds = load_hub_dataset(args.repo, config_name, token=token, console=console)
    source_revision = getattr(ds, "_iwac_source_revision", None)

    if TEXT_COLUMN not in ds.column_names:
        console.print(f"[red]✗[/red] '{TEXT_COLUMN}' is missing from this subset.")
        return 1
    if not confirm_full_recompute(args, update_mode, f"'{COUNT_COLUMN}'"):
        console.print("[yellow]ℹ[/yellow] Cancelled; existing counts kept.")
        return 0

    before = ds
    ds = count_subset(ds, update_mode, args.batch_size)
    ds = reorder_columns_after(ds, [COUNT_COLUMN], TEXT_COLUMN, console=console)
    print_summary(ds[COUNT_COLUMN][:], config_name)
    changed = changed_row_count(before, ds, OUTPUT_COLUMNS)
    console.print(f"[blue]→[/blue] Rows whose count or provenance changes: [bold]{changed:,}[/bold]")

    if args.dry_run:
        print_dry_run_panel(
            repo_id=args.repo, config_name=config_name, n_rows=len(ds), console=console,
            extra=f"[dim]Rows that would change: {changed:,}[/dim]",
        )
        return 0
    if not changed:
        console.print("[green]✓[/green] Every count and its provenance are already current. Nothing to push.")
        return 0

    if push_dataset(
        ds,
        repo_id=args.repo,
        config_name=config_name,
        token=token,
        max_shard_size=args.max_shard_size,
        commit_message=(
            f"Add/update '{COUNT_COLUMN}' from '{TEXT_COLUMN}' "
            f"(language-aware count, tokenizer v{TOKENIZER_VERSION}; "
            f"config: {config_name}, mode: {update_mode})"
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
