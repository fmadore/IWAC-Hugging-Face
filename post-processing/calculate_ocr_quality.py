#!/usr/bin/env python3
"""
calculate_ocr_quality.py
========================

Adds an ``ocr_quality`` float column (0-1) to a subset of the IWAC dataset:
the share of scoreable word tokens found in a frequency lexicon
(``wordfreq``). Downstream analyses can use it to weight or filter noisy
scans.

Metric definition
-----------------
- Tokenize the RAW ``OCR`` text with ``iwac_common.text_utils.tokenize_words``
  (elision-aware, lowercased).
- Denominator = alphabetic tokens (``t.isalpha()``) of length >= 3, minus
  likely proper nouns (see below). Digits, short tokens and proper nouns
  count neither way.
- A token is a dictionary hit when ``wordfreq.zipf_frequency(token, lang)
  >= 1.5`` — ``lang`` is "en" if the row's primary (first-listed) ``language``
  is "Anglais", else "fr".
- Proper-noun heuristic (kept deliberately simple): a lowercased form is
  skipped when, in the ORIGINAL text, it (a) never occurs lowercase and
  (b) occurs capitalized at least once mid-sentence (i.e. the preceding
  non-quote/non-bracket character is not a sentence terminator ``.!?…:`` or
  the start of the text). All-caps OCR headlines can occasionally sweep a
  common word into this set; that is an accepted trade-off for not punishing
  scans full of West African names absent from the lexicons.
- Rows with fewer than 20 scoreable tokens (incl. empty/missing OCR) get
  ``None`` — too short to judge.

Usage
-----
    python post-processing/calculate_ocr_quality.py --config articles --dry-run
    python post-processing/calculate_ocr_quality.py --config articles --update-mode all -y

Environment variables
---------------------
HF_TOKEN    Hugging Face Hub access token (otherwise interactive login).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import uuid
from functools import lru_cache
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Make ``post-processing/_common.py`` and ``iwac_common`` importable.
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from _common import ensure_hf_token, PRIVATE_REPO_ID  # noqa: E402
from iwac_common.text_utils import tokenize_words  # noqa: E402

load_dotenv()

console = Console()

QUALITY_COLUMN = "ocr_quality"
TEXT_COLUMN = "OCR"
LANGUAGE_COLUMN = "language"
OCR_CONFIGS = ["articles", "publications", "references", "documents", "audiovisual"]

ZIPF_THRESHOLD = 1.5
MIN_SCOREABLE_TOKENS = 20
MIN_TOKEN_LEN = 3

# Alphabetic runs, case preserved (no digits/underscore) — used only by the
# proper-noun heuristic, which needs the ORIGINAL casing.
_CASED_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_SENTENCE_END = ".!?…:"
# Characters that may sit between a sentence terminator and the next word
# (quotes, brackets, dashes, whitespace) — ignored when looking back.
_LOOKBACK_STRIP = " \t\r\n\"'«»“”‘’()[]—–-"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


# ---------------------------------------------------------------------------
# Pure scoring functions (importable without network / model downloads;
# wordfreq is imported lazily and loads its lexicons on first lookup).
# ---------------------------------------------------------------------------

@lru_cache(maxsize=262_144)
def _zipf(token: str, lang: str) -> float:
    """Cached ``wordfreq.zipf_frequency`` lookup (lazy import)."""
    from wordfreq import zipf_frequency

    return zipf_frequency(token, lang)


def resolve_lang(language_value) -> str:
    """Map a (possibly pipe-separated) ``language`` cell to a wordfreq code.

    "en" if the primary (first-listed) language is "Anglais", else "fr".
    """
    if language_value is None:
        return "fr"
    primary = str(language_value).split("|")[0].strip()
    return "en" if primary == "Anglais" else "fr"


def proper_noun_candidates(text: str) -> frozenset:
    """Lowercased forms that look like proper nouns in ``text``.

    A form qualifies when it never occurs lowercase in the original text AND
    occurs capitalized at least once mid-sentence (preceding meaningful
    character is not a sentence terminator / start of text). Simple by
    design — see module docstring for the accepted trade-offs.
    """
    seen_lower: set = set()
    cap_mid_sentence: set = set()
    for m in _CASED_WORD_RE.finditer(text):
        tok = m.group()
        low = tok.lower()
        if tok[0].isupper():
            before = text[: m.start()].rstrip(_LOOKBACK_STRIP)
            if before and before[-1] not in _SENTENCE_END:
                cap_mid_sentence.add(low)
        else:
            seen_lower.add(low)
    return frozenset(cap_mid_sentence - seen_lower)


def score_ocr_quality(
    text: Optional[str],
    lang: str = "fr",
    min_scoreable: int = MIN_SCOREABLE_TOKENS,
) -> Optional[float]:
    """Dictionary hit-rate of the scoreable tokens of ``text`` in [0, 1].

    Returns ``None`` for empty/missing text or fewer than ``min_scoreable``
    scoreable tokens (too short to judge).
    """
    if text is None or not str(text).strip():
        return None
    text = str(text)
    proper_nouns = proper_noun_candidates(text)
    scoreable = [
        t
        for t in tokenize_words(text)
        if t.isalpha() and len(t) >= MIN_TOKEN_LEN and t not in proper_nouns
    ]
    if len(scoreable) < min_scoreable:
        return None
    hits = sum(1 for t in scoreable if _zipf(t, lang) >= ZIPF_THRESHOLD)
    return hits / len(scoreable)


def add_ocr_quality_batch(
    batch: Dict[str, list],
    text_col: str = TEXT_COLUMN,
    lang_col: str = LANGUAGE_COLUMN,
    out_col: str = QUALITY_COLUMN,
    update_mode: str = "all",
) -> Dict[str, list]:
    """Batched ``ds.map`` function adding/updating ``out_col``.

    ``update_mode="missing"`` keeps existing non-null scores untouched.
    """
    first_col = next(iter(batch), None)
    n = len(batch[first_col]) if first_col is not None else 0
    texts = batch.get(text_col, [None] * n)
    langs = batch.get(lang_col, [None] * n)
    existing = batch.get(out_col, [None] * n)

    scores: List[Optional[float]] = []
    for text, lang_val, prev in zip(texts, langs, existing):
        if update_mode == "missing" and prev is not None:
            scores.append(prev)
            continue
        scores.append(score_ocr_quality(text, resolve_lang(lang_val)))
    batch[out_col] = scores
    return batch


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def show_distribution(df) -> None:
    """Rich table with distribution stats of the quality scores."""
    scores = df[QUALITY_COLUMN].dropna()
    table = Table(title="ocr_quality distribution", box=box.ROUNDED)
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Rows", f"{len(df):,}")
    table.add_row("Scored rows", f"{len(scores):,}")
    table.add_row("None (too short / empty)", f"{df[QUALITY_COLUMN].isna().sum():,}")
    if len(scores):
        table.add_row("Mean", f"{scores.mean():.4f}")
        table.add_row("Median", f"{scores.median():.4f}")
        table.add_row("Q1", f"{scores.quantile(0.25):.4f}")
        table.add_row("Q3", f"{scores.quantile(0.75):.4f}")
        table.add_row("Min", f"{scores.min():.4f}")
        table.add_row("Max", f"{scores.max():.4f}")
    console.print(table)


def show_examples(df, n: int = 5) -> None:
    """Show the worst and best scored rows by title (dry-run preview)."""
    scored = df.dropna(subset=[QUALITY_COLUMN]).sort_values(QUALITY_COLUMN)
    if scored.empty:
        console.print("[yellow]ℹ[/yellow] No scored rows to preview.")
        return
    title_col = "title" if "title" in scored.columns else "o:id"
    table = Table(title=f"Worst / best {n} by {QUALITY_COLUMN}", box=box.ROUNDED)
    table.add_column("", style="dim")
    table.add_column("o:id", style="cyan")
    table.add_column("Title", style="white", max_width=70)
    table.add_column("Score", style="green", justify="right")
    for _, row in scored.head(n).iterrows():
        table.add_row("worst", str(row.get("o:id", "?")), str(row[title_col])[:70], f"{row[QUALITY_COLUMN]:.3f}")
    for _, row in scored.tail(n).iterrows():
        table.add_row("best", str(row.get("o:id", "?")), str(row[title_col])[:70], f"{row[QUALITY_COLUMN]:.3f}")
    console.print(table)


def reorder_after_anchor(ds: Dataset, column: str, anchors: List[str]) -> Dataset:
    """Move ``column`` right after the first present anchor column."""
    cols = list(ds.column_names)
    if column not in cols:
        return ds
    cols.remove(column)
    for anchor in anchors:
        if anchor in cols:
            idx = cols.index(anchor)
            new_order = cols[: idx + 1] + [column] + cols[idx + 1:]
            return ds.select_columns(new_order)
    console.print(
        f"[yellow]⚠[/yellow] Anchor column not found ({', '.join(anchors)}); "
        f"keeping [bold]{column}[/bold] at the end."
    )
    return ds.select_columns(cols + [column])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging()

    console.print(Panel.fit(
        "[bold cyan]OCR Quality Calculator[/bold cyan]\n"
        "[dim]Score per-document OCR quality (dictionary hit-rate via wordfreq)[/dim]",
        border_style="cyan",
    ))

    parser = argparse.ArgumentParser(
        description="Add/update the 'ocr_quality' column (dictionary hit-rate in [0,1]) on an IWAC subset."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument(
        "--config",
        choices=OCR_CONFIGS,
        default=None,
        help="Subset to process (skips the interactive prompt)",
    )
    parser.add_argument(
        "--update-mode",
        choices=["missing", "all"],
        default="missing",
        help="'missing' (default): only rows where ocr_quality is null; 'all': recompute everything",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and show distribution stats + worst/best examples; push nothing",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Push without confirmation",
    )
    args = parser.parse_args()

    # --- Config selection (CLI or interactive prompt) ---
    if args.config:
        config_name = args.config
    else:
        try:
            config_name = Prompt.ask(
                "[cyan]Which configuration to process?[/cyan]",
                choices=OCR_CONFIGS,
                default="articles",
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Operation cancelled by user.")
            return
    console.print(f"[green]→[/green] Selected configuration: [bold]{config_name}[/bold]")
    console.print(f"[green]→[/green] Update mode: [bold]{args.update_mode}[/bold]")

    # --- HF authentication ---
    token = ensure_hf_token(console=console)
    console.print("[green]✓[/green] Hugging Face authentication successful")

    # --- Load dataset ---
    console.print(f"\n[blue]→[/blue] Loading dataset [bold]{args.repo}[/bold], configuration [bold]{config_name}[/bold]...")
    try:
        with console.status("[bold green]Loading...", spinner="dots"):
            ds = load_dataset(args.repo, name=config_name, split="train", token=token)
        console.print(f"[green]✓[/green] Dataset loaded: [bold]{len(ds):,}[/bold] rows")
    except Exception as e:
        console.print(f"[red]✗[/red] Error loading dataset: {e}")
        return

    if TEXT_COLUMN not in ds.column_names:
        console.print(f"[red]✗[/red] Text column [bold]{TEXT_COLUMN}[/bold] not found in this subset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        return
    if LANGUAGE_COLUMN not in ds.column_names:
        console.print(
            f"[yellow]⚠[/yellow] No [bold]{LANGUAGE_COLUMN}[/bold] column — all rows scored with the French lexicon."
        )

    if QUALITY_COLUMN in ds.column_names and args.update_mode == "missing":
        console.print(
            f"[yellow]ℹ[/yellow] Column [bold]{QUALITY_COLUMN}[/bold] exists — "
            f"only rows with a null score will be computed (use --update-mode all to recompute)."
        )

    # --- Compute scores ---
    console.print(f"\n[blue]→[/blue] Scoring OCR quality from column [bold]{TEXT_COLUMN}[/bold]...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task(f"Computing '{QUALITY_COLUMN}'", total=None)
        ds_processed = ds.map(
            add_ocr_quality_batch,
            batched=True,
            batch_size=200,
            fn_kwargs={
                "text_col": TEXT_COLUMN,
                "lang_col": LANGUAGE_COLUMN,
                "out_col": QUALITY_COLUMN,
                "update_mode": args.update_mode,
            },
            # Never reuse a stale .map() cache on re-runs (same pattern as
            # calculate_word_count.py / calculate_lexical_richness.py).
            load_from_cache_file=False,
            new_fingerprint=str(uuid.uuid4()),
        )
    console.print("[green]✓[/green] Scoring complete")

    # Nullable float typing + stats via pandas.
    with console.status("[bold blue]Converting to nullable float...", spinner="dots"):
        df = ds_processed.to_pandas()
        df[QUALITY_COLUMN] = df[QUALITY_COLUMN].astype("Float64")
        ds_processed = Dataset.from_pandas(df, preserve_index=False)
    console.print(f"[green]✓[/green] Column [bold]{QUALITY_COLUMN}[/bold] cast to nullable float (Float64)")

    show_distribution(df)

    if args.dry_run:
        show_examples(df)
        console.print(Panel(
            "[yellow]Dry run — nothing pushed to the Hub.[/yellow]",
            border_style="yellow",
        ))
        return

    # --- Column reorder: right after nb_mots if present (else after OCR) ---
    console.print(f"\n[blue]→[/blue] Placing [bold]{QUALITY_COLUMN}[/bold] after [bold]nb_mots[/bold] (or {TEXT_COLUMN})")
    ds_processed = reorder_after_anchor(ds_processed, QUALITY_COLUMN, ["nb_mots", TEXT_COLUMN])
    console.print(f"[dim]New order: {', '.join(ds_processed.column_names[:6])}{'...' if len(ds_processed.column_names) > 6 else ''}[/dim]")

    # --- Push ---
    if not args.yes:
        try:
            if not Confirm.ask(
                f"Push updated dataset to [bold]{args.repo}[/bold] (config: {config_name})?",
                default=False,
            ):
                console.print("[yellow]ℹ[/yellow] Push cancelled.")
                return
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Operation cancelled by user.")
            return

    console.print(f"\n[blue]→[/blue] Pushing to [bold]{args.repo}[/bold] (config: [bold]{config_name}[/bold])...")
    try:
        with console.status("[bold green]Uploading...", spinner="dots"):
            ds_processed.push_to_hub(
                args.repo,
                config_name=config_name,
                token=token,
                max_shard_size="1GB",
                commit_message="Add/update ocr_quality (dictionary hit-rate)",
            )
        console.print(Panel(
            f"[green]✓[/green] Column [bold]{QUALITY_COLUMN}[/bold] pushed successfully\n"
            f"[dim]Configuration: {config_name}\n"
            f"Update mode: {args.update_mode}\n"
            f"Rows: {len(ds_processed):,}\n"
            f"Repository: {args.repo}[/dim]",
            title="[bold green]Done[/bold green]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[red]✗[/red] Error pushing dataset to the Hub: {e}")


if __name__ == "__main__":
    main()
