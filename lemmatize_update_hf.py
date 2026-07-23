#!/usr/bin/env python3
"""
lemmatize_update_hf.py
======================

Add French lemmatisation and stop‑word–filtered text to an existing Hugging Face
Hub dataset.  It downloads the dataset, processes the chosen text column with
spaCy, adds two new columns (lemmatised text and lemmatised text with French
stop‑words removed) and pushes the updated dataset back to the repository.

Usage
-----
    # Articles (default subset)
    python lemmatize_update_hf.py --config articles

    # Publications (periodical issues) — same OCR text column as articles
    python lemmatize_update_hf.py --config publications

    # Full form
    python lemmatize_update_hf.py \
        --repo fmadore/islam-west-africa-collection \
        --config articles \
        --text-column OCR \
        --lemma-column lemma_text \
        --clean-column lemma_nostop \
        --spacy-model fr_core_news_lg \
        --max-shard-size 1GB

Note: The default model (fr_core_news_lg) is optimized for CPU. For GPU users,
consider using fr_dep_news_trf for better accuracy (but much slower on CPU).

Environment variables
---------------------
HF_TOKEN   Personal access token for the Hugging Face Hub (alternatively you
           will be prompted to log‑in interactively).

Dependencies
------------
    pip install datasets huggingface_hub spacy rich python-dotenv
    python -m spacy download fr_core_news_lg

"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List
import re
import unicodedata

import datasets
from datasets import Dataset
import spacy

# Make ``post-processing/_common.py`` and ``_embedding_utils.py`` importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "post-processing"))
from _common import ensure_hf_token, PRIVATE_REPO_ID  # noqa: E402
from _embedding_utils import load_cache, save_cache, delete_cache  # noqa: E402

# Rich console imports for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.prompt import Prompt
from rich import box

# Initialize Rich console
console = Console()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )


# Preprocessing constants and function
RE_DASH  = re.compile(r"[–—−]")
RE_SPACE = re.compile(r"\s{2,}")  # Fixed: was incorrectly escaped
# Fancy quotes to normalize (using Unicode codepoints for reliability)
QUOTE_REPLACEMENTS = [
    ("\u2018", "'"), ("\u2019", "'"), ("\u201a", "'"), ("\u201b", "'"),  # single quotes
    ("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'), ("\u201f", '"'),  # double quotes
    ("\u00ab", '"'), ("\u00bb", '"'),  # guillemets « »
]
MAP_LIG = str.maketrans({"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE"})

# Cap the characters handed to spaCy in a single call. A full periodical issue
# (publications) can top 1M chars; tagging it whole-hog blows up tok2vec memory
# on CPU, so long OCR is split into chunks of this size first.
SPACY_MAX_CHUNK_CHARS = 100_000

# Crash-resumable cache: freshly computed lemmas are checkpointed here (keyed
# by o:id) every CHECKPOINT_EVERY rows, reloaded on restart, and deleted after
# a successful push. Gitignored, like the other .cache_* dirs.
CACHE_DIR = Path(__file__).resolve().parent / ".cache_lemmas"
CHECKPOINT_EVERY = 100


def normalize(text: str) -> str:
    if not text:  # None or empty string (e.g. blank OCR rows) → nothing to lemmatise
        return ""
    text = unicodedata.normalize("NFC", text)
    for old, new in QUOTE_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.translate(MAP_LIG)
    text = RE_DASH.sub("-", text).replace("\u00a0", " ")
    return RE_SPACE.sub(" ", text).strip()


def load_spacy_model(name: str):
    """Load a spaCy model, downloading it the first time it is required."""
    try:
        return spacy.load(name, disable=["parser", "ner", "textcat"])
    except OSError:
        console.print(f"[yellow]⚠[/yellow] SpaCy model '{name}' not found – downloading…")
        from spacy.cli import download as spacy_download

        spacy_download(name)
        return spacy.load(name, disable=["parser", "ner", "textcat"])


def chunk_on_whitespace(text: str, max_chars: int) -> List[str]:
    """Split ``text`` into <= ``max_chars`` pieces, breaking at whitespace."""
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws
        chunks.append(text[start:end])
        start = end
    return chunks


def lemmatise_one(nlp, text: str) -> List[tuple]:
    """Lemmatise one already-normalized text, chunking long text to bound memory.

    Returns a list of ``(lemma, is_stop)`` pairs. Stop-word status is spaCy's
    ``token.is_stop`` on the SURFACE token — the semantically correct check —
    rather than testing the lemma against the surface-form stop-word list.

    A full periodical issue can top 1M chars; feeding that to spaCy in one go
    builds a huge tok2vec tensor and OOMs on CPU. Splitting into
    SPACY_MAX_CHUNK_CHARS pieces keeps each call small. Parser/NER are
    disabled, so chunk boundaries do not change the per-token lemmas.
    """
    if not text:
        return []
    tokens: List[tuple] = []
    for chunk in chunk_on_whitespace(text, SPACY_MAX_CHUNK_CHARS):
        doc = nlp(chunk)
        tokens.extend((tok.lemma_.lower(), tok.is_stop) for tok in doc if tok.is_alpha)
    return tokens


def lemmatise_dataset(
    ds,
    nlp,
    *,
    text_col: str,
    lemma_col: str,
    clean_col: str,
    process_choice: str,
    cache_file: Path,
    language_filter: str | None = None,
):
    """Add lemma columns to ``ds``, checkpointing to ``cache_file`` for resume.

    Returns the updated dataset, or ``None`` when there is nothing to do.

    Resilience: each freshly lemmatised row is written to an on-disk cache
    (keyed by ``o:id``) and flushed every CHECKPOINT_EVERY rows, so a crash
    loses at most that many rows. On restart, cached rows are reused and only
    the remainder is recomputed. Rows that already have a non-empty lemma are
    kept as-is in "empty" mode; blank-text rows get empty lemmas.

    ``language_filter`` skips rows whose PRIMARY language (the first
    pipe-separated component of ``language``, whitespace-stripped) is not the
    given label (e.g. "Français" or "Anglais"): a wrong-language pipeline
    would emit garbage lemmas. Matching on the primary language only makes
    bilingual rows ("Français|Anglais") deterministic — each row belongs to
    exactly one language pass, so the result does not depend on --mode or
    pass order. Skipped rows keep their existing values, so per-language
    passes compose (French pass with the French model, then English pass
    with the English model).
    """
    n = len(ds)
    row_ids = [str(x) for x in ds["o:id"]]
    texts = ds[text_col]
    existing_lemmas = ds[lemma_col] if lemma_col in ds.column_names else [None] * n
    existing_clean = ds[clean_col] if clean_col in ds.column_names else [None] * n

    in_scope = [True] * n
    if language_filter:
        if "language" not in ds.column_names:
            console.print(f"[yellow]⚠[/yellow] --language '{language_filter}' requested but no 'language' column; processing all rows.")
        else:
            # A row is in scope only when its FIRST listed (primary) language
            # matches the filter: bilingual rows ("Français|Anglais") are
            # handled by exactly one pass, so output is deterministic
            # regardless of --mode or pass order. Components are stripped so
            # "Français | Anglais" parses correctly.
            in_scope = [
                str(lang).split("|")[0].strip() == language_filter if lang else False
                for lang in ds["language"]
            ]
            console.print(
                f"[blue]→[/blue] Language filter '{language_filter}': [bold]{sum(in_scope):,}[/bold] of {n:,} rows eligible"
            )

    cache = load_cache(cache_file)

    final_lemmas: List[str] = [""] * n
    final_clean: List[str] = [""] * n
    indices_to_process: List[int] = []
    for i in range(n):
        if not in_scope[i]:
            # Out of scope: keep whatever the row already has.
            final_lemmas[i] = existing_lemmas[i] or ""
            final_clean[i] = existing_clean[i] or ""
            continue
        cached = cache.get(row_ids[i])
        if cached is not None:
            final_lemmas[i], final_clean[i] = cached[0], cached[1]
            continue
        if process_choice == "empty" and existing_lemmas[i] and existing_lemmas[i].strip():
            final_lemmas[i] = existing_lemmas[i]
            final_clean[i] = existing_clean[i] or ""
            continue
        if not texts[i] or not str(texts[i]).strip():
            continue  # blank text -> empty lemmas (already "")
        indices_to_process.append(i)

    if not indices_to_process and not cache:
        return None  # nothing new to compute and nothing cached to flush

    console.print(
        f"[blue]→[/blue] [bold]{len(indices_to_process):,}[/bold] of {n:,} rows to lemmatise"
        + (f" ([green]{len(cache):,}[/green] restored from cache)" if cache else "")
    )

    since_ckpt = 0
    if indices_to_process:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]lemmatising", total=len(indices_to_process))
            for i in indices_to_process:
                tokens = lemmatise_one(nlp, normalize(texts[i]))
                lemma_text = " ".join(lemma for lemma, _ in tokens)
                # Stop-word filtering uses token.is_stop on the surface token
                # (not the lemma vs. the surface-form stop-word list).
                clean_text = " ".join(lemma for lemma, is_stop in tokens if not is_stop)
                final_lemmas[i] = lemma_text
                final_clean[i] = clean_text
                cache[row_ids[i]] = [lemma_text, clean_text]
                since_ckpt += 1
                progress.update(task, advance=1)
                if since_ckpt >= CHECKPOINT_EVERY:
                    save_cache(cache, cache_file)
                    since_ckpt = 0
        if since_ckpt > 0:
            save_cache(cache, cache_file)

    for col in (lemma_col, clean_col):
        if col in ds.column_names:
            ds = ds.remove_columns([col])
    ds = ds.add_column(lemma_col, final_lemmas)
    ds = ds.add_column(clean_col, final_clean)
    return ds


# Subsets that carry an OCR text column worth lemmatising. Passing --config
# with another subset still works; this list only drives the interactive menu.
LEMMATIZABLE_SUBSETS = ["articles", "publications", "references"]

# Default spaCy model per --language label (CPU-friendly large models).
LANGUAGE_MODEL_DEFAULTS = {
    "Français": "fr_core_news_lg",
    "Anglais": "en_core_web_lg",
}


def choose_subset() -> str:
    """Prompt the user to pick which dataset subset to lemmatise."""
    console.print("\n[bold]Dataset subset:[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Subset", style="green")
    table.add_column("Text column", style="white")
    for i, name in enumerate(LEMMATIZABLE_SUBSETS, 1):
        table.add_row(str(i), name, "OCR")
    console.print(table)
    choice = Prompt.ask(
        "Choose subset",
        choices=[str(i) for i in range(1, len(LEMMATIZABLE_SUBSETS) + 1)],
        default="1",
    )
    return LEMMATIZABLE_SUBSETS[int(choice) - 1]


def main():
    configure_logging()

    parser = argparse.ArgumentParser(description="Add lemmatised columns to a Hugging Face dataset")
    parser.add_argument("--repo", default=PRIVATE_REPO_ID, help="Dataset repo on the Hugging Face Hub (default: private full mirror)")
    parser.add_argument("--config", default=None, help="Dataset subset/config to lemmatise (e.g. 'articles', 'publications', 'references'). If omitted, you are prompted to choose. Any subset that has the --text-column is accepted.")
    parser.add_argument("--text-column", default="OCR", help="Name of the column containing the raw French text to process")
    parser.add_argument("--lemma-column", default="lemma_text", help="Column name for the lemmatised text")
    parser.add_argument("--clean-column", default="lemma_nostop", help="Column name for the lemmatised text with stop-words removed")
    parser.add_argument("--spacy-model", default=None, help="spaCy model to use (default: chosen from --language via LANGUAGE_MODEL_DEFAULTS, else fr_core_news_lg)")
    parser.add_argument("--max-shard-size", default="1GB", help="Maximum Parquet shard size when pushing to the Hub")
    parser.add_argument("--update-mode", choices=["all", "missing"], default=None,
                        help="Process all rows ('all') or only rows with an empty lemma column ('missing'). "
                             "Preferred spelling; skips the interactive prompt.")
    parser.add_argument("--mode", choices=["all", "empty"], default=None,
                        help="Deprecated alias for --update-mode ('empty' == 'missing').")
    parser.add_argument("--language", default=None, metavar="LABEL", help="Only lemmatise rows whose PRIMARY (first-listed) 'language' is LABEL (e.g. 'Français', 'Anglais'); picks the matching spaCy model unless --spacy-model is given. Other rows keep their existing values — run one pass per language on mixed subsets like 'references'.")
    parser.add_argument("--french-only", action="store_true", help="Deprecated alias for --language Français")
    parser.add_argument("--dry-run", action="store_true",
                        help="Lemmatise and report what would change, but do not push to the Hub or delete the cache")
    args = parser.parse_args()

    # Resolve the recompute mode: --update-mode wins; --mode is the deprecated
    # alias (all|empty), where 'empty' maps to 'missing'.
    resolved_mode = args.update_mode
    if resolved_mode is None and args.mode is not None:
        console.print("[yellow]⚠[/yellow] --mode is deprecated; use --update-mode {all,missing} ('empty'→'missing').")
        resolved_mode = "missing" if args.mode == "empty" else args.mode
    # The downstream code speaks the legacy vocabulary ('all'/'empty').
    args.mode = None if resolved_mode is None else ("empty" if resolved_mode == "missing" else "all")

    language_filter = args.language or ("Français" if args.french_only else None)
    spacy_model = args.spacy_model or LANGUAGE_MODEL_DEFAULTS.get(language_filter or "", "fr_core_news_lg")

    # ------------------------------------------------------------------
    # Resolve which subset to process (interactive menu if --config omitted)
    # ------------------------------------------------------------------
    config_name = args.config or choose_subset()

    # ------------------------------------------------------------------
    # Authenticate with the Hub
    # ------------------------------------------------------------------
    token = ensure_hf_token(console=console)

    # ------------------------------------------------------------------
    # Load dataset from the Hub
    # ------------------------------------------------------------------
    console.print(Panel(
        f"[bold]Repository:[/bold] {args.repo}\n"
        f"[bold]Subset:[/bold] {config_name}\n"
        f"[bold]Text column:[/bold] {args.text_column}\n"
        f"[bold]Language filter:[/bold] {language_filter or 'none (all rows)'}\n"
        f"[bold]spaCy model:[/bold] {spacy_model}",
        title="[bold blue]Lemmatization Configuration[/bold blue]",
        border_style="blue"
    ))

    with console.status("[bold green]Loading dataset from Hugging Face Hub...", spinner="dots"):
        ds: Dataset = datasets.load_dataset(args.repo, name=config_name, split="train", token=token)
    console.print(f"[green]✓[/green] Loaded {len(ds):,} rows from '{args.repo}' (subset: {config_name})")

    if args.text_column not in ds.column_names:
        console.print(f"[red]✗[/red] Column '{args.text_column}' not found in the dataset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        raise ValueError(f"Column '{args.text_column}' not found in the dataset.")

    # ------------------------------------------------------------------
    # Ask user for processing preference (skipped when --mode is given)
    # ------------------------------------------------------------------
    if args.mode:
        process_choice = args.mode
        console.print(f"[green]✓[/green] Mode: [cyan]{process_choice}[/cyan] (from --mode)")
    else:
        process_choice = Prompt.ask(
            f"Process all articles or only those with empty '[cyan]{args.lemma_column}[/cyan]'?",
            choices=["all", "empty"],
            default="empty"
        )

    # ------------------------------------------------------------------
    # Load the spaCy model once
    # ------------------------------------------------------------------
    with console.status(f"[bold green]Loading spaCy model '{spacy_model}'...", spinner="dots"):
        nlp = load_spacy_model(spacy_model)
    # Long OCR is chunked to SPACY_MAX_CHUNK_CHARS before tagging (see
    # lemmatise_one), keeping each spaCy call well under the default 1M-char
    # limit, so nlp.max_length does not need raising.
    console.print(f"[green]✓[/green] Loaded spaCy model '{spacy_model}'")

    # ------------------------------------------------------------------
    # Lemmatise — crash-resumable: checkpoints to .cache_lemmas, resumes on
    # restart, and the cache is deleted only after a successful push.
    # ------------------------------------------------------------------
    # Per-language cache files: a crashed English pass must not feed its
    # rows into a resumed French pass (and vice versa).
    cache_suffix = f"_{language_filter}" if language_filter else ""
    cache_file = CACHE_DIR / f"{config_name}{cache_suffix}.json.gz"
    console.print(f"[blue]→[/blue] Applying lemmatisation – this can take a while…")
    result = lemmatise_dataset(
        ds,
        nlp,
        text_col=args.text_column,
        lemma_col=args.lemma_column,
        clean_col=args.clean_column,
        process_choice=process_choice,
        cache_file=cache_file,
        language_filter=language_filter,
    )
    if result is None:
        console.print("[green]✓[/green] No rows needed lemmatising. Nothing to do.")
        return
    ds = result

    # ------------------------------------------------------------------
    # Dry-run: stop before mutating the Hub or the cache
    # ------------------------------------------------------------------
    if args.dry_run:
        console.print(Panel(
            "[yellow]Dry run — dataset lemmatised in memory but NOT pushed; "
            "the resume cache is kept for a real run.[/yellow]",
            border_style="yellow",
        ))
        return

    # ------------------------------------------------------------------
    # Push updated dataset to the Hub
    # ------------------------------------------------------------------
    console.print(f"[blue]→[/blue] Pushing updated dataset back to {args.repo} (subset: {config_name})…")
    try:
        with console.status("[bold green]Uploading to Hugging Face Hub...", spinner="dots"):
            ds.push_to_hub(
                args.repo,
                config_name=config_name,
                token=token,
                max_shard_size=args.max_shard_size,
                commit_message=f"Add/update columns '{args.lemma_column}' and '{args.clean_column}' for {config_name} (French lemmatisation, mode: {process_choice})",
            )
    except Exception as e:  # noqa: BLE001
        console.print(Panel(
            f"[bold red]Push failed:[/bold red] {e}\n\n"
            f"[yellow]Computed lemmas remain cached in {cache_file}.[/yellow]\n"
            f"Re-run the script to resume from the cache without re-lemmatising.",
            title="[red]Error[/red]", border_style="red",
        ))
        logging.getLogger(__name__).error("Push error", exc_info=True)
        return

    # The cache only exists for resume; the push succeeded, so drop it.
    delete_cache(cache_file)

    # Summary table
    table = Table(title="Lemmatization Complete", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Repository", args.repo)
    table.add_row("Subset", config_name)
    table.add_row("Rows in subset", f"{len(ds):,}")
    table.add_row("Lemma column", args.lemma_column)
    table.add_row("Clean column", args.clean_column)
    table.add_row("Mode", process_choice)
    console.print(table)
    console.print("[green]✓[/green] Done!")

    
if __name__ == "__main__":
    main()
