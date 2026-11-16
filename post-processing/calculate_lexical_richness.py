#!/usr/bin/env python3
"""
calculate_lexical_richness.py
=============================

Adds lexical richness (MATTR - Moving Average Type-Token Ratio) and readability
(Flesch Reading Ease) columns to an existing Hugging Face dataset, based on the
'OCR' column.

The user is prompted to choose the dataset configuration. Column names:
"Richesse_Lexicale_OCR" (MATTR) and "Lisibilite_OCR" (Flesch).

Readability is only defined for French. ``Lisibilite_OCR`` applies the
Kandel–Moles French constants and French syllabification, so it is computed
only for rows whose sole ``language`` value is ``Français``; every other row — English, Arabic, Hausa, Ewé, Kabiyè, Dendi,
or no language at all — is stored as null, in every update mode, so a value
computed before this rule existed is cleared rather than kept. A French
formula applied to an African-language source ranks correct text as
unreadable, which is worse than no score. MATTR is kept for every language:
it counts types over tokens and carries no lexicon.

A Flesch score also needs enough evidence to mean anything: a text with fewer
than ``MIN_READABILITY_WORDS`` words or ``MIN_READABILITY_SENTENCES``
sentences, more than ``MAX_WORDS_PER_SENTENCE`` words per sentence (OCR that
lost its full stops), or no letters at all is stored as null. The thresholds
are part of the readability configuration hash, so changing them recomputes.

MATTR tokenizes through :func:`iwac_common.text_utils.tokenize_words` with the
row's ``language``, exactly as ``nb_mots`` does, so both count the same tokens.

Usage
-----
    python post-processing/calculate_lexical_richness.py [--repo USER/DATASET]
        [--config SUBSET] [--update-mode missing|all] [--dry-run] [-y]

Examples:
    python post-processing/calculate_lexical_richness.py          # fully interactive
    python post-processing/calculate_lexical_richness.py --config articles -y   # headless, missing
    python post-processing/calculate_lexical_richness.py --config articles --update-mode all -y --dry-run

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
from importlib.metadata import version
from collections import Counter
from typing import List, Dict, Any, Optional

from datasets import Value
from textstat.textstat import textstatistics

# Make ``post-processing/_common.py`` and ``iwac_common`` importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from _common import (  # noqa: E402
    display_text_stats,
    confirm_full_recompute,
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
)
from iwac_common.text_utils import tokenize_words, language_labels, TOKENIZER_VERSION  # noqa: E402
from iwac_common.enrichment import (  # noqa: E402
    changed_row_count, config_fingerprint, fingerprint, provenance_columns, config_json,
)

# Disable symlinks warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich import box
from dotenv import load_dotenv
from iwac_common.paths import workspace_root

# Load .env so a non-interactive run finds HF_TOKEN instead of falling through
# to an interactive login that cannot read stdin.
load_dotenv(workspace_root() / ".env")

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)

logger = logging.getLogger("lexical_richness")

TEXT_COLUMN = "OCR"
RICHNESS_COLUMN = "Richesse_Lexicale_OCR"
READABILITY_COLUMN = "Lisibilite_OCR"
LANGUAGE_COLUMN = "language"
METRIC_COLUMNS = [RICHNESS_COLUMN, READABILITY_COLUMN]
#: Subsets with an OCR column; ``index`` and ``images`` have none.
LEXICAL_SUBSETS = ["articles", "publications", "documents", "references", "audiovisual"]

#: The only language the Flesch column is defined for (the label Omeka uses).
READABILITY_LANGUAGE = "Français"
READABILITY_TEXTSTAT_LANG = "fr"
# Minimum evidence for a Flesch score. Flesch averages words per sentence and
# syllables per word; below these counts one OCR glitch dominates the mean,
# and beyond 100 words per sentence the full stops were lost, not the reader.
MIN_READABILITY_WORDS = 100
MIN_READABILITY_SENTENCES = 3
MAX_WORDS_PER_SENTENCE = 100

# A private instance: the language is set here, where the score is computed,
# never through textstat's process-wide ``set_lang``. With "fr", textstat's
# flesch_reading_ease applies the Kandel–Moles French constants and French
# syllabification, not the English Flesch formula.
_FRENCH_TEXTSTAT = textstatistics()
_FRENCH_TEXTSTAT.set_lang(READABILITY_TEXTSTAT_LANG)


def calculate_mattr(text: str, window_size: int = 50, language: Any = "Français") -> Optional[float]:
    """Compute Moving Average Type-Token Ratio (MATTR).

    Unlike raw TTR, MATTR is not biased by text length because it uses a
    fixed-size sliding window. ``language`` is the row's raw ``language``
    value and goes to :func:`tokenize_words` unchanged, as in ``nb_mots``:
    for a French row an elided clitic is its own token (``l'``), counted
    like ``le``/``la``; other languages keep ``da'wa`` whole.

    Texts with fewer tokens than ``window_size`` return None — a plain TTR
    fallback would mix two incomparable metrics in the same column.

    Returns None if the text is missing, has no tokens, or is too short
    for MATTR.
    """
    if window_size < 1:
        raise ValueError("MATTR window size must be positive")
    if not text or not isinstance(text, str):
        return None

    tokens = tokenize_words(text, language=language)

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


def readability_evidence(text: Any) -> Optional[str]:
    """Why ``text`` is too thin for a Flesch score, or ``None`` when it is not.

    Reasons: ``"no_letters"``, ``"too_few_words"``, ``"too_few_sentences"``,
    ``"implausible_sentence_length"``. Words and sentences are counted with
    textstat's own counters, the ones the score itself uses.
    """
    if not isinstance(text, str) or not any(char.isalpha() for char in text):
        return "no_letters"
    words = _FRENCH_TEXTSTAT.lexicon_count(text)
    if words < MIN_READABILITY_WORDS:
        return "too_few_words"
    sentences = _FRENCH_TEXTSTAT.sentence_count(text)
    if sentences < MIN_READABILITY_SENTENCES:
        return "too_few_sentences"
    if words / sentences > MAX_WORDS_PER_SENTENCE:
        return "implausible_sentence_length"
    return None


def calculate_readability(text: str) -> Optional[float]:
    """French Flesch Reading Ease of ``text``.

    Returns None when the text lacks the evidence :func:`readability_evidence`
    requires, or when textstat fails.
    """
    if readability_evidence(text) is not None:
        return None
    try:
        return _FRENCH_TEXTSTAT.flesch_reading_ease(text)
    except Exception:
        return None


def metric_settings(window_size: int) -> Dict[str, Dict[str, Any]]:
    """The configuration each metric's ``*_config_hash`` is computed from."""
    return {
        RICHNESS_COLUMN: {"metric": "MATTR", "window": window_size, "tokenizer": TOKENIZER_VERSION,
                          "tokenizer_language": "row language"},
        READABILITY_COLUMN: {
            "metric": "French Flesch", "language_policy": "monolingual-fr",
            "textstat": version("textstat"), "textstat_lang": READABILITY_TEXTSTAT_LANG,
            "min_words": MIN_READABILITY_WORDS, "min_sentences": MIN_READABILITY_SENTENCES,
            "max_words_per_sentence": MAX_WORDS_PER_SENTENCE,
        },
    }


def _bump(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def compute_metrics_batch(
    batch: Dict[str, List[Any]],
    *,
    text_col: str,
    richness_col: str,
    readability_col: str,
    update_mode: str,
    window_size: int,
    error_counter: Dict[str, int],
    language_col: str = "language",
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
            readability computation failures (no provenance, retried next
            run); 'readability_insufficient' counts French texts below the
            evidence thresholds (null with valid provenance);
            'readability_not_french' counts rows nulled because their
            language is unknown, multilingual, or not French.
        language_col: Column holding the pipe-joined ``language`` labels.
            Without it no row can be shown to be French, so readability is
            null throughout.

    Returns:
        Batch with metrics added.
    """
    texts = batch[text_col]
    existing_richness = batch.get(richness_col, [None] * len(texts))
    existing_readability = batch.get(readability_col, [None] * len(texts))

    languages = batch.get(language_col, [None] * len(texts))

    richness_results = list(existing_richness)
    readability_results = list(existing_readability)
    settings = metric_settings(window_size)
    settings = {richness_col: settings[RICHNESS_COLUMN], readability_col: settings[READABILITY_COLUMN]}
    configs = {column: config_fingerprint(value) for column, value in settings.items()}
    descriptions = {column: config_json(value) for column, value in settings.items()}
    for column in configs:
        batch[f"{column}_config_json"] = [None] * len(texts)
    hashes = {column: provenance_columns(column) for column in configs}
    previous = {name: list(batch.get(name, [None] * len(texts)))
                for columns in hashes.values() for name in columns}
    for name in previous:
        batch[name] = list(previous[name])

    def needs(column: str, row: int, input_hash: str) -> bool:
        if update_mode != "missing":
            return True
        input_col, config_col = hashes[column]
        return previous[input_col][row] != input_hash or previous[config_col][row] != configs[column]

    for i, text in enumerate(texts):
        text_str = str(text) if text is not None else ""
        has_content = text is not None and text_str.strip()
        labels = language_labels(languages[i])
        is_french = labels == [READABILITY_LANGUAGE]
        input_hash = fingerprint(text_str, labels)
        readability_failed = False

        if needs(richness_col, i, input_hash):
            result = calculate_mattr(text_str, window_size, language=languages[i])
            if result is None and has_content:
                # Non-empty text without a MATTR value means it has fewer
                # tokens than the window — too short for MATTR, not an error.
                _bump(error_counter, "richness_too_short")
            richness_results[i] = result

        if not is_french:
            # Regardless of update mode: a French formula applied to another
            # language is not a stale value to keep, it is a wrong one.
            if has_content:
                _bump(error_counter, "readability_not_french")
            readability_results[i] = None
        elif needs(readability_col, i, input_hash):
            if has_content and readability_evidence(text_str) is not None:
                # Deterministically too thin: a null with valid provenance.
                _bump(error_counter, "readability_insufficient")
                result = None
            else:
                result = calculate_readability(text_str)
                if result is None and has_content:
                    _bump(error_counter, "readability_failed")
                    readability_failed = True
            readability_results[i] = result
        for column in configs:
            input_col, config_col = hashes[column]
            successful = not (column == readability_col and readability_failed)
            batch[input_col][i] = input_hash if successful else None
            batch[config_col][i] = configs[column] if successful else None
            batch[f"{column}_config_json"][i] = descriptions[column] if successful else None

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


def output_columns() -> List[str]:
    """The metric columns plus the private provenance columns written with them."""
    names = list(METRIC_COLUMNS)
    for column in METRIC_COLUMNS:
        names += [*provenance_columns(column), f"{column}_config_json"]
    return names


def compute_metrics(ds, *, update_mode: str, window_size: int, batch_size: int):
    """Map :func:`compute_metrics_batch` over ``ds``; return ``(dataset, counters)``."""
    error_counter: Dict[str, int] = {
        "richness_too_short": 0, "readability_failed": 0,
        "readability_insufficient": 0, "readability_not_french": 0,
    }
    if LANGUAGE_COLUMN not in ds.column_names:
        console.print(
            f"[yellow]⚠[/yellow] No '{LANGUAGE_COLUMN}' column: no row can be shown to be "
            f"French, so {READABILITY_COLUMN} will be null throughout."
        )
    # Declared, not inferred: a first batch of non-French rows is all None
    # and would otherwise fix the column's type as null.
    output_types = {name: Value("string") for name in output_columns()}
    output_types.update({column: Value("float64") for column in METRIC_COLUMNS})
    processed = map_with_progress(
        ds,
        lambda batch: compute_metrics_batch(
            batch, text_col=TEXT_COLUMN, richness_col=RICHNESS_COLUMN,
            readability_col=READABILITY_COLUMN, update_mode=update_mode,
            window_size=window_size, error_counter=error_counter,
            language_col=LANGUAGE_COLUMN,
        ),
        batch_size=batch_size,
        description="[cyan]Computing metrics",
        console=console,
        output_types=output_types,
    )
    console.print("[green]✓[/green] Metrics computation complete.")
    return processed, error_counter


def report_counters(error_counter: Dict[str, int], window_size: int) -> None:
    messages = (
        ("richness_too_short", "ℹ", f"texts too short for MATTR (< {window_size} tokens) — stored as None."),
        ("readability_insufficient", "ℹ",
         f"French texts below the readability evidence thresholds (< {MIN_READABILITY_WORDS} "
         f"words, < {MIN_READABILITY_SENTENCES} sentences or > {MAX_WORDS_PER_SENTENCE} "
         f"words/sentence) — {READABILITY_COLUMN} stored as None."),
        ("readability_not_french", "ℹ",
         f"non-French texts: {READABILITY_COLUMN} stored as None (the French Flesch formula does not apply)."),
        ("readability_failed", "⚠", "readability failures (stored as None, retried next run)."),
    )
    for key, icon, message in messages:
        if error_counter.get(key):
            console.print(f"[yellow]{icon}[/yellow] {error_counter[key]:,} {message}")


def print_sample(ds, limit: int = 5) -> None:
    table = Table(title="Sample results", box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("MATTR", style="green", justify="right")
    table.add_column("Flesch", style="green", justify="right")
    sample = ds.select(range(min(limit, len(ds))))
    for i, (r, f) in enumerate(zip(sample[RICHNESS_COLUMN], sample[READABILITY_COLUMN]), 1):
        table.add_row(
            str(i),
            f"{r:.4f}" if r is not None else "[dim]None[/dim]",
            f"{f:.2f}" if f is not None else "[dim]None[/dim]",
        )
    console.print(table)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
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
        choices=LEXICAL_SUBSETS,
        default=None,
        help="Dataset configuration/subset to process (skips the interactive picker)."
    )
    add_update_mode_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics and report how many rows would change, but do not push to Hub."
    )
    args = parser.parse_args(argv)
    if args.window_size < 1 or args.batch_size < 1:
        parser.error("--window-size and --batch-size must be positive")
    return args


def push_metrics(ds, *, args, config_name: str, update_mode: str, token: str,
                 source_revision: Optional[str], changed: int) -> int:
    commit_message = (
        f"Add/update '{RICHNESS_COLUMN}' (MATTR, window={args.window_size}, tokenizer "
        f"v{TOKENIZER_VERSION}) and '{READABILITY_COLUMN}' (French Flesch, textstat "
        f"{version('textstat')}, monolingual-French rows with >= {MIN_READABILITY_WORDS} words) "
        f"from '{TEXT_COLUMN}' (config: {config_name}, mode: {update_mode})"
    )
    if not push_dataset(
        ds,
        repo_id=args.repo,
        config_name=config_name,
        commit_message=commit_message,
        token=token,
        max_shard_size=args.max_shard_size,
        console=console,
        expected_revision=source_revision,
    ):
        return 1
    console.print(Panel(
        f"[green]✓[/green] {RICHNESS_COLUMN}, {READABILITY_COLUMN} written for "
        f"[bold]{config_name}[/bold]\n"
        f"[dim]Rows changed: {changed:,} of {len(ds):,} · MATTR window: {args.window_size} tokens · "
        f"Repository: {args.repo}\n"
        "Remember: the public dataset only changes when "
        "post-processing/publish_public.py is re-run.[/dim]",
        title="[bold green]Done[/bold green]",
        border_style="green",
    ))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    token = ensure_hf_token(console=console)
    config_name = resolve_config(
        args.repo, token=token, cli_config=args.config,
        restrict_to=LEXICAL_SUBSETS, console=console,
    )
    update_mode = resolve_update_mode(args, console=console)
    display_config_panel(args.repo, config_name, update_mode, args.batch_size,
                         args.window_size, args.dry_run)

    ds = load_hub_dataset(args.repo, config_name, token=token, console=console)
    source_revision = getattr(ds, "_iwac_source_revision", None)
    if TEXT_COLUMN not in ds.column_names:
        console.print(f"[red]✗[/red] Source column '{TEXT_COLUMN}' not found in dataset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        return 1
    if not confirm_full_recompute(args, update_mode, f"{RICHNESS_COLUMN} and {READABILITY_COLUMN}"):
        console.print("[yellow]ℹ[/yellow] Cancelled; existing metrics kept.")
        return 0
    if update_mode == "missing":
        console.print("[blue]→[/blue] Rows whose provenance matches the current text, language "
                      "and configuration are kept; every other row is recomputed.")
    display_text_stats(ds[TEXT_COLUMN], TEXT_COLUMN, console=console)

    processed, error_counter = compute_metrics(
        ds, update_mode=update_mode, window_size=args.window_size, batch_size=args.batch_size,
    )
    report_counters(error_counter, args.window_size)
    processed = reorder_columns_after(processed, METRIC_COLUMNS, "nb_mots", console=console)
    print_sample(processed)
    changed = changed_row_count(ds, processed, output_columns())
    console.print(f"[blue]→[/blue] Rows whose metrics or provenance change: [bold]{changed:,}[/bold]")

    if args.dry_run:
        print_dry_run_panel(repo_id=args.repo, config_name=config_name, n_rows=len(processed),
                            extra=f"[dim]Rows that would change: {changed:,}[/dim]", console=console)
        return 0
    if not changed:
        console.print("[green]✓[/green] Every metric and its provenance are already current. Nothing to push.")
        return 0
    return push_metrics(processed, args=args, config_name=config_name, update_mode=update_mode,
                        token=token, source_revision=source_revision, changed=changed)


if __name__ == "__main__":
    raise SystemExit(main())
