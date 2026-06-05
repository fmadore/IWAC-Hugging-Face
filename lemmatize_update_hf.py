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
    python lemmatize_update_hf.py \
        --repo fmadore/islam-west-africa-collection \
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
from typing import List
import re
import unicodedata

import datasets
from datasets import Dataset
import spacy

# Make ``post-processing/_common.py`` importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "post-processing"))
from _common import ensure_hf_token  # noqa: E402

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


def normalize(text: str) -> str:
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


def lemmatise_batch(
    batch: dict,
    nlp,
    text_col: str,
    lemma_col: str,
    clean_col: str,
    process_choice: str,  # Added process_choice
) -> dict:
    """Apply lemmatisation and stop‑word removal to one batch of examples."""

    texts_in_batch: List[str] = batch[text_col]
    num_items_in_batch = len(texts_in_batch)

    # Initialize final lists to hold results for the entire batch
    final_lemmas: List[str] = [""] * num_items_in_batch
    final_clean: List[str] = [""] * num_items_in_batch

    indices_to_process = []
    content_to_process = []

    # Determine which items in the batch need processing
    if process_choice == "empty" and lemma_col in batch:
        for i in range(num_items_in_batch):
            # Check if existing lemma is present and not just whitespace
            if batch[lemma_col][i] and batch[lemma_col][i].strip():
                final_lemmas[i] = batch[lemma_col][i]
                # If lemma exists, clean version should also exist from a previous run
                if clean_col in batch and i < len(batch[clean_col]):
                    final_clean[i] = batch[clean_col][i]
                else:
                    # This case (lemma exists, clean does not) implies inconsistency
                    # or clean_col is new. For safety, it will remain "" or be generated
                    # if we decide to regenerate clean if missing.
                    # For now, it will be "" if not in batch[clean_col]
                    pass # final_clean[i] remains "" as initialized
            else:
                indices_to_process.append(i)
                content_to_process.append(texts_in_batch[i])
    else:  # Process all items if 'all' or if lemma_col doesn't exist yet
        indices_to_process = list(range(num_items_in_batch))
        content_to_process = texts_in_batch

    # Perform lemmatisation only on the content that needs it
    if content_to_process:
        # Normalize text before spaCy processing
        normalized_content_to_process = [normalize(text) for text in content_to_process]

        processed_lemmas_for_subset = []
        processed_clean_for_subset = []
        for doc in nlp.pipe(normalized_content_to_process, batch_size=32, n_process=1): # Use normalized content
            lemma_tokens = [tok.lemma_.lower() for tok in doc if tok.is_alpha]
            processed_lemmas_for_subset.append(" ".join(lemma_tokens))

            nostop_tokens = [tok for tok in lemma_tokens if tok not in nlp.Defaults.stop_words]
            processed_clean_for_subset.append(" ".join(nostop_tokens))
        
        # Populate the final lists with the newly processed data at correct original indices
        for idx_in_subset, original_batch_idx in enumerate(indices_to_process):
            final_lemmas[original_batch_idx] = processed_lemmas_for_subset[idx_in_subset]
            final_clean[original_batch_idx] = processed_clean_for_subset[idx_in_subset]

    return {lemma_col: final_lemmas, clean_col: final_clean}


def main():
    configure_logging()

    parser = argparse.ArgumentParser(description="Add lemmatised columns to a Hugging Face dataset")
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection", help="Dataset repo on the Hugging Face Hub (e.g. fmadore/islam-west-africa-collection)")
    parser.add_argument("--text-column", default="OCR", help="Name of the column containing the raw French text to process")
    parser.add_argument("--lemma-column", default="lemma_text", help="Column name for the lemmatised text")
    parser.add_argument("--clean-column", default="lemma_nostop", help="Column name for the lemmatised text with stop‑words removed")
    parser.add_argument("--spacy-model", default="fr_core_news_lg", help="spaCy model to use for French lemmatisation (use fr_core_news_lg for CPU)")
    parser.add_argument("--max-shard-size", default="1GB", help="Maximum Parquet shard size when pushing to the Hub")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Authenticate with the Hub
    # ------------------------------------------------------------------
    token = ensure_hf_token(console=console)

    # ------------------------------------------------------------------
    # Load dataset from the Hub
    # ------------------------------------------------------------------
    console.print(Panel(
        f"[bold]Repository:[/bold] {args.repo}\n"
        f"[bold]Text column:[/bold] {args.text_column}\n"
        f"[bold]spaCy model:[/bold] {args.spacy_model}",
        title="[bold blue]Lemmatization Configuration[/bold blue]",
        border_style="blue"
    ))
    
    with console.status("[bold green]Loading dataset from Hugging Face Hub...", spinner="dots"):
        ds: Dataset = datasets.load_dataset(args.repo, name="articles", split="train", token=token)
    console.print(f"[green]✓[/green] Loaded {len(ds):,} articles from '{args.repo}'")

    if args.text_column not in ds.column_names:
        console.print(f"[red]✗[/red] Column '{args.text_column}' not found in the dataset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        raise ValueError(f"Column '{args.text_column}' not found in the dataset.")

    # ------------------------------------------------------------------
    # Ask user for processing preference
    # ------------------------------------------------------------------
    process_choice = Prompt.ask(
        f"Process all articles or only those with empty '[cyan]{args.lemma_column}[/cyan]'?",
        choices=["all", "empty"],
        default="empty"
    )

    if process_choice == "empty":
        if args.lemma_column not in ds.column_names:
            console.print(
                f"[yellow]⚠[/yellow] Lemma column '{args.lemma_column}' not found. Processing all articles instead."
            )
        else:
            with console.status("[bold green]Filtering dataset...", spinner="dots"):
                original_row_count = len(ds)
                ds = ds.filter(lambda example: not example[args.lemma_column] or example[args.lemma_column].strip() == "")
                filtered_row_count = len(ds)
            console.print(f"[blue]→[/blue] Selected [bold]{filtered_row_count:,}[/bold] articles out of {original_row_count:,} for processing.")
            if filtered_row_count == 0:
                console.print(f"[green]✓[/green] No articles found with an empty '{args.lemma_column}'. Nothing to do.")
                return

    # ------------------------------------------------------------------
    # Load the spaCy model once
    # ------------------------------------------------------------------
    with console.status(f"[bold green]Loading spaCy model '{args.spacy_model}'...", spinner="dots"):
        nlp = load_spacy_model(args.spacy_model)
    console.print(f"[green]✓[/green] Loaded spaCy model '{args.spacy_model}'")

    # ------------------------------------------------------------------
    # Map lemmatisation over the dataset
    # ------------------------------------------------------------------
    console.print(f"[blue]→[/blue] Applying lemmatisation – this can take a while…")
    ds = ds.map(
        lemmatise_batch,
        batched=True,
        batch_size=1000,  # Reverted to larger batch size for full processing
        fn_kwargs={
            "nlp": nlp,
            "text_col": args.text_column,
            "lemma_col": args.lemma_column,
            "clean_col": args.clean_column,
            "process_choice": process_choice,  # Pass process_choice to the map function
        },
        desc="lemmatising",
    )

    # ------------------------------------------------------------------
    # Push updated dataset to the Hub
    # ------------------------------------------------------------------
    console.print(f"[blue]→[/blue] Pushing updated dataset back to {args.repo}…")
    with console.status("[bold green]Uploading to Hugging Face Hub...", spinner="dots"):
        ds.push_to_hub(
            args.repo,
            config_name="articles",
            token=token,
            max_shard_size=args.max_shard_size,
            commit_message=f"Add/update columns '{args.lemma_column}' and '{args.clean_column}' (French lemmatisation, mode: {process_choice})",
        )

    # Summary table
    table = Table(title="Lemmatization Complete", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Repository", args.repo)
    table.add_row("Articles processed", f"{len(ds):,}")
    table.add_row("Lemma column", args.lemma_column)
    table.add_row("Clean column", args.clean_column)
    table.add_row("Mode", process_choice)
    console.print(table)
    console.print("[green]✓[/green] Done!")

    
if __name__ == "__main__":
    main()
