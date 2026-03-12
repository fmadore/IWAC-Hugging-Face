#!/usr/bin/env python3
"""
semantic_embedding.py
=====================

Adds semantic embedding column to the 'articles' subset of a Hugging Face
dataset, based on the 'OCR' column (full article text). Uses Google's
gemini-embedding-2-preview model via the Gemini API for high-quality
multilingual embeddings.

Long texts exceeding the model's 8192-token limit are split into overlapping
chunks, each chunk is embedded separately, and the chunk embeddings are
averaged into a single vector per row.

The new column name is: "embedding_OCR".

Usage
-----
    python post-processing/semantic_embedding.py [--repo USER/DATASET]

Example:
    python post-processing/semantic_embedding.py --dry-run
    python post-processing/semantic_embedding.py --dimensionality 768

Environment Variables
---------------------
GOOGLE_API_KEY   API key for the Gemini API (or GEMINI_API_KEY).
HF_TOKEN         Personal access token for the Hugging Face Hub.

Dependencies
------------
    pip install google-genai datasets huggingface_hub rich
"""
import argparse
import gzip
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import get_token, login
from google import genai
from google.genai import types
import uuid

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Disable symlinks warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn,
)
from rich.logging import RichHandler
from rich.prompt import Prompt
from rich import box

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)

# --- Constants ---
CONFIG_NAME = "articles"
TEXT_COLUMN = "OCR"
EMBEDDING_COLUMN = "embedding_OCR"
MODEL_NAME = "gemini-embedding-2-preview"
# Rough estimate: ~3.5 chars/token for French → 8192 tokens ≈ 28K chars
CHUNK_SIZE = 28_000
CHUNK_OVERLAP = 2_000
DEFAULT_DIMENSIONALITY = 768
DEFAULT_BATCH_SIZE = 20
MAX_RETRIES = 6
BASE_RETRY_DELAY = 5  # seconds
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache_embeddings"
CACHE_FILE = CACHE_DIR / "ocr_embeddings.json.gz"
CHECKPOINT_EVERY = 5  # save cache every N API batches


# --- Cache helpers ---

def load_cache() -> Dict[str, List[float]]:
    """Load cached embeddings from gzipped JSON. Returns {o_id: embedding}."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with gzip.open(CACHE_FILE, "rt", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} cached embeddings from {CACHE_FILE}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load cache ({e}), starting fresh.")
        return {}


def save_cache(cache: Dict[str, List[float]]) -> None:
    """Save embeddings cache to gzipped JSON."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".tmp.gz")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(cache, f)
        tmp.replace(CACHE_FILE)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
        if tmp.exists():
            tmp.unlink()


def delete_cache() -> None:
    """Remove the cache file after a successful push."""
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            logger.info("Cache file deleted after successful push.")
    except Exception as e:
        logger.warning(f"Failed to delete cache: {e}")


def is_empty_embedding(emb: Any) -> bool:
    """Check whether an embedding value is missing or invalid."""
    if emb is None:
        return True
    if isinstance(emb, list):
        if len(emb) == 0:
            return True
        if all(x == 0.0 for x in emb):
            return True
    return False


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks that fit within the Gemini token limit.

    Short texts are returned as a single-element list. Long texts are split at
    chunk_size boundaries with `overlap` characters of overlap between
    consecutive chunks to preserve context continuity.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """Average multiple embedding vectors into a single vector (mean pooling)."""
    if len(embeddings) == 1:
        return embeddings[0]
    dim = len(embeddings[0])
    averaged = [0.0] * dim
    for emb in embeddings:
        for j in range(dim):
            averaged[j] += emb[j]
    n = len(embeddings)
    return [v / n for v in averaged]


def choose_update_mode() -> str:
    """Prompt the user to choose the embedding update mode."""
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


def validate_existing_embeddings(
    ds, embedding_col: str, expected_dim: int
) -> None:
    """Validate that existing embeddings are compatible with the target dimensionality."""
    if embedding_col not in ds.column_names:
        return

    for emb in ds[embedding_col]:
        if not is_empty_embedding(emb):
            actual_dim = len(emb)
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Existing embeddings have dimension {actual_dim}, but target "
                    f"dimensionality is {expected_dim}. "
                    f"Use update mode 'all' to recompute all embeddings."
                )
            break


def embed_texts_with_retry(
    client: genai.Client,
    texts: List[str],
    task_type: str,
    dimensionality: int,
) -> List[List[float]]:
    """Call Gemini embed_content with exponential backoff for rate limiting."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=dimensionality,
                ),
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate limited (attempt {attempt + 1}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                wait = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def display_config_panel(
    repo_id: str, model_name: str, update_mode: str,
    batch_size: int, dimensionality: int, task_type: str, dry_run: bool,
):
    """Display configuration in a Rich panel."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repository", repo_id)
    table.add_row("Configuration", CONFIG_NAME)
    table.add_row("Source Column", TEXT_COLUMN)
    table.add_row("Embedding Column", EMBEDDING_COLUMN)
    table.add_row("Model", model_name)
    table.add_row("Task Type", task_type)
    table.add_row("Output Dimensionality", str(dimensionality))
    table.add_row("API Batch Size", str(batch_size))
    table.add_row("Update Mode", update_mode)
    table.add_row("Chunk Size", f"{CHUNK_SIZE:,} chars")
    table.add_row("Chunk Overlap", f"{CHUNK_OVERLAP:,} chars")
    table.add_row("Long Text Strategy", "chunk → embed each → average")
    if dry_run:
        table.add_row("Dry Run", "[yellow]YES — no changes will be pushed[/yellow]")

    console.print(Panel(table, title="[bold blue]Semantic Embedding Configuration", border_style="blue"))


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
        lengths = [len(str(t)) for t in non_empty_texts]
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        will_chunk = sum(1 for l in lengths if l > CHUNK_SIZE)
        table.add_row("Avg. text length", f"{avg_length:,.0f} characters")
        table.add_row("Max text length", f"{max_length:,} characters")
        if will_chunk > 0:
            total_chunks = sum(len(chunk_text(str(t))) for t in non_empty_texts if len(str(t)) > CHUNK_SIZE)
            table.add_row("Will be chunked", f"[yellow]{will_chunk}[/yellow] texts > {CHUNK_SIZE:,} chars ({total_chunks} total chunks)")

    console.print(table)
    return len(non_empty_texts)


def display_embedding_stats(existing_embeddings: List[Any]) -> tuple[int, int]:
    """Display statistics about existing embeddings and return (valid, missing) counts."""
    valid_embeddings = 0
    empty_embeddings = 0

    for emb in existing_embeddings:
        if is_empty_embedding(emb):
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


def _save_completed_to_cache(
    cache: Dict[str, List[float]],
    row_chunks: List[tuple[int, List[str]]],
    flat_embeddings: List[Any],
    row_ids: List[Any],
    all_embeddings: List[Any],
) -> None:
    """Reassemble completed chunk embeddings into row embeddings and save cache."""
    flat_offset = 0
    updated = 0
    for row_idx, chunks in row_chunks:
        chunk_embs = [
            flat_embeddings[flat_offset + k]
            for k in range(len(chunks))
            if flat_embeddings[flat_offset + k] is not None
        ]
        if chunk_embs:
            averaged = average_embeddings(chunk_embs)
            all_embeddings[row_idx] = averaged
            oid_str = str(row_ids[row_idx])
            if oid_str not in cache:
                cache[oid_str] = averaged
                updated += 1
        flat_offset += len(chunks)
    if updated > 0:
        save_cache(cache)
        logger.info(f"Checkpoint: saved {len(cache)} total embeddings to cache")


def main():
    parser = argparse.ArgumentParser(
        description="Add semantic embedding column ('embedding_OCR') to the "
                    "'articles' subset using Google Gemini embeddings on the full OCR text."
    )
    parser.add_argument(
        "--repo",
        default="fmadore/islam-west-africa-collection",
        help="Repository ID on Hugging Face Hub.",
    )
    parser.add_argument(
        "--dimensionality",
        type=int,
        default=DEFAULT_DIMENSIONALITY,
        help=f"Output embedding dimensionality (default: {DEFAULT_DIMENSIONALITY}). "
             "Supported: 128-3072.",
    )
    parser.add_argument(
        "--task-type",
        default="RETRIEVAL_DOCUMENT",
        choices=[
            "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY",
            "CLASSIFICATION", "CLUSTERING",
        ],
        help="Embedding task type (default: RETRIEVAL_DOCUMENT).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of texts per Gemini API call (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls to avoid rate limits (default: 0.5).",
    )
    parser.add_argument(
        "--max-shard-size",
        default="1GB",
        help="Maximum Parquet shard size when pushing to Hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute embeddings but do not push to Hub.",
    )

    args = parser.parse_args()

    repo_id = args.repo
    dimensionality = args.dimensionality
    task_type = args.task_type
    batch_size = args.batch_size
    delay = args.delay
    max_shard_size = args.max_shard_size
    dry_run = args.dry_run

    # --- Step 1: Authentication ---
    console.print("\n[bold cyan]Step 1:[/bold cyan] Authenticating...")

    # Gemini API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]✗[/red] GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment.")
        return
    console.print("[green]✓[/green] Gemini API key found.")

    # Hugging Face token
    hf_token = os.getenv("HF_TOKEN") or get_token()
    if not hf_token:
        console.print("[yellow]ℹ[/yellow] HF token not found. Attempting interactive login...")
        try:
            login()
            hf_token = get_token()
            if not hf_token:
                console.print("[red]✗[/red] Interactive login failed. Please set HF_TOKEN.")
                return
        except Exception as e:
            console.print(f"[red]✗[/red] Login error: {e}")
            return
    console.print("[green]✓[/green] Hugging Face authenticated.")

    # --- Step 2: Initialize Gemini client ---
    console.print("\n[bold cyan]Step 2:[/bold cyan] Initializing Gemini client...")
    try:
        client = genai.Client(api_key=api_key)
        # Test the connection with a tiny embedding
        test_response = client.models.embed_content(
            model=MODEL_NAME,
            contents=["test"],
            config=types.EmbedContentConfig(output_dimensionality=dimensionality),
        )
        actual_dim = len(test_response.embeddings[0].values)
        console.print(f"[green]✓[/green] Gemini client ready. Model: [cyan]{MODEL_NAME}[/cyan]")
        console.print(f"[blue]→[/blue] Output dimensionality: [cyan]{actual_dim}[/cyan]")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize Gemini client: {e}")
        return

    # --- Update mode selection ---
    update_mode = choose_update_mode()
    console.print(f"[green]✓[/green] Update mode: [cyan]{update_mode}[/cyan]")

    # --- Display configuration ---
    console.print()
    display_config_panel(repo_id, MODEL_NAME, update_mode, batch_size, dimensionality, task_type, dry_run)

    # --- Step 3: Load dataset ---
    console.print(f"\n[bold cyan]Step 3:[/bold cyan] Loading dataset...")
    try:
        with console.status(f"[bold green]Loading '{repo_id}' (config: {CONFIG_NAME})...", spinner="dots"):
            ds = load_dataset(repo_id, name=CONFIG_NAME, split="train", token=hf_token)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load dataset: {e}")
        return

    console.print(f"[green]✓[/green] Dataset loaded: [cyan]{len(ds)}[/cyan] rows")

    # --- Column checks ---
    if TEXT_COLUMN not in ds.column_names:
        console.print(f"[red]✗[/red] Source column '{TEXT_COLUMN}' not found in dataset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        return

    # --- Dimension consistency check ---
    if update_mode == "missing" and EMBEDDING_COLUMN in ds.column_names:
        try:
            validate_existing_embeddings(ds, EMBEDDING_COLUMN, actual_dim)
            console.print(f"[green]✓[/green] Existing embeddings are compatible (dim={actual_dim}).")
        except ValueError as e:
            console.print(f"[red]✗[/red] {e}")
            return

    if EMBEDDING_COLUMN in ds.column_names:
        if update_mode == "all":
            console.print(f"[yellow]⚠[/yellow] Embedding column '{EMBEDDING_COLUMN}' exists and will be overwritten.")
        else:
            console.print(f"[yellow]ℹ[/yellow] Embedding column '{EMBEDDING_COLUMN}' exists. Only missing values will be computed.")
    else:
        console.print(f"[blue]→[/blue] Embedding column '{EMBEDDING_COLUMN}' will be created.")

    # --- Step 4: Analyze source data ---
    console.print(f"\n[bold cyan]Step 4:[/bold cyan] Analyzing source data...")
    texts = ds[TEXT_COLUMN]
    display_text_stats(texts, TEXT_COLUMN)

    # --- Load embedding cache ---
    cache = load_cache()
    if cache:
        console.print(f"[green]✓[/green] Resuming with [cyan]{len(cache)}[/cyan] cached embeddings")

    # --- Identify rows to process ---
    existing_embeddings = ds[EMBEDDING_COLUMN] if EMBEDDING_COLUMN in ds.column_names else [None] * len(ds)
    row_ids = ds["o:id"]  # stable row identifier for cache keys

    if update_mode == "missing" and EMBEDDING_COLUMN in ds.column_names:
        valid_count, missing_count = display_embedding_stats(existing_embeddings)
        if missing_count == 0 and not cache:
            console.print(Panel(
                "[green]All embeddings are already computed![/green]\n\nNo processing needed.",
                title="Nothing to do",
                border_style="green",
            ))
            return

    # Build the full embeddings list, pre-filling from cache
    all_embeddings: List[Any] = list(existing_embeddings)
    cache_hits = 0
    for i, oid in enumerate(row_ids):
        oid_str = str(oid)
        if oid_str in cache:
            all_embeddings[i] = cache[oid_str]
            cache_hits += 1

    if cache_hits > 0:
        console.print(f"[green]✓[/green] Restored [cyan]{cache_hits}[/cyan] embeddings from cache")

    # Determine which rows still need embedding
    error_counter = {"failed": 0, "chunked": 0, "total_chunks": 0}
    indices_to_process = []
    for i, (text, emb) in enumerate(zip(texts, all_embeddings)):
        if update_mode == "all":
            if text is not None and str(text).strip():
                # In 'all' mode, skip only if already in cache (from this run)
                oid_str = str(row_ids[i])
                if oid_str not in cache:
                    indices_to_process.append(i)
        elif update_mode == "missing":
            if is_empty_embedding(emb) and text is not None and str(text).strip():
                indices_to_process.append(i)

    if not indices_to_process:
        console.print(Panel(
            "[green]No rows to process![/green]\n\n"
            "All embeddings are computed (including cached results).\n"
            "Proceeding to update dataset and push.",
            title="Cache Complete",
            border_style="green",
        ))
    else:
        console.print(f"[blue]→[/blue] [cyan]{len(indices_to_process)}[/cyan] rows to embed")

        # --- Step 5: Compute embeddings ---
        console.print(f"\n[bold cyan]Step 5:[/bold cyan] Computing embeddings via Gemini API...")

        # Pre-compute chunks for all texts to process.
        # Each entry: (row_index, [chunk1, chunk2, ...])
        row_chunks: List[tuple[int, List[str]]] = []
        for idx in indices_to_process:
            text = str(texts[idx])
            chunks = chunk_text(text)
            if len(chunks) > 1:
                error_counter["chunked"] += 1
                error_counter["total_chunks"] += len(chunks)
            row_chunks.append((idx, chunks))

        # Flatten all chunks into a single list for batched API calls.
        flat_chunks: List[str] = []
        for _, chunks in row_chunks:
            flat_chunks.extend(chunks)

        console.print(
            f"[blue]→[/blue] {len(flat_chunks)} total chunks from "
            f"{len(indices_to_process)} articles "
            f"({error_counter['chunked']} multi-chunk)"
        )

        # Embed all chunks in batches
        flat_embeddings: List[Any] = [None] * len(flat_chunks)
        batch_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Embedding chunks", total=len(flat_chunks))

            for batch_start in range(0, len(flat_chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(flat_chunks))
                batch_texts = flat_chunks[batch_start:batch_end]

                try:
                    embeddings = embed_texts_with_retry(
                        client, batch_texts, task_type, dimensionality,
                    )
                    for i, emb in enumerate(embeddings):
                        flat_embeddings[batch_start + i] = emb
                except Exception as e:
                    logger.error(f"Batch failed (chunks {batch_start}-{batch_end - 1}): {e}")
                    error_counter["failed"] += len(batch_texts)

                progress.update(task, advance=len(batch_texts))
                batch_count += 1

                # Checkpoint: reassemble completed rows and save cache
                if batch_count % CHECKPOINT_EVERY == 0:
                    _save_completed_to_cache(
                        cache, row_chunks, flat_embeddings, row_ids, all_embeddings,
                    )

                # Delay between API calls to respect rate limits
                if delay > 0 and batch_end < len(flat_chunks):
                    time.sleep(delay)

        # Final reassemble: group chunk embeddings by row and average
        _save_completed_to_cache(
            cache, row_chunks, flat_embeddings, row_ids, all_embeddings,
        )

        console.print("[green]✓[/green] Embedding computation complete.")

        if error_counter["chunked"] > 0:
            console.print(
                f"[yellow]ℹ[/yellow] {error_counter['chunked']} long articles were split "
                f"into {error_counter['total_chunks']} chunks (averaged back to 1 vector each)."
            )
        if error_counter["failed"] > 0:
            console.print(
                f"[yellow]⚠[/yellow] {error_counter['failed']} chunks failed during "
                f"encoding."
            )

    # --- Update the dataset ---
    console.print(f"\n[bold cyan]Step 6:[/bold cyan] Updating dataset...")

    ds_processed = ds.map(
        lambda batch, idx: {EMBEDDING_COLUMN: [all_embeddings[i] for i in idx]},
        batched=True,
        batch_size=1000,
        with_indices=True,
        load_from_cache_file=False,
        new_fingerprint=str(uuid.uuid4()),
    )

    # --- Verify results ---
    console.print("\n[bold]Sample embeddings (first 3 non-empty):[/bold]")
    shown = 0
    for i, emb in enumerate(ds_processed[EMBEDDING_COLUMN]):
        if shown >= 3:
            break
        if not is_empty_embedding(emb):
            console.print(f"  [cyan]#{i+1}[/cyan]: dim={len(emb)}, values=[{emb[0]:.4f}, {emb[1]:.4f}, ...]")
            shown += 1

    # --- Reorder columns: place embedding_OCR after OCR ---
    if TEXT_COLUMN in ds_processed.column_names:
        existing_columns = list(ds_processed.column_names)
        insert_index = existing_columns.index(TEXT_COLUMN) + 1

        new_columns = existing_columns[:insert_index]
        if EMBEDDING_COLUMN in existing_columns and EMBEDDING_COLUMN not in new_columns:
            new_columns.append(EMBEDDING_COLUMN)
        for col in existing_columns[insert_index:]:
            if col not in new_columns:
                new_columns.append(col)

        ds_processed = ds_processed.select_columns(new_columns)
        console.print(f"[blue]→[/blue] Columns reordered ('{EMBEDDING_COLUMN}' after '{TEXT_COLUMN}')")

    # --- Step 7: Push to Hub ---
    if dry_run:
        console.print(Panel(
            "[yellow]Dry run mode — no changes pushed to Hub.[/yellow]\n\n"
            f"Would have pushed [cyan]{len(ds_processed)}[/cyan] rows to "
            f"[cyan]{repo_id}[/cyan] (config: {CONFIG_NAME}).\n\n"
            f"[yellow]Embeddings are cached in {CACHE_FILE}[/yellow]\n"
            f"Re-run without --dry-run to push (cached results will be reused).",
            title="Dry Run Complete",
            border_style="yellow",
        ))
        return

    console.print(f"\n[bold cyan]Step 7:[/bold cyan] Pushing to Hugging Face Hub...")

    try:
        commit_message = (
            f"Add/update '{EMBEDDING_COLUMN}' embeddings using {MODEL_NAME} "
            f"(dim={dimensionality}, task={task_type}, config={CONFIG_NAME})"
        )

        with console.status("[bold green]Pushing dataset to Hub...", spinner="dots"):
            ds_processed.push_to_hub(
                repo_id=repo_id,
                config_name=CONFIG_NAME,
                commit_message=commit_message,
                token=hf_token,
                max_shard_size=max_shard_size,
            )

        action = "updated" if EMBEDDING_COLUMN in ds.column_names else "created"
        total_valid = sum(1 for e in all_embeddings if not is_empty_embedding(e))

        summary = (
            f"[bold green]Dataset successfully published![/bold green]\n\n"
            f"Repository: [cyan]{repo_id}[/cyan]\n"
            f"Configuration: [cyan]{CONFIG_NAME}[/cyan]\n"
            f"Column: [cyan]{EMBEDDING_COLUMN}[/cyan] ({action})\n"
            f"Model: [cyan]{MODEL_NAME}[/cyan]\n"
            f"Embedding dimension: [cyan]{dimensionality}[/cyan]\n"
            f"Task type: [cyan]{task_type}[/cyan]\n"
            f"Valid embeddings: [cyan]{total_valid}[/cyan] / {len(ds_processed)}"
        )
        if error_counter["failed"] > 0:
            summary += f"\n[yellow]Encoding failures: {error_counter['failed']}[/yellow]"

        console.print(Panel(summary, title="Upload Complete", border_style="green"))

        # Clean up cache after successful push
        delete_cache()

    except Exception as e:
        console.print(Panel(
            f"[bold red]Failed to push dataset[/bold red]\n\n"
            f"{e}\n\n"
            f"[yellow]Embeddings are cached in {CACHE_FILE}[/yellow]\n"
            f"Re-run the script to resume from where it left off.",
            title="Error",
            border_style="red",
        ))
        logger.error("Push error details:", exc_info=True)


if __name__ == "__main__":
    main()
