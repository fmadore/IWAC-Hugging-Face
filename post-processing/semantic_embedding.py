#!/usr/bin/env python3
"""
semantic_embedding.py
=====================

Adds semantic embedding columns to Hugging Face dataset subsets using Google's
gemini-embedding-2 model via the Gemini API for high-quality
multilingual embeddings.

Supported configurations:
- **articles**: embeds the 'OCR' column → 'embedding_OCR'
- **publications**: embeds the 'tableOfContents' column → 'embedding_tableOfContents'
  (rows without tableOfContents are left with empty embeddings)

Texts are split at whitespace into overlapping UTF-8-byte-bounded chunks;
token-limit responses are split again adaptively. Chunk vectors are pooled
with a length-weighted average. Documents use Embedding 2's task instructions
and one Content object per independently embedded chunk.

Progress is checkpointed transactionally to SQLite in ``.cache_embeddings/``. The
cache filename embeds the repository and a fingerprint of (model,
dimensionality, task type), so a cache written under one embedding
configuration — or for another repository — is never restored. Each entry
also carries a hash of the row's text and chunking settings, so a row whose
text changed since it was cached is embedded afresh. Only rows whose chunks
ALL embedded successfully are cached — partially-embedded rows are retried on
re-run.

Usage
-----
    python post-processing/semantic_embedding.py [--config articles|publications]

Example:
    python post-processing/semantic_embedding.py --config publications --dry-run
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
from importlib.metadata import version
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
# Make ``post-processing/_common.py`` and ``_embedding_utils.py`` importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    PRIVATE_REPO_ID,
    ensure_hf_token,
    load_hub_dataset,
    push_dataset,
)
from iwac_common.schema import SUBSETS  # noqa: E402
from iwac_common.paths import workspace_root  # noqa: E402
from iwac_common.enrichment import compatible_rows, config_fingerprint, set_provenance, invalidate_columns  # noqa: E402
from _embedding_utils import (  # noqa: E402
    average_embeddings,
    cache_fingerprint,
    cached_value,
    delete_cache,
    input_fingerprint,
    is_empty_embedding,
    load_cache,
    make_entry,
    repo_slug,
    save_cache,
)
from _gemini_client import (  # noqa: E402
    call_with_retry,
    restore_from_cache,
    set_embedding_column,
    validate_response,
)
from google import genai
from google.genai import types

# Load .env from project root
load_dotenv(workspace_root() / ".env")

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
MODEL_NAME = "gemini-embedding-2"
# Conservative byte-sized chunks avoid assuming a French chars/token ratio
# for multilingual OCR. Token-overflow responses are split again on demand.
CHUNK_SIZE = 7_000
CHUNK_OVERLAP = 400
FORMAT_VERSION = 2
DEFAULT_DIMENSIONALITY = 768
# Wrap each document as Content: bare string lists aggregate into one vector.
DEFAULT_BATCH_SIZE = 16
# Retry ladder (MAX_RETRIES / BASE_RETRY_DELAY) is shared with the image
# embedding script and lives in _gemini_client.call_with_retry.
CACHE_DIR = workspace_root() / ".cache_embeddings"
CHECKPOINT_EVERY = 5  # save cache every N API batches

# Per-config settings: (text_column, embedding_column, cache_stem).
# The resume cache filename is "<stem>_<cache_fingerprint(model, dim, task)>.json.gz",
# so a cache written at one embedding configuration is never restored into a
# run with different parameters.
CONFIG_SETTINGS = {
    "articles": ("OCR", "embedding_OCR", "ocr_embeddings"),
    "publications": ("tableOfContents", "embedding_tableOfContents", "toc_embeddings"),
    "references": ("OCR", "embedding_OCR", "references_ocr_embeddings"),
}


# Cache + chunking + averaging helpers live in _embedding_utils. The
# module-level CHUNK_SIZE / CHUNK_OVERLAP constants are still applied —
# we wrap chunk_text() so call sites stay parameter-free.


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Whitespace-aware chunks bounded by UTF-8 bytes, preserving all text."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        while len(text[start:end].encode("utf-8")) > chunk_size:
            if end - start == 1:
                raise ValueError("chunk_size cannot hold one UTF-8 character")
            end = start + max(1, (end - start) // 2)
        if end < len(text):
            boundary = text.rfind(" ", start + (end - start) // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - min(overlap, (end - start) // 4))
    return chunks or [text]


def choose_config() -> str:
    """Prompt the user to choose which dataset configuration to process."""
    console.print("\n[bold]Dataset Configuration:[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Config", style="green")
    table.add_column("Source Column", style="white")
    table.add_column("Embedding Column", style="white")

    config_keys = list(CONFIG_SETTINGS.keys())
    for i, (name, (text_col, emb_col, _)) in enumerate(CONFIG_SETTINGS.items(), 1):
        table.add_row(str(i), name, text_col, emb_col)

    console.print(table)

    choice = Prompt.ask(
        "Choose configuration",
        choices=[str(i) for i in range(1, len(config_keys) + 1)],
        default="1",
    )
    return config_keys[int(choice) - 1]


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
    titles: List[str] | None = None,
) -> List[List[float]]:
    """Call Gemini embed_content with the shared 429/backoff retry ladder."""
    def _call() -> List[List[float]]:
        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=[types.Content(parts=[types.Part.from_text(text=format_embedding_text(
                text, task_type, title
            ))]) for text, title in zip(texts, titles or ["none"] * len(texts), strict=True)],
            config=types.EmbedContentConfig(
                output_dimensionality=dimensionality,
            ),
        )
        # A short response, or a null vector inside it, is a FAILURE and has
        # to raise: the caller fills its slots positionally, so anything it
        # never receives stays None silently — no retry, no log line, and the
        # row is written empty as though nothing had gone wrong.
        return validate_response(response, len(texts), dimensionality)

    try:
        return call_with_retry(_call)
    except Exception as exc:
        message = str(exc).lower()
        if getattr(exc, "code", None) != 400 or "token" not in message or not any(
            word in message for word in ("exceed", "limit", "maximum", "too long")
        ):
            raise
        # Keep one output per original chunk if the model rejects its size.
        output = []
        for text, title in zip(texts, titles or ["none"] * len(texts), strict=True):
            if len(text) < 2:
                raise
            middle = len(text) // 2
            boundary = text.rfind(" ", len(text) // 4, middle + 1)
            middle = boundary if boundary > 0 else middle
            pieces = [text[:middle], text[middle:]]
            vectors = embed_texts_with_retry(client, pieces, task_type, dimensionality,
                                             titles=[title, title])
            output.append(average_embeddings(vectors, weights=[len(piece) for piece in pieces]))
        return output


def format_embedding_text(text: str, task_type: str, title: str = "none") -> str:
    """Embedding 2 uses text instructions, never EmbedContentConfig.task_type."""
    if task_type == "RETRIEVAL_DOCUMENT":
        return f"title: {title or 'none'} | text: {text}"
    prefixes = {"RETRIEVAL_QUERY": "search result", "SEMANTIC_SIMILARITY": "sentence similarity",
                "CLASSIFICATION": "classification", "CLUSTERING": "clustering"}
    return f"task: {prefixes[task_type]} | query: {text}"


def display_config_panel(
    repo_id: str, config_name: str, text_column: str, embedding_column: str,
    model_name: str, update_mode: str,
    batch_size: int, dimensionality: int, task_type: str, dry_run: bool,
):
    """Display configuration in a Rich panel."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repository", repo_id)
    table.add_row("Configuration", config_name)
    table.add_row("Source Column", text_column)
    table.add_row("Embedding Column", embedding_column)
    table.add_row("Model", model_name)
    table.add_row("Task Type", task_type)
    table.add_row("Output Dimensionality", str(dimensionality))
    table.add_row("API Batch Size", str(batch_size))
    table.add_row("Update Mode", update_mode)
    table.add_row("Chunk Budget", f"{CHUNK_SIZE:,} UTF-8 bytes")
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


def text_fingerprint(text: Any, title: str = "none") -> str:
    """Input identity of one row's embedding: its text plus the chunking
    settings that shape the pooled vector."""
    return input_fingerprint(str(text or ""), title, CHUNK_SIZE, CHUNK_OVERLAP, FORMAT_VERSION)


def _save_completed_to_cache(
    cache: Dict[str, Any],
    row_chunks: List[tuple[int, List[str]]],
    flat_embeddings: List[Any],
    row_ids: List[Any],
    all_embeddings: List[Any],
    cache_file: Path,
    fingerprints: List[str],
    state: Dict[str, int] | None = None,
    processed_chunks: int | None = None,
) -> None:
    """Reassemble fully-completed chunk embeddings into row embeddings and save cache.

    A row is assembled and cached ONLY when every one of its chunks has a
    vector. Partially-embedded rows (some chunk vectors None because a batch
    failed) are left untouched — caching them would freeze a permanently
    truncated average. Chunk vectors are pooled with a length-weighted mean
    so short tail chunks don't count as much as full-size ones.
    """
    state = state if state is not None else {}
    flat_offset = state.get("offset", 0)
    next_row = state.get("row", 0)
    updated = {}
    for position in range(next_row, len(row_chunks)):
        row_idx, chunks = row_chunks[position]
        if processed_chunks is not None and flat_offset + len(chunks) > processed_chunks:
            break
        chunk_embs = flat_embeddings[flat_offset:flat_offset + len(chunks)]
        if cached_value(cache, row_ids[row_idx], fingerprints[row_idx]) is None and all(emb is not None for emb in chunk_embs):
            averaged = average_embeddings(chunk_embs, weights=[len(c) for c in chunks])
            all_embeddings[row_idx] = averaged
            oid_str = str(row_ids[row_idx])
            fingerprint = fingerprints[row_idx]
            if cached_value(cache, oid_str, fingerprint) is None:
                cache[oid_str] = make_entry(averaged, fingerprint)
                updated[oid_str] = cache[oid_str]
        flat_offset += len(chunks)
        next_row += 1
    state.update(offset=flat_offset, row=next_row)
    if updated:
        save_cache(updated if cache_file.suffix == ".sqlite3" else cache, cache_file)
        logger.info(f"Checkpoint: saved {len(cache)} total embeddings to cache")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add semantic embedding column to a dataset subset using "
                    "Google Gemini embeddings."
    )
    parser.add_argument(
        "--repo",
        default=PRIVATE_REPO_ID,
        help="Repository ID on Hugging Face Hub (default: private full mirror).",
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIG_SETTINGS.keys()),
        help="Dataset configuration to process. If omitted, an interactive menu is shown.",
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
    parser.add_argument(
        "--update-mode",
        choices=["missing", "all"],
        default=None,
        help="Update only missing embeddings or recompute all (skips the interactive prompt).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a crashed --update-mode all run from its cache instead of starting fresh.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Push even if Gemini chunks failed. By default the cache is kept "
             "and the Hub is left untouched for a safe retry.",
    )

    args = parser.parse_args()
    if args.batch_size < 1 or args.delay < 0:
        parser.error("--batch-size must be positive and --delay nonnegative")

    repo_id = args.repo
    dimensionality = args.dimensionality
    task_type = args.task_type
    configuration_settings = {
        "model": MODEL_NAME, "dimension": dimensionality, "task": task_type,
        "format_version": FORMAT_VERSION, "chunk_bytes": CHUNK_SIZE,
        "overlap_chars": CHUNK_OVERLAP, "pooling": "character-weighted-mean",
        "google_genai": version("google-genai"),
    }
    configuration = config_fingerprint(configuration_settings)
    batch_size = args.batch_size
    delay = args.delay
    max_shard_size = args.max_shard_size
    dry_run = args.dry_run

    # --- Configuration selection ---
    if args.config:
        config_name = args.config
    else:
        config_name = choose_config()
    console.print(f"[green]✓[/green] Configuration: [cyan]{config_name}[/cyan]")

    # Resolve per-config settings
    text_column, embedding_column, cache_stem = CONFIG_SETTINGS[config_name]
    expected_dimension = (SUBSETS[config_name].embedding_columns or {})[embedding_column]
    if dimensionality != expected_dimension:
        parser.error(
            f"--dimensionality must be {expected_dimension} for the canonical "
            f"{config_name}.{embedding_column} schema"
        )
    # Key the resume cache by (model, dimensionality, task type) so a cache
    # written at one embedding configuration can never be restored into a run
    # with different parameters. Old un-fingerprinted cache files (e.g.
    # "ocr_embeddings.json.gz") are simply ignored (fresh start), not migrated.
    # The repository is part of the name too: a scratch run and a production
    # run must not share results for identical o:id values. Entries inside
    # carry a hash of their input text (see text_fingerprint).
    cache_file = CACHE_DIR / (
        f"{cache_stem}_{repo_slug(repo_id)}_"
        f"{cache_fingerprint(MODEL_NAME, dimensionality, task_type)}_{configuration[:16]}.sqlite3"
    )

    # --- Step 1: Authentication ---
    console.print("\n[bold cyan]Step 1:[/bold cyan] Authenticating...")

    # Gemini API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]✗[/red] GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment.")
        return 1
    console.print("[green]✓[/green] Gemini API key found.")

    # Hugging Face token
    try:
        hf_token = ensure_hf_token(console=console)
    except SystemExit:
        return 1
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
        validate_response(test_response, 1, dimensionality)
        actual_dim = dimensionality
        console.print(f"[green]✓[/green] Gemini client ready. Model: [cyan]{MODEL_NAME}[/cyan]")
        console.print(f"[blue]→[/blue] Output dimensionality: [cyan]{actual_dim}[/cyan]")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize Gemini client: {e}")
        return 1

    # --- Update mode selection ---
    if args.update_mode:
        update_mode = args.update_mode
    else:
        update_mode = choose_update_mode()
    console.print(f"[green]✓[/green] Update mode: [cyan]{update_mode}[/cyan]")

    # 'all' means recompute everything: start from a fresh cache unless the
    # user explicitly resumes a crashed run. 'missing' always reuses the cache.
    if update_mode == "all" and cache_file.exists():
        if args.resume:
            console.print("[yellow]ℹ[/yellow] --resume: reusing the existing cache for this 'all' run.")
        else:
            cache_file.unlink()
            console.print(
                "[yellow]ℹ[/yellow] Update mode 'all': deleted existing resume cache "
                "(pass --resume to reuse a crashed run's cache)."
            )

    # --- Display configuration ---
    console.print()
    display_config_panel(repo_id, config_name, text_column, embedding_column, MODEL_NAME, update_mode, batch_size, dimensionality, task_type, dry_run)

    # --- Step 3: Load dataset ---
    console.print(f"\n[bold cyan]Step 3:[/bold cyan] Loading dataset...")
    ds = load_hub_dataset(repo_id, config_name, token=hf_token, console=console)
    source_revision = getattr(ds, "_iwac_source_revision", None)

    # --- Column checks ---
    if text_column not in ds.column_names:
        console.print(f"[red]✗[/red] Source column '{text_column}' not found in dataset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        return 1

    # Provenance below establishes compatibility; legacy vectors are recomputed.

    if embedding_column in ds.column_names:
        if update_mode == "all":
            console.print(f"[yellow]⚠[/yellow] Embedding column '{embedding_column}' exists and will be overwritten.")
        else:
            console.print(f"[yellow]ℹ[/yellow] Embedding column '{embedding_column}' exists. Only missing values will be computed.")
    else:
        console.print(f"[blue]→[/blue] Embedding column '{embedding_column}' will be created.")

    # --- Step 4: Analyze source data ---
    console.print(f"\n[bold cyan]Step 4:[/bold cyan] Analyzing source data...")
    texts = ds[text_column]
    display_text_stats(texts, text_column)

    # --- Load embedding cache ---
    cache = load_cache(cache_file)
    if cache:
        console.print(f"[green]✓[/green] Resuming with [cyan]{len(cache)}[/cyan] cached embeddings")

    # --- Identify rows to process ---
    # Use [] instead of None for missing embeddings so PyArrow infers a consistent list<float> type
    existing_embeddings = ds[embedding_column] if embedding_column in ds.column_names else [[] for _ in range(len(ds))]
    row_ids = ds["o:id"]  # stable row identifier for cache keys
    # Titles affect retrieval input and must participate in invalidation.
    titles = [str(t or "none")[:200] for t in ds["title"]] if "title" in ds.column_names else ["none"] * len(ds)
    fingerprints = [text_fingerprint(t, title) for t, title in zip(texts, titles, strict=True)]
    compatible = compatible_rows(ds, embedding_column, fingerprints, configuration)
    all_embeddings: List[Any] = [
        emb if update_mode == "missing" and ok and text is not None and str(text).strip()
        and not is_empty_embedding(emb) else []
        for emb, ok, text in zip(existing_embeddings, compatible, texts, strict=True)
    ]
    cache_hits = restore_from_cache(all_embeddings, row_ids, cache, fingerprints)

    if cache_hits > 0:
        console.print(f"[green]✓[/green] Restored [cyan]{cache_hits}[/cyan] embeddings from cache")

    # Determine which rows still need embedding
    error_counter = {"failed": 0, "chunked": 0, "total_chunks": 0}
    indices_to_process = []
    for i, (text, emb) in enumerate(zip(texts, all_embeddings)):
        if update_mode == "all":
            if text is not None and str(text).strip():
                # In 'all' mode, skip only if already in cache (this run's
                # checkpoints, or a crashed run's cache kept via --resume)
                if cached_value(cache, row_ids[i], fingerprints[i]) is None:
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
        flat_titles: List[str] = []
        for row_idx, chunks in row_chunks:
            flat_chunks.extend(chunks)
            flat_titles.extend([titles[row_idx]] * len(chunks))

        console.print(
            f"[blue]→[/blue] {len(flat_chunks)} total chunks from "
            f"{len(indices_to_process)} rows "
            f"({error_counter['chunked']} multi-chunk)"
        )

        # Embed all chunks in batches
        flat_embeddings: List[Any] = [None] * len(flat_chunks)
        batch_count = 0
        checkpoint_state: Dict[str, int] = {}

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
                        titles=flat_titles[batch_start:batch_end],
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
                        cache_file, fingerprints, checkpoint_state, batch_end,
                    )

                # Delay between API calls to respect rate limits
                if delay > 0 and batch_end < len(flat_chunks):
                    time.sleep(delay)

        # Final reassemble: group chunk embeddings by row and length-weighted
        # average (only rows whose chunks ALL succeeded are assembled/cached).
        _save_completed_to_cache(
            cache, row_chunks, flat_embeddings, row_ids, all_embeddings,
            cache_file, fingerprints, checkpoint_state, len(flat_chunks),
        )

        # Rows with any failed chunk must end EMPTY — never a truncated
        # average, never a stale pre-existing vector — so a re-run in
        # 'missing' mode retries them.
        partial_rows = 0
        flat_offset = 0
        for row_idx, chunks in row_chunks:
            chunk_embs = flat_embeddings[flat_offset:flat_offset + len(chunks)]
            if any(emb is None for emb in chunk_embs):
                all_embeddings[row_idx] = []
                partial_rows += 1
            flat_offset += len(chunks)
        if partial_rows > 0:
            console.print(
                f"[yellow]⚠[/yellow] {partial_rows} row(s) had failed chunks and were "
                f"left empty — re-run in 'missing' mode to retry them."
            )

        console.print("[green]✓[/green] Embedding computation complete.")

        if error_counter["chunked"] > 0:
            console.print(
                f"[yellow]ℹ[/yellow] {error_counter['chunked']} long texts were split "
                f"into {error_counter['total_chunks']} chunks (averaged back to 1 vector each)."
            )
        if error_counter["failed"] > 0:
            console.print(
                f"[yellow]⚠[/yellow] {error_counter['failed']} chunks failed during "
                f"encoding."
            )

    if error_counter["failed"] > 0 and not args.allow_partial:
        console.print(Panel(
            "[bold red]Embedding run incomplete; Hub left untouched.[/bold red]\n\n"
            f"{error_counter['failed']} chunk(s) failed. Successful work remains "
            f"in {cache_file}; re-run to retry, or pass --allow-partial explicitly.",
            title="Fail-closed derived-data write",
            border_style="red",
        ))
        return 1

    # --- Update the dataset ---
    console.print(f"\n[bold cyan]Step 6:[/bold cyan] Updating dataset...")

    # Build the column as a typed PyArrow array (nulls + float lists coexist)
    # to avoid type-inference issues with sparse embeddings.
    ds_processed = set_embedding_column(ds, embedding_column, all_embeddings)
    ds_processed = set_provenance(ds_processed, embedding_column, fingerprints, configuration,
                                  [not is_empty_embedding(e) for e in all_embeddings], settings=configuration_settings)
    vector_changed = any(
        (not is_empty_embedding(old) or not is_empty_embedding(new))
        and (not ok or update_mode == "all" or is_empty_embedding(new))
        for old, new, ok in zip(existing_embeddings, all_embeddings, compatible, strict=True)
    )
    # A neighbor rank depends on all vectors, not just on its own document.
    ds_processed = invalidate_columns(ds_processed, ["related_articles"], [vector_changed] * len(ds))

    # --- Verify results ---
    console.print("\n[bold]Sample embeddings (first 3 non-empty):[/bold]")
    shown = 0
    for i, emb in enumerate(ds_processed[embedding_column]):
        if shown >= 3:
            break
        if not is_empty_embedding(emb):
            console.print(f"  [cyan]#{i+1}[/cyan]: dim={len(emb)}, values=[{emb[0]:.4f}, {emb[1]:.4f}, ...]")
            shown += 1

    # --- Reorder columns: place embedding column after source text column ---
    if text_column in ds_processed.column_names:
        existing_columns = list(ds_processed.column_names)
        insert_index = existing_columns.index(text_column) + 1

        new_columns = existing_columns[:insert_index]
        if embedding_column in existing_columns and embedding_column not in new_columns:
            new_columns.append(embedding_column)
        for col in existing_columns[insert_index:]:
            if col not in new_columns:
                new_columns.append(col)

        ds_processed = ds_processed.select_columns(new_columns)
        console.print(f"[blue]→[/blue] Columns reordered ('{embedding_column}' after '{text_column}')")

    # --- Step 7: Push to Hub ---
    if dry_run:
        console.print(Panel(
            "[yellow]Dry run mode — no changes pushed to Hub.[/yellow]\n\n"
            f"Would have pushed [cyan]{len(ds_processed)}[/cyan] rows to "
            f"[cyan]{repo_id}[/cyan] (config: {config_name}).\n\n"
            f"[yellow]Embeddings are cached in {cache_file}[/yellow]\n"
            f"(filename is fingerprinted by model/dim/task)\n"
            f"Re-run without --dry-run to push (cached results will be reused; "
            f"in --update-mode all, add --resume to keep them).",
            title="Dry Run Complete",
            border_style="yellow",
        ))
        return 0

    console.print(f"\n[bold cyan]Step 7:[/bold cyan] Pushing to Hugging Face Hub...")

    commit_message = (
        f"Add/update '{embedding_column}' embeddings using {MODEL_NAME} "
        f"(dim={dimensionality}, task={task_type}, config={config_name})"
    )

    if push_dataset(
        ds_processed,
        repo_id=repo_id,
        config_name=config_name,
        commit_message=commit_message,
        token=hf_token,
        max_shard_size=max_shard_size,
        console=console,
        expected_revision=source_revision,
    ):

        action = "updated" if embedding_column in ds.column_names else "created"
        total_valid = sum(1 for e in all_embeddings if not is_empty_embedding(e))

        summary = (
            f"[bold green]Dataset successfully published![/bold green]\n\n"
            f"Repository: [cyan]{repo_id}[/cyan]\n"
            f"Configuration: [cyan]{config_name}[/cyan]\n"
            f"Column: [cyan]{embedding_column}[/cyan] ({action})\n"
            f"Model: [cyan]{MODEL_NAME}[/cyan]\n"
            f"Embedding dimension: [cyan]{dimensionality}[/cyan]\n"
            f"Task type: [cyan]{task_type}[/cyan]\n"
            f"Valid embeddings: [cyan]{total_valid}[/cyan] / {len(ds_processed)}"
        )
        if error_counter["failed"] > 0:
            summary += f"\n[yellow]Encoding failures: {error_counter['failed']}[/yellow]"

        console.print(Panel(summary, title="Upload Complete", border_style="green"))

        # Clean up cache after successful push
        delete_cache(cache_file)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
