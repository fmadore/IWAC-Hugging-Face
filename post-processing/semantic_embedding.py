#!/usr/bin/env python3
"""
semantic_embedding.py
=====================

Adds semantic embedding columns to Hugging Face dataset subsets with
``google/embeddinggemma-2``, run locally (ONNX, fp32) at pinned revisions, so
the vectors can be recomputed offline without an API key. See
``_gemma_encoder.py`` for the model identity and its verification.

Supported configurations:
- **articles**: embeds the 'OCR' column → 'embedding_OCR'
- **publications**: embeds the 'tableOfContents' column → 'embedding_tableOfContents'
  (rows without tableOfContents are left with empty embeddings)
- **references**: embeds the 'abstract' column → 'embedding_abstract' (rows
  without an abstract are left empty; full texts are book-length)
- **documents**: embeds the 'OCR' column → 'embedding_OCR'

Texts are split at whitespace into overlapping UTF-8-byte-bounded chunks, each
embedded as ``title: {title} | text: {chunk}`` (the model card's document
prompt). Chunk vectors are pooled with a length-weighted average.

Interruption-safe: every chunk vector is written to a transactional SQLite
cache in ``.cache_embeddings/`` as soon as its batch finishes, so an
interrupted run loses at most one batch. Re-running the same command resumes,
in either update mode: a local model at a pinned revision gives the same vector
for the same input, so cached work is reused instead of discarded. The cache
filename embeds the repository and a fingerprint of the configuration, and each
entry carries a hash of its row's text, title and chunking, so a changed row is
embedded afresh. ``--fresh-cache`` discards the cache explicitly.

Usage
-----
    python post-processing/semantic_embedding.py [--config articles|publications|references|documents]

Example:
    python post-processing/semantic_embedding.py --config publications --dry-run
    python post-processing/semantic_embedding.py --config articles --update-mode all

Environment Variables
---------------------
HF_TOKEN         Personal access token for the Hugging Face Hub.

Dependencies
------------
    pip install onnxruntime tokenizers datasets huggingface_hub rich
"""
import argparse
import logging
import os
import sys
import time
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
from _embedding_columns import set_embedding_column  # noqa: E402
from _gemma_encoder import (  # noqa: E402
    DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    GemmaEncoder,
    document_text,
    model_settings,
    runtime_versions,
    token_batches,
)

# Load .env from project root
load_dotenv(workspace_root() / ".env")

# Disable symlinks warning from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.progress import (  # noqa: E402
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.logging import RichHandler  # noqa: E402
from rich.prompt import Prompt  # noqa: E402
from rich import box  # noqa: E402

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)

# --- Constants ---
# Conservative byte-sized chunks avoid assuming a French chars/token ratio
# for multilingual OCR; 7,000 bytes stay far below the 8,192-token context.
CHUNK_SIZE = 7_000
CHUNK_OVERLAP = 400
FORMAT_VERSION = 2
# Padded tokens per encoder call. ~16k keeps a CPU batch near one minute, which
# bounds the work an interruption can lose; raise it on a GPU.
DEFAULT_BATCH_TOKENS = 16_384
CACHE_DIR = workspace_root() / ".cache_embeddings"

# Per-config settings: (text_column, embedding_column, cache_stem).
# The resume cache filename is "<stem>_<repo>_<cache_fingerprint>_<config hash>.sqlite3",
# so a cache written at one embedding configuration is never restored into a
# run with different parameters.
CONFIG_SETTINGS = {
    "articles": ("OCR", "embedding_OCR", "ocr_embeddings"),
    "publications": ("tableOfContents", "embedding_tableOfContents", "toc_embeddings"),
    "references": ("abstract", "embedding_abstract", "references_abstract_embeddings"),
    "documents": ("OCR", "embedding_OCR", "documents_ocr_embeddings"),
}

# Vector columns a subset no longer produces. The upload merge preserves every
# Hub-only column, so a superseded vector would otherwise survive forever beside
# its replacement; the run that writes the replacement removes it.
RETIRED_COLUMNS = {
    "references": ("embedding_OCR",),
}


def drop_retired_columns(ds, config_name: str):
    """Remove retired vectors and their provenance; return (dataset, dropped)."""
    from iwac_common.enrichment import provenance_columns

    dropped = []
    for column in RETIRED_COLUMNS.get(config_name, ()):
        dropped += [name for name in (column, *provenance_columns(column), f"{column}_config_json")
                    if name in ds.column_names]
    return (ds.remove_columns(dropped) if dropped else ds), dropped


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


def chunk_key(row_id: Any, index: int) -> str:
    """Resume-cache key of one chunk; its entry carries the row's input hash."""
    return f"{row_id}#{index}"


def load_encoder(args, token: str) -> GemmaEncoder:
    """Separate so tests can substitute an offline encoder."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    return GemmaEncoder(providers=providers, threads=args.threads, token=token)


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

    table.add_row("1", "missing", "Update only rows without compatible embeddings (recommended)")
    table.add_row("2", "all", "Recalculate all embeddings (may take longer)")

    console.print(table)

    while True:
        try:
            choice = Prompt.ask("Choose update mode", choices=["1", "2"], default="1")
            return "missing" if choice == "1" else "all"
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.[/yellow]")
            raise SystemExit(0)


def display_config_panel(
    repo_id: str, config_name: str, text_column: str, embedding_column: str,
    update_mode: str, batch_tokens: int, providers: List[str], dry_run: bool,
):
    """Display configuration in a Rich panel."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Repository", repo_id)
    table.add_row("Configuration", config_name)
    table.add_row("Source Column", text_column)
    table.add_row("Embedding Column", embedding_column)
    table.add_row("Model", f"{MODEL_ID}@{MODEL_REVISION[:12]}")
    table.add_row("Runtime", f"onnxruntime fp32 ({', '.join(providers)})")
    table.add_row("Output Dimensionality", str(DIMENSION))
    table.add_row("Batch Budget", f"{batch_tokens:,} padded tokens")
    table.add_row("Update Mode", update_mode)
    table.add_row("Chunk Budget", f"{CHUNK_SIZE:,} UTF-8 bytes")
    table.add_row("Chunk Overlap", f"{CHUNK_OVERLAP:,} chars")
    table.add_row("Long Text Strategy", "chunk → embed each → average")
    table.add_row("Checkpoint", "every batch (resume by re-running)")
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
            table.add_row("Will be chunked", f"[yellow]{will_chunk}[/yellow] texts > {CHUNK_SIZE:,} chars")

    console.print(table)
    return len(non_empty_texts)


def text_fingerprint(text: Any, title: str = "none") -> str:
    """Input identity of one row's embedding: its text plus the chunking
    settings that shape the pooled vector."""
    return input_fingerprint(str(text or ""), title, CHUNK_SIZE, CHUNK_OVERLAP, FORMAT_VERSION)


def resume_hint(cache_file, config_name: str, update_mode: str) -> str:
    return (f"Finished batches are saved in {cache_file}.\n"
            f"Re-run the same command to resume:\n"
            f"  iwac-process embeddings --config {config_name} --update-mode {update_mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a semantic embedding column to a dataset subset using "
                    f"{MODEL_ID}, run locally."
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
        "--batch-tokens",
        type=int,
        default=DEFAULT_BATCH_TOKENS,
        help=f"Padded tokens per encoder call (default: {DEFAULT_BATCH_TOKENS:,}). "
             "Each finished batch is checkpointed.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="onnxruntime intra-op threads (default: onnxruntime's choice).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="cuda requires the onnxruntime-gpu package instead of onnxruntime.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="1GB",
        help="Maximum Parquet shard size when pushing to Hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute embeddings but do not push to Hub (the cache is kept for the real run).",
    )
    parser.add_argument(
        "--update-mode",
        choices=["missing", "all"],
        default=None,
        help="Update only missing embeddings or recompute all (skips the interactive prompt).",
    )
    parser.add_argument(
        "--fresh-cache",
        action="store_true",
        help="Discard this configuration's resume cache before starting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=argparse.SUPPRESS,  # resuming is now the default; kept so old commands still parse
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Push even if some chunks failed. By default the cache is kept "
             "and the Hub is left untouched for a safe retry.",
    )

    args = parser.parse_args()
    if args.batch_tokens < 1 or (args.threads is not None and args.threads < 1):
        parser.error("--batch-tokens and --threads must be positive")

    repo_id = args.repo
    configuration_settings = {
        **model_settings(), "document_format": "title: {title} | text: {text}",
        "format_version": FORMAT_VERSION, "chunk_bytes": CHUNK_SIZE,
        "overlap_chars": CHUNK_OVERLAP, "pooling": "character-weighted-mean",
    }
    configuration = config_fingerprint(configuration_settings)
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
    if DIMENSION != expected_dimension:
        parser.error(f"{MODEL_ID} produces {DIMENSION} dims; schema expects {expected_dimension}")
    # The repository is part of the name: a scratch run and a production run
    # must not share results for identical o:id values. Entries inside carry a
    # hash of their row's input (see text_fingerprint).
    cache_file = CACHE_DIR / (
        f"{cache_stem}_{repo_slug(repo_id)}_"
        f"{cache_fingerprint(MODEL_ID, DIMENSION, 'document')}_{configuration[:16]}.sqlite3"
    )
    if args.fresh_cache and cache_file.exists():
        cache_file.unlink()
        console.print("[yellow]ℹ[/yellow] --fresh-cache: deleted the existing resume cache.")

    # --- Step 1: Authentication ---
    console.print("\n[bold cyan]Step 1:[/bold cyan] Authenticating...")
    try:
        hf_token = ensure_hf_token(console=console)
    except SystemExit:
        return 1
    console.print("[green]✓[/green] Hugging Face authenticated.")

    # --- Step 2: Load the encoder ---
    console.print(f"\n[bold cyan]Step 2:[/bold cyan] Loading {MODEL_ID} (first run downloads ~1.1 GB)...")
    try:
        encoder = load_encoder(args, hf_token)
        encoder.encode([document_text("test")])  # readiness probe; validates the output
        versions = runtime_versions()
        console.print(f"[green]✓[/green] Encoder ready on {', '.join(encoder.providers)} "
                      f"(onnxruntime {versions['onnxruntime']}, tokenizers {versions['tokenizers']})")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load the encoder: {e}")
        return 1

    # --- Update mode selection ---
    if args.update_mode:
        update_mode = args.update_mode
    else:
        update_mode = choose_update_mode()
    console.print(f"[green]✓[/green] Update mode: [cyan]{update_mode}[/cyan]")

    # --- Display configuration ---
    console.print()
    display_config_panel(repo_id, config_name, text_column, embedding_column, update_mode,
                         args.batch_tokens, encoder.providers, dry_run)

    # --- Step 3: Load dataset ---
    console.print("\n[bold cyan]Step 3:[/bold cyan] Loading dataset...")
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
            console.print(f"[yellow]ℹ[/yellow] Embedding column '{embedding_column}' exists. "
                          "Rows without compatible provenance will be computed.")
    else:
        console.print(f"[blue]→[/blue] Embedding column '{embedding_column}' will be created.")

    # --- Step 4: Analyze source data ---
    console.print("\n[bold cyan]Step 4:[/bold cyan] Analyzing source data...")
    texts = ds[text_column]
    display_text_stats(texts, text_column)

    cache = load_cache(cache_file)
    if cache:
        console.print(f"[green]✓[/green] Resume cache holds [cyan]{len(cache)}[/cyan] chunk vectors")

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
    indices_to_process = [
        i for i, (text, emb) in enumerate(zip(texts, all_embeddings))
        if text is not None and str(text).strip() and is_empty_embedding(emb)
    ]

    # Every chunk of every row to process; those already in the cache for the
    # row's current input are reused, the rest are queued.
    row_chunks: Dict[int, List[str]] = {i: chunk_text(str(texts[i])) for i in indices_to_process}
    chunk_vectors: Dict[tuple[int, int], Any] = {}
    queue: List[tuple[int, int]] = []
    for i, chunks in row_chunks.items():
        for c in range(len(chunks)):
            vector = cached_value(cache, chunk_key(row_ids[i], c), fingerprints[i])
            if vector is None:
                queue.append((i, c))
            else:
                chunk_vectors[(i, c)] = vector
    cache.clear()  # vectors needed are now in chunk_vectors

    failed = 0
    if not indices_to_process:
        console.print(Panel(
            "[green]No rows to process![/green]\n\n"
            "Every row has a compatible embedding. Proceeding to update dataset and push.",
            title="Up to Date",
            border_style="green",
        ))
    else:
        console.print(
            f"[blue]→[/blue] [cyan]{len(indices_to_process)}[/cyan] rows, "
            f"{sum(len(c) for c in row_chunks.values())} chunks; "
            f"[cyan]{len(chunk_vectors)}[/cyan] restored from cache, [cyan]{len(queue)}[/cyan] to embed"
        )

    if queue:
        # --- Step 5: Compute embeddings ---
        console.print(f"\n[bold cyan]Step 5:[/bold cyan] Computing embeddings with {MODEL_ID}...")
        queued_texts = [document_text(row_chunks[i][c], titles[i]) for i, c in queue]
        lengths = encoder.token_lengths(queued_texts)
        started = time.time()
        done_tokens = 0
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.fields[rate]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Embedding tokens", total=sum(lengths), rate="")
                for batch in token_batches(lengths, args.batch_tokens):
                    try:
                        vectors = encoder.encode([queued_texts[k] for k in batch])
                    except Exception as e:
                        logger.error(f"Batch of {len(batch)} chunks failed: {e}")
                        failed += len(batch)
                    else:
                        # Checkpoint this batch before starting the next one.
                        completed = {}
                        for k, vector in zip(batch, vectors, strict=True):
                            i, c = queue[k]
                            chunk_vectors[(i, c)] = vector.tolist()
                            completed[chunk_key(row_ids[i], c)] = make_entry(chunk_vectors[(i, c)], fingerprints[i])
                        save_cache(completed, cache_file)
                    batch_tokens = sum(lengths[k] for k in batch)
                    done_tokens += batch_tokens
                    rate = done_tokens / max(time.time() - started, 1e-9)
                    progress.update(task, advance=batch_tokens, rate=f"{rate:,.0f} tok/s")
        except KeyboardInterrupt:
            console.print(Panel(
                "[yellow]Interrupted; Hub left untouched.[/yellow]\n\n"
                + resume_hint(cache_file, config_name, update_mode),
                title="Embedding paused",
                border_style="yellow",
            ))
            return 130
        console.print("[green]✓[/green] Embedding computation complete.")

    # Rows with any failed chunk must end EMPTY — never a truncated average,
    # never a stale pre-existing vector — so a re-run retries them.
    partial_rows = 0
    multi_chunk = 0
    for i, chunks in row_chunks.items():
        vectors = [chunk_vectors.get((i, c)) for c in range(len(chunks))]
        if any(v is None for v in vectors):
            all_embeddings[i] = []
            partial_rows += 1
            continue
        multi_chunk += len(chunks) > 1
        all_embeddings[i] = average_embeddings(vectors, weights=[len(c) for c in chunks])
    if multi_chunk:
        console.print(f"[yellow]ℹ[/yellow] {multi_chunk} long texts were averaged from several chunks.")
    if partial_rows:
        console.print(f"[yellow]⚠[/yellow] {partial_rows} row(s) had failed chunks and were left empty.")

    if failed and not args.allow_partial:
        console.print(Panel(
            "[bold red]Embedding run incomplete; Hub left untouched.[/bold red]\n\n"
            f"{failed} chunk(s) failed. " + resume_hint(cache_file, config_name, update_mode)
            + "\nOr pass --allow-partial explicitly.",
            title="Fail-closed derived-data write",
            border_style="red",
        ))
        return 1

    # --- Update the dataset ---
    console.print("\n[bold cyan]Step 6:[/bold cyan] Updating dataset...")

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
    ds_processed, retired = drop_retired_columns(ds_processed, config_name)
    if retired:
        console.print(f"[blue]→[/blue] Removing retired columns: {', '.join(retired)}")

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
            f"[yellow]Chunk vectors are cached in {cache_file}[/yellow]\n"
            "Re-run without --dry-run to push; cached results are reused.",
            title="Dry Run Complete",
            border_style="yellow",
        ))
        return 0

    console.print("\n[bold cyan]Step 7:[/bold cyan] Pushing to Hugging Face Hub...")

    versions = runtime_versions()
    commit_message = (
        f"Add/update '{embedding_column}' embeddings using {MODEL_ID}@{MODEL_REVISION[:12]} "
        f"(dim={DIMENSION}, onnxruntime {versions['onnxruntime']}, config={config_name})"
    )

    if push_dataset(
        ds_processed,
        repo_id=repo_id,
        config_name=config_name,
        commit_message=commit_message,
        token=hf_token,
        max_shard_size=args.max_shard_size,
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
            f"Model: [cyan]{MODEL_ID}@{MODEL_REVISION[:12]}[/cyan]\n"
            f"Embedding dimension: [cyan]{DIMENSION}[/cyan]\n"
            f"Valid embeddings: [cyan]{total_valid}[/cyan] / {len(ds_processed)}"
        )
        if failed:
            summary += f"\n[yellow]Encoding failures: {failed}[/yellow]"

        console.print(Panel(summary, title="Upload Complete", border_style="green"))

        # Clean up cache after successful push
        delete_cache(cache_file)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
