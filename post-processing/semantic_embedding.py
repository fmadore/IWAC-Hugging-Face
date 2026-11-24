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

When no vector or provenance cell would change (and no retired column needs
removing), nothing is pushed: the run says so and exits successfully.

Usage
-----
    python post-processing/semantic_embedding.py [--config articles|publications|references|documents]
                                                 [--update-mode missing|all] [-y]

``--update-mode`` defaults to ``missing`` with ``-y`` or when not run from a
terminal; otherwise the mode is asked for.

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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence

import numpy as np

# Make ``post-processing/_common.py`` and ``_embedding_utils.py`` importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    PRIVATE_REPO_ID,
    add_update_mode_args,
    choose_config,
    display_text_stats,
    ensure_hf_token,
    load_environment,
    load_hub_dataset,
    print_dry_run_panel,
    push_dataset,
    reorder_columns_after,
    resolve_update_mode,
)
from iwac_common.schema import SUBSETS  # noqa: E402
from iwac_common.paths import workspace_root  # noqa: E402
from iwac_common.enrichment import (  # noqa: E402
    compatible_rows, config_fingerprint, invalidate_columns, provenance_columns, set_provenance,
)
from _embedding_utils import (  # noqa: E402
    average_embeddings,
    cache_fingerprint,
    delete_cache,
    input_fingerprint,
    is_empty_embedding,
    iter_cache,
    make_entry,
    repo_slug,
    save_cache,
)
from _embedding_columns import (  # noqa: E402
    changed_columns, column_vectors, same_vector, set_embedding_column,
)
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

load_environment()

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


def text_fingerprint(text: Any, title: str = "none") -> str:
    """Input identity of one row's embedding: its text plus the chunking
    settings that shape the pooled vector."""
    return input_fingerprint(str(text or ""), title, CHUNK_SIZE, CHUNK_OVERLAP, FORMAT_VERSION)


def resume_hint(cache_file, config_name: str, update_mode: str) -> str:
    return (f"Finished batches are saved in {cache_file}.\n"
            f"Re-run the same command to resume:\n"
            f"  iwac-process embeddings --config {config_name} --update-mode {update_mode}")


def _has_text(text: Any) -> bool:
    return text is not None and bool(str(text).strip())


def embedding_settings() -> Dict[str, Any]:
    """Everything that identifies the vectors; hashed into the provenance."""
    return {
        **model_settings(), "document_format": "title: {title} | text: {text}",
        "format_version": FORMAT_VERSION, "chunk_bytes": CHUNK_SIZE,
        "overlap_chars": CHUNK_OVERLAP, "pooling": "character-weighted-mean",
    }


def cache_path(repo_id: str, cache_stem: str, configuration: str) -> Path:
    """The resume cache of one repository and embedding configuration.

    The repository is part of the name: a scratch run and a production run
    must not share results for identical o:id values. Entries inside carry a
    hash of their row's input (see text_fingerprint).
    """
    return CACHE_DIR / (
        f"{cache_stem}_{repo_slug(repo_id)}_"
        f"{cache_fingerprint(MODEL_ID, DIMENSION, 'document')}_{configuration[:16]}.sqlite3"
    )


@dataclass
class RowPlan:
    """Per-row inputs and outputs of one run; ``vectors`` is filled in place."""

    texts: List[Any]
    titles: List[str]
    fingerprints: List[str]
    compatible: List[bool]
    existing: List[Optional[np.ndarray]]
    vectors: List[Optional[np.ndarray]]
    indices: List[int]  # rows to (re)compute


def plan_rows(ds, text_column: str, embedding_column: str, update_mode: str,
              configuration: str) -> RowPlan:
    """Keep compatible vectors (missing mode) and list the rows to compute.

    Provenance establishes compatibility; legacy vectors are recomputed.
    """
    texts = ds[text_column]
    # Titles affect retrieval input and must participate in invalidation.
    titles = ([str(t or "none")[:200] for t in ds["title"]] if "title" in ds.column_names
              else ["none"] * len(ds))
    fingerprints = [text_fingerprint(t, title) for t, title in zip(texts, titles, strict=True)]
    compatible = compatible_rows(ds, embedding_column, fingerprints, configuration)
    existing = column_vectors(ds, embedding_column)
    vectors: List[Optional[np.ndarray]] = [
        emb if update_mode == "missing" and ok and _has_text(text) and not is_empty_embedding(emb) else None
        for emb, ok, text in zip(existing, compatible, texts, strict=True)
    ]
    indices = [i for i, (text, vector) in enumerate(zip(texts, vectors))
               if _has_text(text) and vector is None]
    return RowPlan(texts, titles, fingerprints, compatible, existing, vectors, indices)


def restore_chunks(plan: RowPlan, row_ids: Sequence[Any], cache_file: Path):
    """Chunk every row to compute; reuse cached chunk vectors for its current input.

    Returns ``(row_chunks, chunk_vectors, queue, n_cached)``. The cache is read
    one entry at a time and only matching entries are kept, as float32 arrays.
    """
    row_chunks: Dict[int, List[str]] = {i: chunk_text(str(plan.texts[i])) for i in plan.indices}
    wanted = {chunk_key(row_ids[i], c): (i, c)
              for i, chunks in row_chunks.items() for c in range(len(chunks))}
    chunk_vectors: Dict[tuple[int, int], np.ndarray] = {}
    n_cached = 0
    for key, entry in iter_cache(cache_file):
        n_cached += 1
        target = wanted.get(key)
        if (target is not None and isinstance(entry, dict)
                and entry.get("h") == plan.fingerprints[target[0]] and entry.get("v") is not None):
            chunk_vectors[target] = np.asarray(entry["v"], dtype=np.float32)
    queue = [target for target in wanted.values() if target not in chunk_vectors]
    return row_chunks, chunk_vectors, queue, n_cached


def embed_queue(encoder, queue, row_chunks, plan: RowPlan, row_ids, chunk_vectors,
                cache_file: Path, batch_tokens: int) -> int:
    """Embed queued chunks batch by batch, checkpointing each finished batch.

    Returns the number of failed chunks; KeyboardInterrupt propagates after the
    last finished batch has been saved.
    """
    queued_texts = [document_text(row_chunks[i][c], plan.titles[i]) for i, c in queue]
    lengths = encoder.token_lengths(queued_texts)
    started = time.time()
    done_tokens = 0
    failed = 0
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
        for batch in token_batches(lengths, batch_tokens):
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
                    chunk_vectors[(i, c)] = np.asarray(vector, dtype=np.float32)
                    completed[chunk_key(row_ids[i], c)] = make_entry(
                        chunk_vectors[(i, c)].tolist(), plan.fingerprints[i])
                save_cache(completed, cache_file)
            n_tokens = sum(lengths[k] for k in batch)
            done_tokens += n_tokens
            rate = done_tokens / max(time.time() - started, 1e-9)
            progress.update(task, advance=n_tokens, rate=f"{rate:,.0f} tok/s")
    return failed


def pool_rows(plan: RowPlan, row_chunks, chunk_vectors) -> tuple[int, int]:
    """Length-weighted mean of each row's chunk vectors; ``(partial, multi_chunk)``.

    Rows with any failed chunk end EMPTY — never a truncated average, never a
    stale pre-existing vector — so a re-run retries them.
    """
    partial_rows = multi_chunk = 0
    for i, chunks in row_chunks.items():
        vectors = [chunk_vectors.get((i, c)) for c in range(len(chunks))]
        if any(v is None for v in vectors):
            plan.vectors[i] = None
            partial_rows += 1
            continue
        multi_chunk += len(chunks) > 1
        plan.vectors[i] = np.asarray(
            average_embeddings(vectors, weights=[len(c) for c in chunks]), dtype=np.float32)
    return partial_rows, multi_chunk


def apply_outputs(ds, plan: RowPlan, embedding_column: str, configuration: str,
                  settings: Dict[str, Any], config_name: str):
    """The dataset with vectors, provenance, invalidated neighbours and retired
    columns removed; returns ``(dataset, retired_columns)``."""
    ds_out = set_embedding_column(ds, embedding_column, plan.vectors)
    ds_out = set_provenance(ds_out, embedding_column, plan.fingerprints, configuration,
                            [not is_empty_embedding(e) for e in plan.vectors], settings=settings)
    # Compare the vectors themselves: a full re-run that reproduces every
    # vector must not wipe the neighbour lists (and thereby force a push).
    vector_changed = not all(
        same_vector(old, new) for old, new in zip(plan.existing, plan.vectors, strict=True)
    )
    # A neighbor rank depends on all vectors, not just on its own document.
    ds_out = invalidate_columns(ds_out, ["related_articles"], [vector_changed] * len(ds))
    return drop_retired_columns(ds_out, config_name)


def touched_columns(embedding_column: str) -> List[str]:
    """Every column this stage writes."""
    return [embedding_column, *provenance_columns(embedding_column),
            f"{embedding_column}_config_json", "related_articles"]


def build_parser() -> argparse.ArgumentParser:
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
    add_update_mode_args(parser)
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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_tokens < 1 or (args.threads is not None and args.threads < 1):
        parser.error("--batch-tokens and --threads must be positive")

    repo_id = args.repo
    configuration_settings = embedding_settings()
    configuration = config_fingerprint(configuration_settings)
    dry_run = args.dry_run

    # --- Configuration selection ---
    config_name = args.config or choose_config(list(CONFIG_SETTINGS), console=console)
    console.print(f"[green]✓[/green] Configuration: [cyan]{config_name}[/cyan]")

    text_column, embedding_column, cache_stem = CONFIG_SETTINGS[config_name]
    expected_dimension = (SUBSETS[config_name].embedding_columns or {})[embedding_column]
    if DIMENSION != expected_dimension:
        parser.error(f"{MODEL_ID} produces {DIMENSION} dims; schema expects {expected_dimension}")
    cache_file = cache_path(repo_id, cache_stem, configuration)
    if args.fresh_cache and cache_file.exists():
        cache_file.unlink()
        console.print("[yellow]ℹ[/yellow] --fresh-cache: deleted the existing resume cache.")

    update_mode = resolve_update_mode(args, console=console)
    console.print(f"[green]✓[/green] Update mode: [cyan]{update_mode}[/cyan]")

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

    console.print()
    display_config_panel(repo_id, config_name, text_column, embedding_column, update_mode,
                         args.batch_tokens, encoder.providers, dry_run)

    # --- Step 3: Load dataset ---
    console.print("\n[bold cyan]Step 3:[/bold cyan] Loading dataset...")
    ds = load_hub_dataset(repo_id, config_name, token=hf_token, console=console)
    source_revision = getattr(ds, "_iwac_source_revision", None)
    if text_column not in ds.column_names:
        console.print(f"[red]✗[/red] Source column '{text_column}' not found in dataset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        return 1
    if embedding_column not in ds.column_names:
        console.print(f"[blue]→[/blue] Embedding column '{embedding_column}' will be created.")
    elif update_mode == "all":
        console.print(f"[yellow]⚠[/yellow] Embedding column '{embedding_column}' exists and will be overwritten.")
    else:
        console.print(f"[yellow]ℹ[/yellow] Embedding column '{embedding_column}' exists. "
                      "Rows without compatible provenance will be computed.")

    # --- Step 4: Plan the rows and restore finished chunks ---
    console.print("\n[bold cyan]Step 4:[/bold cyan] Analyzing source data...")
    plan = plan_rows(ds, text_column, embedding_column, update_mode, configuration)
    display_text_stats(plan.texts, text_column, chunk_bytes=CHUNK_SIZE, console=console)
    row_ids = ds["o:id"]  # stable row identifier for cache keys
    row_chunks, chunk_vectors, queue, n_cached = restore_chunks(plan, row_ids, cache_file)
    if n_cached:
        console.print(f"[green]✓[/green] Resume cache holds [cyan]{n_cached}[/cyan] chunk vectors")
    if not plan.indices:
        console.print(Panel(
            "[green]No rows to process![/green]\n\n"
            "Every row has a compatible embedding; checking whether anything else changed.",
            title="Up to Date",
            border_style="green",
        ))
    else:
        console.print(
            f"[blue]→[/blue] [cyan]{len(plan.indices)}[/cyan] rows, "
            f"{sum(len(c) for c in row_chunks.values())} chunks; "
            f"[cyan]{len(chunk_vectors)}[/cyan] restored from cache, [cyan]{len(queue)}[/cyan] to embed"
        )

    failed = 0
    if queue:
        # --- Step 5: Compute embeddings ---
        console.print(f"\n[bold cyan]Step 5:[/bold cyan] Computing embeddings with {MODEL_ID}...")
        try:
            failed = embed_queue(encoder, queue, row_chunks, plan, row_ids, chunk_vectors,
                                 cache_file, args.batch_tokens)
        except KeyboardInterrupt:
            console.print(Panel(
                "[yellow]Interrupted; Hub left untouched.[/yellow]\n\n"
                + resume_hint(cache_file, config_name, update_mode),
                title="Embedding paused",
                border_style="yellow",
            ))
            return 130
        console.print("[green]✓[/green] Embedding computation complete.")

    partial_rows, multi_chunk = pool_rows(plan, row_chunks, chunk_vectors)
    chunk_vectors.clear()
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
    ds_processed, retired = apply_outputs(ds, plan, embedding_column, configuration,
                                          configuration_settings, config_name)
    if retired:
        console.print(f"[blue]→[/blue] Removing retired columns: {', '.join(retired)}")
    changed = changed_columns(ds, ds_processed, touched_columns(embedding_column))

    console.print("\n[bold]Sample embeddings (first 3 non-empty):[/bold]")
    samples = [(i, v) for i, v in enumerate(plan.vectors) if not is_empty_embedding(v)][:3]
    for i, emb in samples:
        console.print(f"  [cyan]#{i+1}[/cyan]: dim={len(emb)}, values=[{emb[0]:.4f}, {emb[1]:.4f}, ...]")

    ds_processed = reorder_columns_after(ds_processed, [embedding_column], text_column, console=console)

    # --- Step 7: Push to Hub ---
    if dry_run:
        print_dry_run_panel(
            repo_id=repo_id, config_name=config_name, n_rows=len(ds_processed), console=console,
            extra=(f"Changed columns: {', '.join(changed) or 'none (nothing would be pushed)'}\n"
                   f"[yellow]Chunk vectors are cached in {cache_file}[/yellow]\n"
                   "Re-run without --dry-run to push; cached results are reused."),
        )
        return 0
    if not changed:
        console.print(Panel(
            "[green]Nothing changed: every vector and provenance cell already matches the Hub.[/green]\n\n"
            "No commit was made.",
            title="Push skipped",
            border_style="green",
        ))
        if not failed:  # partial rows keep their finished chunks for a retry
            delete_cache(cache_file)
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
        total_valid = sum(1 for e in plan.vectors if not is_empty_embedding(e))

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
