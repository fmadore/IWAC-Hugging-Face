#!/usr/bin/env python3
"""
semantic_embedding_images.py
============================

Adds a multimodal semantic embedding column (``embedding_image``) to the
``images`` subset by embedding each **photograph itself** with Google's
natively-multimodal ``gemini-embedding-2`` model.

Because ``gemini-embedding-2`` maps images and text into the *same* vector
space, ``embedding_image`` is directly comparable to the text embeddings on
the other subsets (``embedding_OCR``, ``embedding_tableOfContents``) at the
same dimensionality — enabling cross-modal search (a text query can retrieve
photographs, and vice versa).

This is a sibling of ``semantic_embedding.py`` (which embeds *text* columns).
The image path is different enough — download the picture, downscale it, send
the bytes; no text chunking/averaging — that keeping it separate avoids
complicating the text path.

Pipeline: load ``images`` from the private mirror → download + downscale each
photo (from ``image_url``, fallback ``thumbnail``) → embed the bytes → write
``embedding_image`` → push back to the private mirror.

Progress is checkpointed to a resume cache in ``.cache_embeddings/``; the
cache filename embeds a fingerprint of (model, dimensionality, task), so a
cache written under one embedding configuration is never restored into a run
with different parameters.

Usage
-----
    python post-processing/semantic_embedding_images.py                 # interactive-ish (single config)
    python post-processing/semantic_embedding_images.py --update-mode missing
    python post-processing/semantic_embedding_images.py --dry-run

Environment Variables
---------------------
GOOGLE_API_KEY   API key for the Gemini API (or GEMINI_API_KEY).
HF_TOKEN         Personal access token for the Hugging Face Hub.

Dependencies
------------
    pip install google-genai datasets huggingface_hub rich pillow
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import hashlib
import io
import logging
import os
import sys
import time
import urllib.request
from typing import Any, List, Optional

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
from iwac_common.enrichment import config_fingerprint, provenance_columns, set_provenance  # noqa: E402
from _embedding_utils import (  # noqa: E402
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
    set_embedding_column,
    validate_response,
)
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

load_dotenv(workspace_root() / ".env")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.progress import (  # noqa: E402
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn,
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
MODEL_NAME = "gemini-embedding-2"
CONFIG_NAME = "images"
SOURCE_COLUMN = "image_url"          # falls back to ``thumbnail`` per row
FALLBACK_COLUMN = "thumbnail"
EMBEDDING_COLUMN = "embedding_image"
DEFAULT_DIMENSIONALITY = 768
DEFAULT_MAX_SIDE = 1024              # downscale longest side before embedding
# gemini-embedding-2 accepts up to 6 images per request; one Content per image
# returns one vector each.
IMAGE_BATCH_LIMIT = 6
DEFAULT_BATCH_SIZE = 6
# Retry ladder (MAX_RETRIES / BASE_RETRY_DELAY) is shared with the text
# embedding script and lives in _gemini_client.call_with_retry.
CACHE_DIR = workspace_root() / ".cache_embeddings"
# Resume cache stem; the full filename embeds cache_fingerprint(model, dim,
# task) so a cache written at one embedding configuration is never restored
# into a run with different parameters. No task_type is sent for image
# embedding (it's folded into the model), so the fixed tag "image" stands in.
CACHE_STEM = "image_embeddings"
CHECKPOINT_EVERY = 3  # save cache every N API batches
DOWNLOAD_TIMEOUT = 30
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
IMAGE_PROCESSING_VERSION = 2


def download_image_bytes(url: str, max_side: int) -> Optional[bytes]:
    """Download an image and re-encode it as a bounded-size JPEG.

    Downscaling to ``max_side`` keeps the request payload small and
    deterministic. Returns ``None`` (and logs) on any failure so one bad URL
    never aborts the run.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "iwac-embed/1.0"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            raw = resp.read(MAX_DOWNLOAD_BYTES + 1)
            if len(raw) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Image download exceeds 32 MiB")
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to download/decode image {url}: {e}")
        return None


def embed_images_with_retry(
    client: genai.Client,
    images: List[bytes],
    dimensionality: int,
) -> List[List[float]]:
    """Embed a batch of images (one vector each) with the shared retry ladder.

    ``gemini-embedding-2`` is natively multimodal; task type is folded into
    the model rather than passed as a parameter, so we only set
    ``output_dimensionality`` (verified: image embedding works without a
    ``task_type``).
    """
    contents = [
        types.Content(parts=[types.Part.from_bytes(data=img, mime_type="image/jpeg")])
        for img in images
    ]

    def _call() -> List[List[float]]:
        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=dimensionality),
        )
        return validate_response(response, len(images), dimensionality)

    return call_with_retry(_call)


def display_config_panel(
    repo_id: str, dimensionality: int, update_mode: str, batch_size: int,
    max_side: int, dry_run: bool,
) -> None:
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Repository", repo_id)
    table.add_row("Configuration", CONFIG_NAME)
    table.add_row("Source Column", f"{SOURCE_COLUMN} (fallback: {FALLBACK_COLUMN})")
    table.add_row("Embedding Column", EMBEDDING_COLUMN)
    table.add_row("Model", MODEL_NAME)
    table.add_row("Output Dimensionality", str(dimensionality))
    table.add_row("Image Batch Size", str(batch_size))
    table.add_row("Max Image Side", f"{max_side}px (downscaled JPEG)")
    table.add_row("Update Mode", update_mode)
    if dry_run:
        table.add_row("Dry Run", "[yellow]YES — no changes will be pushed[/yellow]")
    console.print(Panel(table, title="[bold blue]Multimodal Image Embedding Configuration", border_style="blue"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a multimodal image embedding column to the 'images' subset "
                    "using Google gemini-embedding-2."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID,
                        help="Repository ID on Hugging Face Hub (default: private full mirror).")
    parser.add_argument("--config", default=CONFIG_NAME, choices=[CONFIG_NAME],
                        help="Dataset configuration to process (only 'images').")
    parser.add_argument("--dimensionality", type=int, default=DEFAULT_DIMENSIONALITY,
                        help=f"Output embedding dimensionality (default: {DEFAULT_DIMENSIONALITY}). "
                             "Must match the text embeddings for cross-modal comparison.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Images per Gemini API call (default: {DEFAULT_BATCH_SIZE}, "
                             f"capped at {IMAGE_BATCH_LIMIT}).")
    parser.add_argument("--max-image-side", type=int, default=DEFAULT_MAX_SIDE,
                        help=f"Downscale the longest image side to this many px (default: {DEFAULT_MAX_SIDE}).")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay in seconds between API calls (default: 0.5).")
    parser.add_argument("--max-shard-size", default="1GB",
                        help="Maximum Parquet shard size when pushing to Hub.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute embeddings but do not push to Hub.")
    parser.add_argument("--update-mode", choices=["missing", "all"], default="missing",
                        help="Update only missing embeddings (default) or recompute all.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a crashed --update-mode all run from its cache "
                             "instead of starting fresh.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Push despite download/embedding failures. By default "
                             "the cache is kept and the Hub is left untouched.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_image_side < 1 or args.delay < 0:
        parser.error("Batch size/image side must be positive; delay must be nonnegative")

    repo_id = args.repo
    dimensionality = args.dimensionality
    batch_size = max(1, min(args.batch_size, IMAGE_BATCH_LIMIT))
    max_side = args.max_image_side
    delay = args.delay
    update_mode = args.update_mode
    dry_run = args.dry_run
    configuration_settings = {
        "model": MODEL_NAME, "dimension": dimensionality, "max_side": max_side,
        "processing_version": IMAGE_PROCESSING_VERSION, "jpeg_quality": 90,
        "google_genai": version("google-genai"),
        "pillow": version("Pillow"),
    }
    configuration = config_fingerprint(configuration_settings)
    expected_dimension = (SUBSETS[CONFIG_NAME].embedding_columns or {})[EMBEDDING_COLUMN]
    if dimensionality != expected_dimension:
        parser.error(
            f"--dimensionality must be {expected_dimension} for the canonical "
            f"{CONFIG_NAME}.{EMBEDDING_COLUMN} schema"
        )
    # Key the resume cache by (model, dimensionality, task) so a cache written
    # at one embedding configuration can never be restored into a run with
    # different parameters. Old un-fingerprinted cache files
    # ("image_embeddings.json.gz") are simply ignored (fresh start), not migrated.
    # The repository is part of the name, and each entry carries a hash of the
    # image URL and downscale size it was computed from.
    cache_file = CACHE_DIR / (
        f"{CACHE_STEM}_{repo_slug(repo_id)}_"
        f"{cache_fingerprint(MODEL_NAME, dimensionality, 'image')}_{configuration[:16]}.sqlite3"
    )

    # 'all' means recompute everything: start from a fresh cache unless the
    # user explicitly resumes a crashed run. 'missing' always reuses the cache.
    if update_mode == "all" and cache_file.exists():
        if args.resume:
            console.print("[yellow]ℹ[/yellow] --resume: reusing the existing cache for this 'all' run.")
        else:
            cache_file.unlink()
            console.print("[yellow]ℹ[/yellow] Update mode 'all': deleted existing resume cache "
                          "(pass --resume to reuse a crashed run's cache).")

    # --- Step 1: Authentication ---
    console.print("\n[bold cyan]Step 1:[/bold cyan] Authenticating...")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]✗[/red] GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment.")
        return 1
    console.print("[green]✓[/green] Gemini API key found.")
    try:
        hf_token = ensure_hf_token(console=console)
    except SystemExit:
        return 1
    console.print("[green]✓[/green] Hugging Face authenticated.")

    # --- Step 2: Initialize Gemini client ---
    console.print("\n[bold cyan]Step 2:[/bold cyan] Initializing Gemini client...")
    try:
        client = genai.Client(api_key=api_key)
        test = client.models.embed_content(
            model=MODEL_NAME, contents=["test"],
            config=types.EmbedContentConfig(output_dimensionality=dimensionality),
        )
        validate_response(test, 1, dimensionality)
        actual_dim = dimensionality
        console.print(f"[green]✓[/green] Gemini client ready. Model: [cyan]{MODEL_NAME}[/cyan] (dim={actual_dim})")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]✗[/red] Failed to initialize Gemini client: {e}")
        return 1

    console.print()
    display_config_panel(repo_id, dimensionality, update_mode, batch_size, max_side, dry_run)

    # --- Step 3: Load dataset ---
    console.print(f"\n[bold cyan]Step 3:[/bold cyan] Loading '{repo_id}' (config: {CONFIG_NAME})...")
    ds = load_hub_dataset(repo_id, CONFIG_NAME, token=hf_token, console=console)
    source_revision = getattr(ds, "_iwac_source_revision", None)

    if SOURCE_COLUMN not in ds.column_names:
        console.print(f"[red]✗[/red] Source column '{SOURCE_COLUMN}' not found.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        return 1

    # Per-row image URL: prefer image_url, fall back to thumbnail.
    urls = list(ds[SOURCE_COLUMN])
    fallbacks = list(ds[FALLBACK_COLUMN]) if FALLBACK_COLUMN in ds.column_names else [None] * len(ds)
    row_ids = list(ds["o:id"])

    def row_url(i: int) -> str:
        u = urls[i]
        if u is not None and str(u).strip():
            return str(u)
        f = fallbacks[i]
        return str(f) if f is not None and str(f).strip() else ""

    # Check actual processed image bytes on every run: a URL can retain its
    # name while its image changes. Only API inference is skipped on a proven
    # match. Downloads and embedding are streamed in bounded batches.
    existing = list(ds[EMBEDDING_COLUMN]) if EMBEDDING_COLUMN in ds.column_names else [None] * len(ds)
    input_col, config_col = provenance_columns(EMBEDDING_COLUMN)
    old_inputs = list(ds[input_col]) if input_col in ds.column_names else [None] * len(ds)
    old_configs = list(ds[config_col]) if config_col in ds.column_names else [None] * len(ds)
    all_embeddings: List[Any] = [[] for _ in range(len(ds))]
    fingerprints = [""] * len(ds)
    cache = load_cache(cache_file)
    failed_dl = 0
    failed_emb = 0
    pending: List[tuple[int, bytes]] = []

    def flush_batch() -> None:
        nonlocal failed_emb
        if not pending:
            return
        try:
            vectors = embed_images_with_retry(client, [data for _, data in pending], dimensionality)
        except Exception as exc:
            logger.error("Image batch failed: %s", exc)
            failed_emb += len(pending)
        else:
            completed = {}
            for (idx, _), vector in zip(pending, vectors, strict=True):
                all_embeddings[idx] = vector
                entry = make_entry(vector, fingerprints[idx])
                cache[str(row_ids[idx])] = entry
                completed[str(row_ids[idx])] = entry
            save_cache(completed, cache_file)
        pending.clear()
        if delay:
            time.sleep(delay)

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
                  TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Checking and embedding images", total=len(ds))
        for idx in range(len(ds)):
            url = row_url(idx)
            if url:
                data = download_image_bytes(url, max_side)
                if data is None:
                    failed_dl += 1
                else:
                    fingerprints[idx] = input_fingerprint(hashlib.sha256(data).hexdigest(), url)
                    cached = cached_value(cache, row_ids[idx], fingerprints[idx])
                    if cached is not None:
                        all_embeddings[idx] = cached
                    elif (update_mode == "missing" and old_inputs[idx] == fingerprints[idx]
                          and old_configs[idx] == configuration and not is_empty_embedding(existing[idx])):
                        all_embeddings[idx] = existing[idx]
                    else:
                        pending.append((idx, data))
                        if len(pending) >= batch_size:
                            flush_batch()
            progress.update(task, advance=1)
        flush_batch()

    if (failed_dl or failed_emb) and not args.allow_partial:
        console.print(Panel(
            "[bold red]Image embedding run incomplete; Hub left untouched.[/bold red]\n\n"
            f"Downloads failed: {failed_dl}; embedding calls failed: {failed_emb}. "
            f"Successful work remains in {cache_file}; re-run to retry.",
            title="Fail-closed derived-data write",
            border_style="red",
        ))
        return 1

    # --- Step 6: Update dataset ---
    console.print(f"\n[bold cyan]Step 6:[/bold cyan] Updating dataset...")
    ds_out = set_embedding_column(ds, EMBEDDING_COLUMN, all_embeddings)
    ds_out = set_provenance(ds_out, EMBEDDING_COLUMN, fingerprints, configuration,
                            [not is_empty_embedding(e) for e in all_embeddings], settings=configuration_settings)

    # Place the embedding column right after the image URL column.
    cols = list(ds_out.column_names)
    if SOURCE_COLUMN in cols:
        cols.remove(EMBEDDING_COLUMN)
        cols.insert(cols.index(SOURCE_COLUMN) + 1, EMBEDDING_COLUMN)
        ds_out = ds_out.select_columns(cols)

    valid = sum(1 for e in all_embeddings if not is_empty_embedding(e))
    console.print(f"[green]✓[/green] {valid}/{len(ds_out)} rows have an image embedding.")
    for i, e in enumerate(ds_out[EMBEDDING_COLUMN]):
        if not is_empty_embedding(e):
            console.print(f"  [cyan]sample[/cyan] row {i}: dim={len(e)}, values=[{e[0]:.4f}, {e[1]:.4f}, ...]")
            break

    # --- Step 7: Push ---
    if dry_run:
        console.print(Panel(
            f"[yellow]Dry run — nothing pushed.[/yellow]\n\n"
            f"Would push [cyan]{len(ds_out)}[/cyan] rows to [cyan]{repo_id}[/cyan] (config: {CONFIG_NAME}).\n"
            f"Embeddings cached in {cache_file}.",
            title="Dry Run Complete", border_style="yellow"))
        return 0

    console.print(f"\n[bold cyan]Step 7:[/bold cyan] Pushing to Hugging Face Hub...")
    if push_dataset(
        ds_out,
        repo_id=repo_id,
        config_name=CONFIG_NAME,
        commit_message=(
            f"Add/update '{EMBEDDING_COLUMN}' multimodal embeddings using "
            f"{MODEL_NAME} (dim={dimensionality})"
        ),
        token=hf_token,
        max_shard_size=args.max_shard_size,
        console=console,
        expected_revision=source_revision,
    ):
        console.print(Panel(
            f"[bold green]Dataset successfully published![/bold green]\n\n"
            f"Repository: [cyan]{repo_id}[/cyan]\n"
            f"Configuration: [cyan]{CONFIG_NAME}[/cyan]\n"
            f"Column: [cyan]{EMBEDDING_COLUMN}[/cyan]\n"
            f"Model: [cyan]{MODEL_NAME}[/cyan] (dim={dimensionality})\n"
            f"Valid embeddings: [cyan]{valid}[/cyan] / {len(ds_out)}",
            title="Upload Complete", border_style="green"))
        delete_cache(cache_file)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
