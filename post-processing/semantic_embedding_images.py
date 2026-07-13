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
import io
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from datasets import load_dataset

# Make ``post-processing/_common.py`` and ``_embedding_utils.py`` importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ensure_hf_token, PRIVATE_REPO_ID  # noqa: E402
from _embedding_utils import (  # noqa: E402
    cache_fingerprint,
    delete_cache,
    is_empty_embedding,
    load_cache,
    save_cache,
)
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from PIL import Image  # noqa: E402
import pyarrow as pa  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
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
MAX_RETRIES = 6
BASE_RETRY_DELAY = 5  # seconds
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache_embeddings"
# Resume cache stem; the full filename embeds cache_fingerprint(model, dim,
# task) so a cache written at one embedding configuration is never restored
# into a run with different parameters. No task_type is sent for image
# embedding (it's folded into the model), so the fixed tag "image" stands in.
CACHE_STEM = "image_embeddings"
CHECKPOINT_EVERY = 3  # save cache every N API batches
DOWNLOAD_TIMEOUT = 30


def download_image_bytes(url: str, max_side: int) -> Optional[bytes]:
    """Download an image and re-encode it as a bounded-size JPEG.

    Downscaling to ``max_side`` keeps the request payload small and
    deterministic. Returns ``None`` (and logs) on any failure so one bad URL
    never aborts the run.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "iwac-embed/1.0"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            raw = resp.read()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
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
    """Embed a batch of images (one vector each) with exponential backoff.

    ``gemini-embedding-2`` is natively multimodal; task type is folded into
    the model rather than passed as a parameter, so we only set
    ``output_dimensionality`` (verified: image embedding works without a
    ``task_type``).
    """
    contents = [
        types.Content(parts=[types.Part.from_bytes(data=img, mime_type="image/jpeg")])
        for img in images
    ]
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=dimensionality),
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:  # noqa: BLE001
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


def main() -> None:
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
    args = parser.parse_args()

    repo_id = args.repo
    dimensionality = args.dimensionality
    batch_size = max(1, min(args.batch_size, IMAGE_BATCH_LIMIT))
    max_side = args.max_image_side
    delay = args.delay
    update_mode = args.update_mode
    dry_run = args.dry_run
    # Key the resume cache by (model, dimensionality, task) so a cache written
    # at one embedding configuration can never be restored into a run with
    # different parameters. Old un-fingerprinted cache files
    # ("image_embeddings.json.gz") are simply ignored (fresh start), not migrated.
    cache_file = CACHE_DIR / (
        f"{CACHE_STEM}_{cache_fingerprint(MODEL_NAME, dimensionality, 'image')}.json.gz"
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
        return
    console.print("[green]✓[/green] Gemini API key found.")
    try:
        hf_token = ensure_hf_token(console=console)
    except SystemExit:
        return
    console.print("[green]✓[/green] Hugging Face authenticated.")

    # --- Step 2: Initialize Gemini client ---
    console.print("\n[bold cyan]Step 2:[/bold cyan] Initializing Gemini client...")
    try:
        client = genai.Client(api_key=api_key)
        test = client.models.embed_content(
            model=MODEL_NAME, contents=["test"],
            config=types.EmbedContentConfig(output_dimensionality=dimensionality),
        )
        actual_dim = len(test.embeddings[0].values)
        console.print(f"[green]✓[/green] Gemini client ready. Model: [cyan]{MODEL_NAME}[/cyan] (dim={actual_dim})")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]✗[/red] Failed to initialize Gemini client: {e}")
        return

    console.print()
    display_config_panel(repo_id, dimensionality, update_mode, batch_size, max_side, dry_run)

    # --- Step 3: Load dataset ---
    console.print(f"\n[bold cyan]Step 3:[/bold cyan] Loading '{repo_id}' (config: {CONFIG_NAME})...")
    try:
        ds = load_dataset(repo_id, name=CONFIG_NAME, split="train", token=hf_token)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]✗[/red] Failed to load dataset: {e}")
        return
    console.print(f"[green]✓[/green] Dataset loaded: [cyan]{len(ds)}[/cyan] rows")

    if SOURCE_COLUMN not in ds.column_names:
        console.print(f"[red]✗[/red] Source column '{SOURCE_COLUMN}' not found.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        return

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

    # --- Existing embeddings + cache ---
    existing = ds[EMBEDDING_COLUMN] if EMBEDDING_COLUMN in ds.column_names else [[] for _ in range(len(ds))]
    all_embeddings: List[Any] = [e if e is not None else [] for e in existing]

    cache = load_cache(cache_file)
    if cache:
        console.print(f"[green]✓[/green] Resuming with [cyan]{len(cache)}[/cyan] cached embeddings")
    restored = 0
    for i, oid in enumerate(row_ids):
        if str(oid) in cache:
            all_embeddings[i] = cache[str(oid)]
            restored += 1
    if restored:
        console.print(f"[green]✓[/green] Restored [cyan]{restored}[/cyan] embeddings from cache")

    # Validate dimensionality consistency for pre-existing embeddings.
    for e in all_embeddings:
        if not is_empty_embedding(e) and len(e) != actual_dim:
            console.print(f"[red]✗[/red] Existing embeddings have dim {len(e)} ≠ target {actual_dim}. "
                          f"Use --update-mode all to recompute.")
            if update_mode != "all":
                return
            break

    # --- Determine rows to process ---
    to_process: List[int] = []
    no_url = 0
    for i in range(len(ds)):
        if not row_url(i):
            no_url += 1
            continue
        if update_mode == "all":
            if str(row_ids[i]) not in cache:
                to_process.append(i)
        else:  # missing
            if is_empty_embedding(all_embeddings[i]):
                to_process.append(i)
    if no_url:
        console.print(f"[yellow]ℹ[/yellow] {no_url} row(s) have no image URL — skipped.")

    if not to_process:
        console.print(Panel("[green]All image embeddings are already computed![/green]",
                            title="Nothing to do", border_style="green"))
    else:
        console.print(f"[blue]→[/blue] [cyan]{len(to_process)}[/cyan] photos to embed")

        # --- Step 4: Download images ---
        console.print(f"\n[bold cyan]Step 4:[/bold cyan] Downloading + downscaling images...")
        downloaded: List[tuple[int, bytes]] = []
        failed_dl = 0
        with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
                      TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("[cyan]Downloading", total=len(to_process))
            for idx in to_process:
                img = download_image_bytes(row_url(idx), max_side)
                if img is not None:
                    downloaded.append((idx, img))
                else:
                    failed_dl += 1
                progress.update(task, advance=1)
        console.print(f"[green]✓[/green] Downloaded {len(downloaded)} images"
                      + (f" ([red]{failed_dl} failed[/red])" if failed_dl else ""))

        # --- Step 5: Embed in batches ---
        console.print(f"\n[bold cyan]Step 5:[/bold cyan] Embedding images via Gemini API...")
        failed_emb = 0
        with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
                      TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("[cyan]Embedding photos", total=len(downloaded))
            batch_count = 0
            for start in range(0, len(downloaded), batch_size):
                chunk = downloaded[start:start + batch_size]
                try:
                    vecs = embed_images_with_retry(client, [b for _, b in chunk], dimensionality)
                    for (idx, _), vec in zip(chunk, vecs):
                        all_embeddings[idx] = vec
                        cache[str(row_ids[idx])] = vec
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Batch failed (rows {start}-{start + len(chunk) - 1}): {e}")
                    failed_emb += len(chunk)
                progress.update(task, advance=len(chunk))
                batch_count += 1
                if batch_count % CHECKPOINT_EVERY == 0:
                    save_cache(cache, cache_file)
                if delay > 0 and start + batch_size < len(downloaded):
                    time.sleep(delay)
        save_cache(cache, cache_file)
        console.print("[green]✓[/green] Embedding computation complete."
                      + (f" [yellow]{failed_emb} failed[/yellow]" if failed_emb else ""))

    # --- Step 6: Update dataset ---
    console.print(f"\n[bold cyan]Step 6:[/bold cyan] Updating dataset...")
    emb_col_data = [None if is_empty_embedding(e) else e for e in all_embeddings]
    pa_array = pa.array(emb_col_data, type=pa.list_(pa.float64()))
    if EMBEDDING_COLUMN in ds.column_names:
        ds_out = ds.remove_columns([EMBEDDING_COLUMN]).add_column(EMBEDDING_COLUMN, pa_array)
    else:
        ds_out = ds.add_column(EMBEDDING_COLUMN, pa_array)

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
        return

    console.print(f"\n[bold cyan]Step 7:[/bold cyan] Pushing to Hugging Face Hub...")
    try:
        with console.status("[bold green]Pushing dataset to Hub...", spinner="dots"):
            ds_out.push_to_hub(
                repo_id=repo_id, config_name=CONFIG_NAME,
                commit_message=(f"Add/update '{EMBEDDING_COLUMN}' multimodal embeddings using "
                                f"{MODEL_NAME} (dim={dimensionality})"),
                token=hf_token, max_shard_size=args.max_shard_size,
            )
        console.print(Panel(
            f"[bold green]Dataset successfully published![/bold green]\n\n"
            f"Repository: [cyan]{repo_id}[/cyan]\n"
            f"Configuration: [cyan]{CONFIG_NAME}[/cyan]\n"
            f"Column: [cyan]{EMBEDDING_COLUMN}[/cyan]\n"
            f"Model: [cyan]{MODEL_NAME}[/cyan] (dim={dimensionality})\n"
            f"Valid embeddings: [cyan]{valid}[/cyan] / {len(ds_out)}",
            title="Upload Complete", border_style="green"))
        delete_cache(cache_file)
    except Exception as e:  # noqa: BLE001
        console.print(Panel(
            f"[bold red]Failed to push dataset[/bold red]\n\n{e}\n\n"
            f"[yellow]Embeddings are cached in {cache_file}[/yellow] — re-run to resume "
            f"(in --update-mode all, add --resume).",
            title="Error", border_style="red"))
        logger.error("Push error details:", exc_info=True)


if __name__ == "__main__":
    main()
