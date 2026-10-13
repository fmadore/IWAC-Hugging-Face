#!/usr/bin/env python3
"""
semantic_embedding_images.py
============================

Adds a multimodal semantic embedding column (``embedding_image``) to the
``images`` subset by embedding each **photograph itself** with
``google/embeddinggemma-2``, run locally (ONNX, fp32) at pinned revisions.

EmbeddingGemma 2 maps images and text into the *same* vector space, so
``embedding_image`` is directly comparable to the text embeddings on the
other subsets (``embedding_OCR``, ``embedding_tableOfContents``) — enabling
cross-modal search (a text query can retrieve photographs, and vice versa).
Both columns must therefore always come from the same model revision.

This is a sibling of ``semantic_embedding.py`` (which embeds *text* columns).
The image path is different enough — download the picture, resize it to the
model's patch budget, run the vision encoder; no text chunking/averaging — that
keeping it separate avoids complicating the text path.

Pipeline: load ``images`` from the private mirror → download each photo (from
``image_url``, fallback ``thumbnail``) → Gemma 4 aspect-preserving resize
(``_gemma_encoder.image_patches``, checked against the transformers reference:
cosine ≥ 0.99997) → embed → write ``embedding_image`` → push back to the
private mirror.

Each vector is written to a resume cache in ``.cache_embeddings/`` as soon as
it is computed; re-running the same command resumes. The cache filename embeds
a fingerprint of the configuration, and each entry a hash of the downloaded
bytes and URL.

Usage
-----
    python post-processing/semantic_embedding_images.py
    python post-processing/semantic_embedding_images.py --update-mode all
    python post-processing/semantic_embedding_images.py --dry-run

Environment Variables
---------------------
HF_TOKEN         Personal access token for the Hugging Face Hub.

Dependencies
------------
    pip install onnxruntime tokenizers datasets huggingface_hub rich pillow
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import hashlib
import io
import logging
import os
import sys
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
from _embedding_columns import set_embedding_column  # noqa: E402
from _gemma_encoder import (  # noqa: E402
    DIMENSION,
    IMAGE_PROCESSING,
    MAX_SOFT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    GemmaEncoder,
    document_text,
    model_settings,
    runtime_versions,
)
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
CONFIG_NAME = "images"
SOURCE_COLUMN = "image_url"          # falls back to ``thumbnail`` per row
FALLBACK_COLUMN = "thumbnail"
EMBEDDING_COLUMN = "embedding_image"
CACHE_DIR = workspace_root() / ".cache_embeddings"
# Resume cache stem; the full filename embeds cache_fingerprint(model, dim,
# "image") and the configuration hash, so a cache written at one embedding
# configuration is never restored into a run with different parameters.
CACHE_STEM = "image_embeddings"
DOWNLOAD_TIMEOUT = 30
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
IMAGE_PROCESSING_VERSION = 3


def download_image(url: str) -> Optional[bytes]:
    """The image's bytes as served, or ``None`` (logged) so one bad URL never
    aborts the run."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "iwac-embed/1.0"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            raw = resp.read(MAX_DOWNLOAD_BYTES + 1)
        if len(raw) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Image download exceeds 32 MiB")
        return raw
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to download image {url}: {e}")
        return None


def decode_image(raw: bytes) -> Image.Image:
    """EXIF-upright RGB image; resizing is the encoder's job."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


def load_encoder(args, token: str) -> GemmaEncoder:
    """Separate so tests can substitute an offline encoder."""
    return GemmaEncoder(images=True, threads=args.threads, token=token)


def display_config_panel(repo_id: str, update_mode: str, dry_run: bool) -> None:
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Repository", repo_id)
    table.add_row("Configuration", CONFIG_NAME)
    table.add_row("Source Column", f"{SOURCE_COLUMN} (fallback: {FALLBACK_COLUMN})")
    table.add_row("Embedding Column", EMBEDDING_COLUMN)
    table.add_row("Model", f"{MODEL_ID}@{MODEL_REVISION[:12]}")
    table.add_row("Output Dimensionality", str(DIMENSION))
    table.add_row("Image Processing", f"{IMAGE_PROCESSING}, ≤{MAX_SOFT_TOKENS} soft tokens")
    table.add_row("Update Mode", update_mode)
    table.add_row("Checkpoint", "every image (resume by re-running)")
    if dry_run:
        table.add_row("Dry Run", "[yellow]YES — no changes will be pushed[/yellow]")
    console.print(Panel(table, title="[bold blue]Multimodal Image Embedding Configuration", border_style="blue"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a multimodal image embedding column to the 'images' subset "
                    f"using {MODEL_ID}, run locally."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID,
                        help="Repository ID on Hugging Face Hub (default: private full mirror).")
    parser.add_argument("--config", default=CONFIG_NAME, choices=[CONFIG_NAME],
                        help="Dataset configuration to process (only 'images').")
    parser.add_argument("--threads", type=int, default=None,
                        help="onnxruntime intra-op threads (default: onnxruntime's choice).")
    parser.add_argument("--max-shard-size", default="1GB",
                        help="Maximum Parquet shard size when pushing to Hub.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute embeddings but do not push to Hub (the cache is kept).")
    parser.add_argument("--update-mode", choices=["missing", "all"], default="missing",
                        help="Update only missing embeddings (default) or recompute all.")
    parser.add_argument("--fresh-cache", action="store_true",
                        help="Discard this configuration's resume cache before starting.")
    parser.add_argument("--resume", action="store_true",
                        help=argparse.SUPPRESS)  # resuming is now the default
    parser.add_argument("--allow-partial", action="store_true",
                        help="Push despite download/embedding failures. By default "
                             "the cache is kept and the Hub is left untouched.")
    args = parser.parse_args()
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")

    repo_id = args.repo
    update_mode = args.update_mode
    dry_run = args.dry_run
    configuration_settings = {
        **model_settings(), "image_processing": IMAGE_PROCESSING, "max_soft_tokens": MAX_SOFT_TOKENS,
        "processing_version": IMAGE_PROCESSING_VERSION,
        # Resampling can differ across Pillow releases; 30 images are cheap to redo.
        "pillow": version("Pillow"),
    }
    configuration = config_fingerprint(configuration_settings)
    expected_dimension = (SUBSETS[CONFIG_NAME].embedding_columns or {})[EMBEDDING_COLUMN]
    if DIMENSION != expected_dimension:
        parser.error(f"{MODEL_ID} produces {DIMENSION} dims; schema expects {expected_dimension}")
    # The repository is part of the name, and each entry carries a hash of the
    # image bytes and URL it was computed from.
    cache_file = CACHE_DIR / (
        f"{CACHE_STEM}_{repo_slug(repo_id)}_"
        f"{cache_fingerprint(MODEL_ID, DIMENSION, 'image')}_{configuration[:16]}.sqlite3"
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
    console.print(f"\n[bold cyan]Step 2:[/bold cyan] Loading {MODEL_ID} with its vision encoder...")
    try:
        encoder = load_encoder(args, hf_token)
        encoder.encode([document_text("test")])  # readiness probe; validates the output
        versions = runtime_versions()
        console.print(f"[green]✓[/green] Encoder ready (onnxruntime {versions['onnxruntime']}, "
                      f"tokenizers {versions['tokenizers']})")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]✗[/red] Failed to load the encoder: {e}")
        return 1

    console.print()
    display_config_panel(repo_id, update_mode, dry_run)

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

    # Check the actual image bytes on every run: a URL can retain its name
    # while its image changes. Only inference is skipped on a proven match.
    existing = list(ds[EMBEDDING_COLUMN]) if EMBEDDING_COLUMN in ds.column_names else [None] * len(ds)
    input_col, config_col = provenance_columns(EMBEDDING_COLUMN)
    old_inputs = list(ds[input_col]) if input_col in ds.column_names else [None] * len(ds)
    old_configs = list(ds[config_col]) if config_col in ds.column_names else [None] * len(ds)
    all_embeddings: List[Any] = [[] for _ in range(len(ds))]
    fingerprints = [""] * len(ds)
    cache = load_cache(cache_file)
    failed_dl = 0
    failed_emb = 0

    try:
        with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
                      TaskProgressColumn(), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("[cyan]Checking and embedding images", total=len(ds))
            for idx in range(len(ds)):
                url = row_url(idx)
                raw = download_image(url) if url else None
                if url and raw is None:
                    failed_dl += 1
                elif raw is not None:
                    fingerprints[idx] = input_fingerprint(hashlib.sha256(raw).hexdigest(), url)
                    cached = cached_value(cache, row_ids[idx], fingerprints[idx])
                    if cached is not None:
                        all_embeddings[idx] = cached
                    elif (update_mode == "missing" and old_inputs[idx] == fingerprints[idx]
                          and old_configs[idx] == configuration and not is_empty_embedding(existing[idx])):
                        all_embeddings[idx] = existing[idx]
                    else:
                        try:
                            vector = encoder.encode_image(decode_image(raw)).tolist()
                        except Exception as exc:  # noqa: BLE001
                            logger.error("Image %s failed: %s", row_ids[idx], exc)
                            failed_emb += 1
                        else:
                            all_embeddings[idx] = vector
                            entry = make_entry(vector, fingerprints[idx])
                            cache[str(row_ids[idx])] = entry
                            save_cache({str(row_ids[idx]): entry}, cache_file)
                progress.update(task, advance=1)
    except KeyboardInterrupt:
        console.print(Panel(
            f"[yellow]Interrupted; Hub left untouched.[/yellow]\n\nFinished images are saved in "
            f"{cache_file}.\nRe-run the same command to resume.",
            title="Embedding paused", border_style="yellow"))
        return 130

    if (failed_dl or failed_emb) and not args.allow_partial:
        console.print(Panel(
            "[bold red]Image embedding run incomplete; Hub left untouched.[/bold red]\n\n"
            f"Downloads failed: {failed_dl}; embeddings failed: {failed_emb}. "
            f"Successful work remains in {cache_file}; re-run to retry.",
            title="Fail-closed derived-data write",
            border_style="red",
        ))
        return 1

    # --- Step 4: Update dataset ---
    console.print("\n[bold cyan]Step 4:[/bold cyan] Updating dataset...")
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

    # --- Step 5: Push ---
    if dry_run:
        console.print(Panel(
            f"[yellow]Dry run — nothing pushed.[/yellow]\n\n"
            f"Would push [cyan]{len(ds_out)}[/cyan] rows to [cyan]{repo_id}[/cyan] (config: {CONFIG_NAME}).\n"
            f"Embeddings cached in {cache_file}.",
            title="Dry Run Complete", border_style="yellow"))
        return 0

    console.print("\n[bold cyan]Step 5:[/bold cyan] Pushing to Hugging Face Hub...")
    if push_dataset(
        ds_out,
        repo_id=repo_id,
        config_name=CONFIG_NAME,
        commit_message=(
            f"Add/update '{EMBEDDING_COLUMN}' multimodal embeddings using "
            f"{MODEL_ID}@{MODEL_REVISION[:12]} (dim={DIMENSION})"
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
            f"Model: [cyan]{MODEL_ID}@{MODEL_REVISION[:12]}[/cyan] (dim={DIMENSION})\n"
            f"Valid embeddings: [cyan]{valid}[/cyan] / {len(ds_out)}",
            title="Upload Complete", border_style="green"))
        delete_cache(cache_file)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
