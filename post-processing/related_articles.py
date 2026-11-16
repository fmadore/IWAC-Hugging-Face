#!/usr/bin/env python3
"""
related_articles.py
===================

Precompute the top-k most similar items per row from the existing
embeddings (cosine similarity), for a "related articles" feature and as a
near-duplicate / reprint detector (agency dispatches and communiqués are
frequently reprinted across West African newspapers).

Supported configurations:
- **articles**:     'embedding_OCR'             → 'related_articles'
- **publications**: 'embedding_tableOfContents' → 'related_articles'
- **documents**:    'embedding_OCR'             → 'related_articles'

Column format: ``"o:id:cosine|o:id:cosine|..."`` (top-k, descending
similarity, 4 decimals).

**Only public items are neighbours.** The candidate pool is the rows whose
``item_is_public`` is exactly True (null is private; a missing column aborts).
A public item whose full text is private stays in the pool: its embedding is a
reviewed public derivative. Private rows get None, as rows without an
embedding do: the column exists for the public "related items" feature, where
private rows never appear, and a private item turning public needs a re-run
anyway because it is absent from every other list. The near-duplicate report
and the candidate-pair file are computed over the same public pool. The
public projection strips, and the write gateway refuses, any entry naming an
unpublished id (``iwac_common.write_policy``), which covers items that turned
private after this ran.

The default run is report-only: the column is written to
``analyses/output/related_articles_<config>.parquet`` together with a
near-duplicate report. Use --push to add the column to the Hub dataset
(requires --source hub so row alignment matches the live data). The commit
message and the run manifest record the embedding configuration hash and
top-k that produced the column.

Usage
-----
    python post-processing/related_articles.py [--config articles] [--source hub|local]
                                               [--topk 10] [--push]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    PRIVATE_REPO_ID,
    REPO_ROOT,
    SOURCE_CHOICES,
    add_columns_by_id,
    ensure_hf_token,
    load_environment,
    load_hub_dataset,
    load_subset_dataframe,
    normalize_source,
    push_dataset,
    write_run_manifest,
)
from iwac_common.write_policy import PublicationPolicyError, strict_public_flag  # noqa: E402

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_environment()

console = Console()
OUTPUT_DIR = REPO_ROOT / "analyses" / "output"

# config → (embedding column, context columns for the report)
CONFIG_SETTINGS = {
    "articles": ("embedding_OCR", ["title", "newspaper", "pub_date"]),
    "publications": ("embedding_tableOfContents", ["title", "pub_date"]),
    "documents": ("embedding_OCR", ["title", "pub_date"]),
}

# Cosine-only near-duplicate cutoff, and the vector space it was calibrated in.
# Used only for one count in the printed report ("rows with top-1 >= cutoff"):
# it filters nothing, and the pushed column and candidate pairs ignore it.
# Chosen by `iwac-analyze similarity-calibration` on articles (EmbeddingGemma 2,
# 2026-10) with a per-pair split that let a reprint family sit on both sides,
# so its held-out precision was optimistic; re-run the calibration (now split
# by connected component) before citing a figure. Cosine scales differ between models, so any
# other subset or configuration is reported as uncalibrated.
NEAR_DUP_THRESHOLD = 0.97
NEAR_DUP_CALIBRATION = ("articles", "c0cf10d8be87655367ada20867b3f29b0306f68073a9ec9060f6a2b093898a98")


def near_dup_label(config: str, configurations) -> str:
    """Whether the cutoff was calibrated for this subset and embedding space."""
    calibrated = (config, *sorted(map(str, configurations))) == NEAR_DUP_CALIBRATION
    return "lexical-proxy calibrated cutoff" if calibrated else "uncalibrated for this subset/vector space"


def parse_embeddings(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (matrix[n_valid, dim] float32, valid_row_positions).

    Handles both list values (Hub parquet) and JSON strings (CSV mirror).
    """
    vecs: List[np.ndarray] = []
    positions: List[int] = []
    for pos, v in enumerate(series.to_numpy()):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, str):
            if not v or v == "[]":
                continue
            v = json.loads(v)
        arr = np.asarray(v, dtype=np.float32)
        if arr.size == 0 or not np.any(arr):
            continue
        if arr.ndim != 1 or not np.isfinite(arr).all():
            raise ValueError(f"Embedding at row {pos} must be a finite, flat vector")
        if vecs and len(arr) != len(vecs[0]):
            raise ValueError(f"Embedding at row {pos} has a different dimensionality")
        vecs.append(arr)
        positions.append(pos)
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=int)
    return np.vstack(vecs), np.array(positions, dtype=int)


def topk_neighbors(matrix: np.ndarray, k: int, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbors (indices into `matrix`, self excluded)."""
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or k < 1 or chunk < 1:
        raise ValueError("Expected finite 2-D vectors, positive k and positive chunk size")
    if len(matrix) < 2:
        return np.empty((len(matrix), 0), dtype=np.int32), np.empty((len(matrix), 0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero vectors do not have cosine neighbors")
    unit = matrix / norms
    n = len(unit)
    k_eff = min(k, n - 1)
    nn_idx = np.zeros((n, k_eff), dtype=np.int32)
    nn_sim = np.zeros((n, k_eff), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = np.clip(unit[start:end] @ unit.T, -1.0, 1.0)
        for row in range(end - start):
            sims[row, start + row] = -np.inf  # exclude self
        part = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        for row in range(end - start):
            # Include ties at the k-th boundary before ordering by row index,
            # otherwise argpartition chooses an arbitrary subset of ties.
            cutoff = np.min(sims[row, part[row]])
            cand = np.flatnonzero(sims[row] >= cutoff)
            order = np.lexsort((cand, -sims[row, cand]))[:k_eff]
            nn_idx[start + row] = cand[order]
            nn_sim[start + row] = sims[row, cand[order]]
    return nn_idx, nn_sim


def candidate_pairs(ids: np.ndarray, nn_idx: np.ndarray, nn_sim: np.ndarray) -> pd.DataFrame:
    """Unique pairs across all top-k neighbors; similarity is not a reprint verdict."""
    pairs = {}
    for row, (neighbors, scores) in enumerate(zip(nn_idx, nn_sim, strict=True)):
        for neighbor, score in zip(neighbors, scores, strict=True):
            key = tuple(sorted((str(ids[row]), str(ids[int(neighbor)]))))
            pairs[key] = max(pairs.get(key, -1.0), float(score))
    return pd.DataFrame(
        [{"id_a": a, "id_b": b, "cosine_similarity": score}
         for (a, b), score in sorted(pairs.items(), key=lambda pair: (-pair[1], pair[0]))],
        columns=["id_a", "id_b", "cosine_similarity"],
    )


def public_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows eligible as neighbours: ``item_is_public`` exactly True.

    Fails closed: a frame without the column cannot show any item is public,
    and a non-boolean value is refused rather than read by truthiness.
    """
    if "item_is_public" not in df.columns:
        raise PublicationPolicyError(
            "'item_is_public' is missing, so no item can be shown to be public; "
            "refresh this subset from Omeka (or the local mirror) before computing neighbours"
        )
    return df["item_is_public"].map(lambda v: strict_public_flag(v, field="item_is_public")).to_numpy(dtype=bool)


def format_neighbors(n_rows: int, positions: np.ndarray, pool_ids: np.ndarray,
                     nn_idx: np.ndarray, nn_sim: np.ndarray) -> List[Optional[str]]:
    """The column values: ``"id:cos|..."`` at each pool row, None elsewhere."""
    related: List[Optional[str]] = [None] * n_rows
    for local_row, pos in enumerate(positions):
        related[pos] = "|".join(
            f"{pool_ids[j]}:{s:.4f}" for j, s in zip(nn_idx[local_row], nn_sim[local_row])
        ) or None
    return related


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Top-k related items per row from existing embeddings. "
                    "Report-only by default (nothing is written); pass --push to add "
                    "the output column and push it to the Hub."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID,
                        help="Repository ID on Hugging Face Hub (default: private full mirror).")
    parser.add_argument("--config", choices=list(CONFIG_SETTINGS), default="articles",
                        help="Dataset configuration (subset) to process (default: articles).")
    parser.add_argument("--source", choices=list(SOURCE_CHOICES), default="hub",
                        help="hub = live dataset (default); local = revision-pinned data/ mirror "
                             "(csv is accepted as its historical alias)")
    parser.add_argument("--topk", type=int, default=10,
                        help="Number of nearest neighbours per row (default: 10).")
    parser.add_argument("--column", default="related_articles", help="Output column name")
    parser.add_argument("--allow-unverified-embeddings", action="store_true",
                        help="Allow exploratory analysis of legacy/public vectors without processing provenance")
    parser.add_argument("--push", action="store_true",
                        help="Write mode: add the output column and push to the Hub "
                             "(without this flag the script only reports; nothing is written).")
    parser.add_argument("--max-shard-size", default="1GB")
    args = parser.parse_args()
    if args.topk < 1:
        parser.error("--topk must be at least 1")
    args.source = normalize_source(args.source)
    if args.push and args.source != "hub":
        parser.error("--push requires --source hub (row alignment must match live data)")

    emb_col, context_cols = CONFIG_SETTINGS[args.config]
    config_col = f"{emb_col}_config_hash"

    console.print(Panel.fit(
        "[bold cyan]Related Articles (embedding kNN)[/bold cyan]\n"
        f"[dim]{args.config}: top-{args.topk} public cosine neighbors from '{emb_col}'[/dim]",
        border_style="cyan",
    ))

    token = ensure_hf_token(console=console) if (args.source == "hub" or args.push) else None
    df = load_subset_dataframe(
        args.repo, args.config, token=token, source=args.source,
        columns=["o:id", "item_is_public", emb_col, config_col] + context_cols, console=console,
    )
    source_revision = df.attrs.get("iwac_source_revision")
    df = df.sort_values("o:id", kind="stable").reset_index(drop=True)
    if emb_col not in df.columns:
        console.print(f"[red]✗[/red] Embedding column '{emb_col}' not found.")
        return 1
    try:
        public = public_mask(df)
    except PublicationPolicyError as exc:
        console.print(f"[red]✗[/red] {exc}")
        return 1

    with console.status("[bold green]Parsing embeddings...", spinner="dots"):
        # Private rows are neither queries nor candidates.
        matrix, positions = parse_embeddings(df[emb_col].where(public, None))
    if len(positions) < 2:
        console.print("[red]✗[/red] Fewer than 2 public rows with embeddings; nothing to do.")
        return 1
    configs: list = []
    unverified = config_col not in df
    if config_col in df:
        configs = sorted(map(str, df.iloc[positions][config_col].dropna().unique()))
        unverified = df.iloc[positions][config_col].isna().any()
        if len(configs) > 1:
            console.print("[red]✗[/red] Embeddings mix processing configurations; recompute before comparing.")
            return 1
    if unverified:
        if not args.allow_unverified_embeddings:
            console.print("[red]✗[/red] Embedding provenance is missing. Re-embed or explicitly "
                          "use --allow-unverified-embeddings for exploratory analysis.")
            return 1
        console.print("[yellow]Unverified legacy embedding provenance: treat neighbors as exploratory candidates.[/yellow]")
    embedding_config = configs[0] if configs and not unverified else "unverified"
    n_private = int((~public).sum())
    console.print(
        f"[green]✓[/green] {len(positions):,} public rows with embeddings "
        f"(dim={matrix.shape[1]}); {n_private:,} private rows excluded from the pool; "
        f"{len(df) - n_private - len(positions):,} public rows without an embedding"
    )

    with console.status(f"[bold green]Computing top-{args.topk} neighbors...", spinner="dots"):
        nn_idx, nn_sim = topk_neighbors(matrix, args.topk)

    ids = df["o:id"].to_numpy()
    pool_ids = ids[positions]
    out_frame = pd.DataFrame({
        "o:id": ids, args.column: format_neighbors(len(df), positions, pool_ids, nn_idx, nn_sim),
    })
    pairs = candidate_pairs(pool_ids, nn_idx, nn_sim)

    # ---------------- Report ----------------
    top1 = nn_sim[:, 0]
    n_dup = int((top1 >= NEAR_DUP_THRESHOLD).sum())
    stats = Table(title="Nearest-neighbor similarity (public pool)", box=box.ROUNDED)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green", justify="right")
    stats.add_row("Rows with neighbors", f"{len(positions):,}")
    stats.add_row("Median top-1 similarity", f"{np.median(top1):.3f}")
    stats.add_row("Mean top-1 similarity", f"{top1.mean():.3f}")
    stats.add_row(f"Rows with top-1 ≥ {NEAR_DUP_THRESHOLD} ({near_dup_label(args.config, configs)})",
                  f"{n_dup:,}")
    console.print(stats)

    row_of = {str(i): pos for i, pos in zip(pool_ids, positions)}

    def describe(item_id: str) -> str:
        row = df.iloc[row_of[item_id]]
        bits = [str(row.get(c, "") or "") for c in context_cols]
        title = bits[0][:45] + ("…" if len(bits[0]) > 45 else "")
        return f"{title} [{' · '.join(b for b in bits[1:] if b)}]"

    pt = Table(title="Most similar pairs (reprint candidates)", box=box.ROUNDED)
    pt.add_column("cos", justify="right", style="cyan")
    pt.add_column("Item A", style="green", max_width=60)
    pt.add_column("Item B", style="green", max_width=60)
    for pair in pairs.head(10).itertuples(index=False):
        pt.add_row(f"{pair.cosine_similarity:.4f}", describe(pair.id_a), describe(pair.id_b))
    console.print(pt)

    # ---------------- Outputs ----------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"related_articles_{args.config}.parquet"
    out_frame.to_parquet(out_path, index=False)
    pair_path = OUTPUT_DIR / f"related_articles_{args.config}_candidates.parquet"
    pairs.to_parquet(pair_path, index=False)
    console.print(f"[green]✓[/green] Column saved: [cyan]{out_path}[/cyan]")
    write_run_manifest(
        OUTPUT_DIR, script=f"related_articles_{args.config}", repo_id=args.repo,
        revision=source_revision, args=args, outputs=[out_path, pair_path],
        inputs={"embedding_column": emb_col,
                "embedding_configuration": configs,
                "embedding_config_hash": embedding_config,
                "topk": args.topk,
                "neighbour_pool": "item_is_public is True",
                "n_pool_rows": int(len(positions)),
                "n_private_rows_excluded": n_private,
                "interpretation": "Similarity candidates require lexical and historical validation"},
    )

    if not args.push:
        console.print("[yellow]ℹ[/yellow] Report-only run. Use [bold]--push[/bold] to add "
                      f"'{args.column}' to the Hub dataset.")
        return 0

    console.print("\n[bold cyan]Pushing related-articles column to the Hub...[/bold cyan]")
    ds = load_hub_dataset(
        args.repo,
        args.config,
        token=token,
        console=console,
        revision=source_revision,
    )
    import pyarrow as pa

    ds = add_columns_by_id(ds, out_frame, types={args.column: pa.string()})
    if push_dataset(
        ds,
        repo_id=args.repo,
        config_name=args.config,
        token=token,
        max_shard_size=args.max_shard_size,
        commit_message=(
            f"Add '{args.column}' (top-{args.topk} public cosine neighbors from {emb_col}, "
            f"embedding config {embedding_config})"
        ),
        console=console,
        expected_revision=source_revision,
    ):
        console.print("[green]✓[/green] Pushed.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
