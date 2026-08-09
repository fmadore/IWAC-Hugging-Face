#!/usr/bin/env python3
"""
related_articles.py
===================

Precompute the top-k most similar items per row from the existing Gemini
embeddings (cosine similarity), for a "related articles" feature and as a
near-duplicate / reprint detector (agency dispatches and communiqués are
frequently reprinted across West African newspapers).

Supported configurations:
- **articles**:     'embedding_OCR'             → 'related_articles'
- **publications**: 'embedding_tableOfContents' → 'related_articles'

Column format: ``"o:id:cosine|o:id:cosine|..."`` (top-k, descending
similarity, 4 decimals). Rows without an embedding get None.

The default run is report-only: the column is written to
``analyses/output/related_articles_<config>.parquet`` together with a
near-duplicate report. Use --push to add the column to the Hub dataset
(requires --source hub so row alignment matches the live data).

Usage
-----
    python post-processing/related_articles.py [--config articles] [--source hub|csv]
                                               [--topk 10] [--push]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    PRIVATE_REPO_ID,
    REPO_ROOT,
    add_columns_by_id,
    ensure_hf_token,
    load_hub_dataset,
    load_subset_dataframe,
    push_dataset,
)

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
OUTPUT_DIR = REPO_ROOT / "analyses" / "output"

# config → (embedding column, context columns for the report)
CONFIG_SETTINGS = {
    "articles": ("embedding_OCR", ["title", "newspaper", "pub_date"]),
    "publications": ("embedding_tableOfContents", ["title", "pub_date"]),
}

NEAR_DUP_THRESHOLD = 0.95


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
        vecs.append(arr)
        positions.append(pos)
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=int)
    return np.vstack(vecs), np.array(positions, dtype=int)


def topk_neighbors(matrix: np.ndarray, k: int, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbors (indices into `matrix`, self excluded)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    n = len(unit)
    k_eff = min(k, n - 1)
    nn_idx = np.zeros((n, k_eff), dtype=np.int32)
    nn_sim = np.zeros((n, k_eff), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = unit[start:end] @ unit.T
        for row in range(end - start):
            sims[row, start + row] = -np.inf  # exclude self
        part = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        for row in range(end - start):
            cand = part[row]
            order = np.argsort(-sims[row, cand])
            nn_idx[start + row] = cand[order]
            nn_sim[start + row] = sims[row, cand[order]]
    return nn_idx, nn_sim


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Top-k related items per row from existing Gemini embeddings. "
                    "Report-only by default (nothing is written); pass --push to add "
                    "the output column and push it to the Hub."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID,
                        help="Repository ID on Hugging Face Hub (default: private full mirror).")
    parser.add_argument("--config", choices=list(CONFIG_SETTINGS), default="articles",
                        help="Dataset configuration (subset) to process (default: articles).")
    parser.add_argument("--source", choices=["hub", "csv"], default="hub",
                        help="hub = live dataset (default); csv = local data/ mirror")
    parser.add_argument("--topk", type=int, default=10,
                        help="Number of nearest neighbours per row (default: 10).")
    parser.add_argument("--column", default="related_articles", help="Output column name")
    parser.add_argument("--push", action="store_true",
                        help="Write mode: add the output column and push to the Hub "
                             "(without this flag the script only reports; nothing is written).")
    parser.add_argument("--max-shard-size", default="1GB")
    args = parser.parse_args()
    if args.topk < 1:
        parser.error("--topk must be at least 1")

    emb_col, context_cols = CONFIG_SETTINGS[args.config]

    console.print(Panel.fit(
        "[bold cyan]Related Articles (embedding kNN)[/bold cyan]\n"
        f"[dim]{args.config}: top-{args.topk} cosine neighbors from '{emb_col}'[/dim]",
        border_style="cyan",
    ))

    token = ensure_hf_token(console=console) if (args.source == "hub" or args.push) else None
    df = load_subset_dataframe(
        args.repo, args.config, token=token, source=args.source,
        columns=["o:id", emb_col] + context_cols, console=console,
    )
    source_revision = df.attrs.get("iwac_source_revision")
    if emb_col not in df.columns:
        console.print(f"[red]✗[/red] Embedding column '{emb_col}' not found.")
        return 1

    with console.status("[bold green]Parsing embeddings...", spinner="dots"):
        matrix, positions = parse_embeddings(df[emb_col])
    if len(positions) < 2:
        console.print("[red]✗[/red] Fewer than 2 rows with embeddings; nothing to do.")
        return 1
    console.print(
        f"[green]✓[/green] {len(positions):,} rows with embeddings "
        f"(dim={matrix.shape[1]}); {len(df) - len(positions):,} without"
    )

    with console.status(f"[bold green]Computing top-{args.topk} neighbors...", spinner="dots"):
        nn_idx, nn_sim = topk_neighbors(matrix, args.topk)

    ids = df["o:id"].to_numpy()
    valid_ids = ids[positions]
    related: List[Optional[str]] = [None] * len(df)
    for local_row, pos in enumerate(positions):
        related[pos] = "|".join(
            f"{valid_ids[j]}:{s:.4f}" for j, s in zip(nn_idx[local_row], nn_sim[local_row])
        )
    out_frame = pd.DataFrame({"o:id": ids, args.column: related})

    # ---------------- Report ----------------
    top1 = nn_sim[:, 0]
    n_dup = int((top1 >= NEAR_DUP_THRESHOLD).sum())
    stats = Table(title="Nearest-neighbor similarity", box=box.ROUNDED)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green", justify="right")
    stats.add_row("Rows with neighbors", f"{len(positions):,}")
    stats.add_row("Median top-1 similarity", f"{np.median(top1):.3f}")
    stats.add_row("Mean top-1 similarity", f"{top1.mean():.3f}")
    stats.add_row(f"Rows with top-1 ≥ {NEAR_DUP_THRESHOLD} (likely reprint/near-dup)", f"{n_dup:,}")
    console.print(stats)

    # Top near-duplicate pairs (deduplicated i<j)
    pairs = {}
    for local_row in range(len(positions)):
        j, s = int(nn_idx[local_row, 0]), float(nn_sim[local_row, 0])
        a, b = sorted((local_row, j))
        key = (a, b)
        if key not in pairs or s > pairs[key]:
            pairs[key] = s
    top_pairs = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:10]

    def describe(local_row: int) -> str:
        row = df.iloc[positions[local_row]]
        bits = [str(row.get(c, "") or "") for c in context_cols]
        title = bits[0][:45] + ("…" if len(bits[0]) > 45 else "")
        return f"{title} [{' · '.join(b for b in bits[1:] if b)}]"

    pt = Table(title="Most similar pairs (reprint candidates)", box=box.ROUNDED)
    pt.add_column("cos", justify="right", style="cyan")
    pt.add_column("Item A", style="green", max_width=60)
    pt.add_column("Item B", style="green", max_width=60)
    for (a, b), s in top_pairs:
        pt.add_row(f"{s:.4f}", describe(a), describe(b))
    console.print(pt)

    # ---------------- Outputs ----------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"related_articles_{args.config}.parquet"
    out_frame.to_parquet(out_path, index=False)
    console.print(f"[green]✓[/green] Column saved: [cyan]{out_path}[/cyan]")

    if not args.push:
        console.print("[yellow]ℹ[/yellow] Report-only run. Use [bold]--push[/bold] to add "
                      f"'{args.column}' to the Hub dataset.")
        return 0
    if args.source != "hub":
        console.print("[red]✗[/red] --push requires --source hub (row alignment must match live data).")
        return 1

    console.print("\n[bold cyan]Pushing related-articles column to the Hub...[/bold cyan]")
    ds = load_hub_dataset(
        args.repo,
        args.config,
        token=token,
        console=console,
        revision=source_revision,
    )
    ds = add_columns_by_id(ds, out_frame)
    if push_dataset(
        ds,
        repo_id=args.repo,
        config_name=args.config,
        token=token,
        max_shard_size=args.max_shard_size,
        commit_message=(
            f"Add '{args.column}' (top-{args.topk} cosine neighbors from {emb_col})"
        ),
        console=console,
        expected_revision=source_revision,
    ):
        console.print("[green]✓[/green] Pushed.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
