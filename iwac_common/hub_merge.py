"""Shared helper for merging fresh Omeka data with the existing HF Hub dataset.

Every upload script repeats the same load → identify-extra-columns → merge
on ``o:id`` flow. The variations the helper accepts are:

- ``how`` / ``suffixes``: ``reference`` uses an outer merge with explicit
  suffixes; the other 5 use a left merge.
- ``columns_to_exclude``: ``reference`` drops a few legacy/computed columns
  (``o:item_set``, ``o:media/file``, ``iiif_manifest``, ``thumbnail``).
- ``fill_after_merge``: ``reference`` runs ``.ffill(axis=1).bfill(axis=1)``
  on the merged frame.

Subset-specific *post-merge* steps (sentiment-column reordering in
``articles``, mixed-type-column casting in ``reference``, integer dtype
coercion, etc.) intentionally stay in their scripts.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import pandas as pd
from datasets import load_dataset
from huggingface_hub import get_token
from rich.console import Console


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """Return the HF token to use, preferring (in order): caller-supplied,
    ``HF_TOKEN`` env var, the locally stored token from ``huggingface-cli login``.

    May return ``None`` if no token is available; callers are expected to
    handle that (e.g. trigger interactive ``login()``).
    """
    if explicit:
        return explicit
    return os.getenv("HF_TOKEN") or get_token()


def merge_with_hub_dataset(
    new_df: pd.DataFrame,
    repo: str,
    config_name: str,
    *,
    token: Optional[str] = None,
    how: str = "left",
    suffixes: Sequence[str] = ("", "_old"),
    columns_to_exclude: Iterable[str] = (),
    fill_after_merge: bool = False,
    console: Optional[Console] = None,
) -> pd.DataFrame:
    """Merge ``new_df`` with the existing HF Hub config, preserving any
    columns that exist on the Hub but not in ``new_df`` (typically
    post-processing outputs like embeddings, lemmas, topic IDs, …).

    Returns the merged DataFrame. If the Hub config is missing, empty, or
    has no extra columns, returns ``new_df`` unchanged.
    """
    console = console or Console()
    token = resolve_hf_token(token)
    excluded = set(columns_to_exclude)

    if "o:id" not in new_df.columns:
        console.print("[bold red]✗[/bold red] new_df is missing 'o:id'; skipping merge.")
        return new_df

    # Defensive: every script casts 'o:id' to str before merging anyway.
    new_df = new_df.copy()
    new_df["o:id"] = new_df["o:id"].astype(str)

    existing_df = pd.DataFrame()
    try:
        with console.status("[bold green]Loading existing dataset from Hub...", spinner="dots"):
            existing_ds = load_dataset(
                repo,
                name=config_name,
                split="train",
                token=token,
                download_mode="force_redownload",
                verification_mode="no_checks",
            )
            existing_df = existing_ds.to_pandas()

        if "o:id" not in existing_df.columns or existing_df["o:id"].isnull().all():
            console.print(
                "[yellow]⚠[/yellow] 'o:id' column missing or all null in existing Hub dataset. Treating as empty."
            )
            existing_df = pd.DataFrame()
        else:
            existing_df["o:id"] = existing_df["o:id"].astype(str)
            console.print(f"[green]✓[/green] Loaded {len(existing_df)} records from {repo}")
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]⚠[/yellow] Could not load existing dataset (may be first run): {exc}"
        )
        existing_df = pd.DataFrame()

    if existing_df.empty:
        console.print("[yellow]ℹ[/yellow] No existing data on Hub; using new Omeka data directly.")
        return new_df

    console.print(
        f"[blue]→[/blue] Merging new Omeka data ({len(new_df)} records) "
        f"with existing Hub data ({len(existing_df)} records)."
    )

    extra_cols = [
        col
        for col in existing_df.columns
        if col not in new_df.columns and col not in excluded
    ]

    if not extra_cols:
        console.print("[yellow]ℹ[/yellow] No unique columns to preserve from existing dataset.")
        if excluded:
            console.print(f"[dim]Excluded columns: {', '.join(sorted(excluded))}[/dim]")
        return new_df

    console.print(f"[green]✓[/green] Preserving columns: {', '.join(extra_cols)}")
    final_df = pd.merge(
        new_df,
        existing_df[["o:id"] + extra_cols],
        on="o:id",
        how=how,
        suffixes=tuple(suffixes),
    )
    if fill_after_merge:
        final_df = final_df.ffill(axis=1).bfill(axis=1)

    if excluded:
        console.print(f"[dim]Excluded columns: {', '.join(sorted(excluded))}[/dim]")
    console.print(
        f"[green]✓[/green] Merge complete: {len(final_df)} records, {len(final_df.columns)} columns"
    )
    for col_name in extra_cols:
        if col_name in final_df.columns:
            nan_count = final_df[col_name].isnull().sum()
            if nan_count > 0:
                console.print(
                    f"[yellow]ℹ[/yellow] Column '{col_name}' has {nan_count} null values "
                    f"(new items needing processing)"
                )
    return final_df


__all__ = ["merge_with_hub_dataset", "resolve_hf_token"]
