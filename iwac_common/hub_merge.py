"""Shared helper for merging fresh Omeka data with the existing HF Hub dataset.

Every upload script repeats the same load → identify-extra-columns → merge
on ``o:id`` flow. The variations the helper accepts are:

- ``how`` / ``suffixes``: ``reference`` uses an outer merge with explicit
  suffixes; the other 5 use a left merge.
- ``columns_to_exclude``: ``reference`` drops a few legacy/computed columns
  (``o:item_set``, ``o:media/file``, ``iiif_manifest``, ``thumbnail``).

Historical note: ``reference`` used to run ``.ffill(axis=1).bfill(axis=1)``
after the merge (via a ``fill_after_merge`` flag). That filled NaN cells from
*adjacent columns* — fabricating values for Hub-only rows and freshly added
items — and was removed as a data-corruption bug. Outer merges now log the
count of Hub-only rows instead, so genuinely deleted Omeka items are visible.

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


class ShrinkGuardError(RuntimeError):
    """Raised when the fresh Omeka fetch is suspiciously smaller than the
    dataset already on the Hub (likely a truncated/partial API response).
    Pushing would silently delete rows; the caller must pass
    ``allow_shrink=True`` (CLI: ``--force-shrink``) to proceed."""


class DuplicateIdError(ValueError):
    """Raised when either frame carries duplicate ``o:id`` values — merging
    would fan out rows and multiply records on the Hub."""


def _assert_unique_ids(df: pd.DataFrame, label: str) -> None:
    dupes = df["o:id"][df["o:id"].duplicated()]
    if not dupes.empty:
        sample = ", ".join(dupes.astype(str).unique()[:5])
        raise DuplicateIdError(
            f"{label} contains {dupes.nunique()} duplicated 'o:id' value(s) "
            f"(e.g. {sample}); merging would multiply rows. Deduplicate first."
        )


def merge_with_hub_dataset(
    new_df: pd.DataFrame,
    repo: str,
    config_name: str,
    *,
    token: Optional[str] = None,
    how: str = "left",
    suffixes: Sequence[str] = ("", "_old"),
    columns_to_exclude: Iterable[str] = (),
    console: Optional[Console] = None,
    min_row_ratio: float = 0.95,
    allow_shrink: bool = False,
    stale_rows: str = "keep",
) -> pd.DataFrame:
    """Merge ``new_df`` with the existing HF Hub config, preserving any
    columns that exist on the Hub but not in ``new_df`` (typically
    post-processing outputs like embeddings, lemmas, topic IDs, …).

    Safety rails (the Hub data is the irreplaceable artifact):

    - both frames must have unique ``o:id`` (raises :class:`DuplicateIdError`);
    - if ``new_df`` has fewer than ``min_row_ratio`` × existing rows the merge
      raises :class:`ShrinkGuardError` unless ``allow_shrink=True`` — a
      truncated Omeka fetch must not silently delete Hub rows;
    - for outer merges, ``stale_rows`` controls Hub-only rows (items deleted
      on Omeka): ``"keep"`` (historical behavior, logged loudly) or
      ``"drop"``.

    Returns the merged DataFrame. If the Hub config is missing, empty, or
    has no extra columns, returns ``new_df`` unchanged.
    """
    console = console or Console()
    token = resolve_hf_token(token)
    excluded = set(columns_to_exclude)
    if stale_rows not in ("keep", "drop"):
        raise ValueError(f"stale_rows must be 'keep' or 'drop', got {stale_rows!r}")

    if "o:id" not in new_df.columns:
        console.print("[bold red]✗[/bold red] new_df is missing 'o:id'; skipping merge.")
        return new_df

    # Defensive: every script casts 'o:id' to str before merging anyway.
    new_df = new_df.copy()
    new_df["o:id"] = new_df["o:id"].astype(str)
    _assert_unique_ids(new_df, "new Omeka data")

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

    _assert_unique_ids(existing_df, f"existing Hub dataset '{config_name}'")

    # Shrink tripwire: a truncated Omeka fetch (partial API response) flowing
    # into a left merge would silently delete the missing rows from the Hub.
    if len(new_df) < min_row_ratio * len(existing_df):
        msg = (
            f"Fresh Omeka data has {len(new_df):,} rows but the Hub config "
            f"'{config_name}' has {len(existing_df):,} "
            f"(< {min_row_ratio:.0%} threshold). This usually means a truncated "
            f"fetch; pushing would delete rows. Re-run with --force-shrink only "
            f"if the shrink is intentional (items really deleted on Omeka)."
        )
        if allow_shrink:
            console.print(f"[yellow]⚠ Shrink allowed by caller:[/yellow] {msg}")
        else:
            raise ShrinkGuardError(msg)

    # Schema visibility: brand-new columns are normal when a mapper gains a
    # field, but they also catch accidental renames (old column preserved via
    # extra_cols + new column added → near-duplicate columns).
    brand_new_cols = [c for c in new_df.columns if c not in existing_df.columns]
    if brand_new_cols:
        console.print(
            f"[yellow]ℹ[/yellow] New column(s) not on the Hub yet: "
            f"{', '.join(brand_new_cols)} (renamed mapper fields would show up here)"
        )

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
        indicator=how == "outer",
    )
    if how == "outer":
        stale_mask = final_df["_merge"] == "right_only"
        hub_only = int(stale_mask.sum())
        if hub_only and stale_rows == "drop":
            final_df = final_df[~stale_mask]
            console.print(
                f"[yellow]⚠[/yellow] Dropped {hub_only} Hub-only row(s) "
                "(items no longer in Omeka; --stale-rows drop)."
            )
        elif hub_only:
            console.print(
                f"[yellow]⚠[/yellow] {hub_only} row(s) exist on the Hub but not in Omeka "
                "(deleted items?). They are KEPT with empty Omeka fields and will be "
                "re-pushed as-is — pass --stale-rows drop to remove them."
            )
        final_df = final_df.drop(columns="_merge")

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


__all__ = [
    "merge_with_hub_dataset",
    "resolve_hf_token",
    "ShrinkGuardError",
    "DuplicateIdError",
]
