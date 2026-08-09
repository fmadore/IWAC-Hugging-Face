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

from collections.abc import Mapping
from typing import Iterable, MutableMapping, Optional, Sequence

import pandas as pd
from datasets import load_dataset
from rich.console import Console

from .hub import (
    HubBaselineUnavailableError,
    get_repo_configs,
    get_repo_revision,
    resolve_hf_token,
)


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
    allow_initialize: bool = False,
    preserve_existing_ids: Iterable[object] = (),
    preserve_fields_by_id: Optional[Mapping[object, Iterable[str]]] = None,
    revision_out: Optional[MutableMapping[str, str]] = None,
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

    Hub reads fail closed.  A missing config is treated as a first run only
    when ``allow_initialize=True`` and the Hub confirms that the config is not
    declared. ``revision_out`` receives the baseline repository SHA for an
    optimistic-concurrency check immediately before the later push.
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

    baseline_revision = None
    if revision_out is not None:
        baseline_revision = get_repo_revision(repo, token=token)
        revision_out["revision"] = baseline_revision

    existing_df = pd.DataFrame()
    try:
        with console.status("[bold green]Loading existing dataset from Hub...", spinner="dots"):
            existing_ds = load_dataset(
                repo,
                name=config_name,
                split="train",
                token=token,
                revision=baseline_revision,
                download_mode="force_redownload",
                verification_mode="no_checks",
            )
            existing_df = existing_ds.to_pandas()

        if existing_df.empty:
            console.print(
                "[yellow]ℹ[/yellow] Existing Hub config is empty; using new Omeka data."
            )
        elif "o:id" not in existing_df.columns or existing_df["o:id"].isnull().any():
            raise HubBaselineUnavailableError(
                f"Existing Hub dataset '{config_name}' has a missing or null "
                "'o:id' column; refusing to treat a corrupt baseline as empty."
            )
        else:
            existing_df["o:id"] = existing_df["o:id"].astype(str)
            console.print(f"[green]✓[/green] Loaded {len(existing_df)} records from {repo}")
    except HubBaselineUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        if allow_initialize:
            configs = get_repo_configs(repo, token=token)
            if config_name not in configs:
                console.print(
                    f"[yellow]ℹ[/yellow] '{config_name}' is not declared in {repo}; "
                    "initialization explicitly allowed."
                )
                existing_df = pd.DataFrame()
            else:
                raise HubBaselineUnavailableError(
                    f"Could not load existing Hub config '{config_name}' from {repo}: "
                    f"{exc}. The config exists, so refusing to overwrite it."
                ) from exc
        else:
            raise HubBaselineUnavailableError(
                f"Could not load existing Hub config '{config_name}' from {repo}: "
                f"{exc}. Refusing to assume this is a first run; pass "
                "--initialize only for a deliberately new config."
            ) from exc

    if existing_df.empty:
        console.print("[yellow]ℹ[/yellow] No existing data on Hub; using new Omeka data directly.")
        return new_df

    _assert_unique_ids(existing_df, f"existing Hub dataset '{config_name}'")

    # A mapper/media failure is not a deletion.  When the caller explicitly
    # allows such failures, retain the complete existing row for those ids and
    # preserve selected same-name fields that a degraded mapper returned blank.
    preserve_ids = {str(value) for value in preserve_existing_ids}
    fields_by_id = {
        str(row_id): set(fields)
        for row_id, fields in (preserve_fields_by_id or {}).items()
    }
    existing_by_id = existing_df.set_index("o:id", drop=False)
    for row_id, fields in fields_by_id.items():
        if row_id not in existing_by_id.index:
            continue
        mask = new_df["o:id"] == row_id
        for field in fields:
            if field in new_df.columns and field in existing_df.columns:
                new_df.loc[mask, field] = existing_by_id.at[row_id, field]

    missing_preserved = sorted(preserve_ids - set(new_df["o:id"]))
    if missing_preserved:
        unavailable = [row_id for row_id in missing_preserved if row_id not in existing_by_id.index]
        if unavailable:
            raise HubBaselineUnavailableError(
                "Cannot preserve failed mapper rows absent from the Hub baseline: "
                + ", ".join(unavailable[:5])
            )
        console.print(
            f"[yellow]⚠[/yellow] Preserved {len(missing_preserved)} complete Hub "
            "row(s) whose fresh mapper failed."
        )

    def append_preserved_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if not missing_preserved:
            return frame
        retained = existing_by_id.loc[missing_preserved].reindex(columns=frame.columns)
        combined = pd.concat([frame, retained], ignore_index=True)
        _assert_unique_ids(combined, "merged data plus preserved mapper failures")
        return combined

    # Shrink tripwire: a truncated Omeka fetch (partial API response) flowing
    # into a left merge would silently delete the missing rows from the Hub.
    effective_new_count = len(new_df) + len(missing_preserved)
    if effective_new_count < min_row_ratio * len(existing_df):
        msg = (
            f"Fresh Omeka data has {effective_new_count:,} usable rows "
            f"({len(new_df):,} mapped + {len(missing_preserved):,} explicitly "
            f"preserved) but the Hub config "
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

    if not extra_cols and how != "outer":
        console.print("[yellow]ℹ[/yellow] No unique columns to preserve from existing dataset.")
        if excluded:
            console.print(f"[dim]Excluded columns: {', '.join(sorted(excluded))}[/dim]")
        return append_preserved_rows(new_df)

    if extra_cols:
        console.print(f"[green]✓[/green] Preserving columns: {', '.join(extra_cols)}")
    merge_columns = ["o:id"] + extra_cols
    merge_existing = existing_df[
        ~existing_df["o:id"].isin(missing_preserved)
    ]
    final_df = pd.merge(
        new_df,
        merge_existing[merge_columns],
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

    final_df = append_preserved_rows(final_df)

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
    "HubBaselineUnavailableError",
]
