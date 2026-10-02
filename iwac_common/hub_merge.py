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
from rich import box
from rich.console import Console
from rich.table import Table

from .hub import (
    HubBaselineUnavailableError,
    get_repo_configs,
    get_repo_revision,
    resolve_hf_token,
)
from .schema import dataset_to_pandas


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


def _normalized_text(value) -> str:
    """Comparable form of a source value: null → "", whitespace collapsed, so
    a re-wrapped OCR line is not mistaken for a changed text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).split())


def detect_source_changes(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    derived_from: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """``{source column: [o:id, …]}`` for rows present in both frames whose
    source value changed (after whitespace normalisation)."""
    changes: dict[str, list[str]] = {}
    common = new_df[["o:id"]].merge(existing_df[["o:id"]], on="o:id")["o:id"]
    if common.empty:
        return changes
    fresh = new_df.set_index("o:id")
    hub = existing_df.set_index("o:id")
    for source in derived_from:
        if source not in fresh.columns or source not in hub.columns:
            continue
        before = hub.loc[common, source].map(_normalized_text)
        after = fresh.loc[common, source].map(_normalized_text)
        changed = common[(before.to_numpy() != after.to_numpy())]
        if len(changed):
            changes[source] = [str(i) for i in changed]
    return changes


def stable_column_order(
    final_columns: Sequence[str],
    hub_columns: Sequence[str],
    new_columns: Sequence[str],
) -> list[str]:
    """Column order for a merged frame: the Hub's order, with new columns slotted in.

    A plain merge puts every mapper column first and every Hub-only column
    after them, so a column a post-processing script placed deliberately (the
    Hijri date beside ``pub_date``) migrated to the end on each upload and came
    back on the next re-run — needless churn in the published schema, and a card
    repair commit every time. Existing columns keep their Hub position; a column
    the Hub has never seen goes right after its predecessor in the mapper's
    order (or first, if it leads the mapper's output).
    """
    final_set = set(final_columns)
    ordered = [c for c in hub_columns if c in final_set]
    placed = set(ordered)
    mapper_order = [c for c in new_columns if c in final_set]
    for position, column in enumerate(mapper_order):
        if column in placed:
            continue
        anchor = next(
            (mapper_order[i] for i in range(position - 1, -1, -1)
             if mapper_order[i] in placed),
            None,
        )
        index = ordered.index(anchor) + 1 if anchor is not None else 0
        ordered.insert(index, column)
        placed.add(column)
    ordered.extend(c for c in final_columns if c not in placed)
    return ordered


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
    stale_rows: str = "drop",
    allow_initialize: bool = False,
    preserve_existing_ids: Iterable[object] = (),
    preserve_fields_by_id: Optional[Mapping[object, Iterable[str]]] = None,
    revision_out: Optional[MutableMapping[str, str]] = None,
    derived_from: Optional[Mapping[str, Sequence[str]]] = None,
    invalidate_derived: bool = True,
    stale_out: Optional[MutableMapping[str, dict]] = None,
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
      on Omeka): ``"drop"`` (default) or ``"keep"`` (complete historical
      rows marked private so they cannot be republished).

    Hub reads fail closed.  A missing config is treated as a first run only
    when ``allow_initialize=True`` and the Hub confirms that the config is not
    declared. ``revision_out`` receives the baseline repository SHA for an
    optimistic-concurrency check immediately before the later push.

    Stale derived values: with ``derived_from`` (``schema.DERIVED_FROM[config]``)
    rows whose source text changed since the Hub copy are reported, because
    their preserved computed columns (embeddings, lemmas, metrics, topics)
    describe the old text. ``stale_out`` receives
    ``{source: {"ids": [...], "derived": [...]}}``. With
    ``invalidate_derived=True`` those preserved values are nulled for the
    changed rows, so each stage's ``missing`` mode recomputes exactly them.
    """
    console = console or Console()
    token = resolve_hf_token(token)
    excluded = set(columns_to_exclude)
    if stale_rows not in ("keep", "drop"):
        raise ValueError(f"stale_rows must be 'keep' or 'drop', got {stale_rows!r}")

    if "o:id" not in new_df.columns:
        raise ValueError("New Omeka data is missing the required 'o:id' column")
    if new_df["o:id"].isna().any():
        raise ValueError("New Omeka data contains null 'o:id' values")
    if new_df["o:id"].astype(str).str.strip().eq("").any():
        raise ValueError("New Omeka data contains blank 'o:id' values")

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
                # A commit SHA is immutable and the datasets cache is keyed by
                # it, so a pinned read can safely reuse the local copy; only an
                # unpinned read (no revision_out) must bypass the cache.
                download_mode=(
                    "reuse_dataset_if_exists" if baseline_revision else "force_redownload"
                ),
                verification_mode="no_checks",
            )
            # Nullable ints/bools survive: the default to_pandas() turns an
            # int64 column with nulls into float64, which is how preserved
            # columns such as lda_topic_id reached the Hub as floats.
            existing_df = dataset_to_pandas(existing_ds)

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
                for index in new_df.index[mask]:
                    new_df.at[index, field] = existing_by_id.at[row_id, field]
        # Preserve data in the full mirror, but fail closed for publication:
        # a fetch failure could be caused by newly restricted upstream media.
        # Previous visibility is not evidence of current visibility.
        if "private_fields" in new_df:
            for index in new_df.index[mask]:
                new = new_df.at[index, "private_fields"]
                new_df.at[index, "private_fields"] = sorted(set(new) | fields)

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
        # A mapper error cannot establish that its source item is still
        # public. Retain the complete data privately until a successful map.
        retained = retained.copy()
        retained["item_is_public"] = False
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

    # Frozen annotations and other preserved columns keep previously known
    # restrictions even when their source Omeka properties have been retired.
    if "private_fields" in new_df and "private_fields" in existing_df:
        for index, row_id in zip(new_df.index, new_df["o:id"]):
            if row_id not in existing_by_id.index:
                continue
            old = existing_by_id.at[row_id, "private_fields"]
            if hasattr(old, "tolist"):
                old = old.tolist()
            if isinstance(old, (list, tuple, set)):
                new_df.at[index, "private_fields"] = sorted(
                    set(new_df.at[index, "private_fields"]) | (set(old) & set(extra_cols))
                )

    stale: dict[str, dict] = {}
    if derived_from:
        changes = detect_source_changes(new_df, existing_df, derived_from)
        for source, ids in changes.items():
            affected = [c for c in derived_from[source] if c in extra_cols]
            if affected:
                stale[source] = {"ids": ids, "derived": affected}
        # Neighbour rankings depend on every candidate vector, not only the
        # query row. Correcting/removing/adding one article can change another
        # article's neighbours; retain no apparently fresh cross-row links.
        embedding_changed = any(
            any(column.startswith("embedding_") for column in derived_from[source])
            for source in changes
        )
        membership_changed = set(new_df["o:id"]) != set(existing_df["o:id"])
        if "related_articles" in extra_cols and (embedding_changed or membership_changed):
            stale["corpus_embeddings"] = {
                "ids": sorted(set(new_df["o:id"]) | set(existing_df["o:id"])),
                "derived": ["related_articles"],
            }
    if stale:
        table = Table(
            title="Source text changed since the Hub copy", box=box.SIMPLE,
        )
        table.add_column("Source", style="cyan")
        table.add_column("Rows", justify="right")
        table.add_column("Preserved values now stale", style="yellow")
        table.add_column("e.g. o:id", style="dim")
        for source, info in stale.items():
            table.add_row(
                source, f"{len(info['ids']):,}", ", ".join(info["derived"]),
                ", ".join(info["ids"][:3]),
            )
        console.print(table)
        if invalidate_derived:
            console.print(
                "[yellow]⚠[/yellow] --invalidate-derived: those values are cleared "
                "for the changed rows; re-run each stage in 'missing' mode."
            )
        else:
            console.print(
                "[yellow]⚠[/yellow] Explicitly preserved. Re-run with --invalidate-derived to "
                "clear them (then each stage's 'missing' mode recomputes exactly "
                "those rows), or re-run the stages with --update-mode all."
            )
    if stale_out is not None:
        stale_out.clear()
        stale_out.update(stale)

    if not extra_cols and how != "outer":
        console.print("[yellow]ℹ[/yellow] No unique columns to preserve from existing dataset.")
        if excluded:
            console.print(f"[dim]Excluded columns: {', '.join(sorted(excluded))}[/dim]")
        merged = append_preserved_rows(new_df)
        return merged[stable_column_order(
            list(merged.columns), list(existing_df.columns), list(new_df.columns)
        )]

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
            # Preserve complete source records for deliberate historical use,
            # while marking absent records ineligible for public projection.
            # An outer merge of only Hub extras previously fabricated rows
            # with blank bibliographic metadata.
            for column in new_df.columns:
                if column != "o:id" and column in existing_df.columns:
                    for index in final_df.index[stale_mask]:
                        row_id = final_df.at[index, "o:id"]
                        final_df.at[index, column] = existing_by_id.at[row_id, column]
            if "item_is_public" not in final_df:
                final_df["item_is_public"] = False
            final_df.loc[stale_mask, "item_is_public"] = False
            console.print(
                f"[yellow]⚠[/yellow] {hub_only} row(s) exist on the Hub but not in Omeka "
                "(deleted items?). Complete historical rows are KEPT only in "
                "the private mirror; item_is_public=False excludes them from publication."
            )
        final_df = final_df.drop(columns="_merge")

    final_df = append_preserved_rows(final_df)
    final_df = final_df[stable_column_order(
        list(final_df.columns), list(existing_df.columns), list(new_df.columns)
    )]
    if invalidate_derived and stale:
        final_df = final_df.reset_index(drop=True)
        for info in stale.values():
            mask = final_df["o:id"].isin(info["ids"])
            for column in info["derived"]:
                final_df.loc[mask, column] = None

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
    "stable_column_order",
    "detect_source_changes",
    "resolve_hf_token",
    "ShrinkGuardError",
    "DuplicateIdError",
    "HubBaselineUnavailableError",
]
