"""Shared helper for merging fresh Omeka data with the existing HF Hub dataset.

Every upload script repeats the same load → identify-extra-columns → merge
on ``o:id`` flow. The variations the helper accepts are:

- ``how``: ``reference`` uses an outer merge (so ``stale_rows`` can decide
  the fate of Hub-only rows); the others use a left merge. The Hub side of the
  merge carries only ``o:id`` and Hub-only columns, which are disjoint from the
  fresh columns by construction, so no column ever needs a suffix.
- ``columns_to_exclude``: ``reference`` drops a few legacy/computed columns
  (``o:item_set``, ``o:media/file``, ``iiif_manifest``, ``thumbnail``).

Historical note: ``reference`` used to run ``.ffill(axis=1).bfill(axis=1)``
after the merge (via a ``fill_after_merge`` flag). That filled NaN cells from
*adjacent columns* — fabricating values for Hub-only rows and freshly added
items — and was removed as a data-corruption bug. Outer merges now log the
count of Hub-only rows instead, so genuinely deleted Omeka items are visible.

Subset-specific *post-merge* steps (sentiment-column reordering in
``articles``, mixed-type-column casting in ``reference``) intentionally stay in
their scripts; integer dtype coercion is applied by ``upload_runner`` from
``schema.SUBSETS[config].int_columns``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, MutableMapping, Optional, Sequence

import numpy as np
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
from .schema import DuplicateIdError, dataset_to_pandas, validate_ids, with_canonical_ids


class ShrinkGuardError(RuntimeError):
    """Raised when the fresh Omeka fetch is suspiciously smaller than the
    dataset already on the Hub (likely a truncated/partial API response).
    Pushing would silently delete rows; the caller must pass
    ``allow_shrink=True`` (CLI: ``--force-shrink``) to proceed."""


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
    source value changed (after whitespace normalisation).

    A source the Hub has never carried reads as empty there, so its first
    non-empty values count as changes. That is the case of a new sentiment
    panel member: its columns arrive with the upload, and the stored consensus
    of the previous panel stops describing the current one."""
    changes: dict[str, list[str]] = {}
    common = new_df[["o:id"]].merge(existing_df[["o:id"]], on="o:id")["o:id"]
    if common.empty:
        return changes
    fresh = new_df.set_index("o:id")
    hub = existing_df.set_index("o:id")
    for source in derived_from:
        if source not in fresh.columns:
            continue
        if source not in hub.columns:
            after = fresh.loc[common, source].map(_normalized_text)
            changed = common[(after != "").to_numpy()]
            if len(changed):
                changes[source] = [str(i) for i in changed]
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

    - both frames must have valid, unique ``o:id`` (``schema.validate_ids``:
      :class:`DuplicateIdError` for duplicates, ``DataContractError`` for
      missing/null/blank ids);
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

    new_df = with_canonical_ids(new_df, label="new Omeka data")

    baseline_revision = None
    if revision_out is not None:
        baseline_revision = get_repo_revision(repo, token=token)
        revision_out["revision"] = baseline_revision

    existing_df = _load_baseline(
        repo, config_name, token=token, revision=baseline_revision,
        allow_initialize=allow_initialize, console=console,
    )
    if existing_df.empty:
        # Nothing to restore from: a failed item here could only ship with
        # blank, unrestricted fields, so the contract is to abort.
        _refuse_unrecoverable_failures(preserve_fields_by_id or {}, set())
        console.print("[yellow]ℹ[/yellow] No existing data on Hub; using new Omeka data directly.")
        return new_df
    validate_ids(existing_df, label=f"existing Hub dataset '{config_name}'")
    existing_by_id = existing_df.set_index("o:id", drop=False)

    # A mapper/media failure is not a deletion.  When the caller explicitly
    # allows such failures, retain the complete existing row for those ids and
    # preserve selected same-name fields that a degraded mapper returned blank.
    _restore_failed_fields(new_df, existing_by_id, preserve_fields_by_id or {})
    missing_preserved = _preserved_mapper_failures(
        new_df, existing_by_id, preserve_existing_ids, console
    )
    _check_shrink(
        new_count=len(new_df), preserved_count=len(missing_preserved),
        existing_count=len(existing_df), config_name=config_name,
        min_row_ratio=min_row_ratio, allow_shrink=allow_shrink, console=console,
    )
    _report_new_columns(new_df, existing_df, console)
    console.print(
        f"[blue]→[/blue] Merging new Omeka data ({len(new_df)} records) "
        f"with existing Hub data ({len(existing_df)} records)."
    )

    extra_cols = [
        col
        for col in existing_df.columns
        if col not in new_df.columns and col not in excluded
    ]
    _carry_private_fields(new_df, existing_by_id, extra_cols)

    stale = _stale_derived(new_df, existing_df, derived_from, extra_cols)
    _report_stale(stale, invalidate_derived, console)
    if stale_out is not None:
        stale_out.clear()
        stale_out.update(stale)

    def append_preserved_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if not missing_preserved:
            return frame
        # A mapper error cannot establish that its source item is still
        # public. Retain the complete data privately until a successful map.
        retained = _private_baseline_rows(existing_by_id, missing_preserved, frame.columns)
        combined = pd.concat([frame, retained], ignore_index=True)
        validate_ids(combined, label="merged data plus preserved mapper failures")
        return combined

    if not extra_cols and how != "outer":
        # No Hub-only columns can go stale, so ``stale`` is empty here.
        console.print("[yellow]ℹ[/yellow] No unique columns to preserve from existing dataset.")
        if excluded:
            console.print(f"[dim]Excluded columns: {', '.join(sorted(excluded))}[/dim]")
        merged = append_preserved_rows(new_df)
        return merged[stable_column_order(
            list(merged.columns), list(existing_df.columns), list(new_df.columns)
        )]

    if extra_cols:
        console.print(f"[green]✓[/green] Preserving columns: {', '.join(extra_cols)}")
    final_df = _merge_hub_columns(
        new_df, existing_df, existing_by_id, extra_cols, missing_preserved,
        how=how, stale_rows=stale_rows, console=console,
    )
    final_df = append_preserved_rows(final_df)
    final_df = final_df[stable_column_order(
        list(final_df.columns), list(existing_df.columns), list(new_df.columns)
    )]
    if invalidate_derived and stale:
        final_df = _invalidate_stale(final_df, stale)
    _report_merge(final_df, extra_cols, excluded, console)
    return final_df


# ---------------------------------------------------------------------------
# Steps of merge_with_hub_dataset
# ---------------------------------------------------------------------------


def _load_baseline(
    repo: str,
    config_name: str,
    *,
    token: Optional[str],
    revision: Optional[str],
    allow_initialize: bool,
    console: Console,
) -> pd.DataFrame:
    """The existing Hub rows (``o:id`` as ``str``), failing closed.

    A missing config counts as a first run only with ``allow_initialize`` and
    when the Hub confirms that the config is not declared.
    """
    try:
        with console.status("[bold green]Loading existing dataset from Hub...", spinner="dots"):
            existing_ds = load_dataset(
                repo,
                name=config_name,
                split="train",
                token=token,
                revision=revision,
                # A commit SHA is immutable and the datasets cache is keyed by
                # it, so a pinned read can safely reuse the local copy; only an
                # unpinned read (no revision_out) must bypass the cache.
                download_mode=(
                    "reuse_dataset_if_exists" if revision else "force_redownload"
                ),
                verification_mode="no_checks",
            )
            # Nullable ints/bools survive: the default to_pandas() turns an
            # int64 column with nulls into float64, which is how preserved
            # columns such as lda_topic_id reached the Hub as floats.
            existing_df = dataset_to_pandas(existing_ds)
    except Exception as exc:  # noqa: BLE001
        if not allow_initialize:
            raise HubBaselineUnavailableError(
                f"Could not load existing Hub config '{config_name}' from {repo}: "
                f"{exc}. Refusing to assume this is a first run; pass "
                "--initialize only for a deliberately new config."
            ) from exc
        if config_name in get_repo_configs(repo, token=token):
            raise HubBaselineUnavailableError(
                f"Could not load existing Hub config '{config_name}' from {repo}: "
                f"{exc}. The config exists, so refusing to overwrite it."
            ) from exc
        console.print(
            f"[yellow]ℹ[/yellow] '{config_name}' is not declared in {repo}; "
            "initialization explicitly allowed."
        )
        return pd.DataFrame()

    if existing_df.empty:
        console.print(
            "[yellow]ℹ[/yellow] Existing Hub config is empty; using new Omeka data."
        )
        return existing_df
    if "o:id" not in existing_df.columns or existing_df["o:id"].isnull().any():
        raise HubBaselineUnavailableError(
            f"Existing Hub dataset '{config_name}' has a missing or null "
            "'o:id' column; refusing to treat a corrupt baseline as empty."
        )
    existing_df["o:id"] = existing_df["o:id"].astype(str)
    console.print(f"[green]✓[/green] Loaded {len(existing_df)} records from {repo}")
    return existing_df


def _refuse_unrecoverable_failures(
    preserve_fields_by_id: Mapping[object, Iterable[str]], baseline_ids: set[str],
) -> None:
    """Abort when a media failure hit an item the baseline does not hold.

    ``--allow-media-failures`` keeps the previous value of a failed field; a
    new item has none, so it would ship blank or with a fallback value never
    marked private (an audiovisual thumbnail, for one). The ingestion contract
    says such an item aborts the upload.
    """
    new_items = sorted(str(i) for i in preserve_fields_by_id if str(i) not in baseline_ids)
    if new_items:
        raise HubBaselineUnavailableError(
            f"Media lookups failed for {len(new_items)} item(s) absent from the Hub baseline "
            f"({', '.join(new_items[:10])}); a new item cannot be reconstructed, even with "
            "--allow-media-failures. Re-run once the media resolve."
        )


def _restore_failed_fields(
    new_df: pd.DataFrame,
    existing_by_id: pd.DataFrame,
    preserve_fields_by_id: Mapping[object, Iterable[str]],
) -> None:
    """Put the Hub value back into fields whose fresh fetch failed (in place),
    and mark those fields private. A failed item absent from the baseline
    aborts (:func:`_refuse_unrecoverable_failures`)."""
    _refuse_unrecoverable_failures(preserve_fields_by_id, set(existing_by_id.index.astype(str)))
    for raw_id, raw_fields in preserve_fields_by_id.items():
        row_id, fields = str(raw_id), set(raw_fields)
        mask = new_df["o:id"] == row_id
        for field in fields:
            if field in new_df.columns and field in existing_by_id.columns:
                for index in new_df.index[mask]:
                    new_df.at[index, field] = existing_by_id.at[row_id, field]
        # Preserve data in the full mirror, but fail closed for publication:
        # a fetch failure could be caused by newly restricted upstream media.
        # Previous visibility is not evidence of current visibility.
        if "private_fields" in new_df:
            for index in new_df.index[mask]:
                new = new_df.at[index, "private_fields"]
                new_df.at[index, "private_fields"] = sorted(set(new) | fields)


def _preserved_mapper_failures(
    new_df: pd.DataFrame,
    existing_by_id: pd.DataFrame,
    preserve_existing_ids: Iterable[object],
    console: Console,
) -> list[str]:
    """Ids of failed-mapper rows to retain whole from the baseline (sorted)."""
    preserve_ids = {str(value) for value in preserve_existing_ids}
    missing = sorted(preserve_ids - set(new_df["o:id"]))
    if not missing:
        return missing
    unavailable = [row_id for row_id in missing if row_id not in existing_by_id.index]
    if unavailable:
        raise HubBaselineUnavailableError(
            "Cannot preserve failed mapper rows absent from the Hub baseline: "
            + ", ".join(unavailable[:5])
        )
    console.print(
        f"[yellow]⚠[/yellow] Preserved {len(missing)} complete Hub "
        "row(s) whose fresh mapper failed."
    )
    return missing


def _check_shrink(
    *,
    new_count: int,
    preserved_count: int,
    existing_count: int,
    config_name: str,
    min_row_ratio: float,
    allow_shrink: bool,
    console: Console,
) -> None:
    """Shrink tripwire: a truncated Omeka fetch (partial API response) flowing
    into a left merge would silently delete the missing rows from the Hub."""
    effective_new_count = new_count + preserved_count
    if effective_new_count >= min_row_ratio * existing_count:
        return
    msg = (
        f"Fresh Omeka data has {effective_new_count:,} usable rows "
        f"({new_count:,} mapped + {preserved_count:,} explicitly "
        f"preserved) but the Hub config "
        f"'{config_name}' has {existing_count:,} "
        f"(< {min_row_ratio:.0%} threshold). This usually means a truncated "
        f"fetch; pushing would delete rows. Re-run with --force-shrink only "
        f"if the shrink is intentional (items really deleted on Omeka)."
    )
    if not allow_shrink:
        raise ShrinkGuardError(msg)
    console.print(f"[yellow]⚠ Shrink allowed by caller:[/yellow] {msg}")


def _report_new_columns(new_df: pd.DataFrame, existing_df: pd.DataFrame, console: Console) -> None:
    """Brand-new columns are normal when a mapper gains a field, but they also
    catch accidental renames (old column preserved as a Hub-only column + new
    column added → near-duplicate columns)."""
    brand_new_cols = [c for c in new_df.columns if c not in existing_df.columns]
    if brand_new_cols:
        console.print(
            f"[yellow]ℹ[/yellow] New column(s) not on the Hub yet: "
            f"{', '.join(brand_new_cols)} (renamed mapper fields would show up here)"
        )


def _carry_private_fields(
    new_df: pd.DataFrame, existing_by_id: pd.DataFrame, extra_cols: Sequence[str]
) -> None:
    """Frozen annotations and other preserved columns keep previously known
    restrictions even when their source Omeka properties have been retired."""
    if "private_fields" not in new_df or "private_fields" not in existing_by_id:
        return
    preserved = set(extra_cols)
    for index, row_id in zip(new_df.index, new_df["o:id"]):
        if row_id not in existing_by_id.index:
            continue
        old = existing_by_id.at[row_id, "private_fields"]
        if hasattr(old, "tolist"):
            old = old.tolist()
        if isinstance(old, (list, tuple, set)):
            new_df.at[index, "private_fields"] = sorted(
                set(new_df.at[index, "private_fields"]) | (set(old) & preserved)
            )


def _visibility_changed(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> bool:
    """Whether any shared item's ``item_is_public`` differs; unknown counts as changed."""
    if "item_is_public" not in new_df or "item_is_public" not in existing_df:
        return True

    def flags(frame: pd.DataFrame) -> pd.Series:
        return frame.set_index("o:id")["item_is_public"].map(
            lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
        )

    fresh, baseline = flags(new_df), flags(existing_df)
    shared = fresh.index.intersection(baseline.index)
    return bool((fresh.loc[shared] != baseline.loc[shared]).any())


def _stale_derived(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    derived_from: Optional[Mapping[str, Sequence[str]]],
    extra_cols: Sequence[str],
) -> dict[str, dict]:
    """``{source: {"ids": [...], "derived": [...]}}`` for preserved (Hub-only)
    columns that describe an older version of their source."""
    stale: dict[str, dict] = {}
    if not derived_from:
        return stale
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
    # The neighbour pool is the public items, so a visibility flip changes it.
    visibility_changed = _visibility_changed(new_df, existing_df)
    if "related_articles" in extra_cols and (
        embedding_changed or membership_changed or visibility_changed
    ):
        stale["corpus_embeddings"] = {
            "ids": sorted(set(new_df["o:id"]) | set(existing_df["o:id"])),
            "derived": ["related_articles"],
        }
    return stale


def _report_stale(stale: Mapping[str, dict], invalidate_derived: bool, console: Console) -> None:
    if not stale:
        return
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


def _private_baseline_rows(
    existing_by_id: pd.DataFrame, ids: Iterable[str], columns: Sequence[str]
) -> pd.DataFrame:
    """Complete baseline rows for ``ids``, shaped to ``columns`` and marked
    ``item_is_public=False``.

    Built frame-wise on purpose: assigning a list cell through ``.at`` into a
    null cell turns a one-element array into a 0-d array, which Arrow rejects
    ("Only 1D arrays accepted").
    """
    rows = existing_by_id.loc[list(ids)].reindex(columns=columns).copy()
    rows["item_is_public"] = False
    return rows


def _merge_hub_columns(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    existing_by_id: pd.DataFrame,
    extra_cols: Sequence[str],
    missing_preserved: Sequence[str],
    *,
    how: str,
    stale_rows: str,
    console: Console,
) -> pd.DataFrame:
    """Join the Hub-only columns onto the fresh rows; for an outer merge,
    drop or privately keep the Hub-only rows according to ``stale_rows``."""
    merge_existing = existing_df[~existing_df["o:id"].isin(missing_preserved)]
    final_df = pd.merge(
        new_df,
        merge_existing[["o:id", *extra_cols]],
        on="o:id",
        how=how,
        indicator=how == "outer",
    )
    if how != "outer":
        return final_df
    stale_mask = final_df.pop("_merge") == "right_only"
    hub_only = int(stale_mask.sum())
    if hub_only and stale_rows == "drop":
        console.print(
            f"[yellow]⚠[/yellow] Dropped {hub_only} Hub-only row(s) "
            "(items no longer in Omeka; --stale-rows drop)."
        )
        return final_df[~stale_mask]
    if hub_only:
        # Preserve complete source records for deliberate historical use,
        # while marking absent records ineligible for public projection.
        # An outer merge of only Hub extras previously fabricated rows
        # with blank bibliographic metadata.
        if "item_is_public" not in final_df:
            final_df["item_is_public"] = False
        kept = _private_baseline_rows(
            existing_by_id, final_df.loc[stale_mask, "o:id"], final_df.columns
        )
        kept.index = final_df.index[stale_mask]
        final_df = pd.concat([final_df[~stale_mask], kept]).sort_index()
        console.print(
            f"[yellow]⚠[/yellow] {hub_only} row(s) exist on the Hub but not in Omeka "
            "(deleted items?). Complete historical rows are KEPT only in "
            "the private mirror; item_is_public=False excludes them from publication."
        )
    return final_df


def _invalidate_stale(final_df: pd.DataFrame, stale: Mapping[str, dict]) -> pd.DataFrame:
    final_df = final_df.reset_index(drop=True)
    for info in stale.values():
        mask = final_df["o:id"].isin(info["ids"])
        for column in info["derived"]:
            final_df.loc[mask, column] = None
    return final_df


def _report_merge(
    final_df: pd.DataFrame, extra_cols: Sequence[str], excluded: set[str], console: Console
) -> None:
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


__all__ = [
    "merge_with_hub_dataset",
    "stable_column_order",
    "detect_source_changes",
    "resolve_hf_token",
    "ShrinkGuardError",
    "DuplicateIdError",
    "HubBaselineUnavailableError",
]
