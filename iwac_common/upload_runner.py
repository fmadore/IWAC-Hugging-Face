"""Shared orchestration for the seven subset upload scripts.

Every upload script used to repeat the same ~150-line ``build_and_push``
body: config panel → fetch → Rich mapping loop → DataFrame validation →
Hub merge → dtype casts → summary → push. Only the Omeka→column mapper and
a handful of knobs genuinely differ per subset, so each script now declares
an :class:`UploadSpec` and calls :func:`run_upload`.

What the runner standardizes (beyond deduplication):

- Rich console output everywhere (audiovisual/index/images previously used
  tqdm + plain logging, against the repo convention);
- the safety rails from ``hub_merge`` / ``omeka_client``: fetch-count
  reconciliation, o:id uniqueness, the shrink tripwire (override with
  ``--force-shrink``), stale-row policy for outer merges;
- a uniform CLI: ``--repo``, ``--max-shard-size``, ``--no-cache``,
  ``--dry-run``, ``--force-shrink`` (+ ``--stale-rows`` for outer-merge
  subsets);
- non-zero exit codes on failure so scripted runs can detect them.

Subset-specific quirks stay in the subset scripts, expressed as hooks:
``map_item`` (the mapper), ``post_map`` (e.g. index frequency stats, needs
the API), ``post_merge`` (column reordering, dtype fixes).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence

import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .hub import (
    ConcurrentHubWriteError,
    HubBaselineUnavailableError,
    HubWriteError,
    HubWriteLockedError,
    hub_write_lock,
    push_dataset_verified,
    resolve_hf_token,
)
from .hub_merge import ShrinkGuardError, merge_with_hub_dataset
from .omeka_client import (
    Config,
    MediaFetchGuardError,
    OmekaApiClient,
    TruncatedFetchError,
    conn_manager,
    media_stats,
)
from .repos import get_private_repo_id
from .schema import (
    DERIVED_FROM,
    SUBSETS,
    DataContractError,
    DuplicateIdError,
    SubsetDefinition,
    normalize_embedding_nulls,
    validate_frame,
    with_canonical_ids,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class MappingGuardError(RuntimeError):
    """One or more Omeka items could not be mapped safely."""


@dataclass
class UploadSpec:
    """Everything that differs between the seven subset upload scripts.

    The Omeka resource classes and the integer columns are not restated here:
    they come from the subset registry, ``schema.SUBSETS[config_name]``.
    """

    config_name: str  # HF dataset config, e.g. "articles"
    map_item: Callable[[Dict[str, Any], OmekaApiClient], Awaitable[Optional[Dict[str, Any]]]]
    title: str  # Rich panel title, e.g. "📰 IWAC Newspaper Upload"
    cache_dir: str  # per-subset Omeka response cache
    description: str = ""  # argparse description
    merge_how: str = "left"
    columns_to_exclude: Sequence[str] = ()  # legacy Hub columns to drop
    # Async hook after mapping, before merge. Receives (df, api, repo, token)
    # so it can load sibling Hub subsets; used by index for its cross-subset
    # frequency stats. Returns the (possibly enriched) DataFrame.
    post_map: Optional[
        Callable[[pd.DataFrame, OmekaApiClient, str, Optional[str]], Awaitable[pd.DataFrame]]
    ] = None
    # Sync hook after merge (column reordering, dtype fix-ups).
    post_merge: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    # Expose --stale-rows (only meaningful for outer merges).
    supports_stale_rows: bool = False

    def __post_init__(self) -> None:
        if self.config_name not in SUBSETS:
            raise ValueError(f"Unknown IWAC subset: {self.config_name!r}")

    @property
    def subset(self) -> SubsetDefinition:
        return SUBSETS[self.config_name]

    @property
    def resource_class_ids(self) -> tuple[int, ...]:
        """Omeka classes fetched for this subset (>1 for references/index)."""
        return self.subset.resource_class_ids

    @property
    def int_columns(self) -> tuple[str, ...]:
        """Columns cast to nullable ``Int64`` after the merge."""
        return self.subset.int_columns


def cast_int_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Cast ``columns`` (where present) to nullable ``Int64``, in place.

    ``None``/``NaN`` and blank strings become nulls. A fractional value raises
    rather than being truncated: the registry declares the column integer.
    """
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column]
        if not pd.api.types.is_numeric_dtype(values):  # object or pandas str dtype
            values = values.astype(object).map(
                lambda v: None if isinstance(v, str) and not v.strip() else v
            )
        df[column] = values.astype("Int64")
    return df


def _state_root() -> Path:
    configured = os.getenv("IWAC_STATE_DIR")
    if configured:
        return Path(configured)
    from .paths import workspace_root

    return workspace_root() / ".iwac_state"


def stale_worklist_path(repo: str, config_name: str) -> Path:
    return _state_root() / "stale_derived" / f"{repo.replace('/', '__')}__{config_name}.json"


def load_stale_derived(repo: str, config_name: str) -> dict[str, list[str]]:
    """Read a durable worklist, refusing corrupt or misdirected recovery state."""
    path = stale_worklist_path(repo, config_name)
    if not path.exists():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("repository") != repo or record.get("config") != config_name:
        raise ValueError(f"Stale-derived worklist identity mismatch: {path}")
    columns = record.get("columns")
    if not isinstance(columns, dict) or any(
        not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
        for ids in columns.values()
    ):
        raise ValueError(f"Invalid stale-derived worklist: {path}")
    return columns


def replay_stale_derived(df: pd.DataFrame, columns: Dict[str, list[str]]) -> pd.DataFrame:
    """Clear previously recorded stale cells even after source changes landed."""
    for column, ids in columns.items():
        if column in df.columns:
            df.loc[df["o:id"].isin(ids), column] = None
    return df


def record_stale_derived(
    repo: str, config_name: str, stale: Dict[str, dict], revision: Optional[str]
) -> Optional[Path]:
    """Append the rows whose preserved derived values went stale in this push
    to a local worklist, ``.iwac_state/stale_derived/<repo>__<config>.json``
    (or ``IWAC_STATE_DIR``).

    The comparison that detects a changed source text only works in the run
    that lands the change: once pushed, the Hub text equals Omeka again. The
    worklist keeps that knowledge, per derived column, until someone acts on
    it (re-run the stage with --update-mode all, or clear the file after an
    upload with --invalidate-derived).
    """
    if not stale:
        return None
    root = _state_root() / "stale_derived"
    root.mkdir(parents=True, exist_ok=True)
    path = stale_worklist_path(repo, config_name)
    columns = load_stale_derived(repo, config_name)
    record = {"repository": repo, "config": config_name, "columns": columns}
    for info in stale.values():
        for column in info["derived"]:
            ids = set(columns.get(column, [])) | set(info["ids"])
            columns[column] = sorted(ids, key=lambda v: (len(v), v))
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    record["last_revision_before_push"] = revision
    # Stage before the remote write: a crash after a successful commit must
    # not discard the only evidence that preserved enrichments went stale.
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


#: ``report_unmapped_values`` hint for the outlet → country mapping.
COUNTRY_MAPPER_HINT = (
    "Add the missing values to country_mapper.py if they belong to a collection country."
)


def report_unmapped_values(
    source_column: str, derived_column: str, *, limit: int = 15, hint: str = ""
) -> Callable[[pd.DataFrame, OmekaApiClient, str, Optional[str]], Awaitable[pd.DataFrame]]:
    """A ``post_map`` hook listing ``source_column`` values whose
    ``derived_column`` came out blank — e.g. outlets missing from
    ``country_mapper``, which otherwise leave ``country`` empty in silence.
    ``hint`` names the fix for this particular mapping. Output goes to the
    runner's console (the API client's). Report-only: the frame is returned
    unchanged.
    """

    async def hook(df: pd.DataFrame, api, repo, token) -> pd.DataFrame:
        if source_column not in df.columns or derived_column not in df.columns:
            return df
        source = df[source_column].fillna("").astype(str).str.strip()
        derived = df[derived_column].fillna("").astype(str).str.strip()
        unmapped = source[(derived == "") & (source != "")].value_counts()
        if unmapped.empty:
            return df
        console = getattr(api, "console", None) or Console()
        table = Table(
            title=f"{int(unmapped.sum()):,} rows with a {source_column} but no {derived_column}",
            box=box.SIMPLE,
        )
        table.add_column(source_column, style="yellow")
        table.add_column("Rows", justify="right")
        for value, count in unmapped.head(limit).items():
            table.add_row(value, f"{count:,}")
        if len(unmapped) > limit:
            table.add_row(f"… {len(unmapped) - limit} more", "")
        console.print(table)
        advice = hint or f"Check the mapping that derives {derived_column} from {source_column}."
        console.print(
            f"[yellow]⚠[/yellow] {derived_column} stays blank for these "
            f"{source_column} values. {advice}"
        )
        return df

    return hook


def _setup_console_logging() -> tuple[Console, logging.Logger]:
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    return console, logging.getLogger("upload")


def _display_config_panel(
    console: Console, spec: UploadSpec, cfg: Config, args: argparse.Namespace
) -> None:
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("API URL", str(cfg.API_URL))
    table.add_row("Repository", args.repo)
    table.add_row("Config", spec.config_name)
    table.add_row("Resource class(es)", ", ".join(map(str, spec.resource_class_ids)))
    table.add_row("Max Shard Size", args.max_shard_size)
    table.add_row("Cache", "disabled" if args.no_cache else f"{cfg.CACHE_DIR} ({cfg.CACHE_HOURS}h)")
    if args.dry_run:
        table.add_row("Mode", "[yellow]dry-run (no push)[/yellow]")
    console.print(Panel(table, title=f"[bold blue]{spec.title}", border_style="blue"))


def _resolve_cache_dir(spec: UploadSpec) -> Path:
    from .paths import workspace_root

    cache_dir = Path(spec.cache_dir).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = workspace_root() / cache_dir
    return cache_dir


def _require_hf_token(console: Console) -> str:
    """Resolve the HF token (logging in interactively if needed) before any
    long Omeka work, so a missing token fails in seconds, dry runs included:
    even a dry run reads the authenticated Hub baseline."""
    token = resolve_hf_token()
    if token is None:
        console.print("[yellow]⚠[/yellow] No HF token found; starting login before fetching.")
        try:
            login()
        except Exception as exc:  # noqa: BLE001
            raise HubBaselineUnavailableError(f"Hugging Face login failed: {exc}") from exc
        token = resolve_hf_token()
    if token is None:
        raise HubBaselineUnavailableError(
            "No Hugging Face token is available after login; refusing to read or write."
        )
    return token


async def _fetch_items(spec: UploadSpec, api: OmekaApiClient, console: Console) -> list:
    """A failed or truncated class fetch raises and aborts the whole run:
    continuing with a partial item list guarantees rows disappear from the Hub
    (or trips the shrink guard later with wasted work)."""
    console.print("\n[bold cyan]Step 1:[/bold cyan] Fetching items from Omeka API...")
    items: list = []
    for rcid in spec.resource_class_ids:
        items.extend(await api.fetch_items(rcid))
    if items:
        console.print(f"[green]✓[/green] Fetched {len(items)} items from Omeka.")
    return items


async def _map_items(
    spec: UploadSpec,
    items: list,
    api: OmekaApiClient,
    args: argparse.Namespace,
    console: Console,
    logger: logging.Logger,
) -> tuple[list[Dict[str, Any]], list[tuple[str, str]]]:
    """Map every item concurrently; returns ``(records, [(o:id, error), …])``.

    Any mapper failure raises :class:`MappingGuardError` unless
    ``--allow-map-failures`` (the merge then retains those rows whole).
    """
    records_by_position: list[Optional[Dict[str, Any]]] = [None] * len(items)
    mapping_errors: list[tuple[str, str]] = []
    semaphore = asyncio.Semaphore(args.map_concurrency)

    async def map_one(position: int, item: Dict[str, Any]):
        async with semaphore:
            try:
                record = await spec.map_item(item, api)
                if record is None:
                    raise ValueError("mapper returned None")
                return position, record, None
            except Exception as exc:  # noqa: BLE001
                return position, None, exc

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Mapping {spec.config_name}", total=len(items))
        tasks = [
            asyncio.create_task(map_one(position, item))
            for position, item in enumerate(items)
        ]
        for completed in asyncio.as_completed(tasks):
            position, record, exc = await completed
            if exc is None:
                records_by_position[position] = record
            else:
                item_id = str(items[position].get("o:id", "Unknown ID"))
                mapping_errors.append((item_id, str(exc)))
                logger.error(
                    "Error mapping item %s: %s", item_id, exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            progress.update(task, advance=1)
    if mapping_errors and not args.allow_map_failures:
        sample = "; ".join(f"{item_id}: {message}" for item_id, message in mapping_errors[:5])
        raise MappingGuardError(
            f"{len(mapping_errors)} item(s) failed to map ({sample}). "
            "The run fails closed so those rows are not deleted. Re-run after "
            "fixing the cause, or use --allow-map-failures to preserve their "
            "existing Hub rows explicitly."
        )
    if mapping_errors:
        console.print(
            f"[yellow]⚠[/yellow] {len(mapping_errors)} mapper failure(s) explicitly "
            "allowed; their complete existing Hub rows will be retained."
        )
    return [r for r in records_by_position if r is not None], mapping_errors


@dataclass
class _MergeResult:
    frame: pd.DataFrame
    revision: Optional[str]  # baseline SHA for the push's parent check
    stale: Dict[str, dict]  # stale-derived report of this run
    pending: Dict[str, list[str]]  # durable worklist from earlier runs


def _merge_with_hub(
    spec: UploadSpec,
    new_df: pd.DataFrame,
    args: argparse.Namespace,
    token: str,
    mapping_errors: Sequence[tuple[str, str]],
    console: Console,
) -> _MergeResult:
    """Merge with the Hub (preserves computed columns; safety rails raise
    instead of silently shrinking or fanning out), replay the durable stale
    worklist, then apply the subset hook and the registry's integer types."""
    console.print("\n[bold cyan]Step 2:[/bold cyan] Merging with existing Hub dataset...")
    revision: dict[str, str] = {}
    stale: dict[str, dict] = {}
    final_df = merge_with_hub_dataset(
        new_df,
        args.repo,
        config_name=spec.config_name,
        token=token,
        how=spec.merge_how,
        columns_to_exclude=spec.columns_to_exclude,
        console=console,
        allow_shrink=args.force_shrink,
        stale_rows=getattr(args, "stale_rows", "drop"),
        allow_initialize=args.initialize,
        preserve_existing_ids=(
            [item_id for item_id, _ in mapping_errors] if args.allow_map_failures else ()
        ),
        preserve_fields_by_id=(
            media_stats.failed_fields_by_item if args.allow_media_failures else None
        ),
        revision_out=revision,
        derived_from=DERIVED_FROM.get(spec.config_name),
        invalidate_derived=args.invalidate_derived,
        stale_out=stale,
    )
    pending = load_stale_derived(args.repo, spec.config_name)
    if args.invalidate_derived and pending:
        final_df = replay_stale_derived(final_df, pending)
        console.print("[yellow]⚠[/yellow] Replayed pending stale-derived worklist.")
    if spec.post_merge is not None:
        final_df = spec.post_merge(final_df)
    final_df = cast_int_columns(final_df, spec.int_columns)
    return _MergeResult(final_df, revision.get("revision"), stale, pending)


def _print_summary(console: Console, final_df: pd.DataFrame) -> None:
    summary = Table(title="Dataset Summary", box=box.ROUNDED)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Total Records", str(len(final_df)))
    summary.add_row("Total Columns", str(len(final_df.columns)))
    summary.add_row(
        "Columns",
        ", ".join(final_df.columns[:5]) + ("..." if len(final_df.columns) > 5 else ""),
    )
    console.print(summary)


def _push(
    spec: UploadSpec, merged: _MergeResult, args: argparse.Namespace, token: str,
    console: Console,
) -> None:
    """Stage the stale worklist, write through the verified gateway, then
    acknowledge the worklist. Gateway errors propagate to the run's handler."""
    final_df = merged.frame
    ds = Dataset.from_pandas(final_df, preserve_index=False)
    # Stage before the remote write: a crash after a successful commit must
    # not discard the only record of which enrichments went stale.
    worklist = record_stale_derived(args.repo, spec.config_name, merged.stale, merged.revision)
    with console.status("[bold green]Pushing dataset to Hugging Face Hub...", spinner="dots"):
        push_dataset_verified(
            ds,
            repo_id=args.repo,
            config_name=spec.config_name,
            token=token,
            commit_message=f"Refresh '{spec.config_name}' from the IWAC Omeka source",
            max_shard_size=args.max_shard_size,
            console=console,
            expected_revision=merged.revision,
            expected_columns=list(final_df.columns),
            expected_ids=final_df["o:id"].tolist(),
        )
    if args.invalidate_derived:
        stale_worklist_path(args.repo, spec.config_name).unlink(missing_ok=True)
    elif worklist or merged.pending:
        console.print(
            f"[yellow]⚠[/yellow] Rows with stale derived values recorded in "
            f"[cyan]{stale_worklist_path(args.repo, spec.config_name)}[/cyan]."
        )
    console.print(Panel(
        f"[bold green]✓ Dataset successfully published![/bold green]\n\n"
        f"Repository: [cyan]{args.repo}[/cyan]\n"
        f"Config: [cyan]{spec.config_name}[/cyan]\n"
        f"Records: [cyan]{len(final_df)}[/cyan]",
        title="🎉 Upload Complete",
        border_style="green",
    ))


_PROTECTED = "Aborted — Hub data protected"

#: Typed failures, most specific first: (exception types, headline, panel title).
#: ``None`` as headline shows the exception's class name.
_FAILURE_PANELS: tuple[tuple[tuple[type[BaseException], ...], Optional[str], str], ...] = (
    ((TruncatedFetchError,), "Truncated fetch", _PROTECTED),
    ((MediaFetchGuardError,), "Mass media-lookup failure", _PROTECTED),
    ((MappingGuardError,), "Mapper failure", "Aborted — Hub rows protected"),
    ((HubBaselineUnavailableError, ConcurrentHubWriteError, HubWriteError,
      HubWriteLockedError), "Hub safety guard", _PROTECTED),
    ((ShrinkGuardError, DuplicateIdError), None, _PROTECTED),
    ((DataContractError,), "Data contract", _PROTECTED),
)


def _report_failure(console: Console, logger: logging.Logger, exc: Exception) -> int:
    """Show one failure panel and return the exit code (always 1)."""
    for types, headline, title in _FAILURE_PANELS:
        if isinstance(exc, types):
            break
    else:
        headline, title = "Unexpected pipeline failure", "Aborted — see the log"
        logger.error("Unexpected upload failure:", exc_info=(type(exc), exc, exc.__traceback__))
    console.print(Panel(
        f"[bold red]✗ {headline or type(exc).__name__}[/bold red]\n\n{exc}",
        title=title,
        border_style="red",
    ))
    return 1


async def _run(spec: UploadSpec, args: argparse.Namespace, console: Console, logger: logging.Logger) -> int:
    cfg = Config(CACHE_DIR=str(_resolve_cache_dir(spec)))
    _display_config_panel(console, spec, cfg, args)
    api = OmekaApiClient(cfg, use_cache=not args.no_cache, console=console)
    media_stats.reset()

    try:
        cfg.require_full_access_credentials()
        token = _require_hf_token(console)

        items = await _fetch_items(spec, api, console)
        if not items:
            console.print(
                "[bold yellow]⚠ Warning:[/bold yellow] No items returned from "
                "Omeka API. Existing Hub dataset left untouched."
            )
            return 0
        records, mapping_errors = await _map_items(spec, items, api, args, console, logger)
        if not records:
            console.print("[bold red]✗ Error:[/bold red] No records were successfully mapped.")
            return 1

        # Mass media-lookup failure blanks PDF/thumbnail for every item while
        # leaving the row count intact, so no other rail catches it. Check
        # before the merge — there is nothing to salvage downstream.
        media_summary = media_stats.check(allow_failures=args.allow_media_failures)
        if media_summary:
            console.print(
                f"[yellow]⚠[/yellow] {media_summary} — existing Hub values for "
                "the affected media fields will be preserved."
            )

        new_df = with_canonical_ids(pd.DataFrame(records), label="mapped Omeka data")
        if spec.post_map is not None:
            new_df = await spec.post_map(new_df, api, args.repo, token)

        merged = _merge_with_hub(spec, new_df, args, token, mapping_errors, console)

        console.print("\n[bold cyan]Step 3:[/bold cyan] Preparing and pushing to Hub...")
        if merged.frame.empty:
            console.print("[yellow]ℹ[/yellow] Final dataset is empty. No push performed.")
            return 0
        # A left-merge fills a brand-new item's embedding cell with NaN; make
        # it a real null before the contract check and the Arrow conversion.
        merged.frame = normalize_embedding_nulls(merged.frame, spec.config_name)
        validate_frame(merged.frame, spec.config_name)
        _print_summary(console, merged.frame)

        if args.dry_run:
            console.print(Panel("[yellow]Dry run — nothing pushed.[/yellow]", border_style="yellow"))
            return 0
        _push(spec, merged, args, token, console)
        return 0
    except Exception as exc:  # noqa: BLE001 - every failure gets one typed panel
        return _report_failure(console, logger, exc)
    finally:
        await conn_manager.close()


def build_parser(spec: UploadSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=spec.description or f"Upload the IWAC '{spec.config_name}' subset to the HF Hub"
    )
    parser.add_argument(
        "--repo", default=get_private_repo_id(),
        help="Target Hugging Face repository (default: private full mirror)",
    )
    parser.add_argument(
        "--max-shard-size", default="1GB",
        help="Maximum Parquet shard size (e.g. 500MB, 1GB)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass the local Omeka response cache (24h TTL) and fetch fresh data",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch, map and merge, but push nothing",
    )
    parser.add_argument(
        "--force-shrink", action="store_true",
        help="Allow pushing a dataset markedly smaller than the one on the Hub "
             "(normally aborted: a truncated fetch would silently delete rows)",
    )
    parser.add_argument(
        "--allow-media-failures", action="store_true",
        help="Allow pushing when any per-item media lookup failed (normally "
             "aborted; existing media values for failed items are preserved). "
             "Use only when the media really are gone, not when the host is "
             "unreachable",
    )
    parser.add_argument(
        "--allow-map-failures", action="store_true",
        help="Continue after mapper errors while retaining those items' complete "
             "existing Hub rows. The default is to abort on any mapper error.",
    )
    parser.add_argument(
        "--map-concurrency", type=int, default=8,
        help="Maximum concurrent Omeka item mappers (default: 8)",
    )
    derived_group = parser.add_mutually_exclusive_group()
    derived_group.add_argument(
        "--invalidate-derived", action="store_true", default=True,
        help="Where a row's source text (OCR, table of contents, image URL, "
             "date) changed since the Hub copy, clear its preserved derived "
             "values (embeddings, lemmas, metrics, topics, Hijri date) so each "
             "stage's 'missing' mode recomputes them (default). Also replays "
             "any pending stale-derived worklist from a previous run.",
    )
    derived_group.add_argument(
        "--preserve-derived", action="store_false", dest="invalidate_derived",
        help="Explicitly retain stale derived values and stage a durable worklist "
             "before pushing. The next default upload clears those recorded cells.",
    )
    parser.add_argument(
        "--initialize", action="store_true",
        help="Allow creation of a deliberately new Hub config. Existing configs "
             "still fail closed on every read/auth/network error.",
    )
    if spec.supports_stale_rows:
        parser.add_argument(
            "--stale-rows", choices=["keep", "drop"], default="drop",
            help="Hub-only rows (items deleted on Omeka) after the outer merge: "
                 "drop them (default), or retain complete historical rows in the "
                 "private mirror with item_is_public=False",
        )
    return parser


def run_upload(spec: UploadSpec, argv: Optional[Sequence[str]] = None) -> int:
    """Parse the CLI and run the upload; returns a process exit code."""
    console, logger = _setup_console_logging()
    args = build_parser(spec).parse_args(argv)
    if args.map_concurrency < 1:
        console.print("[red]--map-concurrency must be at least 1.[/red]")
        return 2
    try:
        # Separate from the gateway's short repo-write lock: this serializes
        # read/replay/stage/write/ack of the recovery queue for the entire
        # subset run. Otherwise an earlier writer could delete a later
        # writer's newly staged worklist after releasing the gateway lock.
        with hub_write_lock(
            f"ingest-state::{args.repo}::{spec.config_name}", console=console,
            root_dir=_state_root() / "locks",
        ):
            return asyncio.run(_run(spec, args, console, logger))
    except HubWriteLockedError as exc:
        console.print(f"[red]Upload state is locked:[/red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130


__all__ = [
    "COUNTRY_MAPPER_HINT",
    "UploadSpec",
    "cast_int_columns",
    "run_upload",
    "build_parser",
    "report_unmapped_values",
    "record_stale_derived",
    "load_stale_derived",
    "replay_stale_derived",
    "stale_worklist_path",
]
