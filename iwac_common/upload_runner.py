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
import logging
import os
from dataclasses import dataclass, field
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
    push_dataset_verified,
    resolve_hf_token,
)
from .hub_merge import (
    DuplicateIdError,
    ShrinkGuardError,
    merge_with_hub_dataset,
)
from .omeka_client import (
    Config,
    MediaFetchGuardError,
    OmekaApiClient,
    TruncatedFetchError,
    conn_manager,
    media_stats,
)
from .repos import PRIVATE_REPO_ID
from .schema import DataContractError, normalize_embedding_nulls, validate_frame

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class MappingGuardError(RuntimeError):
    """One or more Omeka items could not be mapped safely."""


@dataclass
class UploadSpec:
    """Everything that differs between the seven subset upload scripts."""

    config_name: str  # HF dataset config, e.g. "articles"
    resource_class_ids: Sequence[int]  # Omeka classes; >1 for reference/index
    map_item: Callable[[Dict[str, Any], OmekaApiClient], Awaitable[Optional[Dict[str, Any]]]]
    title: str  # Rich panel title, e.g. "📰 IWAC Newspaper Upload"
    cache_dir: str  # per-subset Omeka response cache
    description: str = ""  # argparse description
    int_columns: Sequence[str] = ()  # cast to nullable Int64 after merge
    merge_how: str = "left"
    merge_suffixes: Sequence[str] = ("", "_old")
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
    extra_config_rows: Sequence[tuple] = field(default_factory=tuple)  # (label, value) panel rows


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
    for label, value in spec.extra_config_rows:
        table.add_row(label, str(value))
    console.print(Panel(table, title=f"[bold blue]{spec.title}", border_style="blue"))


async def _run(spec: UploadSpec, args: argparse.Namespace, console: Console, logger: logging.Logger) -> int:
    cfg = Config(CACHE_DIR=spec.cache_dir)
    _display_config_panel(console, spec, cfg, args)
    api = OmekaApiClient(cfg, use_cache=not args.no_cache, console=console)
    media_stats.reset()

    try:
        # 1. Fetch. A failed or truncated class fetch aborts the whole run:
        # continuing with a partial item list guarantees rows disappear from
        # the Hub (or trips the shrink guard later with wasted work).
        console.print("\n[bold cyan]Step 1:[/bold cyan] Fetching items from Omeka API...")
        items: list = []
        for rcid in spec.resource_class_ids:
            items.extend(await api.fetch_items(rcid))
        if not items:
            console.print(
                "[bold yellow]⚠ Warning:[/bold yellow] No items returned from "
                "Omeka API. Existing Hub dataset left untouched."
            )
            return 0
        console.print(f"[green]✓[/green] Fetched {len(items)} items from Omeka.")

        # 2. Map.
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
            task = progress.add_task(
                f"[cyan]Mapping {spec.config_name}", total=len(items)
            )
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
            sample = "; ".join(
                f"{item_id}: {message}" for item_id, message in mapping_errors[:5]
            )
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
        records = [record for record in records_by_position if record is not None]
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

        new_df = pd.DataFrame(records)
        if "o:id" not in new_df.columns or new_df["o:id"].isnull().any():
            console.print("[bold red]✗ Critical:[/bold red] 'o:id' missing or null. Aborting.")
            return 1
        new_df["o:id"] = new_df["o:id"].astype(str)

        token = resolve_hf_token()
        if token is None:
            console.print("[yellow]⚠[/yellow] No HF token found; starting login before reading the baseline.")
            try:
                login()
            except Exception as exc:  # noqa: BLE001
                raise HubBaselineUnavailableError(
                    f"Hugging Face login failed: {exc}"
                ) from exc
            token = resolve_hf_token()
        if token is None:
            raise HubBaselineUnavailableError(
                "No Hugging Face token is available after login; refusing to read or write."
            )
        if spec.post_map is not None:
            new_df = await spec.post_map(new_df, api, args.repo, token)

        # 3. Merge with the Hub (preserves computed columns; safety rails
        # raise instead of silently shrinking or fanning out).
        console.print("\n[bold cyan]Step 2:[/bold cyan] Merging with existing Hub dataset...")
        revision: dict[str, str] = {}
        final_df = merge_with_hub_dataset(
            new_df,
            args.repo,
            config_name=spec.config_name,
            token=token,
            how=spec.merge_how,
            suffixes=tuple(spec.merge_suffixes),
            columns_to_exclude=spec.columns_to_exclude,
            console=console,
            allow_shrink=args.force_shrink,
            stale_rows=getattr(args, "stale_rows", "keep"),
            allow_initialize=args.initialize,
            preserve_existing_ids=(
                [item_id for item_id, _ in mapping_errors]
                if args.allow_map_failures else ()
            ),
            preserve_fields_by_id=(
                media_stats.failed_fields_by_item
                if args.allow_media_failures else None
            ),
            revision_out=revision,
        )

        if spec.post_merge is not None:
            final_df = spec.post_merge(final_df)

        for int_col in spec.int_columns:
            if int_col in final_df.columns:
                final_df[int_col] = final_df[int_col].astype("Int64")

        # 4. Validate + push.
        console.print("\n[bold cyan]Step 3:[/bold cyan] Preparing and pushing to Hub...")
        if final_df.empty:
            console.print("[yellow]ℹ[/yellow] Final dataset is empty. No push performed.")
            return 0
        if final_df["o:id"].isnull().any():
            console.print("[bold red]✗ Critical:[/bold red] 'o:id' null after merge. Aborting push.")
            return 1
        # A left-merge fills a brand-new item's embedding cell with NaN; make
        # it a real null before the contract check and the Arrow conversion.
        final_df = normalize_embedding_nulls(final_df, spec.config_name)
        try:
            validate_frame(final_df, spec.config_name)
        except DataContractError as exc:
            console.print(f"[bold red]✗ Data contract:[/bold red] {exc}")
            return 1

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

        if args.dry_run:
            console.print(Panel(
                "[yellow]Dry run — nothing pushed.[/yellow]",
                border_style="yellow",
            ))
            return 0

        ds = Dataset.from_pandas(final_df, preserve_index=False)
        try:
            with console.status("[bold green]Pushing dataset to Hugging Face Hub...", spinner="dots"):
                push_dataset_verified(
                    ds,
                    repo_id=args.repo,
                    config_name=spec.config_name,
                    token=token,
                    commit_message=f"Refresh '{spec.config_name}' from the IWAC Omeka source",
                    max_shard_size=args.max_shard_size,
                    console=console,
                    expected_revision=revision.get("revision"),
                    expected_columns=list(final_df.columns),
                    expected_ids=final_df["o:id"].tolist(),
                )
            console.print(Panel(
                f"[bold green]✓ Dataset successfully published![/bold green]\n\n"
                f"Repository: [cyan]{args.repo}[/cyan]\n"
                f"Config: [cyan]{spec.config_name}[/cyan]\n"
                f"Records: [cyan]{len(final_df)}[/cyan]",
                title="🎉 Upload Complete",
                border_style="green",
            ))
            return 0
        except Exception as exc:  # noqa: BLE001
            console.print(Panel(
                f"[bold red]✗ Failed to push dataset[/bold red]\n\n{exc}",
                title="Error",
                border_style="red",
            ))
            logger.error("Details of the exception:", exc_info=True)
            return 1

    except TruncatedFetchError as exc:
        console.print(Panel(
            f"[bold red]✗ Truncated fetch[/bold red]\n\n{exc}",
            title="Aborted — Hub data protected",
            border_style="red",
        ))
        return 1
    except MediaFetchGuardError as exc:
        console.print(Panel(
            f"[bold red]✗ Mass media-lookup failure[/bold red]\n\n{exc}",
            title="Aborted — Hub data protected",
            border_style="red",
        ))
        return 1
    except MappingGuardError as exc:
        console.print(Panel(
            f"[bold red]✗ Mapper failure[/bold red]\n\n{exc}",
            title="Aborted — Hub rows protected",
            border_style="red",
        ))
        return 1
    except (
        HubBaselineUnavailableError,
        ConcurrentHubWriteError,
        HubWriteError,
        HubWriteLockedError,
    ) as exc:
        console.print(Panel(
            f"[bold red]✗ Hub safety guard[/bold red]\n\n{exc}",
            title="Aborted — Hub data protected",
            border_style="red",
        ))
        return 1
    except (ShrinkGuardError, DuplicateIdError) as exc:
        console.print(Panel(
            f"[bold red]✗ {type(exc).__name__}[/bold red]\n\n{exc}",
            title="Aborted — Hub data protected",
            border_style="red",
        ))
        return 1
    except Exception as exc:  # noqa: BLE001
        console.print(Panel(
            f"[bold red]✗ Unexpected pipeline failure[/bold red]\n\n{exc}",
            title="Aborted — Hub left untouched",
            border_style="red",
        ))
        logger.error("Unexpected upload failure:", exc_info=True)
        return 1
    finally:
        await conn_manager.close()


def build_parser(spec: UploadSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=spec.description or f"Upload the IWAC '{spec.config_name}' subset to the HF Hub"
    )
    parser.add_argument(
        "--repo", default=PRIVATE_REPO_ID,
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
    parser.add_argument(
        "--initialize", action="store_true",
        help="Allow creation of a deliberately new Hub config. Existing configs "
             "still fail closed on every read/auth/network error.",
    )
    if spec.supports_stale_rows:
        parser.add_argument(
            "--stale-rows", choices=["keep", "drop"], default="keep",
            help="Hub-only rows (items deleted on Omeka) after the outer merge: "
                 "keep them with empty Omeka fields (default) or drop them",
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
        return asyncio.run(_run(spec, args, console, logger))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130


__all__ = ["UploadSpec", "run_upload", "build_parser"]
