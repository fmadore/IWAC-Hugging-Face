"""Shared helpers for post-processing scripts (lemmatization, lexical
richness, word counts, embeddings, LDA topic modeling).

These scripts all (a) authenticate against the HF Hub, (b) optionally pick a
dataset config interactively, then (c) load → compute → push. This module
holds the canonical versions of that plumbing:

- auth: :func:`ensure_hf_token`
- config picking: :func:`get_available_configs` (dynamic, from the Hub),
  :func:`choose_config`, :func:`resolve_config` (CLI flag > interactive)
- update-mode picking: :func:`choose_update_mode` (missing | all)
- loading: :func:`load_hub_dataset` (Dataset), :func:`load_subset_dataframe`
  (pandas, hub or local CSV mirror)
- computing: :func:`map_with_progress` (``ds.map`` + Rich bar + cache-busting)
- column placement: :func:`reorder_columns_after`
- pushing: :func:`push_dataset`, :func:`print_dry_run_panel`

The per-script metric functions, argparse surfaces, and summary panels stay
in each script — only the genuinely shared flow lives here (no framework).
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from huggingface_hub import dataset_info, get_token, login
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich import box


_DEFAULT_CONFIGS: List[str] = ["articles", "publications", "documents"]

# Repo root (parent of post-processing/), used to locate the data/ CSV mirrors.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical repo IDs (re-exported so post-processing scripts can use
# ``from _common import PRIVATE_REPO_ID`` without touching sys.path).
try:
    from iwac_common.repos import PRIVATE_REPO_ID, PUBLIC_REPO_ID  # noqa: F401
except ImportError:  # venv without the editable install
    sys.path.insert(0, str(REPO_ROOT))
    from iwac_common.repos import PRIVATE_REPO_ID, PUBLIC_REPO_ID  # noqa: F401


def load_subset_dataframe(
    repo_id: str,
    config_name: str,
    *,
    token: Optional[str] = None,
    source: str = "hub",
    csv_path: Optional[Path] = None,
    columns: Optional[List[str]] = None,
    console: Optional[Console] = None,
):
    """Load one IWAC subset as a pandas DataFrame.

    source="hub" downloads the live dataset (authoritative, needs network);
    source="csv" reads the local ``data/iwac_<config>.csv`` mirror written by
    ``data/fetch_datasets.py`` (fast, offline — but may lag the Hub).

    ``columns`` restricts the frame (and, for CSV, what is parsed at all —
    important for the 388 MB articles mirror). ``o:id`` is always cast to str.
    """
    import pandas as pd  # local import: keep module import light

    console = console or Console()
    if source == "csv":
        path = csv_path or (REPO_ROOT / "data" / f"iwac_{config_name}.csv")
        if not path.exists():
            raise FileNotFoundError(
                f"Local mirror not found: {path}. Run data/fetch_datasets.py or use --source hub."
            )
        console.print(f"[blue]→[/blue] Loading local mirror [cyan]{path.name}[/cyan]")
        df = pd.read_csv(path, usecols=columns, dtype={"o:id": str}, low_memory=False)
        console.print(
            f"[yellow]ℹ[/yellow] Local CSV mirror may lag the live Hub dataset "
            f"(file date: {pd.Timestamp(path.stat().st_mtime, unit='s').date()})."
        )
    elif source == "hub":
        from datasets import load_dataset

        with console.status(f"[bold green]Loading '{repo_id}' ({config_name}) from Hub...", spinner="dots"):
            ds = load_dataset(repo_id, name=config_name, split="train", token=token)
        if columns:
            keep = [c for c in columns if c in ds.column_names]
            ds = ds.select_columns(keep)
        df = ds.to_pandas()
    else:
        raise ValueError(f"Unknown source '{source}' (expected 'hub' or 'csv').")

    if "o:id" in df.columns:
        df["o:id"] = df["o:id"].astype(str)
    console.print(f"[green]✓[/green] Loaded {len(df):,} rows ({config_name}, source={source})")
    return df


def ensure_hf_token(console: Optional[Console] = None) -> str:
    """Return a usable HF Hub token.

    Resolution order: ``HF_TOKEN`` env var → locally stored token → interactive
    ``login()``. Exits the process if no token can be obtained.
    """
    console = console or Console()
    token = os.getenv("HF_TOKEN") or get_token()
    if token:
        return token

    console.print("[yellow]⚠[/yellow] No HF token found. Triggering interactive login.")
    try:
        login()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] Interactive login failed: {exc}")
        sys.exit(1)
    token = get_token()
    if not token:
        console.print("[red]✗[/red] No token after login. Set HF_TOKEN and retry.")
        sys.exit(1)
    return token


def get_available_configs(
    repo_id: str,
    token: Optional[str] = None,
    fallback: Optional[List[str]] = None,
) -> List[str]:
    """Look up configs for ``repo_id`` from the HF Hub.

    Falls back to ``fallback`` (or a sensible default) if the lookup fails or
    returns nothing — matches the behavior the post-processing scripts had
    inline before extraction.
    """
    fallback = list(fallback) if fallback is not None else list(_DEFAULT_CONFIGS)
    try:
        info = dataset_info(repo_id, token=token)
        names = getattr(info, "config_names", None)
        if names:
            return list(names)
    except Exception:  # noqa: BLE001
        pass
    return fallback


def choose_config(available: List[str], console: Optional[Console] = None) -> str:
    """Interactive picker for a dataset config name.

    Returns the single config silently when there's only one. Exits the
    process on Ctrl-C.
    """
    console = console or Console()
    if len(available) == 1:
        console.print(f"[yellow]ℹ[/yellow] Single configuration available: [cyan]{available[0]}[/cyan]")
        return available[0]

    table = Table(title="Available Configurations", box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Configuration", style="green")
    for i, cfg in enumerate(available, 1):
        table.add_row(str(i), cfg)
    console.print(table)

    while True:
        try:
            choice = IntPrompt.ask(
                "Choose a configuration",
                choices=[str(i) for i in range(1, len(available) + 1)],
                show_choices=False,
            )
            return available[choice - 1]
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.[/yellow]")
            sys.exit(0)


def resolve_config(
    repo_id: str,
    *,
    token: Optional[str] = None,
    cli_config: Optional[str] = None,
    restrict_to: Optional[List[str]] = None,
    console: Optional[Console] = None,
) -> str:
    """Resolve which dataset config to process.

    ``cli_config`` (a ``--config`` flag) wins as-is when given — scripts that
    want strict CLI validation enforce it via argparse ``choices``. Otherwise
    the config list is fetched dynamically from the Hub, filtered to
    ``restrict_to`` (the subsets that make sense for the calling metric —
    the restriction only drives the interactive menu), and the user picks
    via :func:`choose_config`.
    """
    console = console or Console()
    if cli_config:
        return cli_config
    with console.status("[bold green]Fetching available configurations...", spinner="dots"):
        available = get_available_configs(repo_id, token=token, fallback=restrict_to)
    if restrict_to:
        available = [c for c in available if c in restrict_to] or list(restrict_to)
    return choose_config(available, console=console)


def choose_update_mode(console: Optional[Console] = None, default: str = "missing") -> str:
    """Interactive picker for the update mode ('missing' or 'all')."""
    console = console or Console()
    console.print("\n[bold]Update Mode:[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Mode", style="green")
    table.add_column("Description", style="white")

    table.add_row("1", "missing", "Compute only rows without values (recommended)")
    table.add_row("2", "all", "Recalculate all values (may take longer)")

    console.print(table)

    default_choice = "1" if default == "missing" else "2"
    while True:
        try:
            choice = Prompt.ask("Choose update mode", choices=["1", "2"], default=default_choice)
            return "missing" if choice == "1" else "all"
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.[/yellow]")
            sys.exit(0)


def load_hub_dataset(
    repo_id: str,
    config_name: str,
    *,
    token: Optional[str] = None,
    console: Optional[Console] = None,
):
    """Load one config of a Hub dataset (train split) with a status spinner.

    Returns the ``datasets.Dataset``, or ``None`` after printing the error
    (callers bail out on ``None``).
    """
    from datasets import load_dataset

    console = console or Console()
    try:
        with console.status(f"[bold green]Loading '{repo_id}' (config: {config_name})...", spinner="dots"):
            ds = load_dataset(repo_id, name=config_name, split="train", token=token)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]✗[/red] Failed to load dataset: {e}")
        return None
    console.print(f"[green]✓[/green] Dataset loaded: [cyan]{len(ds):,}[/cyan] rows")
    return ds


def map_with_progress(
    ds,
    batch_fn: Callable[[Dict[str, List[Any]]], Dict[str, List[Any]]],
    *,
    batch_size: int = 1000,
    description: str = "[cyan]Processing",
    console: Optional[Console] = None,
):
    """``ds.map(batched=True)`` with a Rich progress bar and cache-busting.

    Always disables the datasets ``.map()`` cache (``load_from_cache_file=False``
    plus a fresh ``new_fingerprint``) so re-runs never resurface stale computed
    columns. ``batch_fn`` takes and returns a batch dict, exactly like a plain
    ``.map`` callable.
    """
    console = console or Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=len(ds))

        def _with_progress(batch):
            result = batch_fn(batch)
            first = next(iter(result.values()), None)
            progress.update(task, advance=len(first) if first is not None else 0)
            return result

        return ds.map(
            _with_progress,
            batched=True,
            batch_size=batch_size,
            desc=None,
            load_from_cache_file=False,
            new_fingerprint=str(uuid.uuid4()),
        )


def reorder_columns_after(ds, new_cols: List[str], after_col: str, console: Optional[Console] = None):
    """Move ``new_cols`` so they sit immediately after ``after_col``.

    No-ops (with an informational message) when ``after_col`` is absent — the
    new columns then stay where ``.map()``/``add_column`` left them (at the end).
    """
    console = console or Console()
    columns = list(ds.column_names)
    if after_col not in columns:
        console.print(f"[yellow]ℹ[/yellow] Column '{after_col}' not found; new columns appended at end.")
        return ds
    present_new = [c for c in new_cols if c in columns]
    remaining = [c for c in columns if c not in present_new]
    idx = remaining.index(after_col)
    ordered = remaining[: idx + 1] + present_new + remaining[idx + 1:]
    ds = ds.select_columns(ordered)
    console.print(f"[blue]→[/blue] Columns reordered ({', '.join(present_new)} after '{after_col}')")
    return ds


def push_dataset(
    ds,
    *,
    repo_id: str,
    config_name: str,
    token: Optional[str],
    commit_message: str,
    max_shard_size: str = "1GB",
    console: Optional[Console] = None,
) -> bool:
    """Push ``ds`` to the Hub with a status spinner. Returns True on success.

    On failure, prints a Rich error panel (full traceback goes to the log)
    and returns False — callers decide what to do with caches, summaries, etc.
    """
    console = console or Console()
    try:
        with console.status("[bold green]Pushing dataset to Hub...", spinner="dots"):
            ds.push_to_hub(
                repo_id,
                config_name=config_name,
                token=token,
                max_shard_size=max_shard_size,
                commit_message=commit_message,
            )
        return True
    except Exception as e:  # noqa: BLE001
        console.print(Panel(
            f"[bold red]Failed to push dataset[/bold red]\n\n{e}",
            title="Error",
            border_style="red",
        ))
        logging.getLogger(__name__).error("Push error details:", exc_info=True)
        return False


def print_dry_run_panel(
    *,
    repo_id: str,
    config_name: str,
    n_rows: int,
    extra: Optional[str] = None,
    console: Optional[Console] = None,
) -> None:
    """Standard 'dry run complete, nothing pushed' panel."""
    console = console or Console()
    body = (
        "[yellow]Dry run mode — no changes pushed to Hub.[/yellow]\n\n"
        f"Would have pushed [cyan]{n_rows}[/cyan] rows to "
        f"[cyan]{repo_id}[/cyan] (config: {config_name})."
    )
    if extra:
        body += f"\n{extra}"
    console.print(Panel(body, title="Dry Run Complete", border_style="yellow"))


__all__ = [
    "ensure_hf_token",
    "get_available_configs",
    "choose_config",
    "resolve_config",
    "choose_update_mode",
    "load_hub_dataset",
    "load_subset_dataframe",
    "map_with_progress",
    "reorder_columns_after",
    "push_dataset",
    "print_dry_run_panel",
    "REPO_ROOT",
    "PRIVATE_REPO_ID",
    "PUBLIC_REPO_ID",
]
