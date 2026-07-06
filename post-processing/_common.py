"""Shared helpers for post-processing scripts (lemmatization, lexical
richness, word counts, embeddings, LDA topic modeling).

These scripts all (a) authenticate against the HF Hub, (b) optionally pick a
dataset config interactively, then (c) load → compute → push. The auth +
config-picker pieces were duplicated across 4-5 of the 5 scripts; this module
holds the canonical versions.

The actual ``load → compute → push`` orchestration is intentionally **not**
extracted — every script has a different per-row function, batch size,
output-column shape, and update-mode flag, and forcing them through a single
wrapper hurts more than it helps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from huggingface_hub import dataset_info, get_token, login
from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table
from rich import box


_DEFAULT_CONFIGS: List[str] = ["articles", "publications", "documents"]

# Repo root (parent of post-processing/), used to locate the data/ CSV mirrors.
REPO_ROOT = Path(__file__).resolve().parent.parent


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


__all__ = [
    "ensure_hf_token",
    "get_available_configs",
    "choose_config",
    "load_subset_dataframe",
    "REPO_ROOT",
]
