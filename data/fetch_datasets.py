#!/usr/bin/env python3
"""Create a revision-pinned, integrity-checked local mirror of IWAC.

Parquet by default: types survive (nullable ints stay ints, booleans stay
booleans, embeddings stay float vectors instead of JSON strings), and files are
several times smaller than the CSV export — the articles CSV alone was 388 MB.
``--format csv`` keeps the legacy CSV export for spreadsheet use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

# Keep optional ML backends out of the lightweight mirror downloader.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from datasets import load_dataset  # noqa: E402
from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt  # noqa: E402
from rich.table import Table  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iwac_common.hub import get_repo_revision, resolve_hf_token  # noqa: E402
from iwac_common.paths import workspace_root  # noqa: E402
from iwac_common.schema import ALL_CONFIGS, dataset_to_pandas  # noqa: E402
from iwac_common.settings import (  # noqa: E402
    get_private_repo_id,
    get_public_repo_id,
    initialize_environment,
)

# Same precedence as every entry point: process env, then IWAC_ENV_FILE or the
# discovered .env (never overriding the process environment).
initialize_environment()

console = Console()
DATA_DIR = workspace_root() / "data"
MANIFEST_NAME = "mirror_manifest.json"
MIRROR_FORMATS = ("parquet", "csv")


def choose_dataset(cli_choice: str | None = None) -> tuple[str, str]:
    """Resolve which repo to mirror: private full or public projection."""
    private_repo, public_repo = get_private_repo_id(), get_public_repo_id()
    choice = cli_choice
    if choice is None:
        console.print("\n[bold]Which dataset do you want to mirror locally?[/bold]")
        table = Table(box=box.SIMPLE)
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Dataset", style="green")
        table.add_column("Repo", style="white")
        table.add_column("Full text", style="white")
        table.add_row("1", "private (full)", private_repo, "complete OCR + lemmas")
        table.add_row("2", "public", public_repo, "OCR masked per row")
        console.print(table)
        selected = Prompt.ask("Choose dataset", choices=["1", "2"], default="1")
        choice = "private" if selected == "1" else "public"
    return (public_repo, "public") if choice == "public" else (private_repo, "private")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return value


def _write_config(dataset, path: Path, fmt: str) -> tuple[int, int]:
    """Write one config; return ``(rows, columns)``."""
    if fmt == "parquet":
        import pyarrow.parquet as pq

        table = dataset.with_format("arrow")[:]
        pq.write_table(table, path)
        return table.num_rows, table.num_columns
    # Typed conversion: a bare to_pandas() turns nullable ints into float64
    # (pub_year 1995 printed as 1995.0 in the CSV).
    frame = dataset_to_pandas(dataset)
    for column in frame.columns:
        if frame[column].map(
            lambda value: isinstance(value, (list, tuple, np.ndarray))
        ).any():
            frame[column] = frame[column].map(_json_safe)
    frame.to_csv(path, index=False, encoding="utf-8")
    return len(frame), len(frame.columns)


def main(
    dataset_id: str | None = None, label: str = "private", fmt: str = "parquet"
) -> int:
    """Download every config at one Hub revision and publish the mirror atomically."""
    if fmt not in MIRROR_FORMATS:
        console.print(f"[red]✗[/red] Unknown mirror format {fmt!r}; use one of {MIRROR_FORMATS}.")
        return 2
    dataset_id = dataset_id or get_private_repo_id()
    token = resolve_hf_token() if label == "private" else None
    if label == "private" and not token:
        console.print(
            "[red]✗[/red] The private mirror needs HF_TOKEN (set it in .env or "
            "run `hf auth login`), or choose the public dataset."
        )
        return 1

    revision = get_repo_revision(dataset_id, token=token)
    console.print(Panel.fit(
        "[bold cyan]IWAC Dataset Downloader[/bold cyan]\n"
        f"Repository: [cyan]{dataset_id}[/cyan]\n"
        f"Pinned revision: [cyan]{revision}[/cyan]",
        border_style="cyan",
    ))

    results: list[dict] = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".iwac-mirror-", dir=DATA_DIR) as tmp:
            staging = Path(tmp)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Downloading configs", total=len(ALL_CONFIGS))
                for config_name in ALL_CONFIGS:
                    progress.update(task, description=f"[cyan]Downloading {config_name}")
                    dataset = load_dataset(
                        dataset_id,
                        name=config_name,
                        split="train",
                        token=token,
                        revision=revision,
                    )
                    filename = f"iwac_{config_name}.{fmt}"
                    path = staging / filename
                    rows, columns = _write_config(dataset, path, fmt)
                    results.append({
                        "config": config_name,
                        "file": filename,
                        "format": fmt,
                        "rows": rows,
                        "columns": columns,
                        "sha256": _sha256(path),
                    })
                    progress.update(task, advance=1)

            manifest = {
                "schema_version": 2,
                "format": fmt,
                "repository": dataset_id,
                "visibility": label,
                "revision": revision,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "configs": {entry["config"]: entry for entry in results},
            }
            manifest_path = staging / MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            # Replace data first and the manifest last. If interrupted midway,
            # the old manifest hashes no longer match and consumers fail closed.
            for entry in results:
                os.replace(staging / entry["file"], DATA_DIR / entry["file"])
            os.replace(manifest_path, DATA_DIR / MANIFEST_NAME)
    except Exception as exc:  # noqa: BLE001
        console.print(Panel(
            f"[bold red]Mirror refresh aborted[/bold red]\n\n{exc}\n\n"
            "The previous verified manifest remains authoritative; do not use "
            f"partially replaced {fmt} files until the next successful refresh.",
            title="No mixed-revision mirror published",
            border_style="red",
        ))
        return 1

    table = Table(title="Mirror Summary", box=box.ROUNDED)
    table.add_column("Config", style="cyan")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Columns", justify="right", style="blue")
    table.add_column("SHA-256", style="dim")
    for entry in results:
        table.add_row(
            entry["config"], f"{entry['rows']:,}", str(entry["columns"]),
            entry["sha256"][:12],
        )
    console.print(table)
    console.print(
        f"[green]✓[/green] Published {len(results)} configs at revision "
        f"[cyan]{revision}[/cyan]; manifest: [cyan]{MANIFEST_NAME}[/cyan]"
    )
    return 0


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["private", "public"],
        default=None,
        help="Repository to mirror (default: interactive prompt).",
    )
    parser.add_argument(
        "--format",
        choices=MIRROR_FORMATS,
        default="parquet",
        help="Mirror file format (default: parquet; csv for spreadsheet use).",
    )
    return parser


def cli(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Command-line entry point shared by ``python fetch_datasets.py`` and ``iwac-mirror``."""
    parsed = build_parser(prog).parse_args(argv)
    resolved_id, resolved_label = choose_dataset(parsed.dataset)
    return main(dataset_id=resolved_id, label=resolved_label, fmt=parsed.format)


if __name__ == "__main__":
    raise SystemExit(cli())
