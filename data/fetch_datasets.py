#!/usr/bin/env python3
"""Create a revision-pinned, integrity-checked local CSV mirror of IWAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Keep optional ML backends out of the lightweight mirror downloader.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from datasets import load_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import get_token  # noqa: E402
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

from iwac_common.hub import get_repo_revision  # noqa: E402
from iwac_common.repos import PRIVATE_REPO_ID, PUBLIC_REPO_ID  # noqa: E402
from iwac_common.schema import ALL_CONFIGS  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

console = Console()
DATA_DIR = Path(__file__).resolve().parent
MANIFEST_NAME = "mirror_manifest.json"


def choose_dataset(cli_choice: str | None = None) -> tuple[str, str]:
    """Resolve which repo to mirror: private full or public projection."""
    choice = cli_choice
    if choice is None:
        console.print("\n[bold]Which dataset do you want to mirror locally?[/bold]")
        table = Table(box=box.SIMPLE)
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Dataset", style="green")
        table.add_column("Repo", style="white")
        table.add_column("Full text", style="white")
        table.add_row("1", "private (full)", PRIVATE_REPO_ID, "complete OCR + lemmas")
        table.add_row("2", "public", PUBLIC_REPO_ID, "OCR masked per row")
        console.print(table)
        selected = Prompt.ask("Choose dataset", choices=["1", "2"], default="1")
        choice = "private" if selected == "1" else "public"
    return (PUBLIC_REPO_ID, "public") if choice == "public" else (PRIVATE_REPO_ID, "private")


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


def main(dataset_id: str = PRIVATE_REPO_ID, label: str = "private") -> int:
    """Download every config at one Hub revision and publish the mirror atomically."""
    token = (os.getenv("HF_TOKEN") or get_token()) if label == "private" else None
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
                    frame = dataset.to_pandas()
                    for column in frame.columns:
                        if frame[column].map(
                            lambda value: isinstance(value, (list, tuple, np.ndarray))
                        ).any():
                            frame[column] = frame[column].map(_json_safe)

                    filename = f"iwac_{config_name}.csv"
                    path = staging / filename
                    frame.to_csv(path, index=False, encoding="utf-8")
                    results.append({
                        "config": config_name,
                        "file": filename,
                        "rows": len(frame),
                        "columns": len(frame.columns),
                        "sha256": _sha256(path),
                    })
                    progress.update(task, advance=1)

            manifest = {
                "schema_version": 1,
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
            "partially replaced CSVs until the next successful refresh.",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["private", "public"],
        default=None,
        help="Repository to mirror (default: interactive prompt).",
    )
    parsed = parser.parse_args()
    resolved_id, resolved_label = choose_dataset(parsed.dataset)
    raise SystemExit(main(dataset_id=resolved_id, label=resolved_label))
