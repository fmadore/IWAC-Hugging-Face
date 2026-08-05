"""Verify that the LIVE public dataset masks full text on every source-private row.

`publish_public.py` guards what gets *written*: it aborts when a content subset
lacks `OCR_is_public`, and on any column outside the allowlist. This script
checks what is already *there* — it reads the published repo and fails if any
row flagged `OCR_is_public = false` still carries `OCR`, `lemma_text` or
`lemma_nostop`.

Run it before minting a DOI. Minting is irreversible and forfeits the ability to
make the repo private, which is the only lever left if a leak is ever found (see
the DOI section in CLAUDE.md).

Read-only. No token needed for the public repo; pass --repo to check a scratch
projection instead.

    .venv\\Scripts\\python post-processing/verify_public_masking.py

Exit status: 0 clean, 1 leak or missing flag.
"""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

# Match the sibling scripts: importable from any working directory, with a
# sys.path fallback for venvs where the package was never installed.
try:
    from iwac_common.repos import PUBLIC_REPO_ID, CONTENT_COLUMNS
except ModuleNotFoundError:  # pragma: no cover - fallback path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from iwac_common.repos import PUBLIC_REPO_ID, CONTENT_COLUMNS

console = Console()


def _is_populated(value) -> bool:
    """True when a cell still holds text (as opposed to None / '' / whitespace)."""
    return value is not None and str(value).strip() != ""


def check_subset(repo_id: str, subset: str, columns: list[str]) -> tuple[dict, str | None]:
    """Inspect one subset. Returns (row for the report, failure reason or None)."""
    ds = load_dataset(repo_id, name=subset, split="train")
    names = ds.column_names

    if "OCR_is_public" not in names:
        return (
            {"subset": subset, "rows": len(ds), "private": "?", "status": "NO FLAG"},
            f"{subset}: missing OCR_is_public",
        )

    flags = ds["OCR_is_public"]
    n_private = sum(1 for f in flags if not f)

    leaked = []
    for col in (c for c in columns if c in names):
        n = sum(1 for f, v in zip(flags, ds[col]) if not f and _is_populated(v))
        if n:
            leaked.append(f"{col}={n}")

    if leaked:
        return (
            {
                "subset": subset,
                "rows": len(ds),
                "private": n_private,
                "status": "LEAK: " + ", ".join(leaked),
            },
            f"{subset}: {', '.join(leaked)}",
        )

    return (
        {"subset": subset, "rows": len(ds), "private": n_private, "status": "clean"},
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=PUBLIC_REPO_ID,
        help=f"Dataset repo to inspect (default: {PUBLIC_REPO_ID})",
    )
    args = parser.parse_args()

    console.print(f"[cyan]ℹ[/cyan] Inspecting [bold]{args.repo}[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("subset")
    table.add_column("rows", justify="right")
    table.add_column("source-private", justify="right")
    table.add_column("status")

    failures: list[str] = []
    total = 0

    for subset in sorted(CONTENT_COLUMNS):
        try:
            row, failure = check_subset(args.repo, subset, CONTENT_COLUMNS[subset])
        except Exception as exc:  # noqa: BLE001 - report and keep going
            console.print(f"[red]✗[/red] {subset}: {type(exc).__name__}: {exc}")
            failures.append(f"{subset}: load failed")
            continue

        total += row["rows"]
        style = "green" if row["status"] == "clean" else "red"
        table.add_row(
            row["subset"],
            str(row["rows"]),
            str(row["private"]),
            f"[{style}]{row['status']}[/{style}]",
        )
        if failure:
            failures.append(failure)

    console.print(table)

    if failures:
        console.print(f"[red]✗[/red] NOT CLEAN — {total} rows inspected")
        for f in failures:
            console.print(f"  [red]•[/red] {f}")
        console.print("[red]Do not mint a DOI. Re-run publish_public.py first.[/red]")
        return 1

    console.print(
        f"[green]✓[/green] Clean — no full text on any source-private row "
        f"({total} rows inspected)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
