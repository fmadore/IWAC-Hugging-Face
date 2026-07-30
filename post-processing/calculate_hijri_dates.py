#!/usr/bin/env python3
"""
calculate_hijri_dates.py
========================

Adds the Hijri (Umm al-Qura) date columns ``hijri_year`` / ``hijri_month`` /
``hijri_day`` to a subset of the IWAC dataset, converted from ``pub_date``.

Why precompute rather than convert at read time
-----------------------------------------------
Every consumer that needs a lunar date would otherwise convert it itself, and
they do not agree. ``hijridate`` uses the Umm al-Qura tables throughout, while
the ICU tables behind a browser's ``Intl`` (and behind Node's, which is what
the IWAC MCP server would have to use) fall back to a tabular approximation
for older dates.

Measured on the live ``articles`` subset (12,220 complete dates, hijridate
2.6.0 vs Node 24 ICU):

===========================  =======  ==========================
Rows                         Count    Hijri DAY differs
===========================  =======  ==========================
pre-2000 (25.8 % of dated)     3,152   2,365 — **75.0 %**
2000 onward                    9,068   0 — 0.0 %
===========================  =======  ==========================

So converting in each consumer would mean the website (IwacVisualizations'
on-this-day block, which already uses ``hijridate``), the MCP server and any
notebook file the same 1990s article under different lunar days three times
over.

The disagreement is almost entirely sub-monthly: only **0.86 %** of articles
(105) land in a different Hijri *month*, because a one-day shift only crosses
a boundary for items on the last day of a month. A month-level distribution is
therefore robust to the choice; a day-level feature — "on this day", an exact
date label — is not. That asymmetry is the whole reason this is computed once.

Computing it once here makes them agree by construction. ``hijridate`` is the
converter of record, chosen to match ``generate_on_this_day.py`` in the
IwacVisualizations repo, so the buckets on the website and the counts served
over MCP are the same numbers.

Precision
---------
Only a complete ``YYYY-MM-DD`` carries a lunar date. Partial values —
year-only, ``YYYY-MM``, and the ``1981-04/1981-06`` ranges that appear in
``publications`` — are left ``None`` rather than guessed: a lunar month
straddles two Gregorian ones, so a missing day is not a rounding problem but a
genuinely unanswerable question. The run reports how many rows that affects so
the gap is visible rather than silent.

Usage
-----
    python post-processing/calculate_hijri_dates.py [--config articles] [--dry-run]

    python post-processing/calculate_hijri_dates.py                       # interactive
    python post-processing/calculate_hijri_dates.py --config articles -y  # non-interactive

Environment
-----------
HF_TOKEN    Hugging Face token (otherwise an interactive login is requested).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Make ``post-processing/_common.py`` and ``iwac_common`` importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from _common import (  # noqa: E402
    PRIVATE_REPO_ID,
    ensure_hf_token,
    load_hub_dataset,
    map_with_progress,
    print_dry_run_panel,
    push_dataset,
    reorder_columns_after,
    resolve_config,
)

load_dotenv()

console = Console()

# Every content subset carries ``pub_date``, but a lunar date is only
# meaningful where the date marks something that happened inside the Islamic
# calendar — a newspaper issue, a periodical number, a document, a recording, a
# photograph. Two subsets are deliberately excluded:
#
# - ``index``   its date fields are first/last-occurrence aggregates, not
#               publication dates.
# - ``references``  academic works (books, journal articles, theses). Their
#               imprint date belongs to an academic publishing calendar; the
#               lunar date of a monograph's copyright year says nothing about
#               the collection and would invite meaningless "references by
#               Ramadan" readings.
HIJRI_SUBSETS = ["articles", "publications", "documents", "audiovisual", "images"]

SOURCE_COLUMN = "pub_date"
HIJRI_COLUMNS = ["hijri_year", "hijri_month", "hijri_day"]

# Umm al-Qura month names, in the academic transliteration IWAC follows (the
# same table as IwacVisualizations' asset/js/charts/shared/hijri.js, so a
# reader moving between a chart and this report meets one spelling).
HIJRI_MONTHS = [
    "Muharram", "Safar", "Rabi' I", "Rabi' II",
    "Jumada I", "Jumada II", "Rajab", "Sha'ban",
    "Ramadan", "Shawwal", "Dhu al-Qa'da", "Dhu al-Hijja",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def parse_gregorian(value: Any) -> Optional[Tuple[int, int, int]]:
    """``(y, m, d)`` for a complete ISO date, else ``None``.

    Deliberately strict: anything that is not exactly ``YYYY-MM-DD`` — a bare
    year, ``YYYY-MM``, or a ``YYYY-MM/YYYY-MM`` range — has no single day to
    convert, so it yields ``None`` instead of a guess.
    """
    if value is None:
        return None
    text = str(value).strip()
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        return None
    try:
        year, month, day = int(text[:4]), int(text[5:7]), int(text[8:10])
    except ValueError:
        return None
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return year, month, day


def to_hijri(year: int, month: int, day: int) -> Optional[Tuple[int, int, int]]:
    """Umm al-Qura ``(year, month, day)``, or ``None`` outside the tables.

    The collection runs 1961–2025, comfortably inside hijridate's 1343–1500 AH
    (1925–2077) range, so ``None`` here means a malformed date slipped past
    :func:`parse_gregorian` rather than a real limitation — skip the row
    instead of failing the run. (Same contract as ``to_hijri`` in
    IwacVisualizations' ``scripts/generate_on_this_day.py``.)
    """
    from hijridate import Gregorian

    try:
        h = Gregorian(year, month, day).to_hijri()
        return h.year, h.month, h.day
    except (ValueError, OverflowError) as exc:
        logging.getLogger(__name__).debug(
            "Hijri conversion failed for %04d-%02d-%02d: %s", year, month, day, exc
        )
        return None


def add_hijri_batch(batch: Dict[str, List[Any]], *, update_mode: str = "all") -> Dict[str, List[Any]]:
    """Fill the three Hijri columns for one ``.map(batched=True)`` batch."""
    if SOURCE_COLUMN not in batch:
        first = next(iter(batch), None)
        n = len(batch[first]) if first is not None else 0
        for col in HIJRI_COLUMNS:
            batch.setdefault(col, [None] * n)
        return batch

    dates = batch[SOURCE_COLUMN]
    existing = {c: batch.get(c) for c in HIJRI_COLUMNS} if update_mode == "missing" else {}
    years: List[Optional[int]] = []
    months: List[Optional[int]] = []
    days: List[Optional[int]] = []

    for i, raw in enumerate(dates):
        prior = existing.get("hijri_year")
        if prior is not None and i < len(prior) and prior[i] is not None:
            years.append(prior[i])
            months.append(existing["hijri_month"][i])
            days.append(existing["hijri_day"][i])
            continue
        gregorian = parse_gregorian(raw)
        converted = to_hijri(*gregorian) if gregorian else None
        if converted is None:
            years.append(None)
            months.append(None)
            days.append(None)
        else:
            years.append(converted[0])
            months.append(converted[1])
            days.append(converted[2])

    batch["hijri_year"], batch["hijri_month"], batch["hijri_day"] = years, months, days
    return batch


def report(df: pd.DataFrame, config_name: str) -> None:
    """Coverage table + the lunar-year distribution the columns exist to enable."""
    total = len(df)
    converted = int(df["hijri_year"].notna().sum())
    parsable = int(df[SOURCE_COLUMN].map(lambda v: parse_gregorian(v) is not None).sum())
    empty = int(df[SOURCE_COLUMN].isna().sum() + (df[SOURCE_COLUMN].astype("string").fillna("") == "").sum())

    coverage = Table(title=f"Hijri coverage — {config_name}", box=box.ROUNDED)
    coverage.add_column("Rows", style="cyan")
    coverage.add_column("Count", style="green", justify="right")
    coverage.add_column("Share", style="dim", justify="right")
    for label, n in (
        ("Total", total),
        ("Complete YYYY-MM-DD", parsable),
        ("Converted to Hijri", converted),
        ("Imprecise date (no lunar day)", parsable - converted + (total - parsable - empty)),
        ("No date at all", empty),
    ):
        coverage.add_row(label, f"{n:,}", f"{(n / total * 100):.1f}%" if total else "—")
    console.print(coverage)

    if not converted:
        return

    counts = df["hijri_month"].dropna().astype(int).value_counts().sort_index()
    expected = converted / 12
    peak = counts.max()
    dist = Table(title="Articles by Hijri month (deviation from an even split)", box=box.ROUNDED)
    dist.add_column("#", style="dim", justify="right")
    dist.add_column("Month", style="cyan")
    dist.add_column("Count", style="green", justify="right")
    dist.add_column("vs even", justify="right")
    dist.add_column("", style="dim")
    for month in range(1, 13):
        n = int(counts.get(month, 0))
        delta = (n / expected - 1) * 100 if expected else 0
        colour = "green" if delta >= 25 else ("red" if delta <= -25 else "white")
        dist.add_row(
            str(month),
            HIJRI_MONTHS[month - 1],
            f"{n:,}",
            f"[{colour}]{'+' if delta >= 0 else ''}{delta:.0f}%[/{colour}]",
            "█" * round(n / (peak / 28)) if peak else "",
        )
    console.print(dist)


def main() -> None:
    configure_logging()
    console.print(
        Panel.fit(
            "[bold cyan]Hijri Date Calculator[/bold cyan]\n"
            "[dim]Umm al-Qura year/month/day from pub_date (hijridate)[/dim]",
            border_style="cyan",
        )
    )

    parser = argparse.ArgumentParser(
        description="Add/refresh hijri_year, hijri_month and hijri_day on an IWAC subset."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", choices=HIJRI_SUBSETS, default=None,
                        help="Subset to process (skips the interactive menu)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Recompute without confirmation when the columns already exist")
    parser.add_argument("--update-mode", choices=["missing", "all"], default="all",
                        help="'all' recomputes every row (default); 'missing' fills only unconverted rows")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report, but push nothing to the Hub")
    args = parser.parse_args()

    try:
        import hijridate  # noqa: F401
    except ImportError:
        console.print(
            "[red]✗[/red] hijridate is not installed — it is the converter of record "
            "for these columns.\n[yellow]ℹ[/yellow] pip install hijridate  "
            "(also listed in requirements.txt)"
        )
        return

    token = ensure_hf_token(console=console)
    config_name = resolve_config(
        args.repo, token=token, cli_config=args.config, restrict_to=HIJRI_SUBSETS, console=console
    )
    console.print(f"[green]→[/green] Subset: [bold]{config_name}[/bold]")

    ds = load_hub_dataset(args.repo, config_name, token=token, console=console)
    if ds is None:
        return

    if SOURCE_COLUMN not in ds.column_names:
        console.print(f"[red]✗[/red] '{SOURCE_COLUMN}' is missing from this subset — nothing to convert.")
        return

    already = [c for c in HIJRI_COLUMNS if c in ds.column_names]
    if already and args.update_mode == "all" and not (args.yes or args.dry_run):
        console.print(f"\n[yellow]⚠[/yellow] Already present: [bold]{', '.join(already)}[/bold]")
        try:
            if not Confirm.ask("Recompute them?", default=False):
                console.print("[yellow]ℹ[/yellow] Cancelled; existing values kept.")
                return
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Cancelled.")
            return

    ds = map_with_progress(
        ds,
        lambda batch: add_hijri_batch(batch, update_mode=args.update_mode),
        description="[cyan]Converting pub_date to Umm al-Qura",
        console=console,
    )

    # Nullable Int64: an imprecise date has no lunar day, and that absence has
    # to survive the round trip rather than become a 0 that charts would plot.
    df = ds.to_pandas()
    for col in HIJRI_COLUMNS:
        df[col] = df[col].astype("Int64")
    from datasets import Dataset

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = reorder_columns_after(ds, HIJRI_COLUMNS, SOURCE_COLUMN, console=console)

    report(df, config_name)

    if args.dry_run:
        print_dry_run_panel(
            repo_id=args.repo,
            config_name=config_name,
            n_rows=len(ds),
            extra=f"[dim]Columns: {', '.join(HIJRI_COLUMNS)}[/dim]",
            console=console,
        )
        return

    ok = push_dataset(
        ds,
        repo_id=args.repo,
        config_name=config_name,
        token=token,
        commit_message=(
            f"Add Hijri date columns ({', '.join(HIJRI_COLUMNS)}) "
            f"converted from {SOURCE_COLUMN} with hijridate (Umm al-Qura) — config: {config_name}"
        ),
        console=console,
    )
    if ok:
        console.print(
            Panel(
                f"[green]✓[/green] {', '.join(HIJRI_COLUMNS)} written for [bold]{config_name}[/bold]\n"
                f"[dim]Rows: {len(ds):,} · Repository: {args.repo}\n"
                "Remember: the public dataset only changes when "
                "post-processing/publish_public.py is re-run.[/dim]",
                title="[bold green]Done[/bold green]",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
