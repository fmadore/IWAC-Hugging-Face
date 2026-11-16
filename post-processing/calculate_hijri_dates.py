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

An ISO datetime (``1999-01-01T00:00:00``, optionally with seconds fraction and
a ``Z``/offset) is converted from its date part — the calendar date it names —
and reported on its own row. A complete date outside hijridate's Umm al-Qura
tables (1924-08-01 to 2077-11-16) and an impossible calendar date
(``1999-02-30``) stay null and are reported separately too. Only ASCII digits
in exactly ``YYYY-MM-DD`` count: ``+199-01-01``, ``1_99-01-01`` or
Arabic-Indic digits are imprecise, not dates.

``--update-mode missing`` (default with ``-y`` or headless) fills only rows
whose three columns are not all set; ``all`` recomputes every row after a
confirmation ``-y`` skips. A run that changes no value pushes nothing.

Usage
-----
    python post-processing/calculate_hijri_dates.py [--config articles] [--dry-run]

    python post-processing/calculate_hijri_dates.py                       # interactive
    python post-processing/calculate_hijri_dates.py --config articles -y  # non-interactive, missing
    python post-processing/calculate_hijri_dates.py --config articles --update-mode all -y

Environment
-----------
HF_TOKEN    Hugging Face token (otherwise an interactive login is requested).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from iwac_common.paths import workspace_root
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
    add_update_mode_args,
    ensure_hf_token,
    load_hub_dataset,
    map_with_progress,
    print_dry_run_panel,
    push_dataset,
    reorder_columns_after,
    resolve_config,
    resolve_update_mode,
)
from iwac_common.enrichment import changed_row_count  # noqa: E402

load_dotenv(workspace_root() / ".env")

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


_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})", re.ASCII)
_ISO_TIME = re.compile(r"T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?", re.ASCII)

# hijridate's Umm al-Qura table range (1343-01-01 to 1500-12-30 AH).
UMM_AL_QURA_RANGE = ("1924-08-01", "2077-11-16")


def _split_date(value: Any) -> Tuple[Optional[Tuple[int, int, int]], bool]:
    """``((y, m, d) or None, is_datetime)`` for a ``pub_date`` value."""
    if value is None:
        return None, False
    text = str(value).strip()
    is_datetime = len(text) > 10 and bool(_ISO_TIME.fullmatch(text[10:]))
    match = _ISO_DATE.fullmatch(text[:10] if is_datetime else text)
    if not match:
        return None, False
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None, False
    return (year, month, day), is_datetime


def parse_gregorian(value: Any) -> Optional[Tuple[int, int, int]]:
    """``(y, m, d)`` for a complete ISO date (or the date part of an ISO
    datetime), else ``None``.

    Deliberately strict: anything that is not exactly ``YYYY-MM-DD`` in ASCII
    digits — a bare year, ``YYYY-MM``, a ``YYYY-MM/YYYY-MM`` range, a signed or
    underscored number, Arabic-Indic digits — has no single day to convert, so
    it yields ``None`` instead of a guess.
    """
    return _split_date(value)[0]


def _convert(year: int, month: int, day: int) -> Tuple[Optional[Tuple[int, int, int]], str]:
    """``(hijri or None, status)``; status is ``converted``, ``outside_tables``
    or ``invalid_date``."""
    from hijridate import Gregorian

    try:
        h = Gregorian(year, month, day).to_hijri()
        return (h.year, h.month, h.day), "converted"
    except OverflowError:
        return None, "outside_tables"
    except ValueError as exc:
        logging.getLogger(__name__).debug(
            "Hijri conversion failed for %04d-%02d-%02d: %s", year, month, day, exc
        )
        return None, "invalid_date"


def to_hijri(year: int, month: int, day: int) -> Optional[Tuple[int, int, int]]:
    """Umm al-Qura ``(year, month, day)``, or ``None``.

    ``None`` means either a date outside hijridate's Umm al-Qura tables
    (:data:`UMM_AL_QURA_RANGE`; ``OverflowError``) or an impossible calendar
    date such as ``1999-02-30`` (``ValueError``); :func:`date_status` tells the
    two apart for the report. The row is skipped instead of failing the run.
    (Same contract as ``to_hijri`` in IwacVisualizations'
    ``scripts/generate_on_this_day.py``.)
    """
    return _convert(year, month, day)[0]


def date_status(value: Any) -> str:
    """Report bucket of one ``pub_date``: ``empty``, ``imprecise``,
    ``invalid_date``, ``outside_tables``, ``converted`` or
    ``converted_datetime``."""
    if value is None or not str(value).strip():
        return "empty"
    parsed, is_datetime = _split_date(value)
    if parsed is None:
        return "imprecise"
    status = _convert(*parsed)[1]
    return "converted_datetime" if status == "converted" and is_datetime else status


def hijridate_version() -> str:
    try:
        return version("hijridate")
    except PackageNotFoundError:
        return "unknown"


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
        if (prior is not None and i < len(prior) and prior[i] is not None
                and all(existing.get(column) is not None and i < len(existing[column])
                        and existing[column][i] is not None for column in HIJRI_COLUMNS)):
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
    statuses = df[SOURCE_COLUMN].map(date_status).value_counts()
    start, end = UMM_AL_QURA_RANGE

    def count(status: str) -> int:
        return int(statuses.get(status, 0))

    coverage = Table(title=f"Hijri coverage — {config_name}", box=box.ROUNDED)
    coverage.add_column("Rows", style="cyan")
    coverage.add_column("Count", style="green", justify="right")
    coverage.add_column("Share", style="dim", justify="right")
    for label, n in (
        ("Total", total),
        ("Complete date (YYYY-MM-DD or ISO datetime)",
         count("converted") + count("converted_datetime") + count("outside_tables") + count("invalid_date")),
        ("Converted to Hijri", converted),
        ("  of which ISO datetime (date part used)", count("converted_datetime")),
        (f"Outside Umm al-Qura tables ({start}..{end})", count("outside_tables")),
        ("Impossible calendar date", count("invalid_date")),
        ("Imprecise date (no lunar day)", count("imprecise")),
        ("No date at all", count("empty")),
    ):
        coverage.add_row(label, f"{n:,}", f"{(n / total * 100):.1f}%" if total else "—")
    console.print(coverage)

    if not converted:
        return

    counts = df["hijri_month"].dropna().astype(int).value_counts().sort_index()
    expected = converted / 12
    peak = counts.max()
    dist = Table(title="Collected items by standardized Hijri month (descriptive counts)", box=box.ROUNDED)
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
    console.print("[dim]Umm al-Qura is a calendar conversion convention, not a locally attested "
                  "observance date. Counts reflect collection coverage and publication dates; "
                  "the even split is descriptive, not an exposure-adjusted baseline.[/dim]")


def main() -> int:
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
    add_update_mode_args(parser)
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
        return 1

    token = ensure_hf_token(console=console)
    config_name = resolve_config(
        args.repo, token=token, cli_config=args.config, restrict_to=HIJRI_SUBSETS, console=console
    )
    console.print(f"[green]→[/green] Subset: [bold]{config_name}[/bold]")
    update_mode = resolve_update_mode(args, console=console)
    console.print(f"[green]✓[/green] Update mode: [cyan]{update_mode}[/cyan]")

    ds = load_hub_dataset(args.repo, config_name, token=token, console=console)
    source_revision = getattr(ds, "_iwac_source_revision", None)

    if SOURCE_COLUMN not in ds.column_names:
        console.print(f"[red]✗[/red] '{SOURCE_COLUMN}' is missing from this subset — nothing to convert.")
        return 1

    already = [c for c in HIJRI_COLUMNS if c in ds.column_names]
    if (already and update_mode == "all" and not (args.yes or args.dry_run)
            and sys.stdin.isatty()):
        console.print(f"\n[yellow]⚠[/yellow] Already present: [bold]{', '.join(already)}[/bold]")
        try:
            if not Confirm.ask("Recompute them for every row?", default=False):
                console.print("[yellow]ℹ[/yellow] Cancelled; existing values kept.")
                return 0
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow] Cancelled.")
            return 0

    from datasets import Value

    before = ds

    # Declared nullable int64: an imprecise date has no lunar day, and that
    # absence stays a null rather than becoming a 0 that charts would plot.
    # Declaring the type (instead of round-tripping the whole subset through
    # pandas) also leaves every other column's type untouched.
    ds = map_with_progress(
        ds,
        lambda batch: add_hijri_batch(batch, update_mode=update_mode),
        description="[cyan]Converting pub_date to Umm al-Qura",
        console=console,
        output_types={col: Value("int64") for col in HIJRI_COLUMNS},
    )
    ds = reorder_columns_after(ds, HIJRI_COLUMNS, SOURCE_COLUMN, console=console)

    report(ds.select_columns([SOURCE_COLUMN, *HIJRI_COLUMNS]).to_pandas(), config_name)
    changed = changed_row_count(before, ds, HIJRI_COLUMNS)
    console.print(f"[blue]→[/blue] Rows whose Hijri date changes: [bold]{changed:,}[/bold]")

    if args.dry_run:
        print_dry_run_panel(
            repo_id=args.repo,
            config_name=config_name,
            n_rows=len(ds),
            extra=f"[dim]Columns: {', '.join(HIJRI_COLUMNS)} · rows that would change: {changed:,}[/dim]",
            console=console,
        )
        return 0
    if not changed:
        console.print("[green]✓[/green] Every Hijri date is already current. Nothing to push.")
        return 0

    ok = push_dataset(
        ds,
        repo_id=args.repo,
        config_name=config_name,
        token=token,
        commit_message=(
            f"Add/update Hijri date columns ({', '.join(HIJRI_COLUMNS)}) "
            f"converted from {SOURCE_COLUMN} with hijridate {hijridate_version()} "
            f"(Umm al-Qura) — config: {config_name}, mode: {update_mode}"
        ),
        console=console,
        expected_revision=source_revision,
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
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
