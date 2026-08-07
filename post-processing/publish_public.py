#!/usr/bin/env python3
"""
publish_public.py
=================

Project the PRIVATE full mirror (fmadore/islam-west-africa-collection-full)
onto the PUBLIC dataset (fmadore/islam-west-africa-collection), masking full
text PER ROW according to whether it is public on the Omeka S source.

This script is the ONLY writer to the public repo. Upload and post-processing
scripts all target the private repo; run this afterwards to refresh the
public projection.

Full text is NOT stripped wholesale. The content columns (see
iwac_common/repos.py CONTENT_COLUMNS: OCR + lemma_text + lemma_nostop) are
blanked ONLY for rows whose ``OCR_is_public`` flag is False — i.e. items
whose ``bibo:content`` is private on Omeka. Items whose full text is already
public on the source (verified: ~61% of articles, ~89% of publications,
25/26 documents, 7/867 references) keep their OCR and lemmas. Lemmas derive
from OCR, so they follow the same per-row mask.

A subset that has content columns but no ``OCR_is_public`` flag aborts the
push (run the upload scripts / backfill first) — never silently leaks.

What always stays public: metadata, embeddings (not invertible to text), LDA
topic columns, AI sentiment (scores + justifications), descriptionAI,
abstract, tableOfContents, lexical metrics, word counts, and the
``OCR_is_public`` flag itself.

Two guards protect against NEW full-text columns slipping through:

1. An explicit per-subset column allowlist (iwac_common/public_columns.json):
   ANY column not listed there aborts the push. Approve reviewed columns with
   --approve-columns (persisted to the JSON; commit the change).
2. A prose-length heuristic as a second layer: an unexpected string/list
   column with prose-length values aborts the push unless it is a known
   content column or allow-listed here.

Both run BEFORE the push. A third check runs after it: ``sync_card_features``
(iwac_common/card_sync.py) verifies that the dataset card declares the schema
that was actually pushed, because ``push_to_hub`` refreshes the card's byte sizes
but not its feature list — which on 2026-08-06 left this very dataset raising
``CastError`` on ``load_dataset``. It repairs the card rather than aborting: the
push has already landed by then, so stopping would leave the citable dataset
broken.

Usage
-----
    # Preview what would change, push nothing
    python post-processing/publish_public.py --dry-run

    # Publish all subsets (asks for confirmation)
    python post-processing/publish_public.py

    # Publish selected subsets non-interactively
    python post-processing/publish_public.py --config articles,references -y

    # One-time: squash the public repo's git history so previously public
    # OCR/lemma parquet files disappear from old revisions. DESTRUCTIVE to
    # history (downloads pinned to old revisions break). Run once after the
    # first stripped publish.
    python post-processing/publish_public.py --squash -y
"""
from __future__ import annotations

import argparse
import os
import sys

# Make ``post-processing/_common.py`` importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ensure_hf_token, PRIVATE_REPO_ID, PUBLIC_REPO_ID  # noqa: E402
from iwac_common.card_sync import CardSchemaError, sync_card_features  # noqa: E402
from iwac_common.repos import (  # noqa: E402
    CONTENT_COLUMNS,
    PUBLIC_COLUMNS_FILE,
    load_public_columns,
)
from iwac_common.sentiment_panel import all_justification_columns  # noqa: E402

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd  # noqa: E402
from datasets import load_dataset  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.prompt import Confirm  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()

ALL_CONFIGS = ["articles", "publications", "index", "references", "audiovisual", "documents", "images"]

# Long-text columns that are legitimately public (shown on the public Omeka
# pages or derived commentary, not the protected full text).
PUBLIC_TEXT_ALLOWLIST = {
    "tableOfContents",
    "descriptionAI",
    "descriptionAI_en",
    "abstract",
    "Description",
    "lda_topic_label",
    # Sentiment justifications, for every panel member including frozen ones.
    # Derived rather than listed so rotating the panel cannot leave a new
    # model's justifications tripping the prose guard.
    *all_justification_columns(),
}

# Heuristic thresholds for "this looks like full text": mean length of
# non-empty values, and a hard per-value ceiling.
SUSPECT_MEAN_CHARS = 3_000
SUSPECT_MAX_CHARS = 30_000


def _value_text_len(v) -> int:
    """Character length of a value for the prose heuristic: strings count
    directly; lists/tuples of strings count their joined length (a full text
    stored as list[str] chunks must not evade the guard)."""
    if isinstance(v, str):
        return len(v)
    if isinstance(v, (list, tuple)):
        return sum(len(x) for x in v if isinstance(x, str))
    return 0


def find_suspect_columns(df, handled: set[str]) -> list[tuple[str, int, int]]:
    """Return (column, mean_len, max_len) for prose-length columns that are
    not already handled (content columns, masked per-row) nor allow-listed —
    i.e. NEW full-text columns that need classifying.

    Second defense layer behind the explicit column allowlist: scans object
    AND pandas-StringDtype columns, and unwraps list values, so chunked or
    typed text columns can't slip past on a technicality."""
    from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype

    suspects = []
    for col in df.columns:
        if col in PUBLIC_TEXT_ALLOWLIST or col in handled:
            continue
        dtype = df[col].dtype
        if is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_datetime64_any_dtype(dtype):
            continue
        lengths = df[col].dropna().map(_value_text_len)
        lengths = lengths[lengths > 0]
        if lengths.empty:
            continue
        mean_len, max_len = int(lengths.mean()), int(lengths.max())
        if mean_len > SUSPECT_MEAN_CHARS or max_len > SUSPECT_MAX_CHARS:
            suspects.append((col, mean_len, max_len))
    return suspects


class MissingFlagError(RuntimeError):
    """A subset has content columns but no ``OCR_is_public`` flag — masking
    is impossible, so publishing must abort rather than leak."""


def mask_content_columns(public_df, cfg: str) -> tuple[list[str], int, int]:
    """Blank the content columns of ``public_df`` IN PLACE for every row
    whose ``OCR_is_public`` flag is not truthy (null → private, conservative).

    Returns ``(content_cols, kept, blanked)``. Raises
    :class:`MissingFlagError` if content columns exist without the flag.
    This is the privacy boundary — covered by tests/test_publish_public.py.
    """
    content_cols = [c for c in CONTENT_COLUMNS.get(cfg, []) if c in public_df.columns]
    if not content_cols:
        return [], 0, 0
    if "OCR_is_public" not in public_df.columns:
        raise MissingFlagError(
            f"'{cfg}' has content columns {content_cols} but no 'OCR_is_public' "
            f"flag; full text cannot be masked per row."
        )
    is_public = public_df["OCR_is_public"].map(lambda v: bool(v) if pd.notna(v) else False)
    private_mask = ~is_public
    for col in content_cols:
        # Blank the full text (and its lemmas) only for private-content rows.
        public_df.loc[private_mask, col] = ""
    return content_cols, int(is_public.sum()), int(private_mask.sum())


def check_column_allowlist(cfg: str, df, approve: set[str]) -> list[str]:
    """Primary guard: every column must be explicitly allow-listed in
    ``iwac_common/public_columns.json`` before it reaches the public repo.

    Returns the list of columns newly approved this run (already persisted).
    Aborts on any unknown column that was not passed via --approve-columns.
    """
    import json

    allowlist = load_public_columns()
    if cfg not in allowlist:
        console.print(
            f"[red]✗[/red] Subset [bold]{cfg}[/bold] has no entry in "
            f"{PUBLIC_COLUMNS_FILE}. Add one (see the file's _readme). Aborting."
        )
        sys.exit(1)

    unknown = [c for c in df.columns if c not in allowlist[cfg]]
    if not unknown:
        return []

    to_approve = [c for c in unknown if c in approve]
    still_unknown = [c for c in unknown if c not in approve]

    if still_unknown:
        console.print(
            f"[red]✗[/red] [bold]{cfg}[/bold]: column(s) not in the public "
            f"allowlist: [red]{', '.join(still_unknown)}[/red]"
        )
        for col, mean_len, max_len in find_suspect_columns(df[still_unknown], handled=set()):
            console.print(
                f"    [red]{col}[/red] looks like prose (mean {mean_len:,} chars, "
                f"max {max_len:,}) — if it can hold private full text, add it to "
                f"CONTENT_COLUMNS instead of the allowlist!"
            )
        console.print(
            "    Review each column, then either re-run with "
            "[cyan]--approve-columns col1,col2[/cyan] or edit "
            f"{PUBLIC_COLUMNS_FILE} directly. Aborting — nothing pushed."
        )
        sys.exit(1)

    # Persist the explicitly approved columns.
    with open(PUBLIC_COLUMNS_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    raw[cfg] = sorted(set(raw[cfg]) | set(to_approve))
    with open(PUBLIC_COLUMNS_FILE, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
        f.write("\n")
    console.print(
        f"[yellow]⚠[/yellow] [bold]{cfg}[/bold]: approved new public column(s) "
        f"{', '.join(to_approve)} (recorded in {os.path.basename(PUBLIC_COLUMNS_FILE)} "
        f"— commit this change)."
    )
    return to_approve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the public projection (private full mirror minus private columns)."
    )
    parser.add_argument("--repo-private", default=PRIVATE_REPO_ID)
    parser.add_argument("--repo-public", default=PUBLIC_REPO_ID)
    parser.add_argument(
        "--config",
        dest="config",
        default=",".join(ALL_CONFIGS),
        help=f"Comma-separated subsets to publish (default: {','.join(ALL_CONFIGS)})",
    )
    # Deprecated alias (pre-2026-07 invocations used --configs); hidden from --help.
    parser.add_argument("--configs", dest="config", help=argparse.SUPPRESS)
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--dry-run", action="store_true", help="Report only; push nothing")
    parser.add_argument(
        "--approve-columns",
        default="",
        help="Comma-separated column names to add to the public allowlist "
             "(iwac_common/public_columns.json) after review. Without this, any "
             "column not already allow-listed aborts the push.",
    )
    parser.add_argument(
        "--squash",
        action="store_true",
        help="After publishing, super-squash the PUBLIC repo history into one commit "
             "(purges previously public OCR/lemma files from old revisions; irreversible).",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()

    if any(a == "--configs" or a.startswith("--configs=") for a in sys.argv[1:]):
        console.print("[yellow]⚠[/yellow] --configs is deprecated; use --config.")

    configs = [c.strip() for c in args.config.split(",") if c.strip()]
    unknown = [c for c in configs if c not in ALL_CONFIGS]
    if unknown:
        console.print(f"[red]✗[/red] Unknown config(s): {unknown}. Valid: {ALL_CONFIGS}")
        sys.exit(1)

    token = ensure_hf_token(console=console)

    console.print(Panel(
        f"[bold]Source (private):[/bold] {args.repo_private}\n"
        f"[bold]Target (public):[/bold] {args.repo_public}\n"
        f"[bold]Subsets:[/bold] {', '.join(configs)}\n"
        f"[bold]Mode:[/bold] {'[yellow]dry-run[/yellow]' if args.dry_run else '[green]publish[/green]'}",
        title="[bold blue]Public Projection", border_style="blue",
    ))

    approve = {c.strip() for c in args.approve_columns.split(",") if c.strip()}

    plans = []
    for cfg in configs:
        with console.status(f"[bold green]Loading '{cfg}' from {args.repo_private}...", spinner="dots"):
            ds = load_dataset(args.repo_private, name=cfg, split="train", token=token)
        public_df = ds.to_pandas()

        # Primary guard: every column must be explicitly allow-listed.
        check_column_allowlist(cfg, public_df, approve)

        try:
            content_cols, kept, blanked = mask_content_columns(public_df, cfg)
        except MissingFlagError as exc:
            console.print(
                f"[red]✗[/red] {exc} Re-run the upload script (or the backfill) "
                f"so full text can be masked per row. Aborting."
            )
            sys.exit(1)

        # Guard: the content columns are handled (masked); flag any OTHER
        # prose-length column that we have not classified.
        suspects = find_suspect_columns(public_df, handled=set(content_cols))
        if suspects:
            console.print(f"[red]✗[/red] [bold]{cfg}[/bold]: unclassified prose-length column(s):")
            for col, mean_len, max_len in suspects:
                console.print(f"    [red]{col}[/red] (mean {mean_len:,} chars, max {max_len:,})")
            console.print("    Add them to CONTENT_COLUMNS (iwac_common/repos.py) or "
                          "PUBLIC_TEXT_ALLOWLIST (this script), then re-run.")
            sys.exit(1)

        plans.append((cfg, content_cols, kept, blanked, public_df))

    table = Table(title="Projection Plan (full text masked per row by OCR_is_public)", box=box.ROUNDED)
    table.add_column("Subset", style="cyan")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Content cols", style="yellow")
    table.add_column("OCR kept", justify="right", style="green")
    table.add_column("OCR blanked", justify="right", style="red")
    table.add_column("Public cols", justify="right", style="blue")
    for cfg, content_cols, kept, blanked, public_df in plans:
        table.add_row(
            cfg, f"{len(public_df):,}", ", ".join(content_cols) or "—",
            f"{kept:,}" if content_cols else "—",
            f"{blanked:,}" if content_cols else "—",
            str(len(public_df.columns)),
        )
    console.print(table)

    if args.dry_run:
        console.print(Panel("[yellow]Dry run — nothing pushed.[/yellow]", border_style="yellow"))
        return

    if not args.yes and not Confirm.ask(
        f"Push {len(plans)} subset(s) to [cyan]{args.repo_public}[/cyan]?", default=False
    ):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    import numpy as np
    from datasets import Dataset

    for cfg, content_cols, kept, blanked, public_df in plans:
        # to_pandas returns embeddings as np.ndarray; from_pandas infers list
        # types more reliably from plain Python lists (nulls stay null).
        for col in public_df.columns:
            if col.startswith("embedding"):
                public_df[col] = public_df[col].map(
                    lambda v: v.tolist() if isinstance(v, np.ndarray) else v
                )
        with console.status(f"[bold green]Pushing '{cfg}' to {args.repo_public}...", spinner="dots"):
            pub_ds = Dataset.from_pandas(public_df, preserve_index=False)
            pub_ds.push_to_hub(
                args.repo_public,
                config_name=cfg,
                token=token,
                max_shard_size=args.max_shard_size,
                commit_message=(
                    f"Public projection of '{cfg}' from private mirror"
                    + (f" ({blanked:,} private-content rows masked)" if content_cols else "")
                ),
            )
        console.print(f"[green]✓[/green] {cfg}: {len(public_df):,} rows, "
                      f"{len(public_df.columns)} cols pushed"
                      + (f" — OCR kept for {kept:,}, blanked {blanked:,}" if content_cols else ""))

        # push_to_hub refreshes the card's byte sizes but not its feature list, so
        # a schema change leaves this subset raising CastError on load. That is
        # worse here than on the private mirror: this is the citable dataset.
        try:
            sync_card_features(
                args.repo_public, cfg, token=token, console=console,
                expected_columns=list(public_df.columns),
            )
        except CardSchemaError as exc:
            console.print(Panel(
                f"[bold red]✗ '{cfg}' pushed, but the card is out of step[/bold red]\n\n"
                f"{exc}\n\n"
                f"The rows ARE public — the declared schema is not, so "
                f"load_dataset('{args.repo_public}', name='{cfg}') raises CastError "
                f"until the card's dataset_info is corrected. Fix the card; do not "
                f"re-run the projection.",
                title="Card schema mismatch", border_style="red",
            ))
            sys.exit(1)

    if args.squash:
        console.print(Panel(
            "[bold yellow]About to super-squash the PUBLIC repo history.[/bold yellow]\n"
            "All previous revisions (including ones still containing OCR/lemma parquet "
            "files) become unreachable. Pinned downloads to old commits will break.\n"
            "This cannot be undone.",
            title="[red]History purge[/red]", border_style="red",
        ))
        if args.yes or Confirm.ask("Proceed with super_squash_history?", default=False):
            api = HfApi(token=token)
            api.super_squash_history(repo_id=args.repo_public, repo_type="dataset")
            console.print(f"[green]✓[/green] History of {args.repo_public} squashed to a single commit.")
        else:
            console.print("[yellow]Squash skipped.[/yellow]")

    console.print("[green]✓[/green] Done!")


if __name__ == "__main__":
    main()
