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

A long-text heuristic guards against NEW full-text columns slipping through:
any unexpected string column with prose-length values aborts the push unless
it is a known content column or allow-listed here.

Usage
-----
    # Preview what would change, push nothing
    python post-processing/publish_public.py --dry-run

    # Publish all subsets (asks for confirmation)
    python post-processing/publish_public.py

    # Publish selected subsets non-interactively
    python post-processing/publish_public.py --configs articles,references -y

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
from iwac_common.repos import CONTENT_COLUMNS  # noqa: E402

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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
    "abstract",
    "Description",
    "lda_topic_label",
    "gemini_centralite_justification",
    "gemini_polarite_justification",
    "gemini_subjectivite_justification",
    "chatgpt_centralite_justification",
    "chatgpt_polarite_justification",
    "chatgpt_subjectivite_justification",
    "mistral_centralite_justification",
    "mistral_polarite_justification",
    "mistral_subjectivite_justification",
}

# Heuristic thresholds for "this looks like full text": mean length of
# non-empty values, and a hard per-value ceiling.
SUSPECT_MEAN_CHARS = 3_000
SUSPECT_MAX_CHARS = 30_000


def find_suspect_columns(df, handled: set[str]) -> list[tuple[str, int, int]]:
    """Return (column, mean_len, max_len) for prose-length string columns
    that are not already handled (content columns, masked per-row) nor
    allow-listed — i.e. NEW full-text columns that need classifying."""
    suspects = []
    for col in df.columns:
        if col in PUBLIC_TEXT_ALLOWLIST or col in handled:
            continue
        if df[col].dtype != object:
            continue
        vals = df[col].dropna()
        vals = vals[vals.map(lambda v: isinstance(v, str))]
        vals = vals[vals.str.strip().ne("")]
        if vals.empty:
            continue
        lengths = vals.str.len()
        mean_len, max_len = int(lengths.mean()), int(lengths.max())
        if mean_len > SUSPECT_MEAN_CHARS or max_len > SUSPECT_MAX_CHARS:
            suspects.append((col, mean_len, max_len))
    return suspects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the public projection (private full mirror minus private columns)."
    )
    parser.add_argument("--repo-private", default=PRIVATE_REPO_ID)
    parser.add_argument("--repo-public", default=PUBLIC_REPO_ID)
    parser.add_argument(
        "--configs",
        default=",".join(ALL_CONFIGS),
        help=f"Comma-separated subsets to publish (default: {','.join(ALL_CONFIGS)})",
    )
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--dry-run", action="store_true", help="Report only; push nothing")
    parser.add_argument(
        "--squash",
        action="store_true",
        help="After publishing, super-squash the PUBLIC repo history into one commit "
             "(purges previously public OCR/lemma files from old revisions; irreversible).",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
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

    plans = []
    for cfg in configs:
        with console.status(f"[bold green]Loading '{cfg}' from {args.repo_private}...", spinner="dots"):
            ds = load_dataset(args.repo_private, name=cfg, split="train", token=token)
        public_df = ds.to_pandas()

        content_cols = [c for c in CONTENT_COLUMNS.get(cfg, []) if c in public_df.columns]
        kept, blanked = 0, 0
        if content_cols:
            if "OCR_is_public" not in public_df.columns:
                console.print(
                    f"[red]✗[/red] [bold]{cfg}[/bold] has content columns "
                    f"{content_cols} but no 'OCR_is_public' flag. Re-run the upload "
                    f"script (or the backfill) so full text can be masked per row. Aborting."
                )
                sys.exit(1)
            is_public = public_df["OCR_is_public"].fillna(False).astype(bool)
            private_mask = ~is_public
            kept, blanked = int(is_public.sum()), int(private_mask.sum())
            for col in content_cols:
                # Blank the full text (and its lemmas) only for private-content rows.
                public_df.loc[private_mask, col] = ""

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
