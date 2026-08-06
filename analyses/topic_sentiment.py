#!/usr/bin/env python3
"""
topic_sentiment.py
==================

Which LDA topics attract which AI sentiment — overall, by country, and
over time — on the `articles` subset.

Per-row sentiment is the consensus of the annotator panel defined in
``iwac_common.sentiment_panel``,
mirroring ``post-processing/sentiment_agreement.py``:

- polarité / centralité : majority label (>= 2 identical votes), else no
  consensus ("Non applicable" still counts as a vote but is excluded from
  ordinal means);
- subjectivité          : median of the available 1-5 scores.

If the dataset already carries ``consensus_*`` columns (pushed by
``sentiment_agreement.py --push``), those are preferred and a note is
printed; otherwise the consensus is computed inline from the model columns.

Rows are kept when they have a real LDA topic (``lda_topic_id`` not NaN and
not the -1 outlier bucket) and at least 2 of the 3 model polarity votes.

Outputs (analyses/output/):
- topic_sentiment_summary.csv     per topic: n, polarity label shares,
                                  mean ordinal polarity, mean subjectivity,
                                  share of Central/Très central rows
- topic_sentiment_by_country.csv  topic x country cells (n, mean polarity)
- topic_sentiment_over_time.csv   topic x year cells (+ decade)

Report-only: this script NEVER pushes anything to the Hub.

Usage
-----
    python analyses/topic_sentiment.py [--source hub|csv]
        [--min-topic-n 50] [--min-cell-n 20] [--min-year-n 10]
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "post-processing"))

from _common import ensure_hf_token, load_subset_dataframe, PRIVATE_REPO_ID  # noqa: E402
# Reuse the canonical sentiment vocabulary + consensus helpers.
from sentiment_agreement import (  # noqa: E402
    CENTRALITY_ORDER,
    MODELS,
    POLARITY_ORDER,
    majority,
    subjectivite_ordinal,
    to_ordinal,
)

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()
OUTPUT_DIR = REPO_ROOT / "analyses" / "output"

POLARITY_COLS = [f"{m}_polarite" for m in MODELS]
CENTRALITY_COLS = [f"{m}_centralite_islam_musulmans" for m in MODELS]
SUBJECTIVITY_COLS = [f"{m}_subjectivite_score" for m in MODELS]

# Centrality labels counted as "central" for the per-topic central share.
CENTRAL_LABELS = {"Central", "Très central"}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_new_analyses.py)
# ---------------------------------------------------------------------------


def slug(label: str) -> str:
    """ASCII snake_case slug for a French label ('Très négatif' -> 'tres_negatif')."""
    norm = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    return "_".join(norm.lower().split())


def consensus_label_series(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Strict-majority label per row across model columns.

    Non-empty strings (including 'Non applicable') count as votes, mirroring
    sentiment_agreement.py — whose ``majority`` this delegates to, so the
    threshold scales with the panel size. Returns '' when no label is held by
    more than half the models that voted.
    """
    present = [c for c in cols if c in df.columns]

    def _row(row: pd.Series) -> str:
        votes = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
        return majority(votes)

    return df[present].apply(_row, axis=1)


def consensus_score_series(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Median of the available subjectivité scores per row (NaN when none).

    Goes through ``subjectivite_ordinal`` rather than ``pd.to_numeric``: since
    generation 2 the column holds a label, and a plain numeric coercion would
    turn every value into NaN and silently report "no data" instead of failing.
    """
    present = [c for c in cols if c in df.columns]
    scores = pd.DataFrame({c: subjectivite_ordinal(df[c]) for c in present})
    return scores.median(axis=1, skipna=True)


def polarity_ordinal(labels: pd.Series) -> pd.Series:
    """Map consensus polarity labels onto the 1-5 scale (NaN for '' / 'Non applicable')."""
    return to_ordinal(labels, POLARITY_ORDER)


def year_from_pub_date(pub_date: pd.Series) -> pd.Series:
    """Year from the YYYY prefix of pub_date; NaN when unparsable/implausible."""
    years = pd.to_numeric(pub_date.astype(str).str.strip().str[:4], errors="coerce")
    return years.where(years.between(1000, 2100))


# ---------------------------------------------------------------------------
# Data loading / consensus resolution
# ---------------------------------------------------------------------------


def load_articles(args: argparse.Namespace) -> pd.DataFrame:
    wanted = (
        ["o:id", "lda_topic_id", "lda_topic_label", "pub_date", "country"]
        + POLARITY_COLS + CENTRALITY_COLS + SUBJECTIVITY_COLS
        + ["consensus_polarite", "consensus_centralite", "consensus_subjectivite_score"]
    )
    columns: Optional[List[str]] = None
    if args.source == "csv":
        # usecols raises on missing columns, so intersect with the mirror header
        # (consensus_* columns are optional).
        csv_path = REPO_ROOT / "data" / f"iwac_{args.config}.csv"
        if csv_path.exists():
            header = pd.read_csv(csv_path, nrows=0).columns
            columns = [c for c in wanted if c in header]
    else:
        columns = wanted  # hub loader intersects with ds.column_names itself

    token = ensure_hf_token(console=console) if args.source == "hub" else None
    return load_subset_dataframe(
        args.repo, args.config, token=token, source=args.source,
        columns=columns, console=console,
    )


def abort(message: str) -> None:
    console.print(f"[red]✗[/red] {message}")
    sys.exit(1)


def resolve_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with pol_label, pol_ord, cent_label, subj_score columns.

    Prefers existing consensus_* columns on the dataset; falls back to
    computing the consensus inline from the 3 model columns.
    """
    out = pd.DataFrame(index=df.index)

    def _existing_label(col: str) -> Optional[pd.Series]:
        if col in df.columns:
            s = df[col].astype("string").str.strip().fillna("")
            if (s != "").any():
                return s.astype(object)
        return None

    # polarité
    pol = _existing_label("consensus_polarite")
    if pol is not None:
        console.print("[yellow]ℹ[/yellow] Using existing [cyan]consensus_polarite[/cyan] column from the dataset.")
    else:
        pol = consensus_label_series(df, POLARITY_COLS)
        console.print("[blue]→[/blue] Computed polarity consensus inline (majority of 3 model votes).")
    out["pol_label"] = pol
    out["pol_ord"] = polarity_ordinal(out["pol_label"])

    # centralité
    cent = _existing_label("consensus_centralite")
    if cent is not None:
        console.print("[yellow]ℹ[/yellow] Using existing [cyan]consensus_centralite[/cyan] column from the dataset.")
    elif sum(c in df.columns for c in CENTRALITY_COLS) >= 2:
        cent = consensus_label_series(df, CENTRALITY_COLS)
        console.print(f"[blue]→[/blue] Computed centrality consensus inline (majority of {len(MODELS)} model votes).")
    else:
        abort("Need consensus_centralite or >= 2 of "
              f"{', '.join(CENTRALITY_COLS)} — none available.")
    out["cent_label"] = cent

    # subjectivité
    if "consensus_subjectivite_score" in df.columns and df["consensus_subjectivite_score"].notna().any():
        console.print("[yellow]ℹ[/yellow] Using existing [cyan]consensus_subjectivite_score[/cyan] column from the dataset.")
        out["subj_score"] = pd.to_numeric(df["consensus_subjectivite_score"], errors="coerce")
    elif sum(c in df.columns for c in SUBJECTIVITY_COLS) >= 2:
        out["subj_score"] = consensus_score_series(df, SUBJECTIVITY_COLS)
        console.print(f"[blue]→[/blue] Computed subjectivity consensus inline (median of {len(MODELS)} model scores).")
    else:
        abort("Need consensus_subjectivite_score or >= 2 of "
              f"{', '.join(SUBJECTIVITY_COLS)} — none available.")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topic x sentiment analysis (consensus of the annotator panel). Report-only."
    )
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", default="articles",
                        help="Subset with lda_topic_id + sentiment columns (articles)")
    parser.add_argument("--source", choices=["hub", "csv"], default="hub",
                        help="hub = live dataset (default); csv = local data/ mirror")
    parser.add_argument("--min-topic-n", type=int, default=50,
                        help="Min scored rows per topic for the ranked tables (default 50)")
    parser.add_argument("--min-cell-n", type=int, default=20,
                        help="Min scored rows per topic x country cell (default 20)")
    parser.add_argument("--min-year-n", type=int, default=10,
                        help="Min scored rows per topic x year cell (default 10)")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Topic x Sentiment[/bold cyan]\n"
        "[dim]Which topics attract which sentiment, where, and when — "
        f"{args.repo} ({args.config})[/dim]",
        border_style="cyan",
    ))

    df = load_articles(args)

    # --- column checks ---
    for col in ("lda_topic_id", "lda_topic_label", "pub_date", "country"):
        if col not in df.columns:
            abort(f"Column '{col}' missing from the {args.config} subset "
                  f"(source={args.source}). Cannot proceed.")
    pol_present = [c for c in POLARITY_COLS if c in df.columns]
    if len(pol_present) < 2:
        abort(f"Need >= 2 of the 3 model polarity columns ({', '.join(POLARITY_COLS)}); "
              f"found {len(pol_present)}.")

    # --- row filter: real topic + >= 2 polarity votes ---
    topic_id = pd.to_numeric(df["lda_topic_id"], errors="coerce")  # float64 on the Hub
    has_topic = topic_id.notna() & (topic_id != -1)

    votes = df[pol_present].apply(
        lambda col: col.map(lambda v: pd.notna(v) and bool(str(v).strip()))
    )
    enough_votes = votes.sum(axis=1) >= 2

    n_total = len(df)
    keep = has_topic & enough_votes
    df = df.loc[keep].copy()
    df["topic_id"] = topic_id.loc[keep].astype(int)
    console.print(
        f"[blue]→[/blue] {len(df):,} / {n_total:,} rows kept "
        f"(topic assigned + >= 2 model polarity votes; "
        f"dropped {int((~has_topic).sum()):,} without topic, "
        f"{int((has_topic & ~enough_votes).sum()):,} without enough votes)"
    )
    if df.empty:
        abort("No rows left after filtering.")

    # --- consensus ---
    cons = resolve_consensus(df)
    df = pd.concat([df, cons], axis=1)
    df["year"] = year_from_pub_date(df["pub_date"])
    df["is_central"] = df["cent_label"].isin(CENTRAL_LABELS)
    df["has_cent"] = df["cent_label"].astype(str).str.strip().astype(bool)

    labels = (
        df.groupby("topic_id")["lda_topic_label"]
        .agg(lambda s: next((str(v) for v in s if pd.notna(v) and str(v).strip()), ""))
    )

    # --- (a) per-topic summary ---
    pol_labels_order = list(POLARITY_ORDER)  # 5 scale labels, in order
    rows = []
    for tid, g in df.groupby("topic_id"):
        n = len(g)
        counts = g["pol_label"].value_counts()
        row: Dict[str, object] = {
            "lda_topic_id": tid,
            "label": labels.get(tid, ""),
            "n": n,
            "n_polarity_scored": int(g["pol_ord"].notna().sum()),
        }
        for lab in pol_labels_order:
            row[f"share_{slug(lab)}"] = counts.get(lab, 0) / n
        row["share_non_applicable"] = counts.get("Non applicable", 0) / n
        row["share_no_consensus"] = counts.get("", 0) / n
        row["mean_polarity"] = g["pol_ord"].mean()
        row["mean_subjectivity"] = g["subj_score"].mean()
        n_cent = int(g["has_cent"].sum())
        row["central_share"] = (g["is_central"].sum() / n_cent) if n_cent else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("lda_topic_id").reset_index(drop=True)

    # --- (b) topic x country ---
    by_country = (
        df[df["country"].notna() & df["country"].astype(str).str.strip().astype(bool)]
        .assign(country=lambda x: x["country"].astype(str).str.strip())
        .groupby(["topic_id", "country"])["pol_ord"]
        .agg(n="count", mean_polarity="mean")
        .reset_index()
    )
    by_country = by_country[by_country["n"] >= args.min_cell_n]
    by_country.insert(1, "label", by_country["topic_id"].map(labels))
    by_country = by_country.rename(columns={"topic_id": "lda_topic_id"})

    # --- (c) topic x year ---
    over_time = (
        df[df["year"].notna()]
        .assign(year=lambda x: x["year"].astype(int))
        .groupby(["topic_id", "year"])["pol_ord"]
        .agg(n="count", mean_polarity="mean")
        .reset_index()
    )
    over_time = over_time[over_time["n"] >= args.min_year_n]
    over_time["decade"] = (over_time["year"] // 10) * 10
    over_time.insert(1, "label", over_time["topic_id"].map(labels))
    over_time = over_time.rename(columns={"topic_id": "lda_topic_id"})

    # --- write outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_summary = OUTPUT_DIR / "topic_sentiment_summary.csv"
    out_country = OUTPUT_DIR / "topic_sentiment_by_country.csv"
    out_time = OUTPUT_DIR / "topic_sentiment_over_time.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8")
    by_country.to_csv(out_country, index=False, encoding="utf-8")
    over_time.to_csv(out_time, index=False, encoding="utf-8")

    # --- report ---
    def topic_table(title: str, frame: pd.DataFrame, style: str) -> Table:
        t = Table(title=title, box=box.ROUNDED, border_style=style)
        t.add_column("Topic", style="cyan", justify="right")
        t.add_column("Label", style="green", max_width=48)
        t.add_column("n", justify="right")
        t.add_column("Mean pol.", justify="right")
        t.add_column("% nég.", justify="right")
        t.add_column("% pos.", justify="right")
        t.add_column("Subj.", justify="right")
        t.add_column("Central", justify="right")
        for _, r in frame.iterrows():
            neg = r["share_tres_negatif"] + r["share_negatif"]
            pos = r["share_tres_positif"] + r["share_positif"]
            t.add_row(
                str(int(r["lda_topic_id"])), str(r["label"]), f"{int(r['n']):,}",
                f"{r['mean_polarity']:.2f}", f"{neg:.1%}", f"{pos:.1%}",
                f"{r['mean_subjectivity']:.2f}",
                f"{r['central_share']:.1%}" if pd.notna(r["central_share"]) else "n/a",
            )
        return t

    ranked = summary[summary["n_polarity_scored"] >= args.min_topic_n].dropna(subset=["mean_polarity"])
    console.print()
    console.print(topic_table(
        f"Most negative topics (mean ordinal polarity, n >= {args.min_topic_n})",
        ranked.nsmallest(10, "mean_polarity"), "red",
    ))
    console.print(topic_table(
        f"Most positive topics (mean ordinal polarity, n >= {args.min_topic_n})",
        ranked.nlargest(10, "mean_polarity"), "green",
    ))

    console.print(Panel(
        f"Topics: [bold]{len(summary):,}[/bold]  |  "
        f"topic x country cells kept (n >= {args.min_cell_n}): [bold]{len(by_country):,}[/bold]  |  "
        f"topic x year cells kept (n >= {args.min_year_n}): [bold]{len(over_time):,}[/bold]\n"
        f"[green]✓[/green] {out_summary}\n"
        f"[green]✓[/green] {out_country}\n"
        f"[green]✓[/green] {out_time}",
        title="Outputs", border_style="blue",
    ))
    console.print("[yellow]ℹ[/yellow] Report-only script — nothing is pushed to the Hub.")


if __name__ == "__main__":
    main()
