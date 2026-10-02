#!/usr/bin/env python3
"""
topic_sentiment.py
==================

Which LDA topics attract which AI sentiment — overall, by country, and
over time — on the `articles` subset.

Per-row sentiment is the consensus of the annotator panel defined in
``iwac_common.sentiment_panel``,
mirroring ``post-processing/sentiment_agreement.py``:

- polarité / centralité : strict-majority label (at least 2 voters), else no
  consensus ("Non applicable" still counts as a vote but is excluded from
  ordinal means);
- subjectivité          : median of at least 2 available 1-5 scores.

Consensus is always derived from the selected generation's raw model columns.
Stored unidentified consensus values are never reused. Every real topic row is
retained for missingness denominators. Topics are keyed by immutable model identity
and topic ID; category shares and ordinal medians are the default summaries.

Outputs (analyses/output/):
- topic_sentiment_summary.csv     per topic: n, polarity label shares,
                                  ordinal medians, missingness counts,
                                  share of Central/Très central rows
- topic_sentiment_by_country.csv  topic x country cells (shares, medians, missingness)
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
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "post-processing"))

try:
    from iwac_pipeline.processing._common import (  # noqa: E402
        PRIVATE_REPO_ID,
        ensure_hf_token,
        load_subset_dataframe,
        write_run_manifest,
    )
except ModuleNotFoundError:  # source scripts before editable installation
    from _common import (  # noqa: E402
        PRIVATE_REPO_ID,
        ensure_hf_token,
        load_subset_dataframe,
        write_run_manifest,
    )

# Reuse the canonical sentiment vocabulary + consensus helpers.
try:
    from iwac_pipeline.processing.sentiment_agreement import (  # noqa: E402
        MODELS,
        POLARITY_ORDER,
        majority,
        models_for,
        CENTRALITY_ORDER,
        subjectivite_ordinal,
        to_ordinal,
    )
except ModuleNotFoundError:  # source scripts before editable installation
    from sentiment_agreement import (  # noqa: E402
        MODELS,
        POLARITY_ORDER,
        majority,
        models_for,
        CENTRALITY_ORDER,
        subjectivite_ordinal,
        to_ordinal,
    )


from iwac_common.sentiment_panel import PANEL, instrument_id

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()
from iwac_common.paths import workspace_root

OUTPUT_DIR = workspace_root() / "analyses" / "output"

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

    if not present:
        return pd.Series("", index=df.index, dtype=object)
    return df[present].apply(_row, axis=1)


def consensus_score_series(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Median of the available subjectivité scores per row (NaN when none).

    Goes through ``subjectivite_ordinal`` rather than ``pd.to_numeric``: since
    generation 2 the column holds a label, and a plain numeric coercion would
    turn every value into NaN and silently report "no data" instead of failing.
    """
    present = [c for c in cols if c in df.columns]
    scores = pd.DataFrame({c: subjectivite_ordinal(df[c]) for c in present}, index=df.index)
    return scores.median(axis=1, skipna=True).where(scores.notna().sum(axis=1) >= 2)


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
    models = models_for(args.generation)
    wanted = (["o:id", "lda_model_name", "lda_topic_id", "lda_topic_label", "pub_date", "country"]
              + [f"{m}_{suffix}" for m in models for suffix in
                 ("polarite", "centralite_islam_musulmans", "subjectivite_score")])
    token = ensure_hf_token(console=console) if args.source == "hub" else None
    return load_subset_dataframe(args.repo, args.config, token=token, source=args.source,
                                 columns=wanted, console=console)


def abort(message: str) -> None:
    console.print(f"[red]✗[/red] {message}")
    raise SystemExit(1)


def resolve_consensus(df: pd.DataFrame, generation: int | None = None) -> pd.DataFrame:
    """Recompute from the selected instrument, never reuse unidentified consensus.

    Keep insufficient-coverage rows so their missingness stays visible in every
    denominator. Invalid labels are missing, not votes for an invented category.
    """
    generation = generation if generation is not None else max(m.generation for m in PANEL)
    models = models_for(generation)
    out = pd.DataFrame(index=df.index)
    for key, suffix, mapping in (("pol", "polarite", POLARITY_ORDER),
                                  ("cent", "centralite_islam_musulmans", CENTRALITY_ORDER)):
        cols = [f"{m}_{suffix}" for m in models]
        votes = df.reindex(columns=cols).astype("string").apply(lambda col: col.str.strip())
        votes = votes.where(votes.isin([*mapping, "Non applicable"]))
        out[f"{key}_n_votes"] = votes.notna().sum(axis=1)
        out[f"{key}_label"] = consensus_label_series(votes, cols)
    cols = [f"{m}_subjectivite_score" for m in models]
    scores = pd.DataFrame({c: subjectivite_ordinal(df[c]) for c in cols if c in df}, index=df.index)
    out["subj_n_votes"] = scores.notna().sum(axis=1)
    out["subj_score"] = scores.median(axis=1).where(out["subj_n_votes"] >= 2)
    out["pol_ord"] = polarity_ordinal(out["pol_label"])
    out["sentiment_generation"] = generation
    out["sentiment_instrument_id"] = instrument_id(generation)
    return out


def prepare_articles(df: pd.DataFrame, generation: int) -> pd.DataFrame:
    """Retain every valid topic assignment; never merge numbered topics across models."""
    topic = pd.to_numeric(df["lda_topic_id"], errors="coerce")
    valid = topic.notna() & topic.ge(0) & topic.mod(1).eq(0)
    out = df.loc[valid].copy()
    if "lda_model_name" not in out or out["lda_model_name"].astype("string").fillna("").str.strip().eq("").any():
        raise ValueError("Every topic assignment needs lda_model_name; regenerate predictions using a saved model bundle.")
    out["lda_model_name"] = out["lda_model_name"].astype(str).str.strip()
    out["lda_topic_id"] = topic.loc[valid].astype(int)
    out = pd.concat([out, resolve_consensus(out, generation)], axis=1)
    out["year"] = year_from_pub_date(out["pub_date"])
    out["country"] = out["country"].astype("string").str.strip().replace("", pd.NA)
    return out


def summarize_cells(df: pd.DataFrame, dimensions: List[str], *, ordinal_means=False) -> pd.DataFrame:
    """Category shares use all assigned articles; medians use applicable consensus.

    ``n`` is the complete cell, including insufficient votes and disagreement.
    Means are available only as an explicit equal-spacing sensitivity summary.
    """
    keys = ["lda_model_name", "lda_topic_id", *dimensions]
    rows = []
    for values, group in df.groupby(keys, dropna=False):
        row = dict(zip(keys, values))
        n = len(group)
        row.update({
            "label": next((str(x) for x in group["lda_topic_label"] if pd.notna(x) and str(x).strip()), ""),
            "n": n,
            "sentiment_generation": int(group["sentiment_generation"].iloc[0]),
            "sentiment_instrument_id": group["sentiment_instrument_id"].iloc[0],
            "n_polarity_scored": int(group["pol_ord"].notna().sum()),
            "n_subjectivity_scored": int(group["subj_score"].notna().sum()),
            "n_polarity_insufficient_votes": int(group["pol_n_votes"].lt(2).sum()),
            "n_centrality_insufficient_votes": int(group["cent_n_votes"].lt(2).sum()),
            "n_subjectivity_insufficient_votes": int(group["subj_n_votes"].lt(2).sum()),
            "n_polarity_no_consensus": int((group["pol_n_votes"].ge(2) & group["pol_label"].eq("")).sum()),
            "n_centrality_no_consensus": int((group["cent_n_votes"].ge(2) & group["cent_label"].eq("")).sum()),
            "median_polarity": group["pol_ord"].median(),
            "median_subjectivity": group["subj_score"].median(),
        })
        for label in POLARITY_ORDER:
            row[f"share_{slug(label)}"] = group["pol_label"].eq(label).sum() / n
        row["share_non_applicable"] = group["pol_label"].eq("Non applicable").sum() / n
        row["share_no_consensus"] = row["n_polarity_no_consensus"] / n
        row["share_insufficient_votes"] = row["n_polarity_insufficient_votes"] / n
        for label in CENTRALITY_ORDER:
            row[f"share_centrality_{slug(label)}"] = group["cent_label"].eq(label).sum() / n
        for rank in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
            row[f"share_subjectivity_{str(rank).replace('.', '_')}"] = group["subj_score"].eq(rank).sum() / n
        row["central_share"] = group["cent_label"].isin(CENTRAL_LABELS).sum() / n
        if ordinal_means:
            row["mean_polarity_equal_spacing"] = group["pol_ord"].mean()
            row["mean_subjectivity_equal_spacing"] = group["subj_score"].mean()
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[*keys, "n", "n_polarity_scored"])


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
    parser.add_argument("--generation", type=int, choices=sorted({m.generation for m in PANEL}),
                        default=max(m.generation for m in PANEL), help="One annotation instrument; newest by default")
    parser.add_argument("--ordinal-means", action="store_true",
                        help="Also report means as an explicit equal-spacing sensitivity summary")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Topic x Sentiment[/bold cyan]\n"
        "[dim]Which topics attract which sentiment, where, and when — "
        f"{args.repo} ({args.config})[/dim]",
        border_style="cyan",
    ))

    df = load_articles(args)
    source_revision = df.attrs.get("iwac_source_revision")

    for col in ("lda_topic_id", "lda_topic_label", "pub_date", "country"):
        if col not in df.columns:
            abort(f"Missing required column: {col}")
    try:
        df = prepare_articles(df, args.generation)
    except ValueError as exc:
        abort(str(exc))
    if df.empty:
        abort("No valid topic assignments.")
    legacy = ~df["lda_model_name"].str.startswith(("lda-sha256:", "lda-legacy-sha256:"))
    if legacy.any():
        abort("Mutable legacy lda_model_name detected; regenerate predictions from a verified model bundle before comparing topics.")
    summary = summarize_cells(df, [], ordinal_means=args.ordinal_means)
    by_country = summarize_cells(df, ["country"], ordinal_means=args.ordinal_means)
    by_country = by_country[by_country["n_polarity_scored"] >= args.min_cell_n]
    over_time = summarize_cells(df[df["year"].notna()], ["year"], ordinal_means=args.ordinal_means)
    over_time = over_time[over_time["n_polarity_scored"] >= args.min_year_n]
    over_time["decade"] = (over_time["year"] // 10) * 10

    # --- write outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_summary = OUTPUT_DIR / "topic_sentiment_summary.csv"
    out_country = OUTPUT_DIR / "topic_sentiment_by_country.csv"
    out_time = OUTPUT_DIR / "topic_sentiment_over_time.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8")
    by_country.to_csv(out_country, index=False, encoding="utf-8")
    over_time.to_csv(out_time, index=False, encoding="utf-8")

    table = Table(title="Topic sentiment (conditional on this corpus and instrument)", box=box.ROUNDED)
    for heading in ("Model / topic", "Label", "All articles", "Scored", "Median polarity"):
        table.add_column(heading)
    ranked = summary[summary["n_polarity_scored"] >= args.min_topic_n]
    for row in ranked.head(25).to_dict("records"):
        table.add_row(f"{row['lda_model_name'][-12:]} / {row['lda_topic_id']}", str(row["label"]),
                      str(row["n"]), str(row["n_polarity_scored"]), str(row["median_polarity"]))
    console.print(table)
    console.print("[yellow]Shares include missing and disputed rows in the denominator. "
                  "Ordinal medians do not assume equal spacing; model agreement is not human validation.[/yellow]")
    write_run_manifest(
        OUTPUT_DIR, script="topic_sentiment", repo_id=args.repo,
        revision=source_revision, args=args,
        outputs=[out_summary, out_country, out_time],
    )
    console.print("[yellow]ℹ[/yellow] Report-only script — nothing is pushed to the Hub.")


if __name__ == "__main__":
    main()
