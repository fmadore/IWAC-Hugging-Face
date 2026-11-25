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

Counts and denominators (every table):
- ``n``                       every assigned row in the cell; denominator of
                              the plain ``share_*`` columns.
- ``n_polarity_consensus``    rows whose polarity panel reached a consensus
                              label, "Non applicable" included; denominator
                              of ``share_<polarity>_of_scored``. ``n_scored``
                              is the same count under its historical name.
- ``n_polarity_scored``       rows with an *ordinal* polarity consensus
                              ("Non applicable" excluded); the base of
                              ``median_polarity``.
- ``n_centrality_scored``     rows with a centrality consensus label;
                              denominator of ``share_centrality_*_of_scored``
                              and ``central_share_of_scored``.
- ``n_subjectivity_scored``   rows with a subjectivity median (>= 2 scores);
                              denominator of ``share_subjectivity_*_of_scored``.
- ``n_consensus_max``         max(n_polarity_consensus, n_centrality_scored):
                              the count the ranking, country and year tables
                              filter on, so a well-powered topic whose finding
                              is "Non applicable" polarity or low centrality
                              stays in them.
The ``*_of_scored`` shares do not move just because the panel covered more or
fewer rows of a cell; the plain shares keep missing and disputed rows in.

Outputs (analyses/output/):
- topic_sentiment_summary.csv     per topic: n, polarity, centrality and
                                  subjectivity shares (of all assigned rows
                                  and of rows with a consensus), ordinal
                                  medians, missingness counts, share of
                                  Central/Très central rows
- topic_sentiment_by_country.csv  topic x country cells (shares, medians, missingness)
- topic_sentiment_over_time.csv   topic x year cells (+ decade; point dates only)
- topic_sentiment_summary.json, topic_sentiment.manifest.json

Report-only: this script NEVER pushes anything to the Hub.

Usage
-----
    python analyses/topic_sentiment.py [--source hub|local] [--revision SHA]
        [--ids-file cohort.csv] [--min-topic-n 50] [--min-cell-n 20] [--min-year-n 10]
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import List

import pandas as pd

try:
    from iwac_pipeline.analyses._research import (
        OUTPUT_DIR, add_input_args, apply_ids_filter, date_exclusions, load_input,
        load_subset_dataframe, point_years, save_reports, write_run_manifest,
    )
    # Reuse the canonical sentiment vocabulary + consensus helpers.
    from iwac_pipeline.processing.sentiment_agreement import (
        POLARITY_ORDER, majority, models_for, CENTRALITY_ORDER, subjectivite_ordinal, to_ordinal,
    )
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _research import (  # noqa: E402
        OUTPUT_DIR, add_input_args, apply_ids_filter, date_exclusions, load_input,
        load_subset_dataframe, point_years, save_reports, write_run_manifest,
    )
    # _research put post-processing/ on sys.path.
    from sentiment_agreement import (  # noqa: E402
        POLARITY_ORDER, majority, models_for, CENTRALITY_ORDER, subjectivite_ordinal, to_ordinal,
    )

from iwac_common.sentiment_panel import PANEL, instrument_id  # noqa: E402

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()

# Centrality labels counted as "central" for the per-topic central share.
CENTRAL_LABELS = {"Central", "Très central"}
# The count the ranking, country and year tables filter and sort on: it
# includes "Non applicable" polarity and every centrality consensus, so a
# topic whose finding is precisely that is not dropped (issue #46).
FILTER_COUNT = "n_consensus_max"


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


def polarity_ordinal(labels: pd.Series) -> pd.Series:
    """Map consensus polarity labels onto the 1-5 scale (NaN for '' / 'Non applicable')."""
    return to_ordinal(labels, POLARITY_ORDER)


# ---------------------------------------------------------------------------
# Data loading / consensus resolution
# ---------------------------------------------------------------------------


def load_articles(args: argparse.Namespace) -> pd.DataFrame:
    models = models_for(args.generation)
    wanted = (["o:id", "lda_model_name", "lda_topic_id", "lda_topic_label", "pub_date", "country"]
              + [f"{m}_{suffix}" for m in models for suffix in
                 ("polarite", "centralite_islam_musulmans", "subjectivite_score")])
    return load_input(args, columns=wanted, loader=load_subset_dataframe)


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
    # "Non applicable" is a polarité answer only; centralité's floor is "Non abordé".
    for key, suffix, mapping, extra in (
            ("pol", "polarite", POLARITY_ORDER, ["Non applicable"]),
            ("cent", "centralite_islam_musulmans", CENTRALITY_ORDER, [])):
        cols = [f"{m}_{suffix}" for m in models]
        votes = df.reindex(columns=cols).astype("string").apply(lambda col: col.str.strip())
        votes = votes.where(votes.isin([*mapping, *extra]))
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
    out["year"], out["date_precision"] = point_years(out["pub_date"])
    out["country"] = out["country"].astype("string").str.strip().replace("", pd.NA)
    return out


def _share(count, denominator):
    return count / denominator if denominator else None


def summarize_cells(df: pd.DataFrame, dimensions: List[str], *, ordinal_means=False) -> pd.DataFrame:
    """Per-cell shares over all assigned rows and over each dimension's consensus rows.

    ``n`` is the complete cell, including insufficient votes and disagreement;
    the module docstring lists every count and the shares it divides. Medians
    use applicable consensus. Means are available only as an explicit
    equal-spacing sensitivity summary.
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
        # Each dimension's of-scored shares divide by the rows where that
        # dimension reached a consensus, so they do not move just because the
        # panel has covered more or fewer rows of a cell; plain shares keep n.
        n_pol = int(group["pol_label"].ne("").sum())
        n_cent = int(group["cent_label"].ne("").sum())
        n_subj = row["n_subjectivity_scored"]
        row["n_polarity_consensus"] = n_pol
        row["n_scored"] = n_pol  # historical name of n_polarity_consensus
        row["n_centrality_scored"] = n_cent
        row[FILTER_COUNT] = max(n_pol, n_cent)
        for label in POLARITY_ORDER:
            count = group["pol_label"].eq(label).sum()
            row[f"share_{slug(label)}"] = count / n
            row[f"share_{slug(label)}_of_scored"] = _share(count, n_pol)
        count = group["pol_label"].eq("Non applicable").sum()
        row["share_non_applicable"] = count / n
        row["share_non_applicable_of_scored"] = _share(count, n_pol)
        row["share_no_consensus"] = row["n_polarity_no_consensus"] / n
        row["share_insufficient_votes"] = row["n_polarity_insufficient_votes"] / n
        for label in CENTRALITY_ORDER:
            count = group["cent_label"].eq(label).sum()
            row[f"share_centrality_{slug(label)}"] = count / n
            row[f"share_centrality_{slug(label)}_of_scored"] = _share(count, n_cent)
        for rank in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
            count = group["subj_score"].eq(rank).sum()
            key = f"share_subjectivity_{str(rank).replace('.', '_')}"
            row[key] = count / n
            row[f"{key}_of_scored"] = _share(count, n_subj)
        count = group["cent_label"].isin(CENTRAL_LABELS).sum()
        row["central_share"] = count / n
        row["central_share_of_scored"] = _share(count, n_cent)
        if ordinal_means:
            row["mean_polarity_equal_spacing"] = group["pol_ord"].mean()
            row["mean_subjectivity_equal_spacing"] = group["subj_score"].mean()
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[*keys, "n", FILTER_COUNT])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Topic x sentiment analysis (consensus of the annotator panel). Report-only."
    )
    add_input_args(parser, source_default="hub", output_dir=OUTPUT_DIR, cohort=True)
    parser.add_argument("--min-topic-n", type=int, default=50,
                        help=f"Min {FILTER_COUNT} (rows with a polarity or centrality consensus, "
                             "whichever is larger) per topic for the ranked table (default 50)")
    parser.add_argument("--min-cell-n", type=int, default=20,
                        help=f"Min {FILTER_COUNT} per topic x country cell (default 20)")
    parser.add_argument("--min-year-n", type=int, default=10,
                        help=f"Min {FILTER_COUNT} per topic x year cell (default 10)")
    parser.add_argument("--generation", type=int, choices=sorted({m.generation for m in PANEL}),
                        default=max(m.generation for m in PANEL), help="One annotation instrument; newest by default")
    parser.add_argument("--ordinal-means", action="store_true",
                        help="Also report means as an explicit equal-spacing sensitivity summary")
    return parser


def ranked_topics(summary: pd.DataFrame, min_topic_n: int, limit: int = 25) -> pd.DataFrame:
    """Console ranking: most consensus-scored articles first (model, topic break ties)."""
    eligible = summary[summary[FILTER_COUNT] >= min_topic_n]
    return eligible.sort_values([FILTER_COUNT, "lda_model_name", "lda_topic_id"],
                                ascending=[False, True, True]).head(limit)


def print_ranking(summary: pd.DataFrame, min_topic_n: int) -> None:
    table = Table(title="Topic sentiment, by number of consensus-scored articles "
                        "(conditional on this corpus and instrument)", box=box.ROUNDED)
    for heading in ("Model / topic", "Label", "All articles", "Polarity consensus",
                    "Centrality consensus", "Median polarity", "Central (of scored)"):
        table.add_column(heading)
    for row in ranked_topics(summary, min_topic_n).to_dict("records"):
        central = row["central_share_of_scored"]
        table.add_row(f"{row['lda_model_name'][-12:]} / {row['lda_topic_id']}", str(row["label"]),
                      str(row["n"]), str(row["n_polarity_consensus"]), str(row["n_centrality_scored"]),
                      str(row["median_polarity"]),
                      "—" if central is None or pd.isna(central) else f"{central:.0%}")
    console.print(table)
    console.print("[yellow]share_* columns include missing and disputed rows in the denominator; "
                  "share_*_of_scored divide by that dimension's consensus count (n_polarity_consensus, "
                  "n_centrality_scored, n_subjectivity_scored). Ordinal medians do not assume equal "
                  "spacing; model agreement is not human validation.[/yellow]")


def main() -> None:
    args = build_parser().parse_args()
    console.print(Panel.fit(
        "[bold cyan]Topic x Sentiment[/bold cyan]\n"
        "[dim]Which topics attract which sentiment, where, and when — "
        f"{args.repo} ({args.config})[/dim]",
        border_style="cyan",
    ))

    loaded = apply_ids_filter(load_articles(args), args)
    for col in ("lda_topic_id", "lda_topic_label", "pub_date", "country"):
        if col not in loaded.columns:
            abort(f"Missing required column: {col}")
    try:
        df = prepare_articles(loaded, args.generation)
    except ValueError as exc:
        abort(str(exc))
    if df.empty:
        abort("No valid topic assignments.")
    legacy = ~df["lda_model_name"].str.startswith(("lda-sha256:", "lda-legacy-sha256:"))
    if legacy.any():
        abort("Mutable legacy lda_model_name detected; regenerate predictions from a verified model bundle before comparing topics.")
    summary = summarize_cells(df, [], ordinal_means=args.ordinal_means)
    by_country = summarize_cells(df, ["country"], ordinal_means=args.ordinal_means)
    by_country = by_country[by_country[FILTER_COUNT] >= args.min_cell_n]
    over_time = summarize_cells(df[df["year"].notna()], ["year"], ordinal_means=args.ordinal_means)
    over_time = over_time[over_time[FILTER_COUNT] >= args.min_year_n]
    over_time["decade"] = (over_time["year"] // 10) * 10

    print_ranking(summary, args.min_topic_n)
    report = {
        "generation": args.generation, "instrument_id": instrument_id(args.generation),
        "rows_loaded": int(len(loaded)), "rows_with_valid_topic": int(len(df)),
        "over_time_dates": date_exclusions(df["date_precision"]),
        "denominators": {
            "share_*": "n, every assigned article in the cell",
            "share_<polarity>_of_scored": "n_polarity_consensus (alias n_scored), articles whose polarity "
                                          "panel reached a consensus label (including Non applicable)",
            "share_centrality_*_of_scored, central_share_of_scored": "n_centrality_scored, articles with "
                                                                     "a centrality consensus label",
            "share_subjectivity_*_of_scored": "n_subjectivity_scored, articles with a subjectivity median",
            "median_polarity": "n_polarity_scored, articles with an ordinal polarity consensus "
                               "(Non applicable excluded)",
        },
        "table_filter": f"{FILTER_COUNT} = max(n_polarity_consensus, n_centrality_scored) >= "
                        "--min-topic-n (ranking), --min-cell-n (country), --min-year-n (year)",
        "console_ranking": f"{FILTER_COUNT} descending, then model and topic",
        "interpretation": "Conditional on this corpus, topic model and annotation instrument; model "
                          "agreement is not human validation.",
    }
    save_reports(args, loaded, "topic_sentiment",
                 {"topic_sentiment_summary.csv": summary, "topic_sentiment_by_country.csv": by_country,
                  "topic_sentiment_over_time.csv": over_time}, report, writer=write_run_manifest)
    console.print("[yellow]ℹ[/yellow] Report-only script — nothing is pushed to the Hub.")


if __name__ == "__main__":
    main()
