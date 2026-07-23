#!/usr/bin/env python3
"""
topic_prevalence.py
===================

Probability-weighted LDA topic prevalence over time.

Instead of counting dominant topics (noisy: a 0.34/0.33/0.33 article counts
fully for one topic), this loads the saved LDA model from ``lda_model/``,
computes the *full* topic distribution for every French article, and
aggregates mean topic probability per year and per year x country.

Outputs (written to analyses/output/):
- topic_prevalence_year.csv          year, topic_id, label, prevalence,
                                     ci_low, ci_high, n_docs
- topic_prevalence_year_country.csv  + country (cells with < --min-docs-cell
                                     docs are dropped)
- topic_labels.csv                   topic_id, label, top_words
- topic_prevalence_summary.json      trends (slope, Mann–Kendall p, BH q, peaks)

Statistics
----------
- Trend slope: *n-weighted* least squares (weights = per-year doc counts) on
  the years passing ``--min-docs-year``; reported in percentage points per
  decade.
- Trend test: Mann–Kendall (normal approximation with tie correction,
  two-sided p) on the same year window, with Benjamini–Hochberg correction
  across all topics → ``q_value`` / ``significant`` (q < 0.05). The console
  rising/declining tables only rank topics with q < 0.05.
- ``mean_prevalence`` is doc-weighted over the *same* solid-year window used
  for the slope (no window inconsistency).
- ``peak_year`` is taken on a 3-year centered rolling mean of the per-year
  prevalence (min_periods=1 at the edges) to damp single-year noise; note the
  smoothing runs over the *ordered sequence of solid years*, which may skip
  thin years excluded by ``--min-docs-year``.
- Per-year confidence bands: percentile bootstrap (``--bootstrap N``, default
  200, 0 disables) resampling documents with replacement *within each year*
  → ``ci_low`` / ``ci_high`` (2.5/97.5 percentiles).

Usage
-----
    python analyses/topic_prevalence.py [--source hub|csv] [--min-docs-year 20]
                                        [--bootstrap 200] [--min-docs-cell 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "post-processing"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # analyses/ for _stats

from _common import ensure_hf_token, load_subset_dataframe, PRIVATE_REPO_ID  # noqa: E402
from _stats import (  # noqa: E402
    bh_adjust,
    bootstrap_mean_ci,
    mann_kendall,
    weighted_least_squares_slope,
)
from iwac_common.text_utils import simple_tokenize  # noqa: E402
from lda_topic_modeling.constants import (  # noqa: E402
    DOMAIN_STOPWORDS,
    LDA_GEO_STOPWORDS,
    LDA_GENERIC_STOPWORDS,
    CUSTOM_COLLOCATIONS,
)
from lda_topic_modeling.modeling import (  # noqa: E402
    apply_custom_collocations,
    apply_phraser,
    get_topic_label,
    load_lda_model,
)

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table

console = Console()
OUTPUT_DIR = REPO_ROOT / "analyses" / "output"


def tokenize_like_training(text: str, stopwords: set, phraser) -> list[str]:
    """Reproduce the prediction-time tokenization of the LDA pipeline."""
    tokens = simple_tokenize(text, stopwords)
    tokens = apply_phraser(tokens, phraser)
    return apply_custom_collocations(tokens, CUSTOM_COLLOCATIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probability-weighted topic prevalence over time.")
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", default="articles")
    parser.add_argument("--source", choices=["hub", "csv"], default="hub")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "lda_model"))
    parser.add_argument("--min-docs-year", type=int, default=20,
                        help="Years with fewer French docs are excluded from trend fitting")
    parser.add_argument("--min-docs-cell", type=int, default=10,
                        help="Drop year×country cells with fewer docs from the country output")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Bootstrap replicates for per-year prevalence CIs (0 disables)")
    parser.add_argument("--year-min", type=int, default=1900)
    parser.add_argument("--year-max", type=int, default=2030)
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Topic Prevalence Over Time[/bold cyan]\n"
        "[dim]Probability-weighted LDA topic shares per year / country[/dim]",
        border_style="cyan",
    ))

    # --- model ---
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_dir = Path(args.model_path)
    lda_model, dictionary, phraser = load_lda_model(model_dir, logging.getLogger(__name__))
    n_topics = lda_model.num_topics
    labels = {tid: get_topic_label(lda_model, tid) for tid in range(n_topics)}

    # --- data ---
    token = ensure_hf_token(console=console) if args.source == "hub" else None
    df = load_subset_dataframe(
        args.repo, args.config, token=token, source=args.source,
        columns=["o:id", "lemma_nostop", "language", "pub_date", "country"],
        console=console,
    )

    years = pd.to_numeric(df["pub_date"].astype(str).str[:4], errors="coerce")
    is_french = df["language"].isna() | (df["language"] == "Français")
    has_text = df["lemma_nostop"].notna() & df["lemma_nostop"].astype(str).str.strip().astype(bool)
    valid_year = years.between(args.year_min, args.year_max)
    mask = is_french & has_text & valid_year
    console.print(f"[blue]→[/blue] {int(mask.sum()):,} French articles with text and a usable year")

    stopwords = set(DOMAIN_STOPWORDS) | LDA_GEO_STOPWORDS | LDA_GENERIC_STOPWORDS

    # --- per-document distributions ---
    # Keep per-year lists of doc vectors (for the bootstrap); year×country
    # only needs running sums + counts.
    year_docs: dict[int, list[np.ndarray]] = defaultdict(list)
    yc_sum: dict[tuple[int, str], np.ndarray] = defaultdict(lambda: np.zeros(n_topics))
    yc_n: dict[tuple[int, str], int] = defaultdict(int)

    idx = df.index[mask]
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
        TaskProgressColumn(), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("[cyan]Computing topic distributions", total=len(idx))
        for i in idx:
            tokens = tokenize_like_training(df.at[i, "lemma_nostop"], stopwords, phraser)
            bow = dictionary.doc2bow(tokens)
            if bow:
                vec = np.zeros(n_topics)
                for tid, p in lda_model.get_document_topics(bow, minimum_probability=0.0):
                    vec[tid] = p
                y = int(years.at[i])
                year_docs[y].append(vec)
                country = df.at[i, "country"]
                if pd.notna(country) and str(country).strip():
                    key = (y, str(country).strip())
                    yc_sum[key] += vec
                    yc_n[key] += 1
            progress.update(task, advance=1)

    # Per-year doc×topic matrices, sums, counts.
    year_mat: dict[int, np.ndarray] = {y: np.vstack(v) for y, v in year_docs.items() if v}
    year_sum: dict[int, np.ndarray] = {y: m.sum(axis=0) for y, m in year_mat.items()}
    year_n: dict[int, int] = {y: m.shape[0] for y, m in year_mat.items()}

    # --- per-year frame with bootstrap CIs ---
    ci_low: dict[int, np.ndarray] = {}
    ci_high: dict[int, np.ndarray] = {}
    if args.bootstrap and args.bootstrap > 0:
        for y, m in year_mat.items():
            # Deterministic per-year seed so re-runs reproduce the bands.
            lo, hi = bootstrap_mean_ci(m, args.bootstrap, seed=42 + y)
            ci_low[y], ci_high[y] = lo, hi

    rows_y = []
    for y in sorted(year_mat):
        prev = year_sum[y] / year_n[y]
        for t in range(n_topics):
            rows_y.append({
                "year": y, "topic_id": t, "label": labels[t],
                "prevalence": prev[t], "n_docs": year_n[y],
                "ci_low": float(ci_low[y][t]) if y in ci_low else None,
                "ci_high": float(ci_high[y][t]) if y in ci_high else None,
            })
    prev_year = pd.DataFrame(rows_y)

    # --- year×country frame (min-docs-cell enforced) ---
    kept_cells = [(y, c) for (y, c) in sorted(yc_sum) if yc_n[(y, c)] >= args.min_docs_cell]
    dropped_cells = len(yc_sum) - len(kept_cells)
    if dropped_cells:
        console.print(
            f"[yellow]ℹ[/yellow] Dropped {dropped_cells} year×country cell(s) "
            f"with < {args.min_docs_cell} docs"
        )
    rows_yc = [
        {"year": y, "country": c, "topic_id": t, "label": labels[t],
         "prevalence": yc_sum[(y, c)][t] / yc_n[(y, c)], "n_docs": yc_n[(y, c)]}
        for (y, c) in kept_cells for t in range(n_topics)
    ]
    prev_yc = pd.DataFrame(rows_yc)

    # --- trends: n-weighted slope + Mann–Kendall on solid years ---
    solid_years = sorted(y for y, n in year_n.items() if n >= args.min_docs_year)
    weights = np.array([year_n[y] for y in solid_years], dtype=float)
    trends = []
    mk_pvals = []
    for t in range(n_topics):
        series = np.array([year_sum[y][t] / year_n[y] for y in solid_years])
        if len(solid_years) >= 5:
            slope = weighted_least_squares_slope(solid_years, series, weights)
            _, _, mk_p = mann_kendall(series)
        else:
            slope, mk_p = float("nan"), float("nan")
        mk_pvals.append(mk_p)
        # mean_prevalence over the SAME solid-year window (doc-weighted).
        mean_prev = (
            float(np.sum(weights * series) / weights.sum())
            if len(solid_years) and weights.sum() > 0 else float("nan")
        )
        # peak on a 3-year centered rolling mean over the ordered solid years.
        if len(series):
            smoothed = pd.Series(series).rolling(3, center=True, min_periods=1).mean().to_numpy()
            peak_year = int(solid_years[int(np.argmax(smoothed))])
            peak_prev = float(smoothed.max())
        else:
            peak_year, peak_prev = None, None
        trends.append({
            "topic_id": t, "label": labels[t], "mean_prevalence": mean_prev,
            "slope_per_decade_pp": slope * 10 * 100,  # percentage points per decade
            "mk_p_value": mk_p, "peak_year": peak_year, "peak_prevalence": peak_prev,
        })

    # Benjamini–Hochberg across topics.
    q_values = bh_adjust(mk_pvals)
    for tr, q in zip(trends, q_values):
        tr["q_value"] = None if np.isnan(q) else float(q)
        tr["significant"] = bool(np.isfinite(q) and q < 0.05)
    trends_df = pd.DataFrame(trends)
    n_sig = int(trends_df["significant"].sum())
    console.print(
        f"[blue]→[/blue] {n_sig}/{n_topics} topics show a significant trend "
        f"(Mann–Kendall, BH q < 0.05)"
    )

    # --- write outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prev_year.to_csv(OUTPUT_DIR / "topic_prevalence_year.csv", index=False, encoding="utf-8")
    prev_yc.to_csv(OUTPUT_DIR / "topic_prevalence_year_country.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [{"topic_id": t, "label": labels[t],
          "top_words": ", ".join(w for w, _ in lda_model.show_topic(t, topn=10))}
         for t in range(n_topics)]
    ).to_csv(OUTPUT_DIR / "topic_labels.csv", index=False, encoding="utf-8")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "model_dir": str(model_dir),
        "num_topics": n_topics,
        "docs_used": int(sum(year_n.values())),
        "years_covered": [int(min(year_n)), int(max(year_n))] if year_n else None,
        "trend_year_window": [solid_years[0], solid_years[-1]] if solid_years else None,
        "min_docs_year": args.min_docs_year,
        "topics": trends,
    }
    with open(OUTPUT_DIR / "topic_prevalence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- report ---
    def topic_table(title: str, frame: pd.DataFrame, show_q: bool = False) -> Table:
        t = Table(title=title, box=box.ROUNDED)
        t.add_column("Topic", style="cyan", justify="right")
        t.add_column("Label", style="green", max_width=48)
        t.add_column("Mean", justify="right")
        t.add_column("pp/decade", justify="right")
        if show_q:
            t.add_column("BH q", justify="right")
        t.add_column("Peak", justify="right")
        for _, r in frame.iterrows():
            cells = [
                str(int(r.topic_id)), r.label, f"{r.mean_prevalence:.1%}",
                f"{r.slope_per_decade_pp:+.2f}",
            ]
            if show_q:
                cells.append("—" if r.q_value is None else f"{r.q_value:.3f}")
            cells.append(str(r.peak_year))
            t.add_row(*cells)
        return t

    console.print()
    console.print(topic_table(
        "Top topics by overall prevalence",
        trends_df.nlargest(8, "mean_prevalence"),
    ))

    sig = trends_df[trends_df["significant"]]
    if sig.empty:
        console.print(
            "[yellow]ℹ[/yellow] No topic trend is significant at BH q < 0.05 — "
            "reporting nothing as rising/declining (avoids over-claiming on noise)."
        )
    else:
        console.print(topic_table(
            f"Rising topics (significant, ≥{args.min_docs_year} docs/yr, BH q < 0.05)",
            sig.nlargest(5, "slope_per_decade_pp"), show_q=True,
        ))
        console.print(topic_table(
            "Declining topics (significant, BH q < 0.05)",
            sig.nsmallest(5, "slope_per_decade_pp"), show_q=True,
        ))
    console.print(f"\n[green]✓[/green] Outputs in [cyan]{OUTPUT_DIR}[/cyan]")


if __name__ == "__main__":
    main()
