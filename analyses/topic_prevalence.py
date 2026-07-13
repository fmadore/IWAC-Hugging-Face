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
- topic_prevalence_year.csv          year, topic_id, label, prevalence, n_docs
- topic_prevalence_year_country.csv  + country
- topic_labels.csv                   topic_id, label, top_words
- topic_prevalence_summary.json      trends (rising/falling topics, peaks)

Usage
-----
    python analyses/topic_prevalence.py [--source hub|csv] [--min-docs-year 20]
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

from _common import ensure_hf_token, load_subset_dataframe, PRIVATE_REPO_ID  # noqa: E402
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

    # --- per-document distributions, accumulated straight into aggregates ---
    year_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(n_topics))
    year_n: dict[int, int] = defaultdict(int)
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
                year_sum[y] += vec
                year_n[y] += 1
                country = df.at[i, "country"]
                if pd.notna(country) and str(country).strip():
                    key = (y, str(country).strip())
                    yc_sum[key] += vec
                    yc_n[key] += 1
            progress.update(task, advance=1)

    # --- long-format frames ---
    rows_y = [
        {"year": y, "topic_id": t, "label": labels[t],
         "prevalence": year_sum[y][t] / year_n[y], "n_docs": year_n[y]}
        for y in sorted(year_sum) for t in range(n_topics)
    ]
    prev_year = pd.DataFrame(rows_y)

    rows_yc = [
        {"year": y, "country": c, "topic_id": t, "label": labels[t],
         "prevalence": yc_sum[(y, c)][t] / yc_n[(y, c)], "n_docs": yc_n[(y, c)]}
        for (y, c) in sorted(yc_sum) for t in range(n_topics)
    ]
    prev_yc = pd.DataFrame(rows_yc)

    # --- trends: linear fit on years with enough docs ---
    solid_years = sorted(y for y, n in year_n.items() if n >= args.min_docs_year)
    trends = []
    for t in range(n_topics):
        series = np.array([year_sum[y][t] / year_n[y] for y in solid_years])
        if len(solid_years) >= 5:
            slope = float(np.polyfit(solid_years, series, 1)[0])
        else:
            slope = float("nan")
        overall = float(
            sum(year_sum[y][t] for y in year_sum) / max(1, sum(year_n.values()))
        )
        peak_year = int(solid_years[int(np.argmax(series))]) if len(solid_years) else None
        trends.append({
            "topic_id": t, "label": labels[t], "mean_prevalence": overall,
            "slope_per_decade_pp": slope * 10 * 100,  # percentage points per decade
            "peak_year": peak_year,
            "peak_prevalence": float(series.max()) if len(series) else None,
        })
    trends_df = pd.DataFrame(trends)

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
    def topic_table(title: str, frame: pd.DataFrame) -> Table:
        t = Table(title=title, box=box.ROUNDED)
        t.add_column("Topic", style="cyan", justify="right")
        t.add_column("Label", style="green", max_width=52)
        t.add_column("Mean", justify="right")
        t.add_column("pp/decade", justify="right")
        t.add_column("Peak", justify="right")
        for _, r in frame.iterrows():
            t.add_row(
                str(int(r.topic_id)), r.label, f"{r.mean_prevalence:.1%}",
                f"{r.slope_per_decade_pp:+.2f}", str(r.peak_year),
            )
        return t

    console.print()
    console.print(topic_table(
        "Top topics by overall prevalence",
        trends_df.nlargest(8, "mean_prevalence"),
    ))
    console.print(topic_table(
        f"Rising topics (linear trend, years with ≥{args.min_docs_year} docs)",
        trends_df.nlargest(5, "slope_per_decade_pp"),
    ))
    console.print(topic_table(
        "Declining topics",
        trends_df.nsmallest(5, "slope_per_decade_pp"),
    ))
    console.print(f"\n[green]✓[/green] Outputs in [cyan]{OUTPUT_DIR}[/cyan]")


if __name__ == "__main__":
    main()
