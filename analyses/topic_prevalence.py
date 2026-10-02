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

Statistics and scope
--------------------
These summaries describe the eligible archived documents conditional on one
fitted topic model. They do not estimate all press coverage or public opinion.
Uncertainty excludes archive selection, transcription and topic-model error.
- Trend slope: *n-weighted* least squares (weights = per-year doc counts) on
  the years passing ``--min-docs-year``; reported in percentage points per
  decade.
- Optional ``--trend-test mann-kendall``: Mann–Kendall (normal approximation with tie correction,
  two-sided p) on the same year window, with Benjamini–Hochberg correction
  across all topics → ``q_value`` / ``significant`` (q < 0.05). This requires
  independent annual observations and does not adjust for serial dependence
  or changing outlet composition. The default reports descriptive slopes only.
- ``mean_prevalence`` is doc-weighted over the *same* solid-year window used
  for the slope (no window inconsistency).
- ``peak_year`` is taken on a 3-year centered rolling mean of the per-year
  prevalence (min_periods=1 at the edges) to damp single-year noise; note the
  smoothing runs over the *ordered sequence of solid years*, which may skip
  thin years excluded by ``--min-docs-year``.
- Per-year conditional uncertainty bands: percentile bootstrap (``--bootstrap N``, default
  200, 0 disables) resampling documents with replacement *within each year*
  → ``ci_low`` / ``ci_high`` (2.5/97.5 percentiles). ``--bootstrap-unit newspaper``
  resamples whole outlet clusters within each year; years with fewer than two
  outlets have undefined bands. This does not resolve temporal dependence.

Usage
-----
    python analyses/topic_prevalence.py [--source hub|csv] [--min-docs-year 20]
                                        [--bootstrap 200] [--min-docs-cell 10]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from iwac_pipeline.processing._common import (  # noqa: E402
    PRIVATE_REPO_ID,
    ensure_hf_token,
    load_subset_dataframe,
    write_run_manifest,
)
from iwac_pipeline.analyses._stats import (  # noqa: E402
    bh_adjust,
    bootstrap_mean_ci,
    mann_kendall,
    weighted_least_squares_slope,
)
from iwac_pipeline.processing.lda_topic_modeling.modeling import (  # noqa: E402
    get_topic_label,
    load_lda_model,
    tokenize_for_prediction,
    predict_document,
)
from iwac_pipeline.processing.lda_topic_modeling.artifacts import (  # noqa: E402
    resolve_bundle, load_preprocessing, prediction_tokenizer_kwargs, digest_file, text_fingerprint,
    preprocessing_fingerprint, effective_model_identity,
)
from iwac_common.paths import workspace_root  # noqa: E402
from iwac_common.field_mappers import parse_pub_date  # noqa: E402

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table

console = Console()
OUTPUT_DIR = workspace_root() / "analyses" / "output"


def load_theta_export(path, model_id, repo_id, config, current_rows, n_topics, preprocessing):
    """Reuse theta only after model, artifact and per-document input verification."""
    path = Path(path)
    metadata = json.loads(path.with_name("doc_topics.metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("model_id"), metadata.get("repo_id"), metadata.get("config_name")) != (model_id, repo_id, config):
        raise ValueError("Theta export belongs to a different model or dataset")
    if metadata.get("sha256") != digest_file(path):
        raise ValueError("Theta export checksum mismatch")
    if metadata.get("preprocessing_sha256") != preprocessing_fingerprint(preprocessing):
        raise ValueError("Theta export preprocessing changed or is unrecorded")
    frame = pd.read_parquet(path)
    required = ["o:id", "lda_model_name", "text_sha256"] + [f"topic_{t}" for t in range(n_topics)]
    if any(c not in frame for c in required) or frame["o:id"].astype(str).duplicated().any():
        raise ValueError("Theta export has invalid IDs or columns")
    current = {str(r["o:id"]): text_fingerprint(r["lemma_nostop"]) for r in current_rows.to_dict("records")}
    result = {}
    for row in frame.to_dict("records"):
        oid = str(row["o:id"])
        if oid not in current:
            continue
        if row["lda_model_name"] != model_id or row["text_sha256"] != current[oid]:
            raise ValueError(f"Theta export input/model changed for item {oid}")
        vec = np.array([row[f"topic_{t}"] for t in range(n_topics)], dtype=float)
        if not np.isfinite(vec).all() or (vec < 0).any() or not np.isclose(vec.sum(), 1.0, atol=1e-5):
            raise ValueError(f"Invalid theta distribution for item {oid}")
        result[oid] = vec
    return result


def directional_trends(frame, direction, limit=5):
    """Direction labels must agree with the reported slope, even in small tables."""
    positive = frame[frame["slope_per_decade_pp"] > 0]
    negative = frame[frame["slope_per_decade_pp"] < 0]
    return positive.nlargest(limit, "slope_per_decade_pp") if direction == "rising" else negative.nsmallest(limit, "slope_per_decade_pp")


def eligible_years(values):
    parsed = values.map(lambda v: parse_pub_date(None if pd.isna(v) else v))
    precision = parsed.map(lambda value: value[1])
    years = pd.to_numeric(parsed.map(lambda value: value[0]), errors="coerce")
    return years.where(precision.isin(["day", "month", "year"])), precision


def main() -> None:
    parser = argparse.ArgumentParser(description="Probability-weighted topic prevalence over time.")
    parser.add_argument("--repo", default=PRIVATE_REPO_ID)
    parser.add_argument("--config", default="articles")
    parser.add_argument("--source", choices=["hub", "csv"], default="hub")
    parser.add_argument("--model-path", default=str(workspace_root() / "lda_model"))
    parser.add_argument("--theta-path", help="Reuse a verified doc_topics.parquet export; infer only missing rows")
    parser.add_argument("--allow-legacy-preprocessing", action="store_true")
    parser.add_argument("--include-unknown-language", action="store_true")
    parser.add_argument("--bootstrap-unit", choices=["article", "newspaper"], default="article",
                        help="Conditional document bootstrap or whole-newspaper cluster bootstrap within years")
    parser.add_argument("--trend-test", choices=["none", "mann-kendall"], default="none",
                        help="Optional Mann–Kendall assumes independent annual observations; default is descriptive only")
    parser.add_argument("--min-docs-year", type=int, default=20,
                        help="Years with fewer French docs are excluded from trend fitting")
    parser.add_argument("--min-docs-cell", type=int, default=10,
                        help="Drop year×country cells with fewer docs from the country output")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Bootstrap replicates for per-year prevalence CIs (0 disables)")
    parser.add_argument("--year-min", type=int, default=1900)
    parser.add_argument("--year-max", type=int, default=2030)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.min_docs_year < 1 or args.min_docs_cell < 1 or args.year_min > args.year_max:
        parser.error("Bootstrap must be nonnegative, count thresholds positive, and year bounds ordered")

    console.print(Panel.fit(
        "[bold cyan]Topic Prevalence Over Time[/bold cyan]\n"
        "[dim]Probability-weighted LDA topic shares per year / country[/dim]",
        border_style="cyan",
    ))

    # --- model ---
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_dir, model_id = resolve_bundle(Path(args.model_path))
    preprocessing, saved_params = load_preprocessing(model_dir, allow_legacy=args.allow_legacy_preprocessing)
    model_id = effective_model_identity(model_id, preprocessing)
    lda_model, dictionary, phraser = load_lda_model(model_dir, logging.getLogger(__name__))
    n_topics = lda_model.num_topics
    saved_labels = saved_params.get("topic_labels", {})
    labels = {tid: saved_labels.get(str(tid), get_topic_label(lda_model, tid)) for tid in range(n_topics)}

    # --- data ---
    token = ensure_hf_token(console=console) if args.source == "hub" else None
    df = load_subset_dataframe(
        args.repo, args.config, token=token, source=args.source,
        columns=["o:id", "lemma_nostop", "language", "pub_date", "country", "newspaper_ids", "newspaper"],
        console=console,
    )
    source_revision = df.attrs.get("iwac_source_revision")

    years, date_precision = eligible_years(df["pub_date"])
    language_values = df["language"].astype("string").str.strip()
    language_mask = language_values.eq(preprocessing["language"]).fillna(False)
    if args.include_unknown_language:
        language_mask |= language_values.isna() | language_values.eq("").fillna(False)
    has_text = df["lemma_nostop"].notna() & df["lemma_nostop"].astype(str).str.strip().astype(bool)
    valid_year = years.between(args.year_min, args.year_max)
    mask = language_mask & has_text & valid_year
    console.print(f"[blue]→[/blue] {int(mask.sum()):,} eligible {preprocessing['language']} documents with text and a usable year")
    if not mask.any():
        raise ValueError("No documents meet the language/text/date requirements")

    reused_theta = load_theta_export(args.theta_path, model_id, args.repo, args.config, df.loc[mask], n_topics, preprocessing) if args.theta_path else {}
    newspaper_column = "newspaper_ids" if "newspaper_ids" in df else "newspaper"
    if args.bootstrap_unit == "newspaper":
        if newspaper_column not in df or df.loc[mask, newspaper_column].astype("string").str.strip().fillna("").eq("").any():
            raise ValueError("Newspaper cluster bootstrap requires a nonblank outlet ID/label for every eligible row")
        if newspaper_column == "newspaper" and df.loc[mask, "country"].astype("string").str.strip().fillna("").eq("").any():
            raise ValueError("Country is required to distinguish homonymous newspaper labels")

    # --- per-document distributions ---
    # Keep per-year lists of doc vectors (for the bootstrap); year×country
    # only needs running sums + counts.
    year_docs: dict[int, list[np.ndarray]] = defaultdict(list)
    year_clusters: dict[int, list[str]] = defaultdict(list)
    yc_sum: dict[tuple[int, str], np.ndarray] = defaultdict(lambda: np.zeros(n_topics))
    yc_n: dict[tuple[int, str], int] = defaultdict(int)

    idx = df.index[mask]
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
        TaskProgressColumn(), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("[cyan]Computing topic distributions", total=len(idx))
        for i in idx:
            vec = reused_theta.get(str(df.at[i, "o:id"]))
            if vec is None:
                tokens = tokenize_for_prediction(
                    df.at[i, "lemma_nostop"], phraser=phraser,
                    **prediction_tokenizer_kwargs(preprocessing),
                )
                result = predict_document(
                    lda_model, dictionary, tokens, chunk_words=preprocessing["chunk_words"],
                    return_distribution=True,
                    topic_labels=labels,
                )
                vec = np.asarray(result[4]) if result[4] is not None else None
            if vec is not None:
                y = int(years.at[i])
                year_docs[y].append(vec)
                if args.bootstrap_unit == "newspaper":
                    outlet = str(df.at[i, newspaper_column]).strip()
                    year_clusters[y].append(outlet if newspaper_column == "newspaper_ids" else json.dumps([str(df.at[i, "country"]).strip(), outlet]))
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
    if not year_n:
        raise ValueError("Eligible documents contain no vocabulary recognized by this model")

    # --- per-year frame with bootstrap CIs ---
    ci_low: dict[int, np.ndarray] = {}
    ci_high: dict[int, np.ndarray] = {}
    if args.bootstrap and args.bootstrap > 0:
        for y, m in year_mat.items():
            # Deterministic per-year seed so re-runs reproduce the bands.
            lo, hi = bootstrap_mean_ci(m, args.bootstrap, seed=42 + y,
                                       clusters=year_clusters[y] if args.bootstrap_unit == "newspaper" else None)
            ci_low[y], ci_high[y] = lo, hi

    rows_y = []
    for y in sorted(year_mat):
        prev = year_sum[y] / year_n[y]
        for t in range(n_topics):
            rows_y.append({
                "year": y, "lda_model_name": model_id, "topic_id": t, "label": labels[t],
                "prevalence": prev[t], "n_docs": year_n[y],
                "n_clusters": len(set(year_clusters[y])) if args.bootstrap_unit == "newspaper" else year_n[y],
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
        {"year": y, "country": c, "lda_model_name": model_id, "topic_id": t, "label": labels[t],
         "prevalence": yc_sum[(y, c)][t] / yc_n[(y, c)], "n_docs": yc_n[(y, c)]}
        for (y, c) in kept_cells for t in range(n_topics)
    ]
    prev_yc = pd.DataFrame(rows_yc, columns=["year", "country", "lda_model_name", "topic_id", "label", "prevalence", "n_docs"])

    # --- trends: n-weighted slope + Mann–Kendall on solid years ---
    solid_years = sorted(y for y, n in year_n.items() if n >= args.min_docs_year)
    weights = np.array([year_n[y] for y in solid_years], dtype=float)
    trends = []
    mk_pvals = []
    for t in range(n_topics):
        series = np.array([year_sum[y][t] / year_n[y] for y in solid_years])
        if len(solid_years) >= 2:
            slope = weighted_least_squares_slope(solid_years, series, weights)
            mk_p = mann_kendall(series)[2] if args.trend_test == "mann-kendall" and len(solid_years) >= 5 else float("nan")
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
            "lda_model_name": model_id, "topic_id": t, "label": labels[t], "mean_prevalence": mean_prev,
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
    if args.trend_test == "mann-kendall":
        console.print(
            f"[yellow]ℹ[/yellow] {n_sig}/{n_topics} trends meet BH q < 0.05 under the "
            "independent-years assumption. Serial dependence and corpus selection are not corrected."
        )
    else:
        console.print("[blue]→[/blue] Descriptive slopes only; no significance claims requested.")

    # --- write outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prev_year.to_csv(OUTPUT_DIR / "topic_prevalence_year.csv", index=False, encoding="utf-8")
    prev_yc.to_csv(OUTPUT_DIR / "topic_prevalence_year_country.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [{"lda_model_name": model_id, "topic_id": t, "label": labels[t],
          "top_words": ", ".join(w for w, _ in lda_model.show_topic(t, topn=10))}
         for t in range(n_topics)]
    ).to_csv(OUTPUT_DIR / "topic_labels.csv", index=False, encoding="utf-8")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "model_dir": str(model_dir),
        "model_id": model_id,
        "estimand": "Mean topic probability among eligible archived documents, conditional on the fitted model",
        "uncertainty": "Conditional resampling; excludes archive selection, OCR and fitted-model uncertainty",
        "bootstrap_unit": args.bootstrap_unit,
        "trend_test": args.trend_test,
        "trend_test_assumption": "Independent annual observations" if args.trend_test != "none" else None,
        "theta_rows_reused": len(reused_theta),
        "num_topics": n_topics,
        "docs_used": int(sum(year_n.values())),
        "docs_loaded": len(df),
        "docs_eligible": int(mask.sum()),
        "docs_without_model_vocabulary": int(mask.sum()) - sum(year_n.values()),
        "docs_with_missing_or_ambiguous_dates": int((~date_precision.isin(["day", "month", "year"])).sum()),
        "years_covered": [int(min(year_n)), int(max(year_n))] if year_n else None,
        "trend_year_window": [solid_years[0], solid_years[-1]] if solid_years else None,
        "min_docs_year": args.min_docs_year,
        "topics": [
            {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in row.items()}
            for row in trends
        ],
    }
    with open(OUTPUT_DIR / "topic_prevalence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)

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

    sig = trends_df[trends_df["significant"]] if args.trend_test != "none" else trends_df
    if sig.empty:
        console.print(
            "[yellow]ℹ[/yellow] No topic trend is significant at BH q < 0.05 — "
            "reporting nothing as rising/declining (avoids over-claiming on noise)."
        )
    else:
        console.print(topic_table(
            "Rising topics (conditional test)" if args.trend_test != "none" else "Positive descriptive slopes",
            directional_trends(sig, "rising"), show_q=args.trend_test != "none",
        ))
        console.print(topic_table(
            "Declining topics (conditional test)" if args.trend_test != "none" else "Negative descriptive slopes",
            directional_trends(sig, "declining"), show_q=args.trend_test != "none",
        ))
    params = model_dir / "training_parameters.json"
    write_run_manifest(
        OUTPUT_DIR, script="topic_prevalence", repo_id=args.repo,
        revision=source_revision, args=args,
        inputs={
            "model_dir": str(model_dir),
            "model_id": model_id,
            "preprocessing_sha256": preprocessing_fingerprint(preprocessing),
            "theta_sha256": digest_file(Path(args.theta_path)) if args.theta_path else None,
            "training_parameters": (
                json.loads(params.read_text(encoding="utf-8")) if params.exists() else None
            ),
        },
        outputs=[
            OUTPUT_DIR / "topic_prevalence_year.csv",
            OUTPUT_DIR / "topic_prevalence_year_country.csv",
            OUTPUT_DIR / "topic_labels.csv",
            OUTPUT_DIR / "topic_prevalence_summary.json",
        ],
    )
    console.print(f"\n[green]✓[/green] Outputs in [cyan]{OUTPUT_DIR}[/cyan]")


if __name__ == "__main__":
    main()
