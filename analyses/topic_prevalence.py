#!/usr/bin/env python3
"""
topic_prevalence.py
===================

Probability-weighted LDA topic prevalence over time.

Instead of counting dominant topics (noisy: a 0.34/0.33/0.33 article counts
fully for one topic), this loads the subset's saved LDA bundle, computes the
*full* topic distribution for every eligible document, and aggregates mean
topic probability per year and per year x country.

Model choice: ``--model-path`` defaults to the ``--config`` subset's
``CONFIG_PRESETS`` model (``--language`` picks the per-language model, e.g.
``--config references --language Anglais``). The bundle's recorded training
subset and language must match the request; a mismatch aborts unless
``--allow-cross-corpus-model`` is given, and the comparison is recorded in the
summary and run manifest either way (``model_training_match``).

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
  200, 0 disables; seeded by ``--seed`` + year) resampling documents with replacement
  *within each year* → ``ci_low`` / ``ci_high`` (2.5/97.5 percentiles).
  ``--bootstrap-unit newspaper`` resamples whole outlet clusters within each
  year; years with fewer than ``_stats.MIN_BOOTSTRAP_CLUSTERS`` outlets get no band
  (``ci_status = insufficient_clusters``); a year with a single document gets
  none either (``insufficient_docs``). Eligible rows are ordered by ``o:id``
  first, so the bands do not depend on input row order. This does not
  resolve temporal dependence.
- Evidence floor: documents with fewer in-vocabulary tokens than
  ``--min-vocab-tokens`` (default: the model's training floor, as used for
  the Hub's ``lda_topic_id``) are left out and counted in
  ``docs_below_min_evidence``; ``--min-vocab-tokens 0`` keeps them all.
- The raw annual series pools every country and outlet, so a shift in which
  countries are archived moves it. ``--standardize country`` adds a directly
  standardised series beside it (fixed country weights = each country's share
  of the analysed documents over the whole window); it is reported only for
  years where every weighted country has documents, and never replaces the raw
  series or the trend inputs.
- Dates: point dates only (day/month/year); ranges and unparsed dates are
  excluded and counted (``_research.point_years``).

Usage
-----
    python analyses/topic_prevalence.py [--source hub|local] [--revision SHA]
        [--ids-file cohort.csv] [--min-docs-year 20] [--bootstrap 200] [--seed 42]
        [--min-docs-cell 10] [--standardize country]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from iwac_pipeline.analyses._research import (
        OUTPUT_DIR, add_input_args, apply_ids_filter, date_exclusions, in_window, load_input,
        load_subset_dataframe, point_years, save_reports, sha256, write_run_manifest,
    )
    from iwac_pipeline.analyses._stats import (
        MIN_BOOTSTRAP_CLUSTERS, bh_adjust, bootstrap_mean_ci, excluded_language_counts,
        language_mask, mann_kendall, weighted_least_squares_slope,
    )
    from iwac_pipeline.processing.lda_topic_modeling.modeling import (
        count_vocab_tokens, get_topic_label, load_lda_model, tokenize_for_prediction, predict_document,
    )
    from iwac_pipeline.processing.lda_topic_modeling.constants import DEFAULT_MIN_TRAIN_TOKENS, resolve_preset
    from iwac_pipeline.processing.lda_topic_modeling.artifacts import (
        resolve_bundle, load_preprocessing, prediction_tokenizer_kwargs, text_fingerprint,
        preprocessing_fingerprint, effective_model_identity,
    )
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _research import (  # noqa: E402
        OUTPUT_DIR, add_input_args, apply_ids_filter, date_exclusions, in_window, load_input,
        load_subset_dataframe, point_years, save_reports, sha256, write_run_manifest,
    )
    from _stats import (  # noqa: E402
        MIN_BOOTSTRAP_CLUSTERS, bh_adjust, bootstrap_mean_ci, excluded_language_counts,
        language_mask, mann_kendall, weighted_least_squares_slope,
    )
    # _research put post-processing/ on sys.path.
    from lda_topic_modeling.modeling import (  # noqa: E402
        count_vocab_tokens, get_topic_label, load_lda_model, tokenize_for_prediction, predict_document,
    )
    from lda_topic_modeling.constants import DEFAULT_MIN_TRAIN_TOKENS, resolve_preset  # noqa: E402
    from lda_topic_modeling.artifacts import (  # noqa: E402
        resolve_bundle, load_preprocessing, prediction_tokenizer_kwargs, text_fingerprint,
        preprocessing_fingerprint, effective_model_identity,
    )
from iwac_common.paths import workspace_root  # noqa: E402

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table  # noqa: E402

console = Console()


def load_theta_export(path, model_id, repo_id, config, current_rows, n_topics, preprocessing):
    """Reuse theta only after model, artifact and per-document input verification."""
    path = Path(path)
    metadata = json.loads(path.with_name("doc_topics.metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("model_id"), metadata.get("repo_id"), metadata.get("config_name")) != (model_id, repo_id, config):
        raise ValueError("Theta export belongs to a different model or dataset")
    if metadata.get("sha256") != sha256(path):
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
    """Point years only (the shared ``_research.point_years`` rule)."""
    return point_years(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probability-weighted topic prevalence over time.")
    add_input_args(parser, source_default="hub", output_dir=OUTPUT_DIR, cohort=True, seed=True)
    parser.add_argument("--model-path", default=None,
                        help="LDA model root or bundle (default: the --config subset's preset model "
                             "for --language, as the LDA stage resolves it)")
    parser.add_argument("--language", default=None,
                        help="Model language for subsets with one model per language (references: "
                             "'Français' or 'Anglais'). Default: the subset preset's language")
    parser.add_argument("--allow-cross-corpus-model", action="store_true",
                        help="Deliberately apply a bundle trained on another subset or language; "
                             "recorded in the summary and run manifest")
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
    parser.add_argument("--standardize", choices=["none", "country"], default="none",
                        help="Also report a directly country-standardised annual series beside the raw one")
    parser.add_argument("--min-vocab-tokens", type=int, default=None,
                        help="Evidence floor: documents with fewer in-vocabulary tokens are left out, "
                             "as their lda_topic_id is null on the Hub (default: the model's own "
                             "training floor; 0 keeps every document)")
    return parser


def default_model_path(config: str, language: str | None = None) -> tuple[Path, str]:
    """``(model root, language)`` the LDA stage uses for this subset and language."""
    language, preset = resolve_preset(config, language)
    return workspace_root() / preset.get("model_path", "lda_model"), language


def training_match(bundle: dict, config: str, language: str) -> dict:
    """Compare the bundle's recorded training subset/language with the request.

    A field the bundle does not record counts as a mismatch: the check fails
    closed rather than trusting an unlabelled model.
    """
    trained = {"config_name": bundle.get("trained_config"), "language": bundle.get("trained_language")}
    requested = {"config_name": config, "language": language}
    mismatched = sorted(k for k in trained if trained[k] != requested[k])
    return {"trained": trained, "requested": requested, "mismatched_fields": mismatched}


def check_training_match(bundle: dict, config: str, language: str, *, allow_cross_corpus: bool) -> dict:
    """Abort on a subset/language mismatch unless cross-corpus use is explicit."""
    match = training_match(bundle, config, language)
    match["cross_corpus_allowed"] = bool(allow_cross_corpus)
    if match["mismatched_fields"] and not allow_cross_corpus:
        raise SystemExit(
            f"LDA bundle {bundle['id']} was trained on config={match['trained']['config_name']!r}, "
            f"language={match['trained']['language']!r}; this run requests config={config!r}, "
            f"language={language!r}. Its topics do not describe these documents' lda_topic_id. "
            "Omit --model-path to use the subset's own model, or pass --allow-cross-corpus-model "
            "to apply it deliberately (recorded in the run manifest)."
        )
    return match


def load_bundle(model_path, *, allow_legacy: bool) -> dict:
    model_dir, model_id = resolve_bundle(Path(model_path))
    preprocessing, saved_params = load_preprocessing(model_dir, allow_legacy=allow_legacy)
    model_id = effective_model_identity(model_id, preprocessing)
    lda_model, dictionary, phraser = load_lda_model(model_dir, logging.getLogger(__name__))
    n_topics = lda_model.num_topics
    saved_labels = saved_params.get("topic_labels", {})
    labels = {tid: saved_labels.get(str(tid), get_topic_label(lda_model, tid, language=preprocessing["language"]))
              for tid in range(n_topics)}
    # Models fitted before the floor was recorded all used the default.
    extra = saved_params.get("extra", {})
    floor = int(extra.get("min_train_tokens", DEFAULT_MIN_TRAIN_TOKENS))
    return {"dir": model_dir, "id": model_id, "preprocessing": preprocessing, "model": lda_model,
            "dictionary": dictionary, "phraser": phraser, "n_topics": n_topics, "labels": labels,
            "min_vocab_tokens": floor, "trained_config": extra.get("config_name"),
            "trained_language": extra.get("language", preprocessing.get("language"))}


def eligibility(df: pd.DataFrame, language: str, args) -> tuple[pd.Series, pd.Series, dict]:
    """``(mask, years, counts)`` for model language, nonblank text and a point year in window."""
    years, precision = eligible_years(df["pub_date"])
    language_ok = language_mask(df["language"], language, include_unknown=args.include_unknown_language)
    has_text = df["lemma_nostop"].astype("string").fillna("").str.strip().ne("")
    valid_year = in_window(years, args.year_min, args.year_max)
    mask = language_ok & has_text & valid_year
    counts = {"docs_loaded": int(len(df)), "docs_eligible": int(mask.sum()),
              "dates": date_exclusions(precision),
              "docs_outside_year_window": int((years.notna() & ~valid_year).sum()),
              "docs_excluded_language": int((~language_ok).sum()),
              "excluded_language_labels": excluded_language_counts(df["language"], language_ok),
              "docs_without_text": int((language_ok & ~has_text).sum())}
    return mask, years, counts


def outlet_clusters(df: pd.DataFrame, idx, unit: str) -> pd.Series | None:
    """Outlet cluster per eligible row (ID, else country-scoped title) for the cluster bootstrap."""
    if unit != "newspaper":
        return None
    column = "newspaper_ids" if "newspaper_ids" in df else "newspaper"
    if column not in df or df.loc[idx, column].astype("string").str.strip().fillna("").eq("").any():
        raise ValueError("Newspaper cluster bootstrap requires a nonblank outlet ID/label for every eligible row")
    outlets = df.loc[idx, column].astype(str).str.strip()
    if column == "newspaper_ids":
        return outlets
    if df.loc[idx, "country"].astype("string").str.strip().fillna("").eq("").any():
        raise ValueError("Country is required to distinguish homonymous newspaper labels")
    return pd.Series([json.dumps([str(c).strip(), o]) for c, o in zip(df.loc[idx, "country"], outlets)],
                     index=outlets.index)


def document_vectors(df: pd.DataFrame, idx, bundle: dict, reused: dict,
                     min_vocab_tokens: int = 0) -> tuple[list, np.ndarray, int]:
    """Full topic distribution per eligible document, and the number left out
    below the evidence floor. Rows without model vocabulary drop out too.

    Below ``min_vocab_tokens`` in-vocabulary tokens a document's theta sits
    near the prior and the Hub leaves its ``lda_topic_id`` null; skipping it
    here keeps the denominator the one topic_sentiment and topic_review use.
    The floor is checked on the tokens, so a reused theta is filtered alike.
    """
    kept, vectors, below = [], [], 0
    tokenizer_kwargs = prediction_tokenizer_kwargs(bundle["preprocessing"])
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
        TaskProgressColumn(), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("[cyan]Computing topic distributions", total=len(idx))
        for i in idx:
            vec = reused.get(str(df.at[i, "o:id"]))
            tokens = None
            if vec is None or min_vocab_tokens:
                tokens = tokenize_for_prediction(df.at[i, "lemma_nostop"], phraser=bundle["phraser"],
                                                 **tokenizer_kwargs)
            if min_vocab_tokens and tokens and count_vocab_tokens(bundle["dictionary"], tokens) < min_vocab_tokens:
                below += 1
                progress.update(task, advance=1)
                continue
            if vec is None:
                result = predict_document(bundle["model"], bundle["dictionary"], tokens,
                                          chunk_words=bundle["preprocessing"]["chunk_words"],
                                          chunk_weighting=bundle["preprocessing"].get(
                                              "chunk_weighting", "all_tokens"),
                                          return_distribution=True, topic_labels=bundle["labels"])
                vec = np.asarray(result[4]) if result[4] is not None else None
            if vec is not None:
                kept.append(i)
                vectors.append(vec)
            progress.update(task, advance=1)
    n_topics = bundle["n_topics"]
    return kept, (np.vstack(vectors) if vectors else np.empty((0, n_topics))), below


def standard_weights(countries: pd.Series) -> dict:
    """Fixed country weights: share of analysed documents with a country, whole window."""
    known = countries.dropna()
    known = known[known.astype(str).str.strip().ne("")]
    return (known.value_counts(normalize=True).sort_index().to_dict()) if len(known) else {}


def standardized_series(theta: np.ndarray, years: pd.Series, countries: pd.Series, weights: dict) -> dict:
    """Year -> (standardised prevalence or None, weight coverage).

    Direct standardisation needs a rate for every weighted country; a year
    missing one has coverage < 1 and no standardised value rather than a
    silently renormalised one.
    """
    out = {}
    for year in sorted(years.unique()):
        in_year = (years == year).to_numpy()
        total, coverage = np.zeros(theta.shape[1]), 0.0
        for country, weight in weights.items():
            # A missing country is NA, not False; it simply matches no stratum.
            rows = in_year & (countries == country).fillna(False).to_numpy(dtype=bool)
            if rows.any():
                total += weight * theta[rows].mean(axis=0)
                coverage += weight
        complete = np.isclose(coverage, 1.0)
        out[int(year)] = (total if complete else None, float(coverage))
    return out


def yearly_prevalence(theta, years, clusters, bundle, args, standardized=None) -> pd.DataFrame:
    rows = []
    for y in sorted(years.unique()):
        in_year = (years == y).to_numpy()
        m = theta[in_year]
        group = clusters[in_year].tolist() if clusters is not None else None
        n_clusters = len(set(group)) if group is not None else int(m.shape[0])
        if not args.bootstrap:
            status, lo, hi = "disabled", None, None
        elif group is not None and n_clusters < MIN_BOOTSTRAP_CLUSTERS:
            status, lo, hi = "insufficient_clusters", None, None
        elif m.shape[0] < 2:
            # Resampling one document always returns it: a zero-width band.
            status, lo, hi = "insufficient_docs", None, None
        else:
            # Deterministic per-year seed so re-runs reproduce the bands.
            lo, hi = bootstrap_mean_ci(m, args.bootstrap, seed=args.seed + int(y), clusters=group)
            status = "ok"
        prev = m.mean(axis=0)
        std, coverage = standardized.get(int(y), (None, None)) if standardized is not None else (None, None)
        for t in range(bundle["n_topics"]):
            row = {"year": int(y), "lda_model_name": bundle["id"], "topic_id": t, "label": bundle["labels"][t],
                   "prevalence": prev[t], "n_docs": int(m.shape[0]), "n_clusters": n_clusters,
                   "ci_low": float(lo[t]) if lo is not None else None,
                   "ci_high": float(hi[t]) if hi is not None else None, "ci_status": status}
            if standardized is not None:
                row["prevalence_standardized"] = float(std[t]) if std is not None else None
                row["standard_weight_coverage"] = coverage
            rows.append(row)
    return pd.DataFrame(rows)


def country_year_prevalence(theta, years, countries, bundle, min_docs_cell) -> tuple[pd.DataFrame, int]:
    columns = ["year", "country", "lda_model_name", "topic_id", "label", "prevalence", "n_docs"]
    frame = pd.DataFrame({"year": years.to_numpy(), "country": countries.to_numpy()})
    frame = frame[frame["country"].notna() & frame["country"].astype(str).str.strip().ne("")]
    rows, dropped = [], 0
    for (y, c), group in frame.groupby(["year", "country"], sort=True):
        if len(group) < min_docs_cell:
            dropped += 1
            continue
        prev = theta[group.index.to_numpy()].mean(axis=0)
        rows += [{"year": int(y), "country": c, "lda_model_name": bundle["id"], "topic_id": t,
                  "label": bundle["labels"][t], "prevalence": prev[t], "n_docs": len(group)}
                 for t in range(bundle["n_topics"])]
    return pd.DataFrame(rows, columns=columns), dropped


def topic_trends(prev_year: pd.DataFrame, bundle, args) -> tuple[pd.DataFrame, list]:
    """n-weighted slope (+ optional Mann–Kendall/BH) on the raw series over solid years."""
    per_year = prev_year.drop_duplicates("year").set_index("year")["n_docs"]
    solid_years = sorted(int(y) for y, n in per_year.items() if n >= args.min_docs_year)
    weights = np.array([per_year[y] for y in solid_years], dtype=float)
    table = prev_year.pivot(index="year", columns="topic_id", values="prevalence")
    trends, mk_pvals = [], []
    for t in range(bundle["n_topics"]):
        series = table.loc[solid_years, t].to_numpy(dtype=float) if solid_years else np.array([])
        if len(solid_years) >= 2:
            slope = weighted_least_squares_slope(solid_years, series, weights)
            mk_p = mann_kendall(series)[2] if args.trend_test == "mann-kendall" and len(solid_years) >= 5 else float("nan")
        else:
            slope, mk_p = float("nan"), float("nan")
        mk_pvals.append(mk_p)
        # mean_prevalence over the SAME solid-year window (doc-weighted).
        mean_prev = float(np.sum(weights * series) / weights.sum()) if len(solid_years) and weights.sum() > 0 else float("nan")
        # peak on a 3-year centered rolling mean over the ordered solid years.
        if len(series):
            smoothed = pd.Series(series).rolling(3, center=True, min_periods=1).mean().to_numpy()
            peak_year, peak_prev = int(solid_years[int(np.argmax(smoothed))]), float(smoothed.max())
        else:
            peak_year, peak_prev = None, None
        trends.append({"lda_model_name": bundle["id"], "topic_id": t, "label": bundle["labels"][t],
                       "mean_prevalence": mean_prev,
                       "slope_per_decade_pp": slope * 10 * 100,  # percentage points per decade
                       "mk_p_value": mk_p, "peak_year": peak_year, "peak_prevalence": peak_prev})
    # Benjamini–Hochberg across topics; undefined (NaN) tests are left out of the family.
    for tr, q in zip(trends, bh_adjust(mk_pvals)):
        tr["q_value"] = None if np.isnan(q) else float(q)
        tr["significant"] = bool(np.isfinite(q) and q < 0.05)
    return pd.DataFrame(trends), solid_years


def topic_table(title: str, frame: pd.DataFrame, show_q: bool = False) -> Table:
    t = Table(title=title, box=box.ROUNDED)
    t.add_column("Topic", style="cyan", justify="right")
    t.add_column("Label", style="green", max_width=48)
    t.add_column("Mean", justify="right")
    t.add_column("pp/decade", justify="right")
    if show_q:
        t.add_column("BH q", justify="right")
    t.add_column("Peak", justify="right")
    for r in frame.to_dict("records"):
        cells = [str(int(r["topic_id"])), r["label"], f"{r['mean_prevalence']:.1%}",
                 f"{r['slope_per_decade_pp']:+.2f}"]
        if show_q:
            cells.append("—" if r["q_value"] is None else f"{r['q_value']:.3f}")
        cells.append(str(r["peak_year"]))
        t.add_row(*cells)
    return t


def print_trends(trends_df: pd.DataFrame, trend_test: str, n_topics: int) -> None:
    tested = trend_test != "none"
    if tested:
        console.print(f"[yellow]ℹ[/yellow] {int(trends_df['significant'].sum())}/{n_topics} trends meet BH q < 0.05 "
                      "under the independent-years assumption. Serial dependence and corpus selection are not corrected.")
    else:
        console.print("[blue]→[/blue] Descriptive slopes only; no significance claims requested.")
    console.print()
    console.print(topic_table("Top topics by overall prevalence", trends_df.nlargest(8, "mean_prevalence")))
    sig = trends_df[trends_df["significant"]] if tested else trends_df
    if sig.empty:
        console.print("[yellow]ℹ[/yellow] No topic trend is significant at BH q < 0.05 — "
                      "reporting nothing as rising/declining (avoids over-claiming on noise).")
        return
    console.print(topic_table("Rising topics (conditional test)" if tested else "Positive descriptive slopes",
                              directional_trends(sig, "rising"), show_q=tested))
    console.print(topic_table("Declining topics (conditional test)" if tested else "Negative descriptive slopes",
                              directional_trends(sig, "declining"), show_q=tested))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.bootstrap < 0 or args.min_docs_year < 1 or args.min_docs_cell < 1 or args.year_min > args.year_max:
        parser.error("Bootstrap must be nonnegative, count thresholds positive, and year bounds ordered")
    if args.min_vocab_tokens is not None and args.min_vocab_tokens < 0:
        parser.error("--min-vocab-tokens must be nonnegative")
    console.print(Panel.fit(
        "[bold cyan]Topic Prevalence Over Time[/bold cyan]\n"
        "[dim]Probability-weighted LDA topic shares per year / country[/dim]",
        border_style="cyan",
    ))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    default_path, language = default_model_path(args.config, args.language)
    model_path = Path(args.model_path) if args.model_path else default_path
    bundle = load_bundle(model_path, allow_legacy=args.allow_legacy_preprocessing)
    training = check_training_match(bundle, args.config, language,
                                    allow_cross_corpus=args.allow_cross_corpus_model)
    if training["mismatched_fields"]:
        console.print(f"[yellow]⚠[/yellow] Cross-corpus model (explicit): trained on "
                      f"{training['trained']}, applied to {training['requested']}")
    preprocessing, n_topics = bundle["preprocessing"], bundle["n_topics"]

    loaded = load_input(args, loader=load_subset_dataframe, columns=[
        "o:id", "lemma_nostop", "language", "pub_date", "country", "newspaper_ids", "newspaper"])
    df = apply_ids_filter(loaded, args)
    mask, years, counts = eligibility(df, preprocessing["language"], args)
    console.print(f"[blue]→[/blue] {counts['docs_eligible']:,} eligible {preprocessing['language']} documents "
                  f"with text and a point year; {counts['dates']['excluded']:,} without a point date")
    if not mask.any():
        raise ValueError("No documents meet the language/text/date requirements")
    # Sorted by item id, so a fixed seed gives the same bands whatever the
    # row order of the input.
    idx = df.loc[mask].sort_values("o:id", key=lambda ids: ids.astype(str), kind="stable").index
    min_vocab = bundle["min_vocab_tokens"] if args.min_vocab_tokens is None else args.min_vocab_tokens
    reused_theta = (load_theta_export(args.theta_path, bundle["id"], args.repo, args.config, df.loc[mask],
                                      n_topics, preprocessing) if args.theta_path else {})
    clusters_all = outlet_clusters(df, idx, args.bootstrap_unit)
    kept, theta, below_floor = document_vectors(df, idx, bundle, reused_theta, min_vocab)
    if below_floor:
        console.print(f"[yellow]ℹ[/yellow] Left out {below_floor:,} document(s) below the evidence floor "
                      f"({min_vocab} in-vocabulary tokens)")
    if not kept:
        raise ValueError("Eligible documents contain no vocabulary recognized by this model")
    doc_years = years.loc[kept].astype(int).reset_index(drop=True)
    countries = df.loc[kept, "country"].astype("string").str.strip().replace("", pd.NA).reset_index(drop=True)
    clusters = clusters_all.loc[kept].reset_index(drop=True) if clusters_all is not None else None

    weights, standardized = {}, None
    if args.standardize == "country":
        weights = standard_weights(countries)
        standardized = standardized_series(theta, doc_years, countries, weights)
    prev_year = yearly_prevalence(theta, doc_years, clusters, bundle, args, standardized)
    prev_yc, dropped_cells = country_year_prevalence(theta, doc_years, countries, bundle, args.min_docs_cell)
    if dropped_cells:
        console.print(f"[yellow]ℹ[/yellow] Dropped {dropped_cells} year×country cell(s) with < {args.min_docs_cell} docs")
    insufficient = prev_year.drop_duplicates("year")["ci_status"].eq("insufficient_clusters").sum()
    single_doc_years = int(prev_year.drop_duplicates("year")["ci_status"].eq("insufficient_docs").sum())
    if insufficient:
        console.print(f"[yellow]⚠[/yellow] {insufficient} year(s) have fewer than {MIN_BOOTSTRAP_CLUSTERS} outlets: "
                      "no band reported (insufficient clusters, not zero width)")
    trends_df, solid_years = topic_trends(prev_year, bundle, args)
    print_trends(trends_df, args.trend_test, n_topics)

    lda_model = bundle["model"]
    labels_df = pd.DataFrame([{"lda_model_name": bundle["id"], "topic_id": t, "label": bundle["labels"][t],
                               "top_words": ", ".join(w for w, _ in lda_model.show_topic(t, topn=10))}
                              for t in range(n_topics)])
    per_year = prev_year.drop_duplicates("year")
    summary = {
        "model_dir": str(bundle["dir"]),
        "model_id": bundle["id"],
        "model_training_match": training,
        "estimand": "Mean topic probability among eligible archived documents, conditional on the fitted model",
        "uncertainty": "Conditional resampling; excludes archive selection, OCR and fitted-model uncertainty",
        "bootstrap_unit": args.bootstrap_unit,
        "bootstrap_seed": args.seed,
        "min_bootstrap_clusters": MIN_BOOTSTRAP_CLUSTERS if args.bootstrap_unit == "newspaper" else None,
        "years_with_insufficient_clusters": int(insufficient),
        "years_with_insufficient_docs": single_doc_years,
        "trend_test": args.trend_test,
        "trend_test_assumption": "Independent annual observations" if args.trend_test != "none" else None,
        "trend_series": "raw (pooled) prevalence",
        "standardization": None if standardized is None else {
            "method": "direct; fixed country weights = share of analysed documents over the whole window",
            "weights": weights,
            "complete_years": int(per_year["standard_weight_coverage"].ge(1 - 1e-9).sum()),
            "documents_without_country": int(countries.isna().sum()),
            "note": "Reported beside the raw series; years missing any weighted country have no standardised value."},
        "theta_rows_reused": len(reused_theta),
        "num_topics": n_topics,
        "docs_used": len(kept),
        **counts,
        "min_vocab_tokens": min_vocab,
        "docs_below_min_evidence": below_floor,
        "docs_without_model_vocabulary": counts["docs_eligible"] - len(kept) - below_floor,
        "years_covered": [int(doc_years.min()), int(doc_years.max())],
        "trend_year_window": [solid_years[0], solid_years[-1]] if solid_years else None,
        "min_docs_year": args.min_docs_year,
        "year_country_cells_dropped": dropped_cells,
        "topics": trends_df.to_dict("records"),
    }
    params = bundle["dir"] / "training_parameters.json"
    save_reports(args, df, "topic_prevalence",
                 {"year": prev_year, "year_country": prev_yc, "topic_labels.csv": labels_df}, summary,
                 writer=write_run_manifest, inputs={
                     "model_dir": str(bundle["dir"]), "model_id": bundle["id"],
                     "model_training_match": training,
                     "preprocessing_sha256": preprocessing_fingerprint(preprocessing),
                     "theta_sha256": sha256(Path(args.theta_path)) if args.theta_path else None,
                     "training_parameters": json.loads(params.read_text(encoding="utf-8")) if params.exists() else None,
                 })
    console.print(f"\n[green]✓[/green] Outputs in [cyan]{args.output_dir}[/cyan]")


if __name__ == "__main__":
    main()
