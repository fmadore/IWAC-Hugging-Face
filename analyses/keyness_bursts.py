#!/usr/bin/env python3
"""
keyness_bursts.py
=================

Two classic corpus-DH analyses over the `articles` subset:

1. **Keyness (Dunning log-likelihood, G²)** — the most distinctive vocabulary
   of each country subcorpus and each decade, computed on ``lemma_nostop``
   with the same stopword sets as the LDA pipeline (geographic + generic
   noise removed, Islamic/domain terms kept, per the DH guidelines).

   G² is the *complete* 2×2 log-likelihood ratio (all four cells of
   token/other × corpus contribute), not the token-present two-cell
   shorthand — the latter is not the statistic a df=1 chi-square tests.

   Statistical control: G² conflates effect size and sample size (a tiny
   rate difference on a huge corpus yields a huge G²), so G² is used only
   for the *significance test* — per-token p = chi2.sf(G², df=1), then
   Benjamini–Hochberg correction within each slice's tested token family
   (q_value). The tested family is restricted to *dispersed* tokens: at
   least ``--min-count`` occurrences, ``--min-docs`` documents and (where an
   outlet column exists) ``--min-outlets`` outlets in the slice, so local
   proper names and OCR variants from a single article cannot enter. Tokens
   with q < 0.05 are *ranked by a conservative effect size*: the lower bound
   of the log-ratio's confidence interval (log2 of relative rates,
   Haldane–Anscombe +0.5 smoothing, Poisson standard error), capped at
   ``--top-n``. Keyness runs on French-labelled rows only.

2. **Burst detection (Kleinberg 2-state automaton)** — periods when a
   authority ID (``subject_ids``, joined to the index subset)
   appears far above its corpus base rate: coverage spikes around events.
   Subjects are deduplicated per document (a document listing the same
   subject twice counts once), and the year axis is a contiguous calendar
   range (missing years zero-filled), so burst intervals can span truly
   empty years and log(T) reflects the real calendar span.

   ``--burst-stratify country`` runs the automaton separately per country,
   each against that country's own yearly exposure, instead of pooling
   every country into one denominator.

Outputs (analyses/output/):
- keyness_country.csv    slice, rank, token, log_ratio, log_ratio_ci_low/high, g2,
                         p_value, q_value, count, n_docs, n_outlets, rate_ratio
- keyness_decade.csv     same columns, per decade
- subject_bursts.csv     stratum, authority ID/label, start/end, weight, mentions, exposure
- subject_burst_exposure.csv stratum x year corpus/tagged counts and denominator
- keyness_bursts_summary.json, keyness_bursts.manifest.json

Usage
-----
    python analyses/keyness_bursts.py [--source hub|local] [--revision SHA]
        [--ids-file cohort.csv] [--min-subject-total 30] [--burst-stratify country]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from iwac_pipeline.analyses._research import (
        OUTPUT_DIR, add_input_args, apply_ids_filter, date_exclusions, in_window, load_companion,
        load_input, load_subset_dataframe, point_years, save_reports, write_run_manifest,
    )
    # bh_adjust lives in analyses/_stats.py (shared with topic_prevalence.py).
    from iwac_pipeline.analyses._stats import bh_adjust, excluded_language_counts, french_language_mask
    from iwac_pipeline.analyses.entity_networks import ambiguous_titles, authority_lookups, resolve_entities
    from iwac_pipeline.processing.lda_topic_modeling.constants import (
        DOMAIN_STOPWORDS, FRAGMENT_STOPWORDS, LDA_GEO_STOPWORDS, LDA_GENERIC_STOPWORDS,
    )
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _research import (  # noqa: E402
        OUTPUT_DIR, add_input_args, apply_ids_filter, date_exclusions, in_window, load_companion,
        load_input, load_subset_dataframe, point_years, save_reports, write_run_manifest,
    )
    from _stats import bh_adjust, excluded_language_counts, french_language_mask  # noqa: E402
    from entity_networks import ambiguous_titles, authority_lookups, resolve_entities  # noqa: E402
    # _research put post-processing/ on sys.path.
    from lda_topic_modeling.constants import (  # noqa: E402
        DOMAIN_STOPWORDS, FRAGMENT_STOPWORDS, LDA_GEO_STOPWORDS, LDA_GENERIC_STOPWORDS,
    )
from iwac_common.text_utils import simple_tokenize  # noqa: E402

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()

# FDR threshold for the keyness report (Benjamini–Hochberg q-values).
ALPHA = 0.05
# Two-sided confidence of the log-ratio interval whose lower bound ranks tokens.
LOG_RATIO_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Keyness — Dunning log-likelihood
# ---------------------------------------------------------------------------


def dunning_g2(a: int, b: int, total_a: int, total_b: int) -> float:
    """Signed G² for token overrepresentation in corpus A vs corpus B.

    The full 2×2 log-likelihood ratio test of independence — every cell of

        |          | token | other tokens |
        | corpus A | a     | total_a - a  |
        | corpus B | b     | total_b - b  |

    contributes ``2·O·log(O/E)``, not just the token-present pair. The
    two-cell form (Rayson & Garside 2000) is the common corpus-linguistics
    shorthand and agrees to within ~0.1 % at realistic lemma rates, but it
    understates G² for frequent terms and is not the statistic that
    ``chi2.sf(G², df=1)`` in :func:`keyness_for_slices` actually tests.

    Signed by the two relative rates: positive when overrepresented in A.
    Antisymmetric under swapping the corpora,
    ``dunning_g2(a, b, ta, tb) == -dunning_g2(b, a, tb, ta)``, so a token
    absent from A but present in B scores a finite *negative* value.

    Raises:
        ValueError: if either corpus total is non-positive, or a count falls
            outside its corpus (``0 <= a <= total_a``, ``0 <= b <= total_b``).
    """
    if total_a <= 0 or total_b <= 0:
        raise ValueError(
            f"corpus totals must be positive (total_a={total_a}, total_b={total_b})"
        )
    if not 0 <= a <= total_a:
        raise ValueError(f"count a={a} outside its corpus [0, {total_a}]")
    if not 0 <= b <= total_b:
        raise ValueError(f"count b={b} outside its corpus [0, {total_b}]")

    n = total_a + total_b
    with_token = a + b
    without_token = n - with_token
    cells = (
        (a, total_a * with_token / n),
        (total_a - a, total_a * without_token / n),
        (b, total_b * with_token / n),
        (total_b - b, total_b * without_token / n),
    )
    # O == 0 contributes nothing (lim O·log O = 0 as O → 0). E == 0 only ever
    # coincides with O == 0 here, so skipping those cells also guards O/E.
    g2 = 2.0 * sum(o * math.log(o / e) for o, e in cells if o > 0)
    # Sign: positive when overrepresented in A
    return g2 if (a / total_a) > (b / total_b) else -g2


def log_ratio(a: int, b: int, total_a: int, total_b: int) -> float:
    """Log2 ratio of relative rates with Haldane–Anscombe +0.5 smoothing.

    Effect size for keyness (Hardie's Log Ratio): +1 means the token is twice
    as frequent (per token) in corpus A as in corpus B. Antisymmetric:
    ``log_ratio(a, b, ta, tb) == -log_ratio(b, a, tb, ta)``. The +0.5 on both
    counts keeps zero counts finite.
    """
    if total_a <= 0 or total_b <= 0:
        return float("nan")
    return math.log2(((a + 0.5) / total_a) / ((b + 0.5) / total_b))


def log_ratio_interval(a: int, b: int, total_a: int, total_b: int,
                       confidence: float = LOG_RATIO_CONFIDENCE) -> tuple[float, float]:
    """Wald interval of :func:`log_ratio` on the log2 scale.

    Treating both counts as Poisson, the smoothed log rate ratio has standard
    error ``sqrt(1/(a+0.5) + 1/(b+0.5)) / ln 2``. A token seen 10 times in one
    slice and never elsewhere has a large ratio but a wide interval; its lower
    bound is what the ranking uses.
    """
    centre = log_ratio(a, b, total_a, total_b)
    if not math.isfinite(centre):
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    se = math.sqrt(1.0 / (a + 0.5) + 1.0 / (b + 0.5)) / math.log(2)
    return centre - z * se, centre + z * se


KEYNESS_COLUMNS = ["slice", "rank", "token", "log_ratio", "log_ratio_ci_low", "log_ratio_ci_high",
                   "g2", "p_value", "q_value", "count", "other_count", "n_docs", "n_outlets",
                   "rate_ratio"]


def keyness_for_slices(
    slice_tokens: dict[str, Counter],
    top_n: int,
    min_count: int,
    alpha: float = ALPHA,
    *,
    doc_freq: dict[str, Counter] | None = None,
    outlet_range: dict[str, Counter] | None = None,
    min_docs: int = 1,
    min_outlets: int = 1,
    confidence: float = LOG_RATIO_CONFIDENCE,
) -> pd.DataFrame:
    """Significant, dispersed overrepresented tokens per slice vs. all other slices pooled.

    G² grows with sample size as well as with effect strength, so it is used
    only as the *test statistic*: p = chi2.sf(G², df=1), BH-corrected within
    each slice's tested token family. The family is every token meeting
    ``min_count`` occurrences and, when supplied, ``min_docs`` documents
    (``doc_freq[slice][token]``) and ``min_outlets`` outlets
    (``outlet_range[slice][token]``) in the slice. Tokens with q < alpha are
    *ranked by the lower confidence bound of log_ratio* (then G², then token)
    and capped at top_n.
    """
    totals = {s: sum(c.values()) for s, c in slice_tokens.items()}
    grand: Counter = Counter()
    for c in slice_tokens.values():
        grand.update(c)
    grand_total = sum(totals.values())

    rows = []
    for s, counter in slice_tokens.items():
        total_a = totals[s]
        total_b = grand_total - total_a
        if total_a == 0 or total_b == 0:
            continue
        docs = doc_freq.get(s, Counter()) if doc_freq is not None else None
        outlets = outlet_range.get(s, Counter()) if outlet_range is not None else None
        # Tested family: every dispersed token (over- AND under-represented)
        # — the BH denominator must cover all tests performed.
        tested = []
        for tok, a in counter.items():
            if a < min_count:
                continue
            if docs is not None and docs[tok] < min_docs:
                continue
            if outlets is not None and outlets[tok] < min_outlets:
                continue
            b = grand[tok] - a
            tested.append((tok, a, b, dunning_g2(a, b, total_a, total_b)))
        if not tested:
            continue
        # dunning_g2 is signed (negative = underrepresented); the chi2 test
        # statistic is the unsigned value.
        pvals = stats.chi2.sf(np.abs([g for _, _, _, g in tested]), df=1)
        qvals = bh_adjust(pvals)
        scored = []
        for (tok, a, b, g2), p, q in zip(tested, pvals, qvals):
            if g2 <= 0 or q >= alpha:
                continue  # report only significant overrepresentation
            rate_b = b / total_b
            ratio = (a / total_a) / rate_b if rate_b else float("inf")
            low, high = log_ratio_interval(a, b, total_a, total_b, confidence)
            scored.append({
                "slice": s, "token": tok,
                "log_ratio": round(log_ratio(a, b, total_a, total_b), 3),
                "log_ratio_ci_low": round(low, 3), "log_ratio_ci_high": round(high, 3),
                "g2": round(g2, 2),
                "p_value": float(p), "q_value": float(q),
                "count": a, "other_count": b,
                "n_docs": int(docs[tok]) if docs is not None else None,
                "n_outlets": int(outlets[tok]) if outlets is not None else None,
                "rate_ratio": round(ratio, 2) if math.isfinite(ratio) else None,
            })
        scored.sort(key=lambda x: (-x["log_ratio_ci_low"], -x["g2"], x["token"]))
        for rank, item in enumerate(scored[:top_n], 1):
            rows.append({"rank": rank, **item})
    frame = pd.DataFrame(rows, columns=KEYNESS_COLUMNS if not rows else None)
    return frame[KEYNESS_COLUMNS] if rows else frame


# ---------------------------------------------------------------------------
# Bursts — Kleinberg's 2-state automaton (batch, document-stream variant)
# ---------------------------------------------------------------------------


def kleinberg_bursts(
    r: np.ndarray, d: np.ndarray, years: np.ndarray, s: float = 2.0, gamma: float = 1.0
) -> list[dict]:
    """Detect burst intervals for one term.

    r[t] = docs mentioning the term in year t; d[t] = all docs in year t.
    Two states: base rate p0 = R/D and burst rate p1 = s*p0. Entering the
    burst state costs gamma*ln(T); staying or leaving is free. Returns the
    maximal state-1 intervals with their total weight (cost saved vs. state 0).

    ``years`` must be a *contiguous calendar range* (the caller zero-fills
    missing years) so that T = ln-cost horizon equals the true calendar span
    and intervals never treat non-consecutive years as adjacent. Years with
    d[t] = 0 are handled: sigma() emits zero cost for either state, so burst
    intervals can legitimately span truly empty years.
    """
    r, d, years = np.asarray(r), np.asarray(d), np.asarray(years)
    if r.ndim != 1 or d.shape != r.shape or years.shape != r.shape:
        raise ValueError("Counts, exposure and years must be aligned vectors")
    if not np.isfinite(r).all() or not np.isfinite(d).all() or (r < 0).any() or (d < r).any():
        raise ValueError("Require finite counts with 0 <= mentions <= exposure")
    if s <= 1 or gamma < 0:
        raise ValueError("Require s > 1 and gamma >= 0")
    if len(years) > 1 and not (np.diff(years) == 1).all():
        raise ValueError("years must be a contiguous calendar range")
    T = len(r)
    R, D = float(r.sum()), float(d.sum())
    if T < 2 or R == 0 or D == 0:
        return []
    p0 = R / D
    p1 = min(s * p0, 0.9999)
    if p1 <= p0:
        return []

    def sigma(p: float, rt: float, dt: float) -> float:
        if dt == 0:
            return 0.0
        return -(rt * math.log(p) + (dt - rt) * math.log(1.0 - p))

    trans = gamma * math.log(T)
    INF = float("inf")
    cost = np.full((T, 2), INF)
    back = np.zeros((T, 2), dtype=int)
    cost[0, 0] = sigma(p0, r[0], d[0])
    cost[0, 1] = trans + sigma(p1, r[0], d[0])
    for t in range(1, T):
        for q in (0, 1):
            emit = sigma(p1 if q else p0, r[t], d[t])
            stay = cost[t - 1, q]
            move = cost[t - 1, 1 - q] + (trans if q == 1 else 0.0)
            if stay <= move:
                cost[t, q] = stay + emit
                back[t, q] = q
            else:
                cost[t, q] = move + emit
                back[t, q] = 1 - q

    # Backtrack optimal path
    q = int(np.argmin(cost[T - 1]))
    path = [q]
    for t in range(T - 1, 0, -1):
        q = int(back[t, q])
        path.append(q)
    path.reverse()

    bursts = []
    t = 0
    while t < T:
        if path[t] == 1:
            start = t
            weight = 0.0
            mentions = 0
            while t < T and path[t] == 1:
                weight += sigma(p0, r[t], d[t]) - sigma(p1, r[t], d[t])
                mentions += int(r[t])
                t += 1
            bursts.append({
                "start": int(years[start]), "end": int(years[t - 1]),
                "weight": round(weight, 2), "mentions_in_burst": mentions,
            })
        else:
            t += 1
    return bursts


def contiguous_year_index(observed_years) -> np.ndarray:
    """Contiguous calendar range min..max over the observed years.

    Calendar gaps (years with zero documents) become explicit d[t] = 0
    entries instead of silently collapsing, so Kleinberg's log(T) and the
    interval bookkeeping see the real calendar span.
    """
    ys = np.asarray(list(observed_years), dtype=int)
    return np.arange(int(ys.min()), int(ys.max()) + 1) if len(ys) else np.array([], dtype=int)



def keyness_rows(df: pd.DataFrame, year_min: int, year_max: int, *, include_unknown=False):
    """One eligibility rule for country and decade keyness: French, text, point year in window.

    ``rows.attrs["eligibility"]`` reports what each rule excluded, including
    the excluded language labels (never scored with a French lexicon).
    """
    if "year" in df:
        years = df["year"].astype("Int64")
    else:
        years = point_years(df["pub_date"])[0]
    language = df["language"] if "language" in df else pd.Series(pd.NA, index=df.index, dtype=object)
    text = df["lemma_nostop"].astype("string").fillna("").str.strip().ne("")
    french = french_language_mask(language, include_unknown=include_unknown)
    dated = in_window(years, year_min, year_max)
    keep = dated & text & french
    counts = {"rows_considered": int(len(df)), "eligible": int(keep.sum()),
              "excluded_no_point_year_in_window": int((~dated).sum()),
              "excluded_no_text": int((dated & ~text).sum()),
              "excluded_language": int((dated & text & ~french).sum()),
              "excluded_language_labels": excluded_language_counts(language[dated & text], french[dated & text])}
    rows = df.loc[keep].assign(year=years.loc[keep].astype(int))
    rows.attrs = {"eligibility": counts}
    return rows


def outlet_keys(df: pd.DataFrame) -> pd.Series | None:
    """Outlet identity per row: newspaper ID, else country-scoped title; None if absent."""
    if "newspaper_ids" in df:
        ids = df["newspaper_ids"].astype("string").str.strip().replace("", pd.NA)
    else:
        ids = pd.Series(pd.NA, index=df.index, dtype="string")
    if "newspaper" in df:
        titles = df["newspaper"].astype("string").str.strip().replace("", pd.NA)
        country = (df["country"].astype("string").str.strip().fillna("")
                   if "country" in df else pd.Series("", index=df.index, dtype="string"))
        ids = ids.fillna(country + "|" + titles)
    if ids.isna().all():
        return None
    return ids


def slice_counts(rows: pd.DataFrame, stopwords):
    """Token counts, document frequency and outlet range per country and per decade."""
    outlets = outlet_keys(rows)
    has_outlets = outlets is not None
    tokens = {"country": defaultdict(Counter), "decade": defaultdict(Counter)}
    docs = {"country": defaultdict(Counter), "decade": defaultdict(Counter)}
    outlet_sets = {"country": defaultdict(lambda: defaultdict(set)),
                   "decade": defaultdict(lambda: defaultdict(set))}
    outlet_values = outlets.tolist() if has_outlets else [None] * len(rows)
    countries = rows["country"].tolist() if "country" in rows else [None] * len(rows)
    for text, country, year, outlet in zip(rows["lemma_nostop"], countries, rows["year"], outlet_values):
        # simple_tokenize lowercases, so tokenization matches the LDA preprocessing.
        toks = simple_tokenize(text, stopwords)
        if not toks:
            continue
        distinct = set(toks)
        slices = [("decade", f"{int(year) // 10 * 10}s")]
        if pd.notna(country) and str(country).strip():
            slices.append(("country", str(country).strip()))
        for kind, key in slices:
            tokens[kind][key].update(toks)
            docs[kind][key].update(distinct)
            if has_outlets and pd.notna(outlet):
                for tok in distinct:
                    outlet_sets[kind][key][tok].add(outlet)
    outlet_range = ({kind: {key: Counter({tok: len(v) for tok, v in per.items()})
                            for key, per in by_slice.items()}
                     for kind, by_slice in outlet_sets.items()} if has_outlets else None)
    return tokens, docs, outlet_range


def subject_exposure(df: pd.DataFrame, index: pd.DataFrame, *, denominator="all"):
    """Authority-ID document counts and an explicit exposure universe.

    Tagged means at least one resolved authority ID. All includes untagged
    articles, while the coverage output makes changing cataloguing visible.
    """
    if denominator not in {"all", "tagged"}:
        raise ValueError("denominator must be all or tagged")
    titles, _, fallback = authority_lookups(index)
    ambiguous = ambiguous_titles(titles)
    subjects = [resolve_entities(row, titles, fallback, ambiguous)[0] for row in df.to_dict("records")]
    years = df["year"].astype(int).to_numpy()
    all_counts = Counter(int(y) for y in years)
    tagged_counts = Counter(int(y) for y, entities in zip(years, subjects) if entities)
    mentions = defaultdict(Counter)
    for year, entities in zip(years, subjects):
        for entity in entities:
            mentions[entity][int(year)] += 1
    year_index = contiguous_year_index(years)
    exposure = all_counts if denominator == "all" else tagged_counts
    d_arr = np.array([exposure[int(y)] for y in year_index], dtype=float)
    coverage = pd.DataFrame([{"year": int(y), "n_articles": all_counts[int(y)],
                              "n_tagged": tagged_counts[int(y)], "exposure": exposure[int(y)],
                              "tagged_share": tagged_counts[int(y)] / all_counts[int(y)] if all_counts[int(y)] else None}
                             for y in year_index],
                            columns=["year", "n_articles", "n_tagged", "exposure", "tagged_share"])
    return mentions, titles, year_index, d_arr, coverage


BURST_COLUMNS = ["stratum", "subject_id", "subject", "start", "end", "weight", "mentions_in_burst",
                 "total_mentions", "exposure_in_burst"]


def burst_strata(df: pd.DataFrame, stratify: str):
    """``[(label, rows)]``: one pooled stream, or one stream per country."""
    if stratify == "none":
        return [("all", df)]
    if stratify != "country":
        raise ValueError("stratify must be none or country")
    country = df["country"].astype("string").str.strip().fillna("") if "country" in df else pd.Series("", index=df.index)
    return [(str(name), df[country.eq(name)]) for name in sorted(country.unique()) if name]


def detect_bursts(df: pd.DataFrame, index: pd.DataFrame, *, denominator="all", stratify="none",
                  min_subject_total=30, s=2.0, gamma=1.0):
    """Kleinberg bursts per stratum, each against its own yearly exposure."""
    rows, coverages, tested, gaps = [], [], 0, 0
    for label, part in burst_strata(df, stratify):
        mentions, titles, year_index, d_arr, coverage = subject_exposure(part, index, denominator=denominator)
        coverages.append(coverage.assign(stratum=label))
        gaps += int((d_arr == 0).sum())
        for subject, counts in mentions.items():
            total = sum(counts.values())
            if total < min_subject_total:
                continue
            tested += 1
            r_arr = np.array([counts.get(int(y), 0) for y in year_index], dtype=float)
            for b in kleinberg_bursts(r_arr, d_arr, year_index, s=s, gamma=gamma):
                window = (year_index >= b["start"]) & (year_index <= b["end"])
                rows.append({"stratum": label, "subject_id": subject, "subject": titles[subject], **b,
                             "total_mentions": total, "exposure_in_burst": int(d_arr[window].sum())})
    bursts = pd.DataFrame(rows, columns=BURST_COLUMNS)
    if not bursts.empty:
        bursts = bursts.sort_values(["weight", "stratum", "subject_id"], ascending=[False, True, True])
    coverage_columns = ["stratum", "year", "n_articles", "n_tagged", "exposure", "tagged_share"]
    coverage = (pd.concat(coverages, ignore_index=True)[coverage_columns] if coverages
                else pd.DataFrame(columns=coverage_columns))
    return bursts, coverage, {"subject_streams_tested": tested, "zero_exposure_stratum_years": gaps,
                              "strata": [label for label, _ in burst_strata(df, stratify)]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyness (Dunning G²) + subject burst detection.")
    add_input_args(parser, source_default="hub", output_dir=OUTPUT_DIR, cohort=True, companion_index=True)
    parser.add_argument("--top-n", type=int, default=25, help="Keywords kept per slice")
    parser.add_argument("--min-count", type=int, default=10, help="Min token count in a slice")
    parser.add_argument("--min-docs", type=int, default=5,
                        help="Min documents containing the token in a slice")
    parser.add_argument("--min-outlets", type=int, default=2,
                        help="Min outlets using the token in a slice (ignored without an outlet column)")
    parser.add_argument("--burst-stratify", choices=["none", "country"], default="none",
                        help="Run burst detection per country against that country's exposure")
    parser.add_argument("--min-subject-total", type=int, default=30,
                        help="Min total mentions for a subject to enter burst detection")
    parser.add_argument("--burst-s", type=float, default=2.0, help="Kleinberg burst rate multiplier")
    parser.add_argument("--burst-gamma", type=float, default=1.0, help="Kleinberg entry cost factor")
    parser.add_argument("--year-min", type=int, default=1900)
    parser.add_argument("--year-max", type=int, default=2030)
    parser.add_argument("--include-unknown-language", action="store_true",
                        help="Explicitly include missing/blank language as presumed French")
    parser.add_argument("--burst-denominator", choices=["all", "tagged"], default="all",
                        help="Exposure universe: all dated articles or only authority-tagged articles")
    return parser


def print_report(keyness_country, keyness_decade, bursts) -> None:
    for title, frame, heading in (("country", keyness_country, "Country"), ("decade", keyness_decade, "Decade")):
        table = Table(title=f"Distinctive vocabulary per {title} (top 8 by log-ratio lower bound, q<{ALPHA})",
                      box=box.ROUNDED)
        table.add_column(heading, style="cyan")
        table.add_column("Keywords", style="green")
        for key in sorted(frame["slice"].unique()) if not frame.empty else []:
            top = frame[frame["slice"] == key].head(8)["token"]
            table.add_row(key, ", ".join(top))
        console.print(table)
    if not bursts.empty:
        bt = Table(title="Strongest subject bursts (Kleinberg)", box=box.ROUNDED)
        bt.add_column("Stratum", style="cyan")
        bt.add_column("Subject", style="green", max_width=46)
        bt.add_column("Period", style="cyan", justify="center")
        bt.add_column("Weight", justify="right")
        bt.add_column("Mentions", justify="right")
        for r in bursts.head(20).to_dict("records"):
            period = f"{r['start']}" if r["start"] == r["end"] else f"{r['start']}–{r['end']}"
            bt.add_row(r["stratum"], r["subject"], period, f"{r['weight']:,.0f}", f"{r['mentions_in_burst']:,}")
        console.print(bt)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.year_min > args.year_max:
        parser.error("--year-min cannot exceed --year-max")
    if min(args.min_count, args.min_docs, args.min_outlets) < 1:
        parser.error("--min-count, --min-docs and --min-outlets must be positive")

    console.print(Panel.fit(
        "[bold cyan]Keyness & Burst Detection[/bold cyan]\n"
        "[dim]Dunning G² per country/decade · Kleinberg bursts on subjects[/dim]",
        border_style="cyan",
    ))
    loaded = load_input(args, loader=load_subset_dataframe, columns=[
        "o:id", "lemma_nostop", "pub_date", "country", "newspaper", "newspaper_ids",
        "subject", "subject_ids", "language"])
    loaded = apply_ids_filter(loaded, args)
    years, precision = point_years(loaded["pub_date"])
    df = loaded.assign(year=years)
    valid_year = in_window(years, args.year_min, args.year_max)

    # Keyness runs no phrase detection, so the fragments that the LDA
    # pipeline filters post-phrase can go straight into the stopword set.
    stopwords = (
        set(DOMAIN_STOPWORDS) | FRAGMENT_STOPWORDS
        | LDA_GEO_STOPWORDS | LDA_GENERIC_STOPWORDS
    )

    # ---------------- Keyness (French rows only) ----------------
    text_rows = keyness_rows(df, args.year_min, args.year_max,
                             include_unknown=args.include_unknown_language)
    eligibility = text_rows.attrs["eligibility"]
    console.print(f"[blue]→[/blue] Keyness corpus: {len(text_rows):,} French articles; "
                  f"{eligibility['excluded_language']:,} excluded by language")
    tokens, docs, outlet_range = slice_counts(text_rows, stopwords)
    keyness = {kind: keyness_for_slices(
        tokens[kind], args.top_n, args.min_count, doc_freq=docs[kind],
        outlet_range=outlet_range[kind] if outlet_range else None,
        min_docs=args.min_docs, min_outlets=args.min_outlets) for kind in ("country", "decade")}

    # ---------------- Bursts ----------------
    index = load_companion(args, loaded, "index", columns=["o:id", "Titre", "Type"],
                           loader=load_subset_dataframe, path=args.index_input)
    bursts, coverage, burst_counts = detect_bursts(
        df.loc[valid_year], index, denominator=args.burst_denominator, stratify=args.burst_stratify,
        min_subject_total=args.min_subject_total, s=args.burst_s, gamma=args.burst_gamma)
    console.print(f"[blue]→[/blue] Burst detection: {burst_counts['subject_streams_tested']:,} subject streams; "
                  f"exposure={args.burst_denominator}; stratify={args.burst_stratify}; "
                  f"{burst_counts['zero_exposure_stratum_years']} zero-exposure years")

    summary = {
        "config": args.config,
        "dates": {**date_exclusions(precision),
                  "outside_year_window": int((precision.isin(("day", "month", "year")) & ~valid_year).sum())},
        "keyness": {
            "countries": sorted(tokens["country"]),
            "decades": sorted(tokens["decade"]),
            "year_range": [args.year_min, args.year_max],
            "include_unknown_language": args.include_unknown_language,
            "eligibility": eligibility,
            "n_eligible_articles": len(text_rows),
            "stopword_count": len(stopwords),
            "top_n": args.top_n,
            "alpha": ALPHA,
            "dispersion": {"min_count": args.min_count, "min_docs": args.min_docs,
                           "min_outlets": args.min_outlets if outlet_range else None,
                           "outlet_identity": "newspaper_ids, else country|newspaper" if outlet_range else "no outlet column"},
            "ranking": f"lower bound of the {LOG_RATIO_CONFIDENCE:.0%} log-ratio interval, then G²",
            "note": ("p = chi2.sf(G², df=1), Benjamini–Hochberg within each "
                     "slice's tested family of dispersed tokens; only q < alpha reported, "
                     "ranked by the log-ratio lower confidence bound (G² conflates effect "
                     "and sample size). Token independence is an approximation: these are exploratory corpus-conditional diagnostics, not population inference; repeated articles and outlet clustering can inflate significance."),
        },
        "bursts": {
            **burst_counts,
            "bursts_found": int(len(bursts)),
            "s": args.burst_s, "gamma": args.burst_gamma,
            "denominator": args.burst_denominator,
            "stratify": args.burst_stratify,
            "identity": "authority o:id; ambiguous title fallbacks excluded",
            "note": ("Subjects deduplicated per document; calendar gaps "
                     "zero-filled, so burst intervals can span truly empty "
                     "years; zero-exposure gaps provide no evidence of a continuing event. Interpret spikes against the exported tagging coverage and corpus exposure. "
                     "Pooled exposure ('all' stratum) mixes countries whose coverage changes at different times; "
                     "--burst-stratify country separates them."),
        },
    }
    print_report(keyness["country"], keyness["decade"], bursts)
    save_reports(args, loaded, "keyness_bursts", {
        "keyness_country.csv": keyness["country"], "keyness_decade.csv": keyness["decade"],
        "subject_bursts.csv": bursts, "subject_burst_exposure.csv": coverage,
    }, summary, writer=write_run_manifest)
    console.print(f"\n[green]✓[/green] Outputs in [cyan]{args.output_dir}[/cyan]")


if __name__ == "__main__":
    main()
