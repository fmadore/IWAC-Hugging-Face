#!/usr/bin/env python3
"""Audit corpus composition, missingness and an explicit analysis inclusion rule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from iwac_pipeline.analyses._research import (
        MISSING, POINT_PRECISIONS, add_input_args, date_exclusions, dates, load_input, present,
        save_reports, series, source_link, string,
    )
    from iwac_pipeline.analyses._stats import language_mask
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _research import (  # noqa: E402
        MISSING, POINT_PRECISIONS, add_input_args, date_exclusions, dates, load_input, present,
        save_reports, series, source_link, string,
    )
    from _stats import language_mask  # noqa: E402


def _access(df: pd.DataFrame) -> pd.Series:
    """Item privacy wins over text privacy; absent flags stay ``unknown``."""
    parent_raw = series(df, "o:is_public") if "o:is_public" in df else series(df, "item_is_public")
    parent = parent_raw.map(lambda v: string(v).lower())
    text = series(df, "OCR_is_public").map(lambda v: string(v).lower())
    access = pd.Series("unknown", index=df.index, dtype=object)
    access[text.isin(("false", "0"))] = "restricted_text"
    access[text.isin(("true", "1"))] = "public_text"
    access[parent.isin(("false", "0"))] = "restricted_item"
    return access


def _ledger(df: pd.DataFrame, metrics) -> pd.DataFrame:
    years, precision = dates(df)
    ledger = pd.DataFrame({"o:id": df["o:id"].astype(str),
                           "iwac_url": [source_link(row) for row in df.to_dict("records")]},
                          index=df.index)
    for column in ("country", "newspaper", "newspaper_ids", "language"):
        ledger[column] = series(df, column).map(lambda x: string(x, MISSING))
    ledger["outlet_key"] = ledger["newspaper_ids"].where(
        ledger["newspaper_ids"].ne(MISSING), ledger["newspaper"])
    ledger["pub_date"] = series(df, "pub_date").map(string)
    ledger["year"] = years
    ledger["date_precision"] = precision
    ledger["access"] = _access(df)
    for column in metrics:
        ledger[f"has:{column}"] = series(df, column).map(present)
    return ledger


def _exclusion_reasons(df, ledger, *, year_min, year_max, languages, require) -> pd.Series:
    """One ``|``-joined reason string per row; empty means included."""
    point = ledger["date_precision"].isin(POINT_PRECISIONS)
    flags = {"missing_or_ambiguous_date": ~point,
             "outside_year_window": point & ~ledger["year"].between(year_min, year_max).fillna(False)}
    if languages:
        selected = pd.Series(False, index=df.index)
        for language in languages:
            selected |= language_mask(series(df, "language"), language)
        flags["language_not_selected"] = ~selected
    for column in require:
        flags[f"missing:{column}"] = ~ledger[f"has:{column}"]
    reasons = pd.Series("", index=df.index, dtype=object)
    for name, mask in flags.items():
        mask = mask.astype(bool)
        reasons[mask] = reasons[mask].map(lambda r, n=name: f"{r}|{n}" if r else n)
    return reasons


def _stable_outlets(ledger, year_min, year_max, min_outlet_year) -> set:
    """Outlets meeting the minimum in EVERY year of the window (gaps count as zero)."""
    n_years = year_max - year_min + 1
    counts = ledger[ledger["included"]].groupby(["country", "outlet_key", "year"]).size()
    return {(country, outlet) for (country, outlet), group in counts.groupby(level=[0, 1])
            if outlet != MISSING and len(group) == n_years and group.min() >= min_outlet_year}


def coverage_report(df, *, year_min=1900, year_max=2100, languages=(), require=(),
                    min_outlet_year=1):
    """Return aggregate counts plus a row-level inclusion ledger, never article text.

    Multi-valued country/language strings remain joint categories, so totals
    have one denominator per document. Ranges are counted but excluded from
    the eligible year-based cohort rather than silently using their start.
    """
    if year_min > year_max or min_outlet_year < 1:
        raise ValueError("Invalid year window or outlet-year minimum")
    for column in require:
        if column not in df:
            raise ValueError(f"Required column is absent: {column}")
    metrics = list(dict.fromkeys(["OCR", "lemma_nostop", "embedding_OCR", "lda_topic_id", *require,
                                *(c for c in df if c.endswith(("_polarite", "_subjectivite_score",
                                                              "_centralite_islam_musulmans")))]))
    ledger = _ledger(df, metrics)
    reasons = _exclusion_reasons(df, ledger, year_min=year_min, year_max=year_max,
                                 languages=languages, require=require)
    ledger["exclusion_reasons"] = reasons
    ledger["included"] = reasons.eq("")
    aggregates = []
    for dims in (("country",), ("newspaper",), ("year",), ("language",),
                 ("access",), ("date_precision",), ("country", "outlet_key", "year")):
        for key, group in ledger.groupby(list(dims), dropna=False, sort=True):
            keys = key if isinstance(key, tuple) else (key,)
            record = {"dimension": "|".join(dims), "category": "|".join(string(k, MISSING) for k in keys),
                      "n_documents": len(group), "n_included": int(group["included"].sum())}
            for column in metrics:
                record[f"n:{column}"] = int(group[f"has:{column}"].sum())
                record[f"fraction:{column}"] = float(group[f"has:{column}"].mean())
            aggregates.append(record)
    stable = _stable_outlets(ledger, year_min, year_max, min_outlet_year)
    ledger["stable_outlet_cohort"] = ledger["included"] & pd.Series(
        list(zip(ledger["country"], ledger["outlet_key"])), index=ledger.index).isin(stable)
    ledger = ledger.reset_index(drop=True)
    exploded = [x for r in reasons if r for x in r.split("|")]
    summary = {"documents": len(df), "included": int(ledger["included"].sum()),
               "excluded": int((~ledger["included"]).sum()),
               "year_window": [year_min, year_max], "stable_outlets": len(stable),
               "stable_outlet_documents": int(ledger["stable_outlet_cohort"].sum()),
               "date_exclusions": date_exclusions(ledger["date_precision"]),
               "column_available": {c: c in df for c in metrics},
               "exclusion_counts_overlapping": {str(k): int(v) for k, v in pd.Series(exploded, dtype=str).value_counts().items()},
               "cohort_reuse": "Pass corpus_coverage_inclusion.csv to another analysis with "
                               "--ids-file ... --ids-where stable_outlet_cohort (or included).",
               "interpretation": "Corpus coverage, not population representativeness. Missing labels stay missing; ranges are not assigned a point year."}
    return pd.DataFrame(aggregates), ledger, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_args(parser)
    parser.add_argument("--year-min", type=int, default=1900)
    parser.add_argument("--year-max", type=int, default=2100)
    parser.add_argument("--languages", nargs="*", default=[])
    parser.add_argument("--require", nargs="*", default=[], help="Nonempty columns required for the analysis cohort")
    parser.add_argument("--min-outlet-year", type=int, default=1)
    args = parser.parse_args()
    from iwac_common.sentiment_panel import all_columns
    annotation_columns = [c for c in all_columns() if c.endswith(
        ("_polarite", "_subjectivite_score", "_centralite_islam_musulmans"))]
    columns = list(dict.fromkeys(["o:id", "iwac_url", "country", "newspaper", "newspaper_ids",
        "pub_date", "language", "OCR_is_public", "o:is_public", "item_is_public", "OCR",
        "lemma_nostop", "embedding_OCR", "lda_topic_id", *annotation_columns, *args.require]))
    df = load_input(args, columns=columns)
    cells, ledger, summary = coverage_report(df, year_min=args.year_min, year_max=args.year_max,
        languages=args.languages, require=args.require, min_outlet_year=args.min_outlet_year)
    save_reports(args, df, "corpus_coverage", {"cells": cells, "inclusion": ledger}, summary)


if __name__ == "__main__":
    main()
