#!/usr/bin/env python3
"""Audit corpus composition, missingness and an explicit analysis inclusion rule."""
from __future__ import annotations

import argparse

import pandas as pd

from iwac_pipeline.analyses._research import (
    MISSING, add_input_args, dates, load_input, present, save_reports, series, source_link, string,
)


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
    years, precision = dates(df)
    years = years.where(precision.isin(("year", "month", "day")))
    ledger = pd.DataFrame({"o:id": df["o:id"].astype(str),
                           "iwac_url": [source_link(row) for _, row in df.iterrows()]})
    for column in ("country", "newspaper", "newspaper_ids", "language"):
        ledger[column] = series(df, column).map(lambda x: string(x, MISSING))
    ledger["outlet_key"] = ledger["newspaper_ids"].where(
        ledger["newspaper_ids"].ne(MISSING), ledger["newspaper"])
    ledger["pub_date"] = series(df, "pub_date").map(string)
    ledger["year"] = years
    ledger["date_precision"] = precision
    def access(row):
        value = row.get("OCR_is_public")
        parent = row.get("o:is_public", row.get("item_is_public"))
        if string(parent).lower() in ("false", "0"):
            return "restricted_item"
        if string(value).lower() in ("true", "1"):
            return "public_text"
        if string(value).lower() in ("false", "0"):
            return "restricted_text"
        return "unknown"
    ledger["access"] = [access(row) for _, row in df.iterrows()]
    reasons = [[] for _ in range(len(df))]
    for pos, (_, row) in enumerate(df.iterrows()):
        year, prec = years.iloc[pos], precision.iloc[pos]
        if prec not in ("year", "month", "day"):
            reasons[pos].append("missing_or_ambiguous_date")
        elif not year_min <= year <= year_max:
            reasons[pos].append("outside_year_window")
        if languages and string(row.get("language")) not in languages:
            reasons[pos].append("language_not_selected")
        for column in require:
            if not present(row.get(column)):
                reasons[pos].append(f"missing:{column}")
    ledger["exclusion_reasons"] = ["|".join(r) for r in reasons]
    ledger["included"] = [not r for r in reasons]
    metrics = list(dict.fromkeys(["OCR", "lemma_nostop", "embedding_OCR", "lda_topic_id", *require,
                                *(c for c in df if c.endswith(("_polarite", "_subjectivite_score",
                                                              "_centralite_islam_musulmans")))]))
    for column in metrics:
        ledger[f"has:{column}"] = series(df, column).map(present)
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
    included = ledger[ledger["included"]]
    # An outlet must meet the minimum in EVERY year of the requested window.
    # Missing years count as zero, preventing a one-year outlet looking stable.
    n_years = year_max - year_min + 1
    counts = included.groupby(["country", "outlet_key", "year"]).size()
    stable = []
    for (country, outlet), group in counts.groupby(level=[0, 1]):
        if outlet != MISSING and len(group) == n_years and group.min() >= min_outlet_year:
            stable.append((country, outlet))
    stable_set = set(stable)
    ledger["stable_outlet_cohort"] = [bool(inc and (country, outlet) in stable_set)
        for inc, country, outlet in zip(ledger["included"], ledger["country"], ledger["outlet_key"])]
    summary = {"documents": len(df), "included": int(ledger["included"].sum()),
               "excluded": int((~ledger["included"]).sum()),
               "year_window": [year_min, year_max], "stable_outlets": len(stable),
               "stable_outlet_documents": int(ledger["stable_outlet_cohort"].sum()),
               "column_available": {c: c in df for c in metrics},
               "exclusion_counts_overlapping": {str(k): int(v) for k, v in pd.Series([x for r in reasons for x in r], dtype=str).value_counts().items()},
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
