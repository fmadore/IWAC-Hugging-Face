#!/usr/bin/env python3
"""Prepare reproducible, blinded human annotation sheets and a separate challenge set."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iwac_pipeline.analyses._research import (
    MISSING, METADATA, add_input_args, dates, load_input, metadata, present, save_reports,
    series, stable_rank, string,
)
from iwac_common.sentiment_panel import generation, latest_generation

DIMENSIONS = ("polarite", "subjectivite_score", "centralite_islam_musulmans")


def annotation_sample(df, *, per_stratum=3, challenge_size=30, seed=42,
                      strata=("country", "decade", "newspaper", "language"),
                      models=None, difficult_ids=(), excerpt_chars=0, text_column="OCR"):
    if per_stratum < 1 or challenge_size < 0 or excerpt_chars < 0:
        raise ValueError("Sample sizes must be positive (challenge and excerpts may be zero)")
    if not strata or len(set(strata)) != len(strata):
        raise ValueError("Supply distinct nonempty strata")
    if "o:id" not in df or df["o:id"].astype(str).duplicated().any():
        raise ValueError("Unique source IDs are required")
    models = tuple(latest_generation() if models is None else models)
    if not models or len({m.generation for m in models}) != 1:
        raise ValueError("Select exactly one annotation generation")
    years, precision = dates(df)
    work = df.copy()
    work["decade"] = [str((int(y) // 10) * 10) if pd.notna(y) and p in ("year", "month", "day") else MISSING
                      for y, p in zip(years, precision)]
    for column in strata:
        if column != "decade" and column not in df:
            raise ValueError(f"Stratum column is absent: {column}")
        work[column] = series(work, column).map(lambda value: string(value, MISSING))
    work["_rank"] = work["o:id"].astype(str).map(lambda item_id: stable_rank(seed, item_id))
    records, audit, chosen = [], [], set()
    def append(row, group, reason, population=None, sampled=None):
        item_id = str(row["o:id"])
        record = metadata(row, excerpt_chars=excerpt_chars, text_column=text_column)
        record["sample_group"] = group
        record["stratum"] = "|".join(string(row.get(c), MISSING) for c in strata)
        record["stratum_population"] = population
        record["stratum_sampled"] = sampled
        record["inclusion_probability"] = sampled / population if population else None
        record["design_weight"] = population / sampled if sampled else None
        for reviewer in ("annotator_a", "annotator_b", "adjudicated"):
            for dimension in ("polarity", "subjectivity", "centrality"):
                record[f"{reviewer}_{dimension}"] = ""
        record.update({"evaluation_target": "", "quoted_stance": "", "negative_event": "",
                       "uncertain": "", "notes": ""})
        records.append(record)
        audit.append({"o:id": item_id, "sample_group": group, "selection_reason": reason,
                      "annotation_generation": models[0].generation})
        chosen.add(item_id)
    for _, group in work.groupby(list(strata), dropna=False, sort=True):
        sampled = group.sort_values(["_rank", "o:id"]).head(per_stratum)
        for _, row in sampled.iterrows():
            append(row, "stratified_probability", "stratified_hash_sample", len(group), len(sampled))
    difficult = set(map(str, difficult_ids))
    unknown = difficult - set(work["o:id"].astype(str))
    if unknown:
        raise ValueError(f"Difficult IDs absent from this snapshot: {sorted(unknown)[:10]}")
    candidates = []
    for idx, row in work.iterrows():
        item_id = str(row["o:id"])
        if item_id in chosen:
            continue
        reasons = []
        if item_id in difficult:
            reasons.append("expert_flagged")
        for dimension in DIMENSIONS:
            values = [string(row.get(m.column(dimension))) for m in models
                      if present(row.get(m.column(dimension)))]
            if len(values) >= 2 and len(set(values)) > 1:
                reasons.append(f"model_disagreement:{dimension}")
        if reasons:
            candidates.append((0 if item_id in difficult else 1, row["_rank"], idx, "|".join(reasons)))
    for _, _, idx, reason in sorted(candidates)[:challenge_size]:
        append(work.loc[idx], "challenge_nonprobability", reason)
    columns = [*METADATA, *( ["text_excerpt"] if excerpt_chars else []), "sample_group", "stratum",
               "stratum_population", "stratum_sampled", "inclusion_probability", "design_weight",
               *(f"{reviewer}_{dimension}" for reviewer in ("annotator_a", "annotator_b", "adjudicated")
                 for dimension in ("polarity", "subjectivity", "centrality")),
               "evaluation_target", "quoted_stance", "negative_event", "uncertain", "notes"]
    sheet = pd.DataFrame(records, columns=columns)
    summary = {"seed": seed, "strata": list(strata), "population": len(df),
               "probability_sample": sum(r["sample_group"] == "stratified_probability" for r in records),
               "challenge_sample": sum(r["sample_group"] == "challenge_nonprobability" for r in records),
               "annotation_generation": models[0].generation,
               "models": [{"id": m.model_id, "prompt_fingerprint": m.prompt_fingerprint} for m in models],
               "interpretation": "Blank independent annotation forms; no gold labels or accuracy estimates. Analyze challenge set separately; use design weights for probability-sample estimates."}
    return sheet, pd.DataFrame(audit, columns=("o:id", "sample_group", "selection_reason", "annotation_generation")), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_args(parser, excerpts=True)
    parser.add_argument("--per-stratum", type=int, default=3)
    parser.add_argument("--challenge-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strata", nargs="+", default=["country", "decade", "newspaper", "language"])
    parser.add_argument("--generation", type=int)
    parser.add_argument("--difficult-ids", type=Path, help="One expert-flagged source ID per line")
    args = parser.parse_args()
    models = generation(args.generation) if args.generation else latest_generation()
    columns = list(dict.fromkeys([*METADATA, *(c for c in args.strata if c != "decade"),
        *(m.column(d) for m in models for d in DIMENSIONS),
        *([args.text_column] if args.excerpt_chars else [])]))
    df = load_input(args, columns=columns)
    difficult = args.difficult_ids.read_text(encoding="utf-8").splitlines() if args.difficult_ids else []
    sheet, audit, summary = annotation_sample(df, per_stratum=args.per_stratum,
        challenge_size=args.challenge_size, seed=args.seed, strata=tuple(args.strata),
        models=models,
        difficult_ids=[x.strip() for x in difficult if x.strip()],
        excerpt_chars=args.excerpt_chars, text_column=args.text_column)
    save_reports(args, df, "annotation_review", {"sheet": sheet, "selection_audit": audit}, summary)


if __name__ == "__main__":
    main()
