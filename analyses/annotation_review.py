#!/usr/bin/env python3
"""Prepare reproducible, blinded human annotation sheets and a separate challenge set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from iwac_pipeline.analyses._research import (
        MISSING, METADATA, add_input_args, dates, load_input, metadata, save_reports,
        series, stable_rank, string,
    )
    from iwac_pipeline.processing.sentiment_agreement import subjectivite_ordinal
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _research import (  # noqa: E402
        MISSING, METADATA, add_input_args, dates, load_input, metadata, save_reports,
        series, stable_rank, string,
    )
    # _research put post-processing/ on sys.path.
    from sentiment_agreement import subjectivite_ordinal  # noqa: E402
from iwac_common.sentiment_panel import generation, latest_generation  # noqa: E402

DIMENSIONS = ("polarite", "subjectivite_score", "centralite_islam_musulmans")
LABEL_FIELDS = ("polarity", "subjectivity", "centrality")
CODING_FIELDS = ("evaluation_target", "quoted_stance", "negative_event", "uncertain", "notes")
ANNOTATORS = ("annotator_a", "annotator_b")
AUDIT_COLUMNS = ("o:id", "stable_rank", "display_rank", "sample_group", "selection_reason", "stratum",
                 "stratum_population", "stratum_sampled", "inclusion_probability",
                 "design_weight", "annotation_generation")
DESIGN_COLUMNS = frozenset(AUDIT_COLUMNS) - {"o:id"}


def _strata_frame(df, strata, seed):
    years, _ = dates(df)
    work = df.copy()
    work["decade"] = [str((int(y) // 10) * 10) if pd.notna(y) else MISSING for y in years]
    for column in strata:
        if column != "decade" and column not in df:
            raise ValueError(f"Stratum column is absent: {column}")
        work[column] = series(work, column).map(lambda value: string(value, MISSING))
    work["_rank"] = work["o:id"].astype(str).map(lambda item_id: stable_rank(seed, item_id))
    # Sheet order needs its own hash: ordering by the selection rank would put
    # low-rank probability items first and rank-blind challenge items last.
    work["_display"] = work["o:id"].astype(str).map(lambda item_id: display_rank(seed, item_id))
    return work


def display_rank(seed: int, item_id: str) -> str:
    """Position on the blinded sheets, independent of the selection rank."""
    return stable_rank(seed, f"display\0{item_id}")


def _probability_sample(work, strata, per_stratum):
    """Up to ``per_stratum`` lowest hash ranks per stratum, with design weights."""
    selected = []
    for _, group in work.groupby(list(strata), dropna=False, sort=True):
        sampled = group.sort_values(["_rank", "o:id"]).head(per_stratum)
        for idx in sampled.index:
            selected.append((idx, "stratified_probability", "stratified_hash_sample",
                             len(group), len(sampled)))
    return selected


def _challenge_sample(work, models, chosen, difficult, challenge_size):
    """Expert-flagged first, then within-generation model disagreement, by hash rank."""
    disagreement = {}
    for dimension in DIMENSIONS:
        columns = [m.column(dimension) for m in models]
        votes = work.reindex(columns=columns)
        if dimension == "subjectivite_score":
            # Compare scale points, not spellings: "3", "3.0" and the matching
            # label are one vote (a float-typed load prints "3.0").
            votes = votes.apply(lambda col: subjectivite_ordinal(col).map(
                lambda v: "" if pd.isna(v) else str(int(v))))
        else:
            votes = votes.apply(lambda col: col.map(string))
        voted = votes.ne("")
        distinct = votes.where(voted).nunique(axis=1)
        disagreement[dimension] = voted.sum(axis=1).ge(2) & distinct.gt(1)
    candidates = []
    for idx, item_id, rank in zip(work.index, work["o:id"].astype(str), work["_rank"]):
        if item_id in chosen:
            continue
        reasons = ["expert_flagged"] if item_id in difficult else []
        reasons += [f"model_disagreement:{d}" for d in DIMENSIONS if disagreement[d].loc[idx]]
        if reasons:
            candidates.append((0 if item_id in difficult else 1, rank, idx, "|".join(reasons)))
    return [(idx, "challenge_nonprobability", reason, None, None)
            for _, _, idx, reason in sorted(candidates)[:challenge_size]]


def _sheets(work, selected, *, excerpt_chars, text_column):
    """Blind per-annotator forms and an adjudication form, in display-hash order.

    Probability and challenge cases are interleaved and carry no design or
    selection field, so neither annotator can tell why a source is present.
    The order uses a hash independent of the one that selected the items.
    """
    order = sorted(selected, key=lambda item: (work.at[item[0], "_display"], str(work.at[item[0], "o:id"])))
    base = [metadata(work.loc[idx], excerpt_chars=excerpt_chars, text_column=text_column)
            for idx, *_ in order]
    meta_columns = [*METADATA, *(["text_excerpt"] if excerpt_chars else [])]
    sheets = {}
    for annotator in ANNOTATORS:
        fields = ["annotator_id", *LABEL_FIELDS, *CODING_FIELDS]
        sheets[annotator] = pd.DataFrame([{**record, **{f: "" for f in fields}} for record in base],
                                         columns=[*meta_columns, *fields])
    fields = ["adjudicator_id", *(f"adjudicated_{f}" for f in LABEL_FIELDS), *CODING_FIELDS]
    sheets["adjudication"] = pd.DataFrame([{**record, **{f: "" for f in fields}} for record in base],
                                          columns=[*meta_columns, *fields])
    return sheets, order


def annotation_sample(df, *, per_stratum=3, challenge_size=30, seed=42,
                      strata=("country", "decade", "newspaper", "language"),
                      models=None, difficult_ids=(), excerpt_chars=0, text_column="OCR"):
    """Return ``(sheets, audit, summary)``.

    ``sheets`` maps ``annotator_a``/``annotator_b``/``adjudication`` to blank
    forms with identical rows; ``audit`` holds every design and selection field
    keyed by ``o:id`` and is for study management only.
    """
    if per_stratum < 1 or challenge_size < 0 or excerpt_chars < 0:
        raise ValueError("Sample sizes must be positive (challenge and excerpts may be zero)")
    if not strata or len(set(strata)) != len(strata):
        raise ValueError("Supply distinct nonempty strata")
    if "o:id" not in df or df["o:id"].astype(str).duplicated().any():
        raise ValueError("Unique source IDs are required")
    models = tuple(latest_generation() if models is None else models)
    if not models or len({m.generation for m in models}) != 1:
        raise ValueError("Select exactly one annotation generation")
    work = _strata_frame(df, strata, seed)
    difficult = set(map(str, difficult_ids))
    unknown = difficult - set(work["o:id"].astype(str))
    if unknown:
        raise ValueError(f"Difficult IDs absent from this snapshot: {sorted(unknown)[:10]}")
    selected = _probability_sample(work, strata, per_stratum)
    chosen = {str(work.at[idx, "o:id"]) for idx, *_ in selected}
    selected += _challenge_sample(work, models, chosen, difficult, challenge_size)
    sheets, order = _sheets(work, selected, excerpt_chars=excerpt_chars, text_column=text_column)
    audit = pd.DataFrame([{
        "o:id": str(work.at[idx, "o:id"]), "stable_rank": work.at[idx, "_rank"],
        "display_rank": work.at[idx, "_display"],
        "sample_group": group, "selection_reason": reason,
        "stratum": "|".join(string(work.at[idx, c], MISSING) for c in strata),
        "stratum_population": population, "stratum_sampled": sampled,
        "inclusion_probability": sampled / population if population else None,
        "design_weight": population / sampled if sampled else None,
        "annotation_generation": models[0].generation,
    } for idx, group, reason, population, sampled in order], columns=AUDIT_COLUMNS)
    summary = {"seed": seed, "strata": list(strata), "population": len(df),
               "probability_sample": int(audit["sample_group"].eq("stratified_probability").sum()),
               "challenge_sample": int(audit["sample_group"].eq("challenge_nonprobability").sum()),
               "annotation_generation": models[0].generation,
               "models": [{"id": m.model_id, "prompt_fingerprint": m.prompt_fingerprint} for m in models],
               "blinding": "Annotator and adjudication sheets share one order, from a hash independent of "
                           "the selection rank (display_rank), and omit sample "
                           "group, stratum, weights and selection reasons; join the selection audit on o:id "
                           "only after annotation.",
               "interpretation": "Blank independent annotation forms; no gold labels or accuracy estimates. Analyze challenge set separately; use design weights for probability-sample estimates."}
    return sheets, audit, summary


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
    sheets, audit, summary = annotation_sample(df, per_stratum=args.per_stratum,
        challenge_size=args.challenge_size, seed=args.seed, strata=tuple(args.strata),
        models=models,
        difficult_ids=[x.strip() for x in difficult if x.strip()],
        excerpt_chars=args.excerpt_chars, text_column=args.text_column)
    save_reports(args, df, "annotation_review", {**sheets, "selection_audit": audit}, summary)


if __name__ == "__main__":
    main()
