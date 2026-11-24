#!/usr/bin/env python3
"""Export source-linked representative and borderline texts for expert topic review.

Uses existing document-topic estimates and never refits a model. Topic IDs are
scoped by immutable model identity. Borderline examples require a runner-up in
the saved top-k distribution; truncated top-k values are not renormalized.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from iwac_pipeline.analyses._research import (
        METADATA, add_input_args, load_input, metadata, present, save_reports, string,
    )
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _research import METADATA, add_input_args, load_input, metadata, present, save_reports, string  # noqa: E402

# Very short texts reach extreme dominant-topic probabilities from a handful of
# tokens; they are poor evidence of what a topic means.
DEFAULT_MIN_WORDS = 100


def parse_topk(value) -> list[tuple[int, float]]:
    if not present(value):
        return []
    result = []
    try:
        for part in str(value).split("|"):
            tid, probability = part.split(":")
            topic_id, probability = int(tid), float(probability)
            if topic_id < 0 or not math.isfinite(probability) or not 0 <= probability <= 1:
                return []
            result.append((topic_id, probability))
    except (TypeError, ValueError):
        return []
    if len({t for t, _ in result}) != len(result) or sum(p for _, p in result) > 1.001:
        return []
    return sorted(result, key=lambda x: (-x[1], x[0]))


def topic_review(df, *, representatives=5, borderline=5, excerpt_chars=0,
                 text_column="OCR", allow_legacy_model_names=False,
                 min_words=DEFAULT_MIN_WORDS, length_column="nb_mots"):
    """Representative documents must have at least ``min_words`` in ``length_column``.

    Rows below the floor, or with an unknown length, stay eligible as
    borderline cases but are never exported as representative; the summary
    counts them. ``min_words=0`` disables the floor.
    """
    if min(representatives, borderline, excerpt_chars, min_words) < 0 or representatives + borderline == 0:
        raise ValueError("Choose non-negative review sizes and at least one document per topic")
    required = {"lda_topic_id", "lda_topic_prob", "lda_model_name"}
    if not required.issubset(df):
        raise ValueError(f"Missing topic columns: {sorted(required - set(df))}")
    if min_words and length_column not in df:
        raise ValueError(f"Length floor needs {length_column}; pass --min-words 0 to disable it explicitly")
    lengths = pd.to_numeric(df[length_column], errors="coerce") if min_words else None
    eligible, exclusions = [], []
    for idx, row in df.iterrows():
        model_id = string(row.get("lda_model_name"))
        verified = bool(re.fullmatch(r"lda-(?:legacy-)?sha256:[0-9a-f]{64}", model_id))
        reason = ""
        try:
            raw_id = float(row["lda_topic_id"])
            tid, probability = int(raw_id), float(row["lda_topic_prob"])
            if tid != raw_id or tid < 0 or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            reason = "missing_or_invalid_topic_estimate"
        if not model_id:
            reason = "missing_model_identity"
        elif not verified and not allow_legacy_model_names:
            reason = "unverified_legacy_model_name"
        if reason:
            exclusions.append({"o:id": string(row["o:id"]), "reason": reason})
            continue
        topk = parse_topk(row.get("lda_topic_topk"))
        # top-k is rounded to four decimals, so a dominant topic tied with the
        # runner-up there may sort second (ties order by id). Find it anywhere
        # among the top entries, provided it is level with the first one.
        dominant = next((p for t, p in topk if t == tid), None)
        # Rounding of saved probabilities may differ at the last few digits.
        consistent = (dominant is not None and topk[0][1] - dominant <= 0.001
                      and abs(dominant - probability) <= 0.001)
        others = [entry for entry in topk if entry[0] != tid] if consistent else []
        runner = others[0] if others else None
        length = lengths.loc[idx] if min_words else None
        eligible.append({"index": idx, "o:id": string(row["o:id"]), "model_id": model_id,
                         "word_count": None if length is None or pd.isna(length) else int(length),
                         "immutable_model_id_recorded": verified, "topic_id": tid,
                         "topic_probability": probability,
                         "runner_up_topic": runner[0] if runner else None,
                         "runner_up_probability": runner[1] if runner else None,
                         # Both values from top-k, so at the same precision.
                         "margin": max(0.0, dominant - runner[1]) if runner else None})
    records = []
    frame = pd.DataFrame(eligible)
    floor = {"min_words": min_words, "length_column": length_column if min_words else None,
             "below_floor": 0, "length_unknown": 0}
    if not frame.empty:
        if min_words:
            words_known = frame["word_count"].notna()
            floor["length_unknown"] = int((~words_known).sum())
            floor["below_floor"] = int((words_known & frame["word_count"].lt(min_words)).sum())
            frame["long_enough"] = words_known & frame["word_count"].ge(min_words)
        else:
            frame["long_enough"] = True
        for _, group in frame.groupby(["model_id", "topic_id"], sort=True):
            representative = (group[group["long_enough"]]
                              .sort_values(["topic_probability", "o:id"], ascending=[False, True])
                              .head(representatives))
            near = group[~group["o:id"].isin(representative["o:id"]) & group["margin"].notna()]
            near = near.sort_values(["margin", "o:id"]).head(borderline)
            for role, selection in (("representative", representative), ("borderline", near)):
                for _, item in selection.iterrows():
                    row = df.loc[item["index"]]
                    record = metadata(row, excerpt_chars=excerpt_chars, text_column=text_column)
                    record.update({k: v for k, v in item.items() if k not in ("index", "o:id")})
                    record.update({"review_role": role, "current_label": string(row.get("lda_topic_label")),
                                   "approved_label": "", "approved_description": "", "fit": "",
                                   "supporting_source_ids": "", "reviewer": "", "notes": ""})
                    records.append(record)
    columns = [*METADATA, *(["text_excerpt"] if excerpt_chars else []), "model_id",
               "immutable_model_id_recorded", "topic_id", "topic_probability", "runner_up_topic",
               "runner_up_probability", "margin", "word_count", "review_role", "current_label",
               "approved_label", "approved_description", "fit", "supporting_source_ids", "reviewer", "notes"]
    summary = {"population": len(df), "eligible": len(eligible), "excluded": len(exclusions),
               "review_documents": len(records), "model_ids": sorted({x["model_id"] for x in eligible}),
               "representative_length_floor": floor,
               "model_identity_scope": "Identity recorded in source rows; this report does not independently load or verify model artifacts.",
               "representative_definition": "Largest stored dominant-topic probability among documents with a known length at or above the floor.",
               "borderline_definition": "Smallest dominant-minus-runner-up probability margin among unselected documents; needs consistent saved top-k with >=2 topics.",
               "interpretation": "Examples support expert interpretation; selection is purposive, not an accuracy estimate. Compare approved descriptions and supporting documents across models, never numeric topic IDs alone."}
    return pd.DataFrame(records, columns=columns), pd.DataFrame(exclusions, columns=("o:id", "reason")), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_args(parser, excerpts=True)
    parser.add_argument("--representatives", type=int, default=5)
    parser.add_argument("--borderline", type=int, default=5)
    parser.add_argument("--allow-legacy-model-names", action="store_true",
                        help="Review mutable historical labels, explicitly marked identity unverified")
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
                        help="Representative documents need at least N words in --length-column (0 disables)")
    parser.add_argument("--length-column", default="nb_mots")
    args = parser.parse_args()
    columns = [*METADATA, "lda_topic_id", "lda_topic_prob", "lda_model_name",
               "lda_topic_topk", "lda_topic_label", *([args.length_column] if args.min_words else []),
               *([args.text_column] if args.excerpt_chars else [])]
    df = load_input(args, columns=columns)
    sheet, exclusions, summary = topic_review(df, representatives=args.representatives,
        borderline=args.borderline, excerpt_chars=args.excerpt_chars, text_column=args.text_column,
        allow_legacy_model_names=args.allow_legacy_model_names,
        min_words=args.min_words, length_column=args.length_column)
    save_reports(args, df, "topic_review", {"sheet": sheet, "exclusions": exclusions}, summary)


if __name__ == "__main__":
    main()
