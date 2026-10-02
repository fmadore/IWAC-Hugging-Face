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

import pandas as pd

from iwac_pipeline.analyses._research import METADATA, add_input_args, load_input, metadata, present, save_reports, string


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
                 text_column="OCR", allow_legacy_model_names=False):
    if min(representatives, borderline, excerpt_chars) < 0 or representatives + borderline == 0:
        raise ValueError("Choose non-negative review sizes and at least one document per topic")
    required = {"lda_topic_id", "lda_topic_prob", "lda_model_name"}
    if not required.issubset(df):
        raise ValueError(f"Missing topic columns: {sorted(required - set(df))}")
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
        # Rounding of saved probabilities may differ at the last few digits.
        consistent = bool(topk and topk[0][0] == tid and abs(topk[0][1] - probability) <= 0.001)
        runner = topk[1] if consistent and len(topk) > 1 else None
        eligible.append({"index": idx, "o:id": string(row["o:id"]), "model_id": model_id,
                         "immutable_model_id_recorded": verified, "topic_id": tid,
                         "topic_probability": probability,
                         "runner_up_topic": runner[0] if runner else None,
                         "runner_up_probability": runner[1] if runner else None,
                         "margin": probability - runner[1] if runner else None})
    records = []
    frame = pd.DataFrame(eligible)
    if not frame.empty:
        for _, group in frame.groupby(["model_id", "topic_id"], sort=True):
            representative = group.sort_values(["topic_probability", "o:id"], ascending=[False, True]).head(representatives)
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
               "runner_up_probability", "margin", "review_role", "current_label",
               "approved_label", "approved_description", "fit", "supporting_source_ids", "reviewer", "notes"]
    summary = {"population": len(df), "eligible": len(eligible), "excluded": len(exclusions),
               "review_documents": len(records), "model_ids": sorted({x["model_id"] for x in eligible}),
               "model_identity_scope": "Identity recorded in source rows; this report does not independently load or verify model artifacts.",
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
    args = parser.parse_args()
    columns = [*METADATA, "lda_topic_id", "lda_topic_prob", "lda_model_name",
               "lda_topic_topk", "lda_topic_label", *([args.text_column] if args.excerpt_chars else [])]
    df = load_input(args, columns=columns)
    sheet, exclusions, summary = topic_review(df, representatives=args.representatives,
        borderline=args.borderline, excerpt_chars=args.excerpt_chars, text_column=args.text_column,
        allow_legacy_model_names=args.allow_legacy_model_names)
    save_reports(args, df, "topic_review", {"sheet": sheet, "exclusions": exclusions}, summary)


if __name__ == "__main__":
    main()
