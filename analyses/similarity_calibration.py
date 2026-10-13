#!/usr/bin/env python3
"""Calibrate cosine cutoffs for one embedding space against lexical evidence.

Cosine scales differ between embedding models, so a cutoff chosen in one vector
space means something else in the next. This report re-derives the two cutoffs
the pipeline uses, from the stored ``related_articles`` neighbour lists:

- the **reprint prefilter** (``reprint_candidates --min-cosine``) only decides
  which pairs receive the lexical check, so it is chosen for *recall* of
  lexical duplicates;
- the **near-duplicate cutoff** (``related_articles.NEAR_DUP_THRESHOLD``) labels
  pairs on cosine alone, so it is chosen for *precision*.

The reference label is a lexical proxy, not a human judgement: word-trigram
Jaccard of the two texts at or above ``--label-jaccard`` (the reprint screen's
own lexical test). OCR errors hide copies and shared boilerplate creates them.
Only top-k neighbour pairs are scored, so a copy that is not among an item's
nearest neighbours is invisible here. Thresholds are chosen on one half of the
pairs and reported on the other; adjudicated pairs remain the validation step
(docs/research-validation.md).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from iwac_pipeline.analyses._research import add_input_args, load_input, save_reports, stable_rank, string
from iwac_pipeline.analyses.reprint_candidates import shingles

GRID = np.round(np.arange(0.80, 0.9951, 0.005), 3)


def neighbor_pairs(df: pd.DataFrame, related_column: str = "related_articles") -> pd.DataFrame:
    """Unique (id_a, id_b, cosine) pairs from ``"o:id:cosine|..."`` neighbour lists."""
    pairs: dict[tuple[str, str], float] = {}
    for source, value in zip(df["o:id"].astype(str), df[related_column], strict=True):
        if not isinstance(value, str) or not value:
            continue
        for part in value.split("|"):
            target, score = part.rsplit(":", 1)
            if target == source:
                continue
            key = tuple(sorted((source, target)))
            pairs[key] = max(pairs.get(key, -1.0), float(score))
    return pd.DataFrame([(a, b, c) for (a, b), c in sorted(pairs.items())],
                        columns=["id_a", "id_b", "cosine"])


def add_lexical_jaccard(pairs: pd.DataFrame, texts: dict[str, str], shingle_size: int = 3) -> pd.DataFrame:
    """Score each pair by word-shingle Jaccard; pairs lacking text on either side are dropped."""
    cache: dict[str, set] = {}

    def grams(item_id: str) -> set:
        if item_id not in cache:
            cache[item_id] = shingles(string(texts.get(item_id)), shingle_size)
        return cache[item_id]

    scores = []
    for a, b in zip(pairs["id_a"], pairs["id_b"], strict=True):
        left, right = grams(a), grams(b)
        scores.append(len(left & right) / len(left | right) if left and right else np.nan)
    return pairs.assign(lexical_jaccard=scores).dropna(subset=["lexical_jaccard"]).reset_index(drop=True)


def curve(pairs: pd.DataFrame, label: pd.Series, grid=GRID) -> pd.DataFrame:
    """Precision and recall of ``cosine >= t`` against the lexical label, per threshold."""
    rows = []
    positives = int(label.sum())
    for threshold in grid:
        selected = pairs["cosine"] >= threshold
        true_positive = int((selected & label).sum())
        rows.append({"threshold": float(threshold), "selected": int(selected.sum()),
                     "precision": true_positive / int(selected.sum()) if selected.any() else None,
                     "recall": true_positive / positives if positives else None})
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame, *, target_recall: float, target_precision: float, min_selected: int = 20) -> dict:
    """Highest cutoff meeting the recall target; lowest meeting the precision target."""
    recall_ok = table[table["recall"].fillna(0) >= target_recall]
    precise = table[(table["precision"].fillna(0) >= target_precision) & (table["selected"] >= min_selected)]
    return {"prefilter": float(recall_ok["threshold"].max()) if not recall_ok.empty else None,
            "near_duplicate": float(precise["threshold"].min()) if not precise.empty else None}


def calibrate(pairs: pd.DataFrame, *, label_jaccard=0.45, target_recall=0.98, target_precision=0.90, seed=42):
    """Choose cutoffs on a calibration half and evaluate them on the held-out half."""
    if pairs.empty:
        raise ValueError("No scored neighbour pairs; run related_articles first")
    half = pairs.apply(lambda row: int(stable_rank(seed, f"{row.id_a}|{row.id_b}"), 16) % 2, axis=1)
    label = pairs["lexical_jaccard"] >= label_jaccard
    fit, held = half == 0, half == 1
    chosen = choose(curve(pairs[fit], label[fit]), target_recall=target_recall,
                    target_precision=target_precision)
    held_curve = curve(pairs[held], label[held], grid=[t for t in chosen.values() if t is not None])
    held_out = {row.threshold: {"selected": row.selected, "precision": row.precision, "recall": row.recall}
                for row in held_curve.itertuples()}
    return chosen, held_out, curve(pairs, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(parser)
    parser.add_argument("--text-column", default="OCR")
    parser.add_argument("--embedding-column", default="embedding_OCR",
                        help="Column whose configuration fingerprint the cutoffs belong to")
    parser.add_argument("--related-column", default="related_articles")
    parser.add_argument("--label-jaccard", type=float, default=0.45)
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--target-recall", type=float, default=0.98)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config_column = f"{args.embedding_column}_config_hash"
    df = load_input(args, columns=["o:id", args.text_column, args.related_column, config_column])
    configurations = sorted(df[config_column].dropna().astype(str).unique()) if config_column in df else []
    if len(configurations) != 1:
        raise SystemExit(f"Expected one embedding configuration, found {configurations or 'none'}")
    texts = dict(zip(df["o:id"].astype(str), df[args.text_column], strict=True))
    pairs = add_lexical_jaccard(neighbor_pairs(df, args.related_column), texts, args.shingle_size)
    chosen, held_out, table = calibrate(pairs, label_jaccard=args.label_jaccard, target_recall=args.target_recall,
                                        target_precision=args.target_precision, seed=args.seed)
    summary = {
        "embedding_configuration": configurations[0], "scored_pairs": len(pairs),
        "lexical_duplicates": int((pairs["lexical_jaccard"] >= args.label_jaccard).sum()),
        "label": f"word {args.shingle_size}-gram Jaccard >= {args.label_jaccard}",
        "chosen_on_calibration_half": chosen, "held_out_half": held_out,
        "interpretation": "Lexical-proxy calibration over top-k neighbour pairs only. It fixes a cosine "
                          "scale, not a validated reprint operating point; adjudicate pairs before citing "
                          "reprint counts.",
    }
    print(summary)
    save_reports(args, df, "similarity_calibration", {"curve": table}, summary)


if __name__ == "__main__":
    main()
