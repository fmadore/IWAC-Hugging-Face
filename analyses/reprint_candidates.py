#!/usr/bin/env python3
"""Suggest reprint candidates using BOTH semantic and lexical evidence.

No rows are deleted or merged. A pair is never called a verified reprint until
independent comparison of the sources. Similar subject matter alone is not
sufficient. Exact pairwise search is blockwise, CPU-only and quadratic in the
eligible corpus size; --top-k limits lexical checks per source, not this cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from iwac_pipeline.analyses._research import (
        ShingleJaccard, add_input_args, dates, load_input, save_reports, shingles, source_link, string, words,
    )
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _research import (  # noqa: E402
        ShingleJaccard, add_input_args, dates, load_input, save_reports, shingles, source_link, string, words,
    )

__all__ = ["evaluate_pairs", "reprint_candidates", "shingles", "vector"]

PAIR_COLUMNS = ("id_a", "id_b", "url_a", "url_b", "cosine", "lexical_jaccard", "day_gap",
                "status", "human_reprint", "reviewer", "notes")


def vector(value):
    try:
        value = json.loads(value) if isinstance(value, str) else value
        result = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(result))
        if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all() or norm <= 0:
            return None
        return result / norm
    except (TypeError, ValueError):
        return None


def _eligible_records(df, *, embedding_column, text_column, shingle_size, max_days):
    """Per-document vector, link and exact day; text stays in ``df``.

    Only the token count is checked here: shingle sets are built later, and
    only for documents that appear in a cosine candidate pair.
    """
    _, precision = dates(df)
    records, exclusions = [], []
    config_column = f"{embedding_column}_config_hash"
    # Sort first so candidate selection and pair identities survive row shuffling.
    ordered = df.sort_values("o:id", key=lambda s: s.astype(str))
    for idx, row in zip(ordered.index, ordered.to_dict("records")):
        reason = ""
        v = vector(row[embedding_column])
        if v is None:
            reason = "missing_or_invalid_embedding"
        elif len(words(string(row[text_column]))) < shingle_size:
            reason = "text_shorter_than_shingle"
        day = pd.Timestamp(string(row.get("pub_date"))) if precision.loc[idx] == "day" else None
        if max_days is not None and day is None:
            reason = "exact_date_required_for_day_window"
        if reason:
            exclusions.append({"o:id": string(row["o:id"]), "reason": reason})
        else:
            records.append({"id": string(row["o:id"]), "vector": v, "url": source_link(row), "day": day,
                            "config_hash": string(row.get(config_column))})
    return records, exclusions


def _check_vector_space(records, allow_unverified_embeddings):
    config_hashes = {row["config_hash"] for row in records if row["config_hash"]}
    unverified = sum(not row["config_hash"] for row in records)
    if len(config_hashes) > 1:
        raise ValueError("Mixed embedding configurations; do not compare incompatible vector spaces")
    if unverified and not allow_unverified_embeddings:
        raise ValueError("Embedding provenance missing; recompute or explicitly use --allow-unverified-embeddings for exploratory screening")
    if len({len(row["vector"]) for row in records}) > 1:
        raise ValueError("Mixed embedding dimensions; compare one model/dimension space only")
    return config_hashes, unverified


def _cosine_candidates(records, *, min_cosine, top_k, block_size, max_days):
    """Yield ``(left, right, cosine)`` once per unordered pair; count capped sources."""
    stats = {"capped": 0}

    def generate():
        seen = set()
        matrix = np.vstack([row["vector"] for row in records])
        for offset in range(0, len(records), block_size):
            similarities = matrix[offset:offset + block_size] @ matrix.T
            for local, similarities_row in enumerate(similarities):
                i = offset + local
                day = records[i]["day"]
                candidate_ids = [int(j) for j in np.flatnonzero(similarities_row >= min_cosine) if j != i and
                                 (max_days is None or abs((day - records[j]["day"]).days) <= max_days)]
                # Stable tie-breaks: deterministic even for identical vectors.
                candidate_ids.sort(key=lambda j: (-float(similarities_row[j]), records[j]["id"]))
                if len(candidate_ids) > top_k:
                    stats["capped"] += 1
                for j in candidate_ids[:top_k]:
                    pair = tuple(sorted((i, j)))
                    if pair not in seen:
                        seen.add(pair)
                        yield (*pair, float(np.clip(similarities_row[j], -1, 1)))

    return (generate() if records else iter(())), stats


def reprint_candidates(df, *, embedding_column="embedding_OCR", text_column="OCR",
                       min_cosine=0.91, min_jaccard=0.45, shingle_size=3, top_k=20,
                       block_size=128, max_days=None, excerpt_chars=0, allow_unverified_embeddings=False):
    if not -1 <= min_cosine <= 1 or not 0 <= min_jaccard <= 1:
        raise ValueError("Similarity thresholds outside their valid ranges")
    if min(shingle_size, top_k, block_size) < 1 or excerpt_chars < 0 or (max_days is not None and max_days < 0):
        raise ValueError("Invalid sizes or date gap")
    if embedding_column not in df or text_column not in df:
        raise ValueError("Reprint screening requires embeddings AND lexical text")
    if "o:id" not in df or df["o:id"].astype(str).duplicated().any():
        raise ValueError("Unique source IDs are required")
    records, exclusions = _eligible_records(df, embedding_column=embedding_column, text_column=text_column,
                                            shingle_size=shingle_size, max_days=max_days)
    config_hashes, unverified = _check_vector_space(records, allow_unverified_embeddings)
    texts = pd.Series(df[text_column].to_numpy(), index=df["o:id"].astype(str).to_numpy())
    jaccard = ShingleJaccard(texts, shingle_size)
    candidates, cosine_stats = _cosine_candidates(records, min_cosine=min_cosine, top_k=top_k,
                                                  block_size=block_size, max_days=max_days)
    pairs, checked = [], 0
    for left, right, cosine in candidates:
        checked += 1
        a, b = records[left], records[right]
        score = jaccard(a["id"], b["id"])
        if not score >= min_jaccard:
            continue
        result = {"id_a": a["id"], "id_b": b["id"], "url_a": a["url"], "url_b": b["url"],
                  "cosine": cosine, "lexical_jaccard": score,
                  "day_gap": abs((a["day"] - b["day"]).days) if a["day"] is not None and b["day"] is not None else None,
                  "status": "candidate_unverified", "human_reprint": "", "reviewer": "", "notes": ""}
        if excerpt_chars:
            result.update({"excerpt_a": string(texts.get(a["id"]))[:excerpt_chars],
                           "excerpt_b": string(texts.get(b["id"]))[:excerpt_chars]})
        pairs.append(result)
    result = pd.DataFrame(pairs, columns=[*PAIR_COLUMNS, *(["excerpt_a", "excerpt_b"] if excerpt_chars else [])])
    if not result.empty:
        result = result.sort_values(["cosine", "lexical_jaccard", "id_a", "id_b"], ascending=[False, False, True, True])
    summary = {"documents": len(df), "eligible": len(records), "excluded": len(exclusions),
               "pairs_checked_lexically": checked, "candidate_pairs": len(result),
               "documents_shingled": len(jaccard.cache),
               "sources_truncated_by_top_k": cosine_stats["capped"], "embedding_config_hashes": sorted(config_hashes),
               "documents_with_unverified_embedding_provenance": unverified,
               "interpretation": "Unverified candidates, not duplicate labels. Top-k, thresholds, missing text and embedding coverage limit recall. Embeddings must share a model and preprocessing; dimensions alone cannot establish compatibility."}
    return result, pd.DataFrame(exclusions, columns=("o:id", "reason")), summary


def evaluate_pairs(candidates, evaluated, *, population_ids=None):
    """Evaluate on externally adjudicated pairs only; never imply corpus recall."""
    required = {"id_a", "id_b", "human_reprint"}
    if not required.issubset(evaluated):
        raise ValueError(f"Evaluation file requires {sorted(required)}")
    predicted = {tuple(sorted((str(row.id_a), str(row.id_b)))) for row in candidates.itertuples()}
    labels = {}
    seen = set()
    unlabelled = 0
    for row in evaluated.itertuples():
        a, b = string(row.id_a), string(row.id_b)
        if not a or not b or a == b:
            raise ValueError("Evaluated pairs require two distinct source IDs")
        if population_ids is not None and (a not in population_ids or b not in population_ids):
            raise ValueError("Evaluated pair refers to an ID outside this input snapshot")
        key = tuple(sorted((a, b)))
        if key in seen:
            raise ValueError(f"Duplicate adjudicated pair: {key}")
        seen.add(key)
        label = string(row.human_reprint).lower()
        if not label:
            unlabelled += 1
            continue
        if label not in ("true", "false", "1", "0", "yes", "no"):
            raise ValueError("human_reprint must be blank, true/false, yes/no, or 1/0")
        labels[key] = label in ("true", "1", "yes")
    tp = sum(value and pair in predicted for pair, value in labels.items())
    fp = sum(not value and pair in predicted for pair, value in labels.items())
    fn = sum(value and pair not in predicted for pair, value in labels.items())
    tn = sum(not value and pair not in predicted for pair, value in labels.items())
    return {"adjudicated_pairs": len(labels), "unlabelled_pairs": unlabelled,
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "precision_on_evaluated_pairs": tp / (tp + fp) if tp + fp else None,
            "recall_on_evaluated_pairs": tp / (tp + fn) if tp + fn else None,
            "scope": "Only the supplied adjudicated pairs. Estimating recall requires independently sampled non-candidate pairs; reviewing candidates alone cannot measure corpus recall."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_args(parser, excerpts=True)
    parser.add_argument("--embedding-column", default="embedding_OCR")
    # Recall-oriented prefilter for EmbeddingGemma 2 article vectors, chosen by
    # iwac-analyze similarity-calibration (2026-10). Its held-out recall came from
    # the old per-pair split; re-run the calibration before quoting it.
    parser.add_argument("--min-cosine", type=float, default=0.91)
    parser.add_argument("--min-jaccard", type=float, default=0.45)
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-days", type=int, help="Requires exact day dates on both items")
    parser.add_argument("--evaluated-pairs", type=Path, help="CSV with id_a,id_b,human_reprint adjudications")
    parser.add_argument("--allow-unverified-embeddings", action="store_true",
                        help="Explicitly allow legacy embeddings without config fingerprints; exploratory only")
    args = parser.parse_args()
    columns = ["o:id", "iwac_url", "pub_date", args.embedding_column, args.text_column,
               f"{args.embedding_column}_config_hash"]
    df = load_input(args, columns=list(dict.fromkeys(columns)))
    pairs, excluded, summary = reprint_candidates(df, embedding_column=args.embedding_column,
        text_column=args.text_column, min_cosine=args.min_cosine, min_jaccard=args.min_jaccard,
        shingle_size=args.shingle_size, top_k=args.top_k, block_size=args.block_size,
        max_days=args.max_days, excerpt_chars=args.excerpt_chars,
        allow_unverified_embeddings=args.allow_unverified_embeddings)
    if args.evaluated_pairs:
        adjudicated = pd.read_csv(args.evaluated_pairs, dtype=str, keep_default_na=False)
        summary["evaluation"] = evaluate_pairs(pairs, adjudicated, population_ids=set(df["o:id"].astype(str)))
    save_reports(args, df, "reprint_candidates", {"pairs": pairs, "exclusions": excluded}, summary)


if __name__ == "__main__":
    main()
