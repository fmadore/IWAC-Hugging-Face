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
nearest neighbours is invisible here. Thresholds are chosen on one half and
reported on the other; adjudicated pairs remain the validation step
(docs/research-validation.md).

The halves keep reprint *families* apart. A family is a connected component
of the label edges alone (pairs at or above ``--label-jaccard``), which is
independent of the cosine being calibrated; the full top-k neighbour graph is
not used, because it percolates into one component and leaves nothing to hold
out. Families are dealt to the halves in hashed order, each to the half with
fewer positive pairs so far, so positives stay balanced. An item outside every
family takes a side from the hash of its own id. A pair whose two items sit on
different sides belongs to neither half and is counted as excluded, so no item
appears in both halves. Held-out recall and precision carry a bootstrap
interval resampling whole units: the family for a pair touching one (the
smaller family key if it touches two), else the pair's smaller item id. Two
non-family pairs sharing an item are therefore resampled independently; that
dependence is not modelled.

Runs before ``SPLIT_VERSION`` 2 split individual pairs (a family leaked across
both halves); version 2 split whole components of the neighbour graph, which on
the real corpus is a single component, so one half was empty. Neither run's
held-out figures are comparable with these.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from iwac_pipeline.analyses._research import (
        ShingleJaccard, add_input_args, load_input, save_reports, stable_rank,
    )
    from iwac_pipeline.analyses._stats import MIN_BOOTSTRAP_CLUSTERS
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("iwac_pipeline"):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _research import ShingleJaccard, add_input_args, load_input, save_reports, stable_rank  # noqa: E402
    from _stats import MIN_BOOTSTRAP_CLUSTERS  # noqa: E402

GRID = np.round(np.arange(0.80, 0.9951, 0.005), 3)
# 1 = per-pair split (leaked families); 2 = components of the whole neighbour
# graph (one giant component, nothing held out); 3 = lexical families,
# positive-balanced, item-disjoint halves.
SPLIT_VERSION = 3


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
    jaccard = ShingleJaccard(texts, shingle_size)
    scores = [jaccard(a, b) for a, b in zip(pairs["id_a"], pairs["id_b"], strict=True)]
    return pairs.assign(lexical_jaccard=scores).dropna(subset=["lexical_jaccard"]).reset_index(drop=True)


def pair_components(pairs: pd.DataFrame) -> pd.Series:
    """Connected-component key (smallest member ID) of each pair's endpoints."""
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for a, b in zip(pairs["id_a"].astype(str), pairs["id_b"].astype(str), strict=True):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            # The smaller ID always becomes the root: keys do not depend on row order.
            low, high = sorted((root_a, root_b))
            parent[high] = low
    return pd.Series([find(a) for a in pairs["id_a"].astype(str)], index=pairs.index, dtype=object)


def lexical_families(pairs: pd.DataFrame, label: pd.Series) -> dict[str, str]:
    """Item id -> family key (smallest member id), linking on label edges only."""
    linked = pairs[label.to_numpy(dtype=bool)]
    keys = pair_components(linked)
    family: dict[str, str] = {}
    for a, b, key in zip(linked["id_a"].astype(str), linked["id_b"].astype(str), keys, strict=True):
        family[a] = family[b] = key
    return family


def family_halves(pairs: pd.DataFrame, label: pd.Series, seed: int) -> pd.DataFrame:
    """Per pair: ``side`` (0 = calibration, 1 = held out, -1 = straddles both,
    excluded), ``unit`` (bootstrap resampling unit) and ``family`` (or None).

    Deterministic for a seed and independent of row order.
    """
    family = lexical_families(pairs, label)
    ids_a, ids_b = pairs["id_a"].astype(str), pairs["id_b"].astype(str)
    positives = ids_a[label.to_numpy(dtype=bool)].map(family).value_counts().to_dict()
    load, family_side = [0, 0], {}
    for key in sorted(positives, key=lambda k: stable_rank(seed, k)):
        if load[0] != load[1]:
            side = 0 if load[0] < load[1] else 1
        else:
            side = int(stable_rank(seed, key), 16) % 2
        family_side[key] = side
        load[side] += positives[key]

    def item_side(item: str) -> int:
        key = family.get(item)
        return family_side[key] if key is not None else int(stable_rank(seed, item), 16) % 2

    side_a, side_b = ids_a.map(item_side), ids_b.map(item_side)
    family_a, family_b = ids_a.map(family), ids_b.map(family)
    families = [min(k for k in (fa, fb) if isinstance(k, str)) if isinstance(fa, str) or isinstance(fb, str)
                else None for fa, fb in zip(family_a, family_b, strict=True)]
    units = [f"family:{f}" if f is not None else f"pair:{min(a, b)}"
             for f, a, b in zip(families, ids_a, ids_b, strict=True)]
    return pd.DataFrame({"side": side_a.where(side_a == side_b, -1).astype(int),
                         "unit": units, "family": families}, index=pairs.index)


def scored_configurations(df: pd.DataFrame, pairs: pd.DataFrame, config_column: str) -> list[str]:
    """The single embedding configuration behind every scored pair.

    Every item in a scored pair must carry a non-blank configuration hash, and
    all of them the same one: cutoffs belong to one vector space. Items outside
    the scored pairs do not matter.
    """
    if config_column not in df:
        raise SystemExit(f"No {config_column} column; the cutoffs cannot be tied to a configuration")
    scored = set(pairs["id_a"].astype(str)) | set(pairs["id_b"].astype(str))
    values = df.loc[df["o:id"].astype(str).isin(scored), config_column].astype("string").str.strip()
    missing = int((values.isna() | values.eq("")).sum())
    if missing:
        raise SystemExit(f"{missing} item(s) in scored pairs have no {config_column}")
    configurations = sorted(values.unique())
    if len(configurations) != 1:
        raise SystemExit(f"Expected one embedding configuration, found {configurations or 'none'}")
    return configurations


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
    """Highest cutoff meeting the recall target; lowest *stable* precision cutoff.

    The near-duplicate cutoff must reach the precision target with at least
    ``min_selected`` pairs AND keep precision at or above target at every
    higher grid threshold that selects any pair, so a lucky dip in the curve
    cannot be chosen when stricter cutoffs are less precise.
    """
    recall_ok = table[table["recall"].fillna(0) >= target_recall]
    ordered = table.sort_values("threshold").reset_index(drop=True)
    defined = ordered["precision"].notna()
    meets = ordered["precision"].fillna(0) >= target_precision
    # For each row: does every higher threshold with a defined precision meet target?
    failing_above = (defined & ~meets)[::-1].cumsum()[::-1].shift(-1, fill_value=0)
    stable = meets & (ordered["selected"] >= min_selected) & failing_above.eq(0)
    precise = ordered[stable]
    return {"prefilter": float(recall_ok["threshold"].max()) if not recall_ok.empty else None,
            "near_duplicate": float(precise["threshold"].min()) if not precise.empty else None}


def _rates(selected: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    true_positive = float((selected & label).sum())
    chosen, positives = float(selected.sum()), float(label.sum())
    return (true_positive / chosen if chosen else np.nan,
            true_positive / positives if positives else np.nan)


def component_bootstrap(pairs: pd.DataFrame, label: pd.Series, components: pd.Series, threshold: float,
                        *, n_boot: int = 1000, seed: int = 42,
                        min_components: int = MIN_BOOTSTRAP_CLUSTERS) -> dict:
    """Percentile 95 % intervals of precision/recall, resampling whole units
    (``components`` labels each pair's unit: its family, else the pair)."""
    keys, codes = np.unique(components.to_numpy(dtype=str), return_inverse=True)
    result = {"n_units": int(len(keys)), "ci_status": "ok",
              "precision_ci": [None, None], "recall_ci": [None, None],
              "finite_replicates": {"precision": 0, "recall": 0}}
    if n_boot <= 0:
        result["ci_status"] = "disabled"
        return result
    if len(keys) < min_components:
        result["ci_status"] = "insufficient_units"
        return result
    selected = (pairs["cosine"] >= threshold).to_numpy()
    positive = label.to_numpy(dtype=bool)
    members = [np.flatnonzero(codes == i) for i in range(len(keys))]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, 2))
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(keys), size=len(keys))])
        draws[b] = _rates(selected[idx], positive[idx])
    for column, (name, rate) in enumerate((("precision_ci", "precision"), ("recall_ci", "recall"))):
        values = draws[:, column][np.isfinite(draws[:, column])]
        # A replicate with no selected (or no positive) pair has no rate; the
        # interval rests on the rest, and the count says how many that is.
        result["finite_replicates"][rate] = int(len(values))
        if len(values):
            result[name] = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    return result


def calibrate(pairs: pd.DataFrame, *, label_jaccard=0.45, target_recall=0.98, target_precision=0.90,
              seed=42, n_boot=1000):
    """Choose cutoffs on calibration components and evaluate on held-out components."""
    if pairs.empty:
        raise ValueError("No scored neighbour pairs; run related_articles first")
    label = pairs["lexical_jaccard"] >= label_jaccard
    halves = family_halves(pairs, label, seed)
    half, units = halves["side"], halves["unit"]
    fit, held = half == 0, half == 1
    chosen = choose(curve(pairs[fit], label[fit]), target_recall=target_recall,
                    target_precision=target_precision)
    held_curve = curve(pairs[held], label[held], grid=[t for t in chosen.values() if t is not None])
    held_out = {}
    for row in held_curve.itertuples():
        held_out[row.threshold] = {
            "selected": row.selected, "precision": row.precision, "recall": row.recall,
            **component_bootstrap(pairs[held], label[held], units[held], row.threshold,
                                  n_boot=n_boot, seed=seed)}
    families = halves["family"]
    split = {"split_version": SPLIT_VERSION,
             "unit": "lexical family (component of label edges); halves are item-disjoint",
             "families": int(families.nunique()),
             "calibration_families": int(families[fit].nunique()),
             "held_out_families": int(families[held].nunique()),
             "calibration_pairs": int(fit.sum()), "held_out_pairs": int(held.sum()),
             "calibration_positives": int(label[fit].sum()), "held_out_positives": int(label[held].sum()),
             "pairs_excluded_straddling": int((half == -1).sum()),
             "leakage": "None at item level: a pair whose items fall on different sides is excluded. "
                        "Non-family pairs sharing an item are separate bootstrap units."}
    return chosen, held_out, curve(pairs, label), split


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
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Component bootstrap replicates for held-out intervals (0 disables)")
    args = parser.parse_args()
    config_column = f"{args.embedding_column}_config_hash"
    df = load_input(args, columns=["o:id", args.text_column, args.related_column, config_column])
    texts = dict(zip(df["o:id"].astype(str), df[args.text_column], strict=True))
    pairs = add_lexical_jaccard(neighbor_pairs(df, args.related_column), texts, args.shingle_size)
    configurations = scored_configurations(df, pairs, config_column)
    chosen, held_out, table, split = calibrate(
        pairs, label_jaccard=args.label_jaccard, target_recall=args.target_recall,
        target_precision=args.target_precision, seed=args.seed, n_boot=args.bootstrap)
    summary = {
        "embedding_configuration": configurations[0], "scored_pairs": len(pairs),
        "lexical_duplicates": int((pairs["lexical_jaccard"] >= args.label_jaccard).sum()),
        "label": f"word {args.shingle_size}-gram Jaccard >= {args.label_jaccard}",
        "split": split,
        "chosen_on_calibration_half": chosen, "held_out_half": held_out,
        "comparability": "Held-out figures from split_version 1 (per-pair, leaking families) or 2 "
                         "(neighbour-graph components, one of which held every pair) are not "
                         "comparable with these.",
        "interpretation": "Lexical-proxy calibration over top-k neighbour pairs only. It fixes a cosine "
                          "scale, not a validated reprint operating point; adjudicate pairs before citing "
                          "reprint counts. Intervals resample held-out components and exclude "
                          "label and neighbour-list (top-k) uncertainty.",
    }
    print(summary)
    save_reports(args, df, "similarity_calibration", {"curve": table}, summary)


if __name__ == "__main__":
    main()
