"""Cosine cutoffs are re-derived per vector space against lexical evidence."""
from __future__ import annotations

import pandas as pd
import pytest

from iwac_pipeline.analyses.similarity_calibration import (
    add_lexical_jaccard, calibrate, choose, component_bootstrap, curve, family_halves,
    neighbor_pairs, pair_components, scored_configurations,
)
from related_articles import NEAR_DUP_CALIBRATION, near_dup_label


def test_neighbor_pairs_are_undirected_and_keep_the_higher_score():
    df = pd.DataFrame({"o:id": ["1", "2", "3"],
                       "related_articles": ["2:0.9100|3:0.5", "1:0.9200|2:1.0", None]})
    pairs = neighbor_pairs(df)
    assert pairs.to_dict("records") == [
        {"id_a": "1", "id_b": "2", "cosine": 0.92},
        {"id_a": "1", "id_b": "3", "cosine": 0.5},
    ]


def test_lexical_jaccard_drops_pairs_without_text():
    pairs = pd.DataFrame({"id_a": ["1", "1"], "id_b": ["2", "3"], "cosine": [0.9, 0.8]})
    texts = {"1": "a b c d", "2": "a b c e", "3": ""}
    scored = add_lexical_jaccard(pairs, texts, shingle_size=3)
    assert scored["id_b"].tolist() == ["2"]
    assert scored["lexical_jaccard"].iloc[0] == pytest.approx(1 / 3)


def test_prefilter_favours_recall_and_near_duplicate_favours_precision():
    pairs = pd.DataFrame({"cosine": [0.99, 0.98, 0.96, 0.92, 0.91, 0.85]})
    label = pd.Series([True, True, False, True, False, False])
    table = curve(pairs, label, grid=[0.85, 0.90, 0.92, 0.95, 0.97])
    chosen = choose(table, target_recall=1.0, target_precision=1.0, min_selected=1)
    assert chosen == {"prefilter": 0.92, "near_duplicate": 0.97}


def test_calibration_reports_held_out_metrics_for_chosen_cutoffs():
    rows = [{"id_a": str(i), "id_b": str(i + 1000), "cosine": 0.99 if i % 2 else 0.85,
             "lexical_jaccard": 0.9 if i % 2 else 0.0} for i in range(200)]
    chosen, held_out, table, split = calibrate(pd.DataFrame(rows), n_boot=200)
    assert chosen["near_duplicate"] is not None and chosen["prefilter"] is not None
    result = held_out[chosen["near_duplicate"]]
    assert result["precision"] == 1.0
    assert result["ci_status"] == "ok" and result["precision_ci"] == [1.0, 1.0]
    low, high = result["recall_ci"]
    assert low <= result["recall"] <= high
    assert split["split_version"] == 3 and split["families"] == 100
    assert split["calibration_positives"] + split["held_out_positives"] == 100
    assert abs(split["calibration_positives"] - split["held_out_positives"]) <= 1
    assert len(table) > 0
    with pytest.raises(ValueError):
        calibrate(pd.DataFrame(columns=["id_a", "id_b", "cosine", "lexical_jaccard"]))


def test_reprint_families_never_straddle_the_split():
    # Ten five-member families: each family's 10 pairs form one component.
    rows = []
    for family in range(10):
        members = [f"{family}-{k}" for k in range(5)]
        rows += [{"id_a": a, "id_b": b, "cosine": 0.95, "lexical_jaccard": 0.9}
                 for i, a in enumerate(members) for b in members[i + 1:]]
    pairs = pd.DataFrame(rows)
    label = pairs["lexical_jaccard"] >= 0.45
    halves = family_halves(pairs, label, seed=42)
    assert halves["family"].nunique() == 10
    assert halves.groupby("family")["side"].nunique().eq(1).all()
    assert set(halves["side"]) == {0, 1}
    # Deterministic and independent of row order.
    shuffled = pairs.sample(frac=1, random_state=3)
    again = family_halves(shuffled, label.loc[shuffled.index], seed=42).sort_index()
    pd.testing.assert_frame_equal(again, halves)


def test_a_giant_thematic_component_still_splits_into_families():
    """The real top-10 graph percolates into one component: thematic
    neighbours chain every item together. Families must still split."""
    rows = []
    hub_items = [f"t{i}" for i in range(300)]
    for i, item in enumerate(hub_items):  # a ring plus chords: one connected component
        for step in (1, 7, 31):
            rows.append({"id_a": item, "id_b": hub_items[(i + step) % 300],
                         "cosine": 0.86, "lexical_jaccard": 0.05})
    for family in range(12):  # reprint families embedded in the thematic graph
        members = [f"t{family * 20}", f"r{family}a", f"r{family}b"]
        rows += [{"id_a": a, "id_b": b, "cosine": 0.98, "lexical_jaccard": 0.9}
                 for i, a in enumerate(members) for b in members[i + 1:]]
    pairs = pd.DataFrame(rows)
    assert pair_components(pairs).nunique() == 1  # what split_version 2 relied on
    label = pairs["lexical_jaccard"] >= 0.45
    halves = family_halves(pairs, label, seed=42)
    sides = halves["side"]
    assert (sides == 0).any() and (sides == 1).any()
    positives = label.groupby(sides).sum()
    assert positives[0] == positives[1] == 18  # 12 families x 3 pairs, balanced
    # No item appears on both sides.
    item_sides = pd.concat([pd.DataFrame({"item": pairs[c], "side": sides}) for c in ("id_a", "id_b")])
    item_sides = item_sides[item_sides["side"] >= 0]
    assert item_sides.groupby("item")["side"].nunique().eq(1).all()
    _, held_out, _, split = calibrate(pairs, n_boot=50)
    assert split["held_out_pairs"] > 0 and split["calibration_pairs"] > 0
    assert split["pairs_excluded_straddling"] == int((sides == -1).sum())


def test_component_bootstrap_requires_enough_components():
    pairs = pd.DataFrame({"id_a": ["1", "3"], "id_b": ["2", "4"], "cosine": [0.99, 0.99],
                          "lexical_jaccard": [0.9, 0.9]})
    result = component_bootstrap(pairs, pairs["lexical_jaccard"] >= 0.45, pair_components(pairs), 0.97)
    assert result["ci_status"] == "insufficient_units"
    assert result["precision_ci"] == [None, None]


def test_near_duplicate_cutoff_needs_precision_at_every_higher_threshold():
    # 0.90 meets the target but 0.95 (higher) does not: 0.90 must not be chosen.
    table = pd.DataFrame({"threshold": [0.85, 0.90, 0.95, 0.97],
                          "selected": [100, 50, 30, 20],
                          "precision": [0.5, 0.95, 0.80, 0.95],
                          "recall": [1.0, 0.9, 0.7, 0.5]})
    chosen = choose(table, target_recall=0.9, target_precision=0.9, min_selected=1)
    assert chosen["near_duplicate"] == 0.97
    table.loc[2, "precision"] = 0.92
    assert choose(table, target_recall=0.9, target_precision=0.9, min_selected=1)["near_duplicate"] == 0.90
    # A higher threshold that selects nothing has no precision and does not veto.
    table.loc[3, ["selected", "precision"]] = [0, None]
    assert choose(table, target_recall=0.9, target_precision=0.9, min_selected=1)["near_duplicate"] == 0.90


def test_near_duplicate_cutoff_is_only_calibrated_for_its_own_space():
    config, digest = NEAR_DUP_CALIBRATION
    assert near_dup_label(config, [digest]).startswith("lexical-proxy calibrated")
    assert near_dup_label("documents", [digest]).startswith("uncalibrated")
    assert near_dup_label(config, ["other-space"]).startswith("uncalibrated")
    assert near_dup_label(config, [digest, "other-space"]).startswith("uncalibrated")


def test_component_bootstrap_reports_finite_replicates():
    # 12 components; only one holds a positive, so many replicates have no recall.
    ids = [(str(2 * i), str(2 * i + 1)) for i in range(12)]
    pairs = pd.DataFrame({"id_a": [a for a, _ in ids], "id_b": [b for _, b in ids],
                          "cosine": [0.99] + [0.5] * 11, "lexical_jaccard": [0.9] + [0.1] * 11})
    label = pairs["lexical_jaccard"] >= 0.45
    result = component_bootstrap(pairs, label, pair_components(pairs), 0.97, n_boot=200, seed=1)
    assert result["ci_status"] == "ok"
    finite = result["finite_replicates"]
    assert 0 < finite["recall"] < 200 and 0 < finite["precision"] < 200


def test_scored_configurations_need_one_non_blank_hash_on_scored_items():
    pairs = pd.DataFrame({"id_a": ["1"], "id_b": ["2"]})
    good = pd.DataFrame({"o:id": ["1", "2", "3"], "h": ["abc", "abc", None]})
    assert scored_configurations(good, pairs, "h") == ["abc"]  # unscored item 3 is irrelevant
    for values in (["abc", ""], ["abc", None], ["abc", "def"]):
        with pytest.raises(SystemExit):
            scored_configurations(pd.DataFrame({"o:id": ["1", "2"], "h": values}), pairs, "h")
