"""Cosine cutoffs are re-derived per vector space against lexical evidence."""
from __future__ import annotations

import pandas as pd
import pytest

from iwac_pipeline.analyses.similarity_calibration import (
    add_lexical_jaccard, calibrate, choose, component_bootstrap, component_halves, curve,
    neighbor_pairs, pair_components,
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
    assert split["split_version"] == 2 and split["components"] == 200
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
    components = pair_components(pairs)
    assert components.nunique() == 10
    halves = component_halves(pairs, seed=42)
    assert halves.groupby(components).nunique().eq(1).all()
    # Deterministic and independent of row order.
    shuffled = pairs.sample(frac=1, random_state=3)
    assert component_halves(shuffled, seed=42).sort_index().equals(halves)


def test_component_bootstrap_requires_enough_components():
    pairs = pd.DataFrame({"id_a": ["1", "3"], "id_b": ["2", "4"], "cosine": [0.99, 0.99],
                          "lexical_jaccard": [0.9, 0.9]})
    result = component_bootstrap(pairs, pairs["lexical_jaccard"] >= 0.45, pair_components(pairs), 0.97)
    assert result["ci_status"] == "insufficient_components"
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
