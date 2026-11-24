"""sentiment_agreement.py: ordinal alpha, the "Non applicable" category,
and null consensus for rows with fewer than two ratings."""
from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pytest
from datasets import Dataset

import _common
import sentiment_agreement as sa
from iwac_common.sentiment_panel import CONSENSUS_RULE_VERSION, consensus_columns, generation

# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", the
# worked example with 4 observers, 12 units and missing values (also the
# reference matrix of R's irr::kripp.alpha). Published: nominal 0.743,
# ordinal 0.815, interval 0.849.
KRIPPENDORFF_UNITS = [
    [1, 1, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4],
    [4, 4, 4, 4], [1, 1, 2, 1], [2, 2, 2, 2], [5, 5, 5], [1, 1], [3],
]


@pytest.mark.parametrize("metric,published", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)])
def test_alpha_matches_krippendorff_worked_example(metric, published):
    assert round(sa.krippendorff_alpha(KRIPPENDORFF_UNITS, metric), 3) == published


def test_ordinal_alpha_ignores_spacing_but_not_order():
    units = [[1, 2], [2, 3], [3, 3], [1, 1]]
    stretched = [[{1: 1, 2: 10, 3: 100}[v] for v in u] for u in units]
    assert sa.krippendorff_alpha(units, "ordinal") == pytest.approx(sa.krippendorff_alpha(stretched, "ordinal"))
    assert sa.krippendorff_alpha(units, "interval") != pytest.approx(sa.krippendorff_alpha(stretched, "interval"))
    assert sa.krippendorff_alpha([[2, 2], [2, 2]], "ordinal") is None
    with pytest.raises(ValueError):
        sa.krippendorff_alpha(units, "ratio")


def test_label_consensus_separates_insufficient_votes_from_ties():
    votes = pd.Series([["Positif"], [], ["Positif", "Négatif"], ["Neutre", "Neutre", "Positif"]])
    consensus, disputed, assessable = sa.label_consensus(votes)
    assert consensus.tolist() == [None, None, "", "Neutre"]
    assert disputed.tolist() == [False, False, True, False]
    assert assessable.tolist() == [False, False, True, True]


def test_score_consensus_and_combined_disagreement():
    ordinal = pd.DataFrame({"a": [1.0, 2.0, None], "b": [None, 4.0, None], "c": [None, 3.0, None]})
    median, disputed, assessable = sa.score_consensus(ordinal)
    assert pd.isna(median[0]) and median[1] == 3.0 and pd.isna(median[2])
    assert disputed.tolist() == [False, True, False]
    flags = {"polarite": pd.Series([False, False, False]), "subjectivite": disputed}
    rated = {"polarite": pd.Series([True, False, False]), "subjectivite": assessable}
    combined = sa.combine_disagreement(flags, rated, ["polarite", "subjectivite"])
    assert combined.tolist() == ["", "subjectivite", None]


def panel(rows):
    """One frame of generation-2 votes; each row lists (polarité, centralité, subjectivité) per model."""
    members = generation(2)
    data = {"o:id": [str(i) for i in range(len(rows))]}
    for k, model in enumerate(members):
        data[model.column("polarite")] = [r[k][0] if k < len(r) else None for r in rows]
        data[model.column("centralite_islam_musulmans")] = [r[k][1] if k < len(r) else None for r in rows]
        data[model.subjectivite_column] = [r[k][2] if k < len(r) else None for r in rows]
    return pd.DataFrame(data)


def run_report(monkeypatch, tmp_path, df):
    monkeypatch.setattr(sa, "load_subset_dataframe", lambda *a, **k: df)
    monkeypatch.setattr(sa, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(sa, "write_run_manifest", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["sentiment_agreement.py", "--source", "local", "--generation", "2"])
    assert sa.main() == 0
    report = json.loads((tmp_path / "sentiment_agreement_articles_g2.json").read_text(encoding="utf-8"))
    consensus = pd.read_csv(tmp_path / "sentiment_consensus_articles_g2.csv", dtype={"o:id": str})
    return report, consensus


def test_single_rater_rows_are_null_and_leave_the_denominator(monkeypatch, tmp_path):
    agree = ("Neutre", "Central", "Mixte")
    rows = [
        [agree] * 6,                                                  # full agreement
        [agree] * 3 + [("Négatif", "Marginal", "Très objectif")] * 3,  # disputed everywhere
        [agree],                                                       # one rater only
        [("Non applicable", "Non abordé", None)] * 2 + [("Positif", "Non abordé", None)] * 2,
    ]
    report, consensus = run_report(monkeypatch, tmp_path, panel(rows))
    names = consensus_columns(2)
    assert pd.isna(consensus.loc[2, names["consensus_polarite"]])
    assert pd.isna(consensus.loc[2, names["sentiment_disagreement"]])
    assert consensus.loc[0, names["consensus_polarite"]] == "Neutre"
    summary = report["consensus"]
    assert summary["rows_with_two_plus_ratings"] == 3
    assert summary["rows_with_fewer_than_two_ratings"] == 1
    assert summary["rows_with_any_disagreement"] == 2
    assert summary["share_with_any_disagreement"] == pytest.approx(2 / 3)
    assert summary["panel_size"] == len(generation(2))
    assert "not comparable" in summary["dispute_rule"]["comparability"]
    polarite = report["dimensions"]["polarite"]
    assert polarite["primary_coefficient"] == "krippendorff_alpha_ordinal"
    assert polarite["krippendorff_alpha_ordinal"] is not None
    # Row 3's "Non applicable" vs "Positif" split is visible only to this alpha.
    assert polarite["krippendorff_alpha_nominal_incl_non_applicable"] is not None
    assert polarite["n_units_incl_non_applicable"] == 3


def test_non_applicable_disagreement_lowers_the_inclusive_alpha(monkeypatch, tmp_path):
    rows = [[("Neutre", "Central", "Mixte")] * 6, [("Positif", "Central", "Mixte")] * 6,
            [("Non applicable", "Central", "Mixte")] * 3 + [("Positif", "Central", "Mixte")] * 3]
    report, _ = run_report(monkeypatch, tmp_path, panel(rows))
    polarite = report["dimensions"]["polarite"]
    assert polarite["krippendorff_alpha_nominal"] == 1  # blind to row 3
    assert polarite["krippendorff_alpha_nominal_incl_non_applicable"] < 1


def test_subjectivite_ordinal_reads_nullable_ints_as_float():
    import numpy as np

    values = sa.subjectivite_ordinal(pd.Series(pd.array([1, 2, pd.NA, 5, 9], dtype="Int64")))
    assert values.dtype == float
    assert np.isnan(values.to_numpy()).tolist() == [False, False, True, False, True]


def test_generation_one_integer_subjectivity_runs(monkeypatch, tmp_path):
    """Generation 1 stores subjectivité as nullable Int64; the report must not crash."""
    members = generation(1)
    rows = 3
    data = {"o:id": [str(i) for i in range(rows)]}
    for k, model in enumerate(members):
        data[model.column("polarite")] = ["Neutre", "Positif", None]
        data[model.column("centralite_islam_musulmans")] = ["Central", "Marginal", None]
        data[model.subjectivite_column] = pd.array([3, 4 if k else 2, pd.NA], dtype="Int64")
    monkeypatch.setattr(sa, "load_subset_dataframe", lambda *a, **k: pd.DataFrame(data))
    monkeypatch.setattr(sa, "OUTPUT_DIR", tmp_path)
    manifests = []
    monkeypatch.setattr(sa, "write_run_manifest", lambda *a, **k: manifests.append(k["script"]))
    monkeypatch.setattr("sys.argv", ["sentiment_agreement.py", "--source", "local", "--generation", "1"])
    assert sa.main() == 0
    report = json.loads((tmp_path / "sentiment_agreement_articles_g1.json").read_text(encoding="utf-8"))
    assert report["dimensions"]["subjectivite"]["n_rated_by_all"] == 2
    # One manifest per generation, so a g1 run cannot overwrite g2's latest.
    assert manifests == ["sentiment_agreement_articles_g1"]


def test_non_applicable_is_a_vote_not_a_missing_rating(monkeypatch, tmp_path):
    rows = [[("Non applicable", "Central", "Mixte")] * 6, [("Positif", "Central", "Mixte")] * 6,
            [("Positif", "Central", "Mixte")] * 5]
    report, _ = run_report(monkeypatch, tmp_path, panel(rows))
    first, last = generation(2)[0].prefix, generation(2)[-1].prefix
    polarite = report["dimensions"]["polarite"]
    assert polarite["n_non_applicable_by_model"][first] == 1
    assert polarite["n_missing_by_model"][first] == 0
    assert polarite["n_missing_by_model"][last] == 1  # the model that did not vote on row 3
    assert "n_non_applicable_by_model" not in report["dimensions"]["centralite"]


def test_non_applicable_is_not_a_centralite_vote(monkeypatch, tmp_path):
    rows = [[("Neutre", "Non applicable", "Mixte")] * 3 + [("Neutre", "Central", "Mixte")] * 3]
    _, consensus = run_report(monkeypatch, tmp_path, panel(rows))
    # Only the three "Central" votes count; with them alone the label is "Central".
    assert consensus.loc[0, consensus_columns(2)["consensus_centralite"]] == "Central"


def test_push_declares_types_for_all_null_columns(monkeypatch):
    ds = Dataset.from_dict({"o:id": ["1", "2"]})
    types = {consensus_columns(2)[name]: pa.type_for_alias(alias) for name, alias in sa.CONSENSUS_TYPES.items()}
    values = pd.DataFrame({"o:id": ["1", "2"], **{name: [None, None] for name in types}})
    updated = _common.add_columns_by_id(ds, values, types=types)
    assert updated.features[consensus_columns(2)["consensus_polarite"]].dtype == "string"
    assert updated.features[consensus_columns(2)["consensus_subjectivite_score"]].dtype == "float64"


def test_consensus_rule_version_was_bumped():
    assert CONSENSUS_RULE_VERSION.endswith(":v2") and "null-below-min2" in CONSENSUS_RULE_VERSION


def test_config_and_source_choices(monkeypatch):
    monkeypatch.setattr("sys.argv", ["sentiment_agreement.py", "--config", "images"])
    with pytest.raises(SystemExit) as exc:
        sa.main()
    assert exc.value.code == 2
