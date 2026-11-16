"""Topic sentiment shares, prevalence standardisation/CI status and topic-review length floor."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from iwac_common.sentiment_panel import generation
from iwac_pipeline.analyses import topic_prevalence as tp
from iwac_pipeline.analyses import topic_sentiment as ts
from iwac_pipeline.analyses.topic_review import topic_review


def sentiment_frame():
    rows = pd.DataFrame({"lda_model_name": ["lda-sha256:a"] * 4, "lda_topic_id": [0] * 4,
                         "lda_topic_label": ["A"] * 4, "country": ["Bénin"] * 4,
                         "pub_date": ["2000", "2000/2001", "2001", "2001"]})
    for model in generation(2):
        rows[model.column("polarite")] = ["Positif", "Négatif", None, None]
        rows[model.column("centralite_islam_musulmans")] = None
        rows[model.subjectivite_column] = None
    return rows


def test_shares_of_scored_use_consensus_rows_as_denominator():
    summary = ts.summarize_cells(ts.prepare_articles(sentiment_frame(), 2), []).iloc[0]
    assert summary["n"] == 4 and summary["n_scored"] == 2
    assert summary["share_positif"] == pytest.approx(0.25)
    assert summary["share_positif_of_scored"] == pytest.approx(0.5)
    assert summary["share_negatif_of_scored"] == pytest.approx(0.5)


def test_range_dates_leave_the_over_time_table():
    prepared = ts.prepare_articles(sentiment_frame(), 2)
    assert prepared["year"].isna().tolist() == [False, True, False, False]
    assert prepared["date_precision"].iloc[1] == "range"


def test_console_ranking_is_sorted_by_scored_rows():
    summary = pd.DataFrame({"lda_model_name": ["m", "m", "m"], "lda_topic_id": [0, 1, 2],
                            "n_polarity_scored": [60, 300, 120]})
    assert ts.ranked_topics(summary, 50)["lda_topic_id"].tolist() == [1, 2, 0]
    assert ts.ranked_topics(summary, 100)["lda_topic_id"].tolist() == [1, 2]


def test_direct_standardisation_fixes_country_weights():
    # Bénin always prevalence 1.0 on topic 0, Togo 0.0; the archive shifts toward Togo.
    theta = np.array([[1, 0]] * 3 + [[0, 1]] * 1 + [[1, 0]] * 1 + [[0, 1]] * 3, dtype=float)
    years = pd.Series([2000] * 4 + [2001] * 4)
    countries = pd.Series(["Bénin"] * 3 + ["Togo"] + ["Bénin"] + ["Togo"] * 3)
    weights = tp.standard_weights(countries)
    assert weights == {"Bénin": 0.5, "Togo": 0.5}
    series = tp.standardized_series(theta, years, countries, weights)
    # Raw: 0.75 then 0.25; standardised: 0.5 both years.
    assert series[2000][0][0] == pytest.approx(0.5) and series[2001][0][0] == pytest.approx(0.5)
    assert series[2000][1] == pytest.approx(1.0)
    # A year missing a weighted country has no standardised value.
    partial = tp.standardized_series(theta[:3], years[:3], countries[:3], weights)
    assert partial[2000] == (None, pytest.approx(0.5))


def test_yearly_frame_marks_insufficient_clusters_and_keeps_raw_series():
    theta = np.tile([0.7, 0.3], (20, 1))
    years = pd.Series([2000] * 20)
    bundle = {"n_topics": 2, "id": "lda-sha256:x", "labels": {0: "a", 1: "b"}}
    args = SimpleNamespace(bootstrap=50, seed=42)
    few = pd.Series([f"o{i % 3}" for i in range(20)])
    frame = tp.yearly_prevalence(theta, years, few, bundle, args,
                                 standardized={2000: (np.array([0.6, 0.4]), 1.0)})
    assert set(frame.ci_status) == {"insufficient_clusters"}
    assert frame.ci_low.isna().all() and frame.n_clusters.eq(3).all()
    assert frame.prevalence.tolist() == pytest.approx([0.7, 0.3])
    assert frame.prevalence_standardized.tolist() == pytest.approx([0.6, 0.4])
    many = pd.Series([f"o{i % 10}" for i in range(20)])
    assert set(tp.yearly_prevalence(theta, years, many, bundle, args).ci_status) == {"ok"}
    rows = tp.yearly_prevalence(theta, years, None, bundle, SimpleNamespace(bootstrap=0, seed=42))
    assert set(rows.ci_status) == {"disabled"}


def test_eligibility_reports_language_and_date_exclusions():
    df = pd.DataFrame({"pub_date": ["2000", "2000/2001", "2001", "2001"],
                       "language": ["Français", "Français", "Kabiyè", "Français"],
                       "lemma_nostop": ["imam", "imam", "imam", ""]})
    args = SimpleNamespace(include_unknown_language=False, year_min=1900, year_max=2030)
    mask, years, counts = tp.eligibility(df, "Français", args)
    assert mask.tolist() == [True, False, False, False]
    assert counts["dates"]["by_precision"] == {"range": 1}
    assert counts["excluded_language_labels"] == {"Kabiyè": 1}
    assert counts["docs_without_text"] == 1


def test_representatives_need_a_length_floor():
    df = pd.DataFrame({"o:id": ["1", "2", "3", "4"], "lda_model_name": ["lda-sha256:" + "a" * 64] * 4,
                       "lda_topic_id": [0] * 4, "lda_topic_prob": [0.99, 0.95, 0.6, 0.55],
                       "lda_topic_topk": ["0:0.99", "0:0.95", "0:0.6|1:0.4", "0:0.55|1:0.45"],
                       "nb_mots": [12, None, 400, 800]})
    sheet, _, summary = topic_review(df, representatives=1, borderline=1, min_words=100)
    rep = sheet[sheet.review_role == "representative"]
    assert rep["o:id"].tolist() == ["3"]
    assert summary["representative_length_floor"]["below_floor"] == 1
    assert summary["representative_length_floor"]["length_unknown"] == 1
    assert topic_review(df, representatives=1, borderline=0, min_words=0)[0]["o:id"].tolist() == ["1"]
    with pytest.raises(ValueError, match="--min-words 0"):
        topic_review(df.drop(columns="nb_mots"), representatives=1, borderline=0)
