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
                            "n_consensus_max": [60, 300, 120]})
    assert ts.ranked_topics(summary, 50)["lda_topic_id"].tolist() == [1, 2, 0]
    assert ts.ranked_topics(summary, 100)["lda_topic_id"].tolist() == [1, 2]


def coverage_frame():
    """Topic 0: six rows, four with a centrality consensus (two of them central),
    two single-rater rows; polarity "Non applicable" throughout (issue #46)."""
    models = list(generation(2))
    rows = pd.DataFrame({"lda_model_name": ["lda-sha256:a"] * 6, "lda_topic_id": [0] * 6,
                         "lda_topic_label": ["A"] * 6, "country": ["Bénin"] * 6, "pub_date": ["2000"] * 6})
    centrality = ["Central", "Très central", "Marginal", "Marginal", None, None]
    subjectivity = [2, 2, 4, 4, None, None]
    for index, model in enumerate(models):
        first = index == 0
        rows[model.column("polarite")] = ["Non applicable"] * 4 + (["Non applicable"] * 2 if first else [None] * 2)
        rows[model.column("centralite_islam_musulmans")] = [
            c if c is not None else ("Central" if first else None) for c in centrality]
        rows[model.subjectivite_column] = [s if s is not None else (5 if first else None) for s in subjectivity]
    return rows


def test_centrality_and_subjectivity_have_of_scored_denominators():
    summary = ts.summarize_cells(ts.prepare_articles(coverage_frame(), 2), []).iloc[0]
    assert summary["n"] == 6
    assert summary["n_centrality_scored"] == 4 and summary["n_subjectivity_scored"] == 4
    assert summary["central_share"] == pytest.approx(2 / 6)
    assert summary["central_share_of_scored"] == pytest.approx(0.5)
    assert summary["share_centrality_marginal"] == pytest.approx(2 / 6)
    assert summary["share_centrality_marginal_of_scored"] == pytest.approx(0.5)
    assert summary["share_subjectivity_2"] == pytest.approx(2 / 6)
    assert summary["share_subjectivity_2_of_scored"] == pytest.approx(0.5)
    # Naming: polarity consensus includes "Non applicable"; the ordinal count does not.
    assert summary["n_polarity_consensus"] == summary["n_scored"] == 4
    assert summary["n_polarity_scored"] == 0
    assert summary["n_consensus_max"] == 4


def test_of_scored_shares_are_null_without_a_consensus():
    frame = coverage_frame().iloc[4:]
    summary = ts.summarize_cells(ts.prepare_articles(frame, 2), []).iloc[0]
    assert summary["n_centrality_scored"] == 0
    assert summary["central_share_of_scored"] is None
    assert summary["share_subjectivity_5_of_scored"] is None


def test_non_applicable_topics_stay_in_country_year_and_ranking_tables(tmp_path, monkeypatch):
    frame = coverage_frame()
    frame["o:id"] = [str(i) for i in range(len(frame))]
    frame.attrs["iwac_source_revision"] = "test-revision"
    monkeypatch.setattr(ts, "load_subset_dataframe", lambda *a, **kw: frame)
    monkeypatch.setattr(ts, "write_run_manifest", lambda *a, **kw: None)
    monkeypatch.setattr("sys.argv", ["topic_sentiment.py", "--source", "csv", "--output-dir", str(tmp_path),
                                     "--min-topic-n", "4", "--min-cell-n", "4", "--min-year-n", "4"])
    ts.main()
    for name in ("topic_sentiment_by_country.csv", "topic_sentiment_over_time.csv"):
        table = pd.read_csv(tmp_path / name)
        # No ordinal polarity at all, yet the topic is well covered: it stays.
        assert table["n_polarity_scored"].tolist() == [0] and table["n_consensus_max"].tolist() == [4]
    summary = pd.read_csv(tmp_path / "topic_sentiment_summary.csv")
    assert len(ts.ranked_topics(summary, 4)) == 1


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


def test_standardisation_tolerates_documents_without_a_country():
    theta = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    years = pd.Series([2000, 2000, 2000])
    countries = pd.Series(["Bénin", "Togo", pd.NA], dtype="string")
    weights = tp.standard_weights(countries)
    series = tp.standardized_series(theta, years, countries, weights)
    assert series[2000][1] == pytest.approx(1.0)
    assert series[2000][0][0] == pytest.approx(0.5)


def test_single_document_year_has_no_zero_width_band():
    theta = np.array([[0.7, 0.3]] + [[0.5, 0.5]] * 5)
    years = pd.Series([1999] + [2000] * 5)
    bundle = {"n_topics": 2, "id": "lda-sha256:x", "labels": {0: "a", 1: "b"}}
    frame = tp.yearly_prevalence(theta, years, None, bundle, SimpleNamespace(bootstrap=20, seed=42))
    status = frame.drop_duplicates("year").set_index("year")["ci_status"]
    assert status[1999] == "insufficient_docs" and status[2000] == "ok"
    assert frame[frame.year == 1999].ci_low.isna().all()


def test_document_vectors_apply_the_evidence_floor_to_reused_theta_too(monkeypatch):
    df = pd.DataFrame({"o:id": ["1", "2", "3"], "lemma_nostop": ["a b c d e f", "a", "a b c d e f"]})
    monkeypatch.setattr(tp, "prediction_tokenizer_kwargs", lambda preprocessing: {})
    monkeypatch.setattr(tp, "tokenize_for_prediction", lambda text, phraser: text.split())
    monkeypatch.setattr(tp, "count_vocab_tokens", lambda dictionary, tokens: len(tokens))
    monkeypatch.setattr(tp, "predict_document",
                        lambda *a, **k: (0, 0.6, "a", "0:0.6", [0.6, 0.4]))
    bundle = {"preprocessing": {"chunk_words": None}, "phraser": None, "dictionary": None,
              "model": None, "labels": {0: "a", 1: "b"}, "n_topics": 2}
    reused = {"2": np.array([0.9, 0.1])}  # below the floor although its theta is reused
    kept, theta, below = tp.document_vectors(df, df.index, bundle, reused, min_vocab_tokens=5)
    assert kept == [0, 2] and below == 1 and theta.shape == (2, 2)
    kept, _, below = tp.document_vectors(df, df.index, bundle, reused, min_vocab_tokens=0)
    assert kept == [0, 1, 2] and below == 0


def test_mann_kendall_p_value_does_not_underflow():
    from iwac_pipeline.analyses._stats import mann_kendall

    _, z, p = mann_kendall(np.arange(200, dtype=float))
    assert z > 20 and 0 < p < 1e-90  # 1 - cdf gives exactly 0 beyond z of about 8.3


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
