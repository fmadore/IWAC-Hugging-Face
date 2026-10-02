"""Behavioral regressions for instrument identity and corpus-bound analyses."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iwac_common.sentiment_panel import consensus_columns, generation, instrument_id

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sa = load("post-processing/sentiment_agreement.py", "sa_regressions")
ts = load("analyses/topic_sentiment.py", "ts_regressions")
kb = load("analyses/keyness_bursts.py", "kb_regressions")
en = load("analyses/entity_networks.py", "en_regressions")
st = load("analyses/_stats.py", "stats_regressions")


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scale", [None, [1, 2, 3, 4, 5]])
def test_constant_kappa_is_undefined(weighted, scale):
    assert sa.cohen_kappa(np.array([2., 2.]), np.array([2., 2.]), weighted, scale) is None


@pytest.mark.parametrize("metric", ["nominal", "interval"])
def test_constant_alpha_is_undefined(metric):
    assert sa.krippendorff_alpha([[2., 2.], [2., 2.]], metric) is None
    assert sa.krippendorff_alpha([[1., 1.], [2., 2.]], metric) == 1


def test_kappa_with_variable_perfect_agreement():
    values = np.array([1., 2., 3., 4., 5.])
    assert sa.cohen_kappa(values, values, scale=list(range(1, 6))) == 1
    # Expected agreement 0.5, observed 0.75 -> kappa 0.5.
    assert sa.cohen_kappa(np.array([1., 1., 2., 2.]), np.array([1., 1., 1., 2.])) == .5


def test_instrument_ids_and_storage_are_generation_specific():
    assert instrument_id(1) != instrument_id(2)
    assert instrument_id(2) == instrument_id(2)
    assert set(consensus_columns(1).values()).isdisjoint(consensus_columns(2).values())


def test_cross_generation_push_rejected_before_any_read(monkeypatch):
    monkeypatch.setattr("sys.argv", ["sentiment_agreement.py", "--generation", "all", "--push"])
    monkeypatch.setattr(sa, "load_subset_dataframe", lambda *a, **k: pytest.fail("read before rejection"))
    with pytest.raises(SystemExit) as exc:
        sa.main()
    assert exc.value.code == 2


def panel_frame():
    rows = pd.DataFrame({"lda_model_name": ["lda-sha256:a", "lda-sha256:a", "lda-sha256:b"],
                         "lda_topic_id": [0, 0, 0], "lda_topic_label": ["A", "A", "B"],
                         "country": ["Bénin"] * 3, "pub_date": ["2000"] * 3})
    for model in generation(2):
        rows[model.column("polarite")] = ["Positif", None, "Négatif"]
        rows[model.column("centralite_islam_musulmans")] = ["Central", None, "Marginal"]
        rows[model.subjectivite_column] = ["Plutôt objectif", None, "Mixte"]
    rows["consensus_polarite"] = "Très négatif"  # Unidentified historical values must be ignored.
    return rows


def test_topic_consensus_ignores_stored_legacy_and_keeps_missing_rows():
    prepared = ts.prepare_articles(panel_frame(), 2)
    assert len(prepared) == 3
    assert prepared["pol_label"].tolist() == ["Positif", "", "Négatif"]
    assert prepared["subj_score"].iloc[0] == 2
    assert pd.isna(prepared["subj_score"].iloc[1])
    assert prepared["sentiment_instrument_id"].eq(instrument_id(2)).all()


def test_topics_do_not_merge_model_ids_and_missingness_denominator_is_visible():
    summary = ts.summarize_cells(ts.prepare_articles(panel_frame(), 2), [])
    assert len(summary) == 2
    first = summary.loc[summary.lda_model_name.eq("lda-sha256:a")].iloc[0]
    assert first["n"] == 2 and first["n_polarity_scored"] == 1
    assert first["n_polarity_insufficient_votes"] == 1
    assert first["share_positif"] == .5
    assert first["share_insufficient_votes"] == .5
    assert first["median_polarity"] == 4
    assert "mean_polarity_equal_spacing" not in summary
    assert "mean_polarity_equal_spacing" in ts.summarize_cells(ts.prepare_articles(panel_frame(), 2), [], ordinal_means=True)


def test_topic_model_identity_required():
    with pytest.raises(ValueError, match="lda_model_name"):
        ts.prepare_articles(panel_frame().drop(columns="lda_model_name"), 2)


def test_selected_generation_and_no_one_rater_subjectivity():
    rows = panel_frame()
    for model in generation(1):
        rows[model.column("polarite")] = "Négatif"
    assert ts.resolve_consensus(rows, 1)["pol_label"].eq("Négatif").all()
    cols = [model.subjectivite_column for model in generation(2)]
    rows.loc[0, cols[1:]] = None
    assert pd.isna(ts.resolve_consensus(rows, 2)["subj_score"].iloc[0])


def test_keyness_country_and_decade_share_year_language_eligibility():
    df = pd.DataFrame({"pub_date": ["1999", "2000", "2010", "2003", "2004", "2004"],
                       "language": ["Français", "Français", "Français", "  ", None, "English"],
                       "lemma_nostop": ["mosquée"] * 6})
    assert kb.keyness_rows(df, 2000, 2009).index.tolist() == [1]
    assert kb.keyness_rows(df, 2000, 2009, include_unknown=True).index.tolist() == [1, 3, 4]


def test_authority_bursts_keep_homonyms_separate_and_expose_missing_tags():
    index = pd.DataFrame({"o:id": [10., 11.], "Titre": ["Ali", "Ali"]})
    df = pd.DataFrame({"year": [2000, 2000, 2001], "subject_ids": ["10|10", "11", None],
                       "subject": ["Ali", "Ali", "Ali"]})
    mentions, titles, years, exposure, coverage = kb.subject_exposure(df, index)
    assert set(mentions) == {"10", "11"}
    assert mentions["10"][2000] == 1
    assert exposure.tolist() == [2., 1.]
    assert coverage.n_tagged.tolist() == [2, 0]
    tagged = kb.subject_exposure(df, index, denominator="tagged")[3]
    assert tagged.tolist() == [2., 0.]
    assert en.resolve_entities({"subject": "Ali"}, titles, {"Ali": "10"}) == (set(), 1)


def test_empty_burst_corpus_is_well_formed():
    empty = pd.DataFrame(columns=["year", "subject", "subject_ids"])
    authority = pd.DataFrame(columns=["o:id", "Titre"])
    mentions, _, years, exposure, coverage = kb.subject_exposure(empty, authority)
    assert not mentions and len(years) == len(exposure) == len(coverage) == 0
    assert list(coverage) == ["year", "n_articles", "n_tagged", "exposure", "tagged_share"]


def test_burst_rejects_mentions_outside_exposure():
    with pytest.raises(ValueError, match="exposure"):
        kb.kleinberg_bursts(np.array([2, 1]), np.array([1, 1]), np.array([2000, 2001]))


def test_cluster_bootstrap_resamples_whole_outlets():
    values = np.array([0.] * 100 + [1.] * 100)
    clusters = ["A"] * 100 + ["B"] * 100
    lo, hi = st.bootstrap_mean_ci(values, 1000, 42, clusters=clusters)
    assert lo == 0 and hi == 1
    assert st.bootstrap_mean_ci(values, 1000, 42, clusters=clusters) == (lo, hi)
    row_lo, row_hi = st.bootstrap_mean_ci(values, 1000, 42)
    assert row_lo > lo and row_hi < hi
    assert np.isnan(st.bootstrap_mean_ci(values, 100, 42, clusters=["A"] * 200)[0])
    with pytest.raises(ValueError, match="missing"):
        st.bootstrap_mean_ci([1., 2.], 10, 42, clusters=["A", None])


def test_agreement_run_exports_only_generation_specific_consensus(monkeypatch, tmp_path):
    frame = panel_frame().assign(**{"o:id": [1, 2, 3]})
    monkeypatch.setattr(sa, "load_subset_dataframe", lambda *a, **k: frame)
    monkeypatch.setattr(sa, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(sa, "write_run_manifest", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["sentiment_agreement.py", "--source", "csv", "--generation", "2"])
    assert sa.main() == 0
    result = pd.read_csv(tmp_path / "sentiment_consensus_articles_g2.csv")
    assert "consensus_polarite" not in result
    assert set(consensus_columns(2).values()).issubset(result.columns)
    assert result["consensus_g2_instrument_id"].eq(instrument_id(2)).all()
    assert pd.isna(result["consensus_g2_subjectivite_score"].iloc[1])


def test_topic_run_emits_missingness_for_country_and_year(monkeypatch, tmp_path):
    monkeypatch.setattr(ts, "load_articles", lambda *a, **k: panel_frame())
    monkeypatch.setattr(ts, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ts, "write_run_manifest", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["topic_sentiment.py", "--source", "csv", "--min-cell-n", "1", "--min-year-n", "1"])
    ts.main()
    for name in ("topic_sentiment_summary.csv", "topic_sentiment_by_country.csv", "topic_sentiment_over_time.csv"):
        report = pd.read_csv(tmp_path / name)
        assert len(report) == 2
        assert report.n.sum() == 3
        assert report.n_polarity_insufficient_votes.sum() == 1
        assert report.sentiment_instrument_id.eq(instrument_id(2)).all()


def test_keyness_run_handles_all_missing_subjects(monkeypatch, tmp_path):
    frame = pd.DataFrame({"o:id": [1, 2], "lemma_nostop": ["mosquée école", "école"],
                          "pub_date": ["2000", "2001"], "country": ["Bénin", "Togo"],
                          "language": ["Français", "Français"], "subject": [None, None]})
    index = pd.DataFrame({"o:id": [10], "Titre": ["Islam"]})
    monkeypatch.setattr(kb, "load_subset_dataframe", lambda repo, config, **kwargs: index if config == "index" else frame)
    monkeypatch.setattr(kb, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(kb, "write_run_manifest", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["keyness_bursts.py", "--source", "csv"])
    kb.main()
    exposure = pd.read_csv(tmp_path / "subject_burst_exposure.csv")
    assert exposure.exposure.tolist() == [1, 1]
    assert exposure.n_tagged.sum() == 0
    assert pd.read_csv(tmp_path / "subject_bursts.csv").empty
