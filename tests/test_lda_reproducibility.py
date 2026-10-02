"""Research contracts: held-out inputs, immutable identities and frozen inference."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from iwac_pipeline.processing.lda_topic_modeling import lda_topic_modeling as cli
from iwac_pipeline.processing.lda_topic_modeling.artifacts import (
    digest_file, frozen_preprocessing, load_preprocessing, prediction_tokenizer_kwargs,
    publish_bundle, resolve_bundle, split_document_indices,
    effective_model_identity,
)
from iwac_pipeline.processing.lda_topic_modeling.modeling import (
    load_lda_model, tokenize_for_prediction,
    predict_document, find_optimal_topics,
)
from iwac_pipeline.analyses import topic_prevalence as prevalence


def test_holdout_keeps_groups_and_exact_duplicates_together():
    texts = ["same text", "same text", "third text", "fourth", "fifth"]
    groups = ["a", "b", "b", "c", "d"]
    train, held = split_document_indices(texts, groups, 0.4)
    assert set(train).isdisjoint(held)
    assert sorted(train + held) == list(range(5))
    # a/b groups and an exact duplicate create one connected group of 3 docs.
    assert set([0, 1, 2]).issubset(train) or set([0, 1, 2]).issubset(held)
    assert set(groups[i] for i in train).isdisjoint(groups[i] for i in held)


def test_evaluation_preprocessing_never_fits_heldout_tokens():
    texts = [f"shared imam mosque unique{suffix} shared imam mosque" for suffix in "abcdef"]
    config = frozen_preprocessing(set(), "Français", 3)
    corpus, dictionary, tokenized, held, train_idx, held_idx = cli.prepare_evaluation(
        texts, list(range(6)), config, no_below=1, no_above=1.0, holdout=0.3,
    )
    assert corpus and held
    assert set(train_idx).isdisjoint(held_idx)
    for i in held_idx:
        word = f"unique{'abcdef'[i]}"
        assert word not in dictionary.token2id
        assert all(word not in doc for doc in tokenized)
    assert len(corpus) > len(train_idx)  # splitting happened before chunking


@pytest.mark.parametrize("fraction", [-0.1, 1, 1.1])
def test_holdout_rejects_invalid_fractions(fraction):
    with pytest.raises(ValueError, match="holdout"):
        split_document_indices(["a", "b"], [1, 2], fraction)


def test_bundle_hashes_are_checked_and_old_bundles_survive(tmp_path):
    def stage(content):
        folder = tmp_path / f"staging-{content}"
        folder.mkdir()
        (folder / "lda_model").write_text(content)
        return folder

    first, first_id = publish_bundle(stage("first"), tmp_path)
    second, second_id = publish_bundle(stage("second"), tmp_path)
    assert first_id != second_id
    assert resolve_bundle(tmp_path) == (second.resolve(), second_id)
    assert resolve_bundle(first) == (first, first_id)
    (first / "lda_model").write_text("damaged")
    with pytest.raises(ValueError, match="changed"):
        resolve_bundle(first)


def test_legacy_preprocessing_requires_explicit_opt_in(tmp_path):
    (tmp_path / "training_parameters.json").write_text(json.dumps({
        "stopwords": {"words": ["special"], "fragments": ["fragment"]},
        "extra": {"language": "Anglais", "chunk_words": 10},
    }))
    with pytest.raises(ValueError, match="Legacy"):
        load_preprocessing(tmp_path)
    settings, _ = load_preprocessing(tmp_path, allow_legacy=True)
    assert settings["stopwords"] == ["special"]
    assert settings["fragment_stopwords"] == ["fragment"]


@pytest.fixture
def fitted_model(tmp_path, monkeypatch):
    docs = [
        "imam mosque prayer mosque imam worship", "imam prayer ramadan mosque fast worship",
        "school student lesson teacher school education", "teacher student school class education lesson",
        "imam mosque worship prayer ramadan fast", "education lesson school student teacher class",
    ]
    ds = Dataset.from_dict({
        "o:id": [str(i) for i in range(len(docs))], "lemma_nostop": docs,
        "language": ["Français"] * len(docs), "pub_date": ["2000-01-01"] * len(docs),
        "country": ["Bénin"] * len(docs), "newspaper_ids": ["one", "two"] * 3,
    })
    ds._iwac_source_revision = "test-source-revision"
    state = {"dataset": ds}
    monkeypatch.setattr(cli, "ensure_hf_token", lambda **kwargs: "test-token")
    monkeypatch.setattr(cli, "load_hub_dataset", lambda *args, **kwargs: state["dataset"])

    def push(dataset, **kwargs):
        state["dataset"] = dataset
        state["push"] = kwargs
        return True

    monkeypatch.setattr(cli, "push_dataset", push)
    model_root = tmp_path / "model"
    stopwords = tmp_path / "extra.txt"
    stopwords.write_text("worship\n")
    monkeypatch.setattr(sys, "argv", [
        "lda", "--repo", "test/repo", "--config", "articles", "--mode", "fit", "--yes",
        "--model-path", str(model_root), "--num-topics", "2", "--passes", "2", "--iterations", "20",
        "--no-below", "1", "--no-above", "1.0", "--skip-coherence", "--holdout", "0",
        "--domain-stopwords-file", str(stopwords),
    ])
    assert cli.main() == 0
    return model_root, state


def test_fit_predict_orchestration_freezes_identity_and_preprocessing(fitted_model, monkeypatch):
    model_root, state = fitted_model
    directory, model_id = resolve_bundle(model_root)
    settings, params = load_preprocessing(directory)
    assert "worship" in settings["stopwords"]
    assert params["extra"]["source_revision"] == "test-source-revision"
    assert set(state["dataset"]["lda_model_name"]) == {model_id}
    original = {p.name: digest_file(p) for p in directory.iterdir() if p.is_file()}
    first_topics = state["dataset"]["lda_topic_id"]
    # Prediction needs no extra stopword file and cannot change the frozen bundle.
    monkeypatch.setattr(sys, "argv", [
        "lda", "--repo", "test/repo", "--config", "articles", "--mode", "predict", "--yes",
        "--model-path", str(model_root),
    ])
    assert cli.main() == 0
    assert state["dataset"]["lda_topic_id"] == first_topics
    assert original == {p.name: digest_file(p) for p in directory.iterdir() if p.is_file()}
    assert len(list((model_root / "predictions").glob("*/doc_topics.parquet"))) == 2
    model, dictionary, phraser = load_lda_model(model_root)
    assert model.num_topics == 2
    assert "worship" not in tokenize_for_prediction(
        "imam worship mosque", phraser=phraser, **prediction_tokenizer_kwargs(settings)
    )


def test_fit_cli_sweep_receives_only_training_documents(fitted_model, monkeypatch, tmp_path):
    _, state = fitted_model
    data = state["dataset"].to_dict()
    data["lemma_nostop"] = [text + f" uniquemarker{suffix}" for text, suffix in zip(data["lemma_nostop"], "abcdef")]
    state["dataset"] = Dataset.from_dict(data)
    captured = {}

    def sweep(corpus, dictionary, tokenized_docs, **kwargs):
        captured["vocabulary"] = set(dictionary.token2id)
        captured["holdout"] = kwargs["holdout_corpus"]
        return 2, [{"k": 2, "c_v": 0.2}]

    monkeypatch.setattr(cli, "find_optimal_topics", sweep)
    root = tmp_path / "evaluated"
    monkeypatch.setattr(sys, "argv", [
        "lda", "--repo", "test/repo", "--config", "articles", "--mode", "fit", "--yes",
        "--model-path", str(root), "--optimize-topics", "--passes", "1", "--iterations", "20",
        "--no-below", "1", "--no-above", "1.0", "--skip-coherence", "--holdout", "0.34",
    ])
    assert cli.main() == 0
    directory, _ = resolve_bundle(root)
    _, params = load_preprocessing(directory)
    split = params["evaluation"]["split"]
    assert set(split["train_document_ids"]).isdisjoint(split["holdout_document_ids"])
    assert captured["holdout"]
    for oid in split["holdout_document_ids"]:
        assert f"uniquemarker{'abcdef'[int(oid)]}" not in captured["vocabulary"]


def test_prevalence_theta_reuse_and_stale_input_rejection(fitted_model):
    root, state = fitted_model
    directory, identity = resolve_bundle(root)
    preprocessing, _ = load_preprocessing(directory)
    theta_path = next((root / "predictions").glob("*/doc_topics.parquet"))
    df = state["dataset"].to_pandas()
    vectors = prevalence.load_theta_export(theta_path, identity, "test/repo", "articles", df, 2, preprocessing)
    assert len(vectors) == len(df)
    assert all(np.isclose(v.sum(), 1) for v in vectors.values())
    with pytest.raises(ValueError, match="preprocessing"):
        prevalence.load_theta_export(theta_path, identity, "test/repo", "articles", df, 2,
                                     {**preprocessing, "chunk_words": 2})
    df.loc[0, "lemma_nostop"] = "changed text"
    with pytest.raises(ValueError, match="changed"):
        prevalence.load_theta_export(theta_path, identity, "test/repo", "articles", df, 2, preprocessing)


@pytest.mark.parametrize("label_only", [False, True])
def test_prevalence_cli_uses_frozen_model_and_conditional_output(fitted_model, tmp_path, monkeypatch, label_only):
    root, state = fitted_model
    frame = state["dataset"].to_pandas()
    if label_only:
        frame = frame.drop(columns=["newspaper_ids"])
        frame["newspaper"] = "A shared outlet title"
        frame["country"] = ["Bénin", "Togo"] * 3
    frame.attrs["iwac_source_revision"] = "test-source-revision"
    monkeypatch.setattr(prevalence, "load_subset_dataframe", lambda *a, **kw: frame)
    monkeypatch.setattr(prevalence, "OUTPUT_DIR", tmp_path / "results")
    monkeypatch.setattr(prevalence, "write_run_manifest", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", [
        "prevalence", "--repo", "test/repo", "--source", "csv", "--model-path", str(root),
        "--bootstrap", "20", "--bootstrap-unit", "newspaper", "--min-docs-year", "1",
        "--theta-path", str(next((root / "predictions").glob("*/doc_topics.parquet"))),
    ])
    prevalence.main()
    summary = json.loads((prevalence.OUTPUT_DIR / "topic_prevalence_summary.json").read_text())
    assert summary["trend_test"] == "none"
    assert summary["theta_rows_reused"] == len(frame)
    assert summary["bootstrap_unit"] == "newspaper"
    assert all(t["q_value"] is None and not t["significant"] for t in summary["topics"])
    yearly = pd.read_csv(prevalence.OUTPUT_DIR / "topic_prevalence_year.csv")
    assert yearly["n_clusters"].eq(2).all()


def test_directional_rankings_never_mix_signs():
    frame = pd.DataFrame({"slope_per_decade_pp": [-0.5, -0.1, 0, 0.3]})
    assert prevalence.directional_trends(frame, "rising")["slope_per_decade_pp"].tolist() == [0.3]
    assert prevalence.directional_trends(frame, "declining")["slope_per_decade_pp"].tolist() == [-0.5, -0.1]


def test_legacy_identity_changes_with_reconstructed_preprocessing():
    first = frozen_preprocessing(set(), "Français", None)
    second = {**first, "custom_collocations": [["new", "phrase"]]}
    assert effective_model_identity("lda-legacy-sha256:abc", first) != effective_model_identity("lda-legacy-sha256:abc", second)


def test_nonfinite_coherence_cannot_choose_a_winner(monkeypatch):
    from gensim.corpora import Dictionary
    from iwac_pipeline.processing.lda_topic_modeling import modeling

    class InvalidCoherence:
        def __init__(self, **kwargs):
            pass

        def get_coherence(self):
            return float("nan")

    monkeypatch.setattr(modeling, "CoherenceModel", InvalidCoherence)
    documents = [["imam", "mosque"], ["teacher", "school"]]
    dictionary = Dictionary(documents)
    with pytest.raises(RuntimeError, match="no model was selected"):
        find_optimal_topics([dictionary.doc2bow(d) for d in documents], dictionary, documents,
                            topic_range_start=2, topic_range_end=2, sweep_passes=1,
                            sweep_iterations=10)


def test_inference_is_independent_of_row_order_and_prior_rng_calls(fitted_model):
    root, _ = fitted_model
    model, dictionary, _ = load_lda_model(root)
    tokens = ["imam", "mosque", "prayer"]
    original_rng = model.random_state
    first = predict_document(model, dictionary, tokens, return_distribution=True)[4]
    predict_document(model, dictionary, ["school", "teacher"], return_distribution=True)
    model.get_document_topics(dictionary.doc2bow(["school", "student"]))
    second = predict_document(model, dictionary, tokens, return_distribution=True)[4]
    assert first == second
    assert model.random_state is original_rng


def test_prevalence_dates_do_not_guess_ambiguous_years():
    years, precision = prevalence.eligible_years(pd.Series([
        "2000/2001", "2000-not-a-date", pd.NA, "2000-02-30", "2000-02", "2000",
    ]))
    assert years.iloc[:4].isna().all()
    assert years.iloc[4:].tolist() == [2000, 2000]
    assert precision.iloc[0] == "range"
