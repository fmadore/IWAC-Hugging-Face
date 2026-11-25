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
    pushes_before = state.get("push")
    monkeypatch.setattr(sys, "argv", [
        "lda", "--repo", "test/repo", "--config", "articles", "--mode", "fit", "--yes",
        "--model-path", str(root), "--optimize-topics", "--passes", "1", "--iterations", "20",
        "--no-below", "1", "--no-above", "1.0", "--skip-coherence", "--holdout", "0.34",
    ])
    assert cli.main() == 0
    # articles pins k, so the sweep is report-only: no bundle, no push.
    assert not (root / "bundles").exists() and not (root / "current.json").exists()
    assert state.get("push") is pushes_before
    report = json.loads(next((root / "sweeps").glob("*/sweep_report.json")).read_text(encoding="utf-8"))
    assert report["k_pinned"] and report["pinned_k"] == 30
    assert "not a document-completion" in report["holdout_metric"]
    split = report["evaluation_split"]
    assert set(split["train_document_ids"]).isdisjoint(split["holdout_document_ids"])
    assert captured["holdout"]
    for oid in split["holdout_document_ids"]:
        assert f"uniquemarker{'abcdef'[int(oid)]}" not in captured["vocabulary"]


def test_pinned_k_refuses_selection_from_the_sweep(fitted_model, monkeypatch, tmp_path):
    for extra in (["--num-topics", "3"], []):  # explicit pin, then the articles preset pin
        monkeypatch.setattr(sys, "argv", [
            "lda", "--repo", "test/repo", "--config", "articles", "--mode", "fit", "--yes",
            "--model-path", str(tmp_path / "refused"), "--select-k-from-sweep", *extra,
        ])
        with pytest.raises(SystemExit, match="pinned"):
            cli.main()
    assert not (tmp_path / "refused").exists()


def test_unpinned_config_selects_k_only_on_request(fitted_model, monkeypatch, tmp_path):
    _, state = fitted_model
    monkeypatch.setattr(cli, "find_optimal_topics", lambda *a, **kw: (3, [
        {"k": 2, "c_v": 0.40, "c_v_sd": 0.05}, {"k": 3, "c_v": 0.42, "c_v_sd": 0.05},
    ]))
    root = tmp_path / "selected"
    base = ["lda", "--repo", "test/repo", "--config", "unpinned-subset", "--mode", "fit", "--yes",
            "--model-path", str(root), "--passes", "1", "--iterations", "20", "--no-below", "1",
            "--no-above", "1.0", "--skip-coherence"]
    monkeypatch.setattr(sys, "argv", base + ["--optimize-topics"])
    assert cli.main() == 0
    assert not (root / "current.json").exists()      # without opt-in: report only
    monkeypatch.setattr(sys, "argv", base + ["--select-k-from-sweep", "--stability-seeds", "2"])
    assert cli.main() == 0
    directory, _ = resolve_bundle(root)
    _, params = load_preprocessing(directory)
    assert params["lda"]["num_topics"] == 3 and params["extra"]["k_source"] == "sweep"
    ranking = params["extra"]["topic_optimization"]["ranking"]
    assert ranking["runner_up_k"] == 2 and ranking["distinguishable"] is False


def test_sweep_ranking_reports_indistinguishable_and_unknown():
    from iwac_pipeline.processing.lda_topic_modeling.modeling import sweep_ranking

    close = sweep_ranking([{"k": 8, "c_v": 0.50, "c_v_sd": 0.02}, {"k": 12, "c_v": 0.51, "c_v_sd": 0.03}])
    assert close["best_k"] == 12 and close["distinguishable"] is False
    clear = sweep_ranking([{"k": 8, "c_v": 0.40, "c_v_sd": 0.01}, {"k": 12, "c_v": 0.51, "c_v_sd": 0.01}])
    assert clear["distinguishable"] is True
    single = sweep_ranking([{"k": 8, "c_v": 0.40, "c_v_sd": None}, {"k": 12, "c_v": 0.51, "c_v_sd": None}])
    assert single["distinguishable"] is None


def test_evidence_floor_nulls_assignments_and_is_recorded(fitted_model, monkeypatch):
    model_root, state = fitted_model
    directory, model_id = resolve_bundle(model_root)
    _, params = load_preprocessing(directory)
    assert params["extra"]["min_train_tokens"] == 5
    monkeypatch.setattr(sys, "argv", [
        "lda", "--repo", "test/repo", "--config", "articles", "--mode", "predict", "--yes",
        "--model-path", str(model_root), "--min-predict-tokens", "1000",
    ])
    assert cli.main() == 0
    assert all(t is None for t in state["dataset"]["lda_topic_id"])
    assert all(t is None for t in state["dataset"]["lda_topic_label"])
    assert set(state["dataset"]["lda_model_name"]) == {model_id}
    assert "min in-vocabulary tokens 1000" in state["push"]["commit_message"]
    assert "_lda_vocab_tokens" not in state["dataset"].column_names
    run = max((model_root / "predictions").iterdir())
    theta = pd.read_parquet(run / "doc_topics.parquet")
    assert theta["below_min_evidence"].all() and (theta["n_vocab_tokens"] > 0).all()
    metadata = json.loads((run / "doc_topics.metadata.json").read_text(encoding="utf-8"))
    assert metadata["min_vocab_tokens"] == 1000


def test_training_reads_only_the_columns_it_needs(fitted_model, monkeypatch, tmp_path):
    _, state = fitted_model
    data = state["dataset"].to_dict()
    data["OCR"] = ["x" * 10_000] * len(data["o:id"])
    state["dataset"] = Dataset.from_dict(data)
    seen = {}
    original = Dataset.select_columns

    def spy(self, columns, *a, **kw):
        seen.setdefault("first", list(columns))
        return original(self, columns, *a, **kw)

    monkeypatch.setattr(Dataset, "select_columns", spy)
    monkeypatch.setattr(sys, "argv", [
        "lda", "--repo", "test/repo", "--config", "articles", "--mode", "fit", "--yes",
        "--model-path", str(tmp_path / "narrow"), "--num-topics", "2", "--passes", "1",
        "--iterations", "20", "--no-below", "1", "--no-above", "1.0", "--skip-coherence",
    ])
    assert cli.main() == 0
    assert seen["first"] == ["o:id", "language", "lemma_nostop"]


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
    summary = json.loads((prevalence.OUTPUT_DIR / "topic_prevalence_summary.json").read_text(encoding="utf-8"))
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


# ── Prevalence model selection (issue #61) ─────────────────────────


def test_prevalence_default_model_follows_subset_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(prevalence, "workspace_root", lambda: tmp_path)
    assert prevalence.default_model_path("articles") == (tmp_path / "lda_model", "Français")
    assert prevalence.default_model_path("publications") == (tmp_path / "lda_model_publications", "Français")
    assert prevalence.default_model_path("references") == (tmp_path / "lda_model_references", "Français")
    assert prevalence.default_model_path("references", "Anglais") == (tmp_path / "lda_model_references_en", "Anglais")


def test_prevalence_training_check_fails_closed():
    bundle = {"id": "lda-sha256:x", "trained_config": "articles", "trained_language": "Français"}
    assert prevalence.check_training_match(bundle, "articles", "Français",
                                           allow_cross_corpus=False)["mismatched_fields"] == []
    with pytest.raises(SystemExit, match="allow-cross-corpus-model"):
        prevalence.check_training_match(bundle, "publications", "Français", allow_cross_corpus=False)
    with pytest.raises(SystemExit, match="language"):
        prevalence.check_training_match({**bundle, "trained_config": "references"}, "references", "Anglais",
                                        allow_cross_corpus=False)
    # A bundle that records no training subset is not trusted.
    with pytest.raises(SystemExit):
        prevalence.check_training_match({**bundle, "trained_config": None}, "articles", "Français",
                                        allow_cross_corpus=False)
    allowed = prevalence.check_training_match(bundle, "publications", "Français", allow_cross_corpus=True)
    assert allowed["mismatched_fields"] == ["config_name"] and allowed["cross_corpus_allowed"]


def _prevalence_cli(state, tmp_path, monkeypatch, *argv):
    frame = state["dataset"].to_pandas()
    frame.attrs["iwac_source_revision"] = "test-source-revision"
    manifests = []
    monkeypatch.setattr(prevalence, "load_subset_dataframe", lambda *a, **kw: frame)
    monkeypatch.setattr(prevalence, "write_run_manifest", lambda *a, **kw: manifests.append(kw))
    monkeypatch.setattr(sys, "argv", [
        "prevalence", "--repo", "test/repo", "--source", "csv", "--bootstrap", "0",
        "--min-docs-year", "1", "--output-dir", str(tmp_path / "results"), *argv,
    ])
    prevalence.main()
    summary = json.loads((tmp_path / "results" / "topic_prevalence_summary.json").read_text(encoding="utf-8"))
    return summary, manifests[-1]["inputs"]


def test_prevalence_resolves_the_periodical_bundle_by_default(fitted_model, tmp_path, monkeypatch):
    press_root, state = fitted_model
    workspace = tmp_path / "workspace"
    periodical_root = workspace / "lda_model_publications"
    monkeypatch.setattr(sys, "argv", [
        "lda", "--repo", "test/repo", "--config", "publications", "--mode", "fit", "--yes",
        "--model-path", str(periodical_root), "--num-topics", "2", "--passes", "1", "--iterations", "20",
        "--no-below", "1", "--no-above", "1.0", "--skip-coherence", "--holdout", "0",
    ])
    assert cli.main() == 0
    periodical_dir, periodical_id = resolve_bundle(periodical_root)
    monkeypatch.setattr(prevalence, "workspace_root", lambda: workspace)
    summary, inputs = _prevalence_cli(state, tmp_path, monkeypatch, "--config", "publications")
    assert summary["model_id"] == periodical_id and summary["model_dir"] == str(periodical_dir)
    assert summary["model_training_match"]["trained"] == {"config_name": "publications", "language": "Français"}
    assert inputs["model_training_match"]["mismatched_fields"] == []
    # The press bundle passed explicitly is refused for the periodicals...
    with pytest.raises(SystemExit, match="trained on config='articles'"):
        _prevalence_cli(state, tmp_path, monkeypatch, "--config", "publications", "--model-path", str(press_root))
    # ...unless cross-corpus use is explicit, which the manifest records.
    summary, inputs = _prevalence_cli(state, tmp_path, monkeypatch, "--config", "publications",
                                      "--model-path", str(press_root), "--allow-cross-corpus-model")
    assert inputs["model_training_match"]["cross_corpus_allowed"] is True
    assert inputs["model_training_match"]["mismatched_fields"] == ["config_name"]
    assert inputs["model_training_match"]["trained"]["config_name"] == "articles"


# ── Reviewed topic-model exclusions ────────────────────────────────


def test_exclusion_registry_lists_the_islam_info_issues():
    from iwac_pipeline.processing.lda_topic_modeling.constants import (
        TOPIC_MODEL_EXCLUSIONS, topic_model_exclusions,
    )

    assert set(TOPIC_MODEL_EXCLUSIONS) == {"publications"}
    registry = topic_model_exclusions("publications")
    assert sorted(registry) == ["23757", "24190", "24233", "24234", "24635", "25286", "25287", "25289"]
    assert all("tabular lists of pilgrims' names" in reason for reason in registry.values())
    assert topic_model_exclusions("articles") == {}
    registry.clear()  # a copy: callers cannot edit the registry
    assert len(topic_model_exclusions("publications")) == 8


def _exclude(monkeypatch, ids):
    monkeypatch.setattr(cli, "topic_model_exclusions",
                        lambda config: {oid: "test reason" for oid in ids} if config == "articles" else {})


def test_excluded_rows_never_reach_phrases_or_dictionary(fitted_model, monkeypatch, tmp_path):
    _, state = fitted_model
    data = state["dataset"].to_dict()
    data["lemma_nostop"][0] += " namelistmarker namelistmarker namelistmarker"
    data["lemma_nostop"][3] += " namelistmarker"
    state["dataset"] = Dataset.from_dict(data)
    _exclude(monkeypatch, ["0"])
    root = tmp_path / "excluded"
    monkeypatch.setattr(sys, "argv", [
        "lda", "--repo", "test/repo", "--config", "articles", "--mode", "fit", "--yes",
        "--model-path", str(root), "--num-topics", "2", "--passes", "1", "--iterations", "20",
        "--no-below", "2", "--no-above", "1.0", "--skip-coherence", "--holdout", "0",
    ])
    assert cli.main() == 0
    directory, model_id = resolve_bundle(root)
    _, params = load_preprocessing(directory)
    assert "0" not in params["extra"]["training_document_ids"]
    assert params["extra"]["topic_model_exclusions"] == {"0": "test reason"}
    # With row 0 gone the marker occurs in one source document, below no_below=2.
    _, dictionary, _ = load_lda_model(root)
    assert "namelistmarker" not in dictionary.token2id
    result = state["dataset"]
    assert result["lda_topic_id"][0] is None and result["lda_topic_label"][0] is None
    assert result["lda_topic_topk"][0] is None and result["lda_topic_prob"][0] is None
    assert result["lda_model_name"][0] == model_id
    assert all(t is not None for t in result["lda_topic_id"][1:])
    run = max((root / "predictions").iterdir())
    assert "0" not in set(pd.read_parquet(run / "doc_topics.parquet")["o:id"])
    metadata = json.loads((run / "doc_topics.metadata.json").read_text(encoding="utf-8"))
    assert metadata["excluded_document_ids"] == ["0"]
    assert "1 reviewed exclusion(s)" in state["push"]["commit_message"]


def test_existing_bundle_predicts_with_registry_rows_nulled(fitted_model, monkeypatch):
    model_root, state = fitted_model
    directory, model_id = resolve_bundle(model_root)
    _, params = load_preprocessing(directory)
    assert "0" in params["extra"]["training_document_ids"]  # trained before the exclusion
    original = {p.name: digest_file(p) for p in directory.iterdir() if p.is_file()}
    before = list(state["dataset"]["lda_topic_id"])
    _exclude(monkeypatch, ["0", "absent-id"])
    monkeypatch.setattr(sys, "argv", [
        "lda", "--repo", "test/repo", "--config", "articles", "--mode", "predict", "--yes",
        "--model-path", str(model_root),
    ])
    assert cli.main() == 0
    result = state["dataset"]
    assert result["lda_topic_id"][0] is None and result["lda_model_name"][0] == model_id
    assert list(result["lda_topic_id"][1:]) == before[1:]
    assert original == {p.name: digest_file(p) for p in directory.iterdir() if p.is_file()}


def test_predict_batch_exclusions_require_ids(fitted_model):
    from iwac_pipeline.processing.lda_topic_modeling.modeling import predict_batch

    root, _ = fitted_model
    model, dictionary, phraser = load_lda_model(root)
    with pytest.raises(ValueError, match="o:id"):
        predict_batch(model, dictionary, {"lemma_nostop": ["imam mosque"], "language": ["Français"]},
                      "lemma_nostop", "t", "p", "l", excluded_ids={"1"})


def test_prevalence_leaves_registry_rows_out():
    from types import SimpleNamespace

    df = pd.DataFrame({"o:id": ["23757", "1"], "pub_date": ["2000", "2000"],
                       "language": ["Français", "Français"], "lemma_nostop": ["kone traore", "imam"]})
    args = SimpleNamespace(include_unknown_language=False, year_min=1900, year_max=2030, config="publications")
    mask, _, counts = prevalence.eligibility(df, "Français", args)
    assert mask.tolist() == [False, True] and counts["docs_excluded_by_registry"] == 1
    args.config = "articles"
    assert prevalence.eligibility(df, "Français", args)[0].tolist() == [True, True]
