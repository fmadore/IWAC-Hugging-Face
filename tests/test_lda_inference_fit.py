"""LDA inference weighting, fit hygiene and stability matching (issue #43)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from gensim.corpora import Dictionary

from iwac_pipeline.processing.lda_topic_modeling import lda_topic_modeling as ltm
from iwac_pipeline.processing.lda_topic_modeling import modeling
from iwac_pipeline.processing.lda_topic_modeling.artifacts import chunk_weighting, frozen_preprocessing


class TwoTopicModel:
    """Topic 0 for any chunk containing "a", else topic 1."""

    num_topics = 2

    def __init__(self, dictionary):
        self.random_state = np.random.RandomState(0)
        self.a_id = dictionary.token2id["a"]

    def get_document_topics(self, bow, minimum_probability=0.0):
        return [(0, 1.0), (1, 0.0)] if any(i == self.a_id for i, _ in bow) else [(0, 0.0), (1, 1.0)]


def test_chunk_weights_count_only_in_vocabulary_tokens():
    dictionary = Dictionary([["a", "b"]])
    model = TwoTopicModel(dictionary)
    # Chunk 1: four known tokens. Chunk 2: one known token among OCR noise.
    tokens = ["a"] * 4 + ["b", "zq", "xw", "vy"]
    legacy = modeling._infer_distribution(model, dictionary, tokens, 4)
    weighted = modeling._infer_distribution(model, dictionary, tokens, 4, "in_vocabulary_tokens")
    assert legacy.tolist() == pytest.approx([0.5, 0.5])
    assert weighted.tolist() == pytest.approx([0.8, 0.2])
    with pytest.raises(ValueError):
        modeling._infer_distribution(model, dictionary, tokens, 4, "bytes")


def test_chunk_weighting_is_part_of_new_fits_only():
    legacy = frozen_preprocessing({"x"}, "Français", 1000)
    assert "chunk_weighting" not in legacy and chunk_weighting(legacy) == "all_tokens"
    new = frozen_preprocessing({"x"}, "Français", 1000, chunk_weighting="in_vocabulary_tokens")
    assert chunk_weighting(new) == "in_vocabulary_tokens"


def test_stability_uses_the_optimal_assignment():
    """Greedy takes a 0.5 pair first and is left with 0.0 (mean 0.25); the optimal
    assignment pairs 2/6 with 0.5."""
    def model(topics):
        return SimpleNamespace(num_topics=len(topics),
                               show_topic=lambda t, topn: [(w, 1.0) for w in topics[t]])

    a = model([["w1", "w2", "w3", "w4", "w5"], ["w1", "w6"]])
    b = model([["w1", "w2", "w3", "w6"], ["w4", "w5", "w9"]])
    # J(a0,b0) = 3/6 = 0.5, J(a0,b1) = 2/6, J(a1,b0) = 2/4 = 0.5, J(a1,b1) = 0
    score = modeling.topic_stability_jaccard([a, b], topn=10)
    assert score == pytest.approx((2 / 6 + 0.5) / 2)
    same = modeling.topic_stability_jaccard([a, a], topn=10)
    assert same == pytest.approx(1.0)


def test_training_corpus_drops_chunks_left_empty_by_the_dictionary():
    prep = frozen_preprocessing(set(), "Français", None)
    texts = ["imam mosquée prière"] * 4 + ["zzrare"]
    chunks, _, dictionary, corpus = ltm.prepare_training_corpus(texts, prep, no_below=2, no_above=1.0)
    assert len(chunks) == len(corpus) == 4
    assert all(corpus)


def test_failed_fit_removes_its_staging_directory(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ltm, "prepare_training_corpus",
                        lambda *a, **k: ([["imam"]], None, Dictionary([["imam"]]), [[(0, 1)]]))
    monkeypatch.setattr(ltm, "compute_corpus_word_probs", lambda *a, **k: None)
    monkeypatch.setattr(ltm, "create_lda_model", lambda *a, **k: SimpleNamespace(num_topics=1))
    monkeypatch.setattr(ltm, "get_topic_label", lambda *a, **k: "imam")
    monkeypatch.setattr(ltm, "save_lda_model", boom)
    settings = SimpleNamespace(model_root=tmp_path, lambda_relevance=None, num_topics=1)
    args = SimpleNamespace(no_below=1, no_above=1.0, topic_label_words=3, passes=1,
                           iterations=1, chunksize=1, workers=None)
    with pytest.raises(RuntimeError, match="disk full"):
        ltm.fit_and_publish(args, settings, SimpleNamespace(texts=["imam"], groups=["1"]),
                            frozen_preprocessing(set(), "Français", None), set(), None, None,
                            SimpleNamespace(info=lambda *a: None))
    assert not any(p.name.startswith(".fit-") for p in tmp_path.iterdir())
