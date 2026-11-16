"""LDA preprocessing: source-document frequency, stopword tiers, labels, evidence.

Synthetic corpora only. Each test pins one mechanism: one long document can
no longer mint vocabulary or compounds; generic words stripped before phrase
detection no longer take domain compounds with them; labels keep relevance
order; and a document with almost no in-vocabulary evidence gets no topic.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset, Sequence, Value
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from lda_topic_modeling.artifacts import frozen_preprocessing, preprocessing_fingerprint
from lda_topic_modeling.constants import (
    DOMAIN_STOPWORDS,
    FRAGMENT_STOPWORDS,
    JUNK_COMPOUND_STOPWORDS,
    LDA_GENERIC_STOPWORDS,
    LDA_GEO_STOPWORDS,
    modeling_stopwords,
)
from lda_topic_modeling.lda_topic_modeling import prepare_training_corpus
from lda_topic_modeling.modeling import (
    _normalize_token,
    _select_label_words,
    build_dictionary,
    get_topic_label,
    predict_batch,
    predict_document,
    tokenize_documents,
    tokenize_for_prediction,
)

FR = modeling_stopwords("Français")


def _phrase_prep(min_docs, chunk_words=None):
    prep = frozen_preprocessing(FR, "Français", chunk_words, phrase_min_docs=min_docs)
    # Toy corpora are too small for the production phrase thresholds.
    return {**prep, "phrase_min_count": 3, "phrase_threshold": 1.0}


def _corpus_with_one_long_document():
    docs = [f"grand imam w{i} mois ramadan v{i} prière" for i in range(12)]
    docs += [f"grand w{i} marché x{i} commerce mois" for i in range(12)]
    # One thesis repeating its running header on every page.
    docs.append("tabl matiere chapitre introduction " * 60)
    return docs


class TestOneLongDocumentCannotMintCompounds:
    def test_repeated_header_in_one_document_is_not_a_compound(self):
        _, (bigram, _) = tokenize_documents(
            _corpus_with_one_long_document(), stopwords=FR,
            phrase_min_count=3, phrase_threshold=1.0, phrase_min_docs=3,
        )
        assert "tabl_matiere" not in bigram.phrasegrams
        assert {"grand_imam", "mois_ramadan"} <= set(bigram.phrasegrams)

    def test_without_the_floor_the_header_would_form(self):
        _, (bigram, _) = tokenize_documents(
            _corpus_with_one_long_document(), stopwords=FR,
            phrase_min_count=3, phrase_threshold=1.0, phrase_min_docs=1,
        )
        assert "tabl_matiere" in bigram.phrasegrams

    def test_pruned_phraser_is_what_prediction_applies(self):
        _, phraser = tokenize_documents(
            _corpus_with_one_long_document(), stopwords=FR,
            phrase_min_count=3, phrase_threshold=1.0, phrase_min_docs=3,
        )
        tokens = tokenize_for_prediction("tabl matiere grand imam", FR, phraser)
        assert tokens == ["tabl", "matiere", "grand_imam"]

    def test_groups_count_reprints_once(self):
        # Five rows that are one source document (shared group) are one vote.
        docs = [f"tabl matiere w{i}" for i in range(5)] + [f"autre texte v{i}" for i in range(20)]
        groups = ["thesis"] * 5 + list(range(20))
        _, (bigram, _) = tokenize_documents(
            docs, stopwords=FR, phrase_min_count=3, phrase_threshold=0.5,
            phrase_min_docs=3, groups=groups,
        )
        assert "tabl_matiere" not in bigram.phrasegrams


class TestSourceDocumentFrequency:
    def test_one_chunked_book_cannot_pass_no_below_alone(self):
        book = [["kontdrissa", "islam"]] * 16          # 16 chunks, one source
        others = [["islam", f"w{i}"] for i in range(10)]
        units = book + others
        groups = ["book"] * 16 + list(range(10))
        by_chunk = build_dictionary(units, no_below=3, no_above=1.0)
        by_source = build_dictionary(units, no_below=3, no_above=1.0, groups=groups)
        assert "kontdrissa" in by_chunk.token2id
        assert "kontdrissa" not in by_source.token2id
        assert "islam" in by_source.token2id

    def test_no_above_counts_sources(self):
        # "page" is in every chunk of one long source but in 1 of 4 sources.
        units = [["page", "a"]] * 10 + [["b", "c"], ["b", "d"], ["c", "d"]]
        groups = ["long"] * 10 + [1, 2, 3]
        by_source = build_dictionary(units, no_below=1, no_above=0.5, groups=groups)
        assert "page" in by_source.token2id

    def test_training_corpus_uses_source_frequency_for_v2_settings(self):
        long_doc = " ".join(["kontdrissa islam mosquée"] * 400)
        docs = [long_doc] + [f"islam mosquée prière w{i}" for i in range(12)]
        prep = _phrase_prep(min_docs=3, chunk_words=50)
        chunks, _, dictionary, _ = prepare_training_corpus(docs, prep, no_below=3, no_above=1.0)
        assert len(chunks) > 10
        assert "kontdrissa" not in dictionary.token2id
        assert "islam" in dictionary.token2id

    def test_legacy_settings_keep_chunk_frequency(self):
        long_doc = " ".join(["kontdrissa islam mosquée"] * 400)
        docs = [long_doc] + [f"islam mosquée prière w{i}" for i in range(12)]
        prep = frozen_preprocessing(FR, "Français", 50)
        assert "document_frequency_unit" not in prep
        _, _, dictionary, _ = prepare_training_corpus(docs, prep, no_below=3, no_above=1.0)
        assert "kontdrissa" in dictionary.token2id

    def test_phrase_floor_is_part_of_the_frozen_identity(self):
        a = frozen_preprocessing({"x"}, "Français", None, phrase_min_docs=5)
        b = frozen_preprocessing({"x"}, "Français", None, phrase_min_docs=2)
        assert a["phrase_min_docs"] == 5 and a["version"] == 2
        assert preprocessing_fingerprint(a) != preprocessing_fingerprint(b)
        legacy = frozen_preprocessing({"x"}, "Français", None)
        assert legacy["version"] == 1 and "phrase_min_docs" not in legacy


class TestFragmentTierKeepsDomainCompounds:
    def test_grand_imam_survives_and_bare_grand_does_not(self):
        docs = _corpus_with_one_long_document()[:-1]
        chunks, _, dictionary, _ = prepare_training_corpus(
            docs, _phrase_prep(min_docs=3), no_below=2, no_above=1.0,
        )
        assert "grand_imam" in dictionary.token2id
        assert "mois_ramadan" in dictionary.token2id
        assert "grand" not in dictionary.token2id
        assert "mois" not in dictionary.token2id
        assert not any(t in ("grand", "mois") for doc in chunks for t in doc)

    def test_moved_words_are_post_phrase_only(self):
        moved = {
            "grand", "mondial", "international", "national", "régional", "regional",
            "local", "social", "premier", "première", "premiere", "dernier", "dernière",
            "derniere", "nouveau", "nouvelle", "nouveaux", "petit", "petite", "lieu",
            "monde", "nom", "ville", "moyen", "non", "pouvoir", "savoir", "travail",
            "mois", "jour", "an", "année", "annee", "heure", "période", "periode", "wa",
        }
        assert moved <= FRAGMENT_STOPWORDS
        assert not moved & FR

    def test_no_underscore_entries_before_phrasing(self):
        # Nothing is underscore-joined when the pre-phrase set is applied.
        for language in ("Français", "Anglais"):
            assert not any("_" in w for w in modeling_stopwords(language))
        assert {"op_cit", "de_la", "porto_novo", "bobo_dioulasso", "burkina_faso"} <= JUNK_COMPOUND_STOPWORDS
        assert {"op", "porto", "novo", "dioulasso"} <= FR

    def test_language_exemptions(self):
        en = modeling_stopwords("Anglais")
        assert "die" in FR and "die" not in en          # German article / English verb
        assert "an" in en and "an" not in FR             # English article / French "year"
        assert "die" in DOMAIN_STOPWORDS                  # shared set unchanged

    def test_nouveau_an_compound_survives_bare_an_does_not(self):
        tokens, _ = tokenize_documents(
            ["nouveau an fête an"], stopwords=FR, detect_phrases=False,
            custom_collocations=[("nouveau", "an")],
        )
        assert tokens[0] == ["nouveau_an", "fête"]

    def test_pre_phrase_sets_keep_their_noise(self):
        for word in ("camscanner", "paris", "press", "the", "und", "faire"):
            assert word in FR
        assert "côte" in LDA_GEO_STOPWORDS and "situation" in LDA_GENERIC_STOPWORDS


class TestCollocationsFireOnLemmas:
    def test_lemma_forms_join(self):
        tokens, _ = tokenize_documents(
            ["oeuvre social société civil liberté religieux radicalisation religieux"],
            stopwords=FR, detect_phrases=False,
        )
        assert tokens[0] == [
            "oeuvre_social", "société_civil", "liberté_religieux", "radicalisation_religieux",
        ]

    def test_bare_social_still_dropped(self):
        tokens, _ = tokenize_documents(["action social islam"], stopwords=FR, detect_phrases=False)
        assert tokens[0] == ["action", "islam"]


class TestHyphenAndApostropheLemmas:
    TEXT = "Sous-préfet jama'at da'wa jama’at côte-d'ivoire côte-d’ivoire porto-novo imam"

    def test_tokenizer_keeps_them_whole_and_strips_geo_forms(self):
        tokens, _ = tokenize_documents([self.TEXT], stopwords=FR, detect_phrases=False)
        assert tokens[0] == ["sous-préfet", "jama'at", "da'wa", "jama’at", "imam"]
        assert tokenize_for_prediction(self.TEXT, FR, None) == tokens[0]

    def test_label_normalisation(self):
        assert _normalize_token("sous-préfet") == "sous-prefet"
        assert _normalize_token("jama'at") == _normalize_token("jama’at") == "jamaat"
        assert _normalize_token("côte-d'ivoire") == "cote-divoire"
        assert _normalize_token("ahl_sunna_wa_jama'a") == "ahl sunna wa jamaa"

    def test_hyphenated_lemma_is_one_label_word(self):
        cands = [("préfet", "prefet"), ("sous-préfet", "sous-prefet")]
        assert _select_label_words(cands, 6) == ["préfet", "sous-préfet"]

    def test_phrases_join_hyphenated_tokens(self):
        docs = [f"sous-préfet jama'at w{i}" for i in range(10)] + [f"v{i} x{i}" for i in range(10)]
        tokens, _ = tokenize_documents(docs, stopwords=FR, phrase_min_count=3,
                                       phrase_threshold=0.5, phrase_min_docs=3)
        assert tokens[0][0] == "sous-préfet_jama'at"


class _StubModel:
    def __init__(self, words):
        self.words = words
        self.num_terms = len(words)

    def show_topic(self, topic_id, topn=10):
        return self.words[:topn]


class TestLabelOrder:
    def test_relevance_order_is_kept(self):
        words = [("islam", .2), ("mosquée", .1), ("imam", .05), ("mémoire_maîtrise_kont_drissa", .01)]
        label = get_topic_label(_StubModel(words), 0, top_n=3)
        # The rare 4-word string used to jump ahead of every unigram.
        assert label == "islam - mosquée - imam"

    def test_unigram_after_its_compound_is_skipped(self):
        cands = [("fête_tabaski", "fete tabaski"), ("fête", "fete"), ("mouton", "mouton")]
        assert _select_label_words(cands, 6) == ["fête_tabaski", "mouton"]

    def test_compound_after_its_unigram_takes_its_slot(self):
        cands = [("tabaski", "tabaski"), ("mouton", "mouton"),
                 ("fête_tabaski", "fete tabaski"), ("prière", "priere")]
        assert _select_label_words(cands, 3) == ["fête_tabaski", "mouton", "prière"]

    def test_compound_covering_two_unigrams_frees_a_slot(self):
        cands = [("grand", "grand"), ("mosquée", "mosquee"), ("imam", "imam"),
                 ("grand_mosquée", "grand mosquee"), ("prière", "priere"), ("fidèle", "fidele")]
        assert _select_label_words(cands, 4) == ["grand_mosquée", "imam", "prière", "fidèle"]

    def test_candidates_past_a_full_label_are_ignored(self):
        # Relevance decides the cut; a lower-ranked compound cannot reach back.
        cands = [("tabaski", "tabaski"), ("mouton", "mouton"), ("fête_tabaski", "fete tabaski")]
        assert _select_label_words(cands, 2) == ["tabaski", "mouton"]


def _toy_model():
    docs = [["imam", "mosquée", "prière"] * 3, ["école", "élève", "maître"] * 3] * 4
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(d) for d in docs]
    model = LdaModel(corpus, id2word=dictionary, num_topics=2, random_state=42, passes=2)
    return model, dictionary


class TestMinimumEvidence:
    def test_below_floor_has_no_assignment_but_keeps_theta(self):
        model, dictionary = _toy_model()
        tid, prob, label, topk, theta = predict_document(
            model, dictionary, ["imam", "inconnu"], return_distribution=True,
            topic_labels={0: "a", 1: "b"}, min_vocab_tokens=5,
        )
        assert (tid, prob, label, topk) == (None, None, None, None)
        assert theta is not None and np.isclose(sum(theta), 1)

    def test_at_floor_is_assigned(self):
        model, dictionary = _toy_model()
        result = predict_document(model, dictionary, ["imam", "mosquée"] * 3,
                                  topic_labels={0: "a", 1: "b"}, min_vocab_tokens=5)
        assert result[0] is not None

    def test_default_has_no_floor(self):
        model, dictionary = _toy_model()
        assert predict_document(model, dictionary, ["imam"], topic_labels={0: "a", 1: "b"})[0] is not None

    def test_batch_nulls_stale_values_and_reports_evidence(self):
        model, dictionary = _toy_model()
        batch = {
            "lemma_nostop": ["imam", "imam mosquée prière imam mosquée prière"],
            "language": ["Français", "Français"],
            "lda_topic_id": [1, None], "lda_topic_prob": [0.9, None],
            "lda_topic_label": ["old", None], "lda_topic_topk": ["1:0.9", None],
            "lda_model_name": ["old-model", None],
        }
        out = predict_batch(
            model, dictionary, batch, "lemma_nostop", "lda_topic_id", "lda_topic_prob",
            "lda_topic_label", topic_topk_col="lda_topic_topk", model_name_col="lda_model_name",
            model_name="m", theta_col="_theta", min_vocab_tokens=5, evidence_col="_n",
            topic_labels={0: "a", 1: "b"}, custom_collocations=[], fragment_stopwords=set(),
        )
        assert out["lda_topic_id"][0] is None and out["lda_topic_label"][0] is None
        assert out["lda_model_name"][0] == "m" and out["_theta"][0] is not None
        assert out["_n"] == [1, 6]
        assert out["lda_topic_id"][1] is not None

    def test_all_null_batch_maps_with_declared_types(self):
        model, dictionary = _toy_model()
        ds = Dataset.from_dict({"lemma_nostop": ["imam", "prière", "école élève"],
                                "language": ["Français"] * 3})
        features = ds.features.copy()
        features.update({
            "lda_topic_id": Value("int64"), "lda_topic_prob": Value("float64"),
            "lda_topic_label": Value("string"), "lda_topic_topk": Value("string"),
            "lda_model_name": Value("string"), "_n": Value("int64"),
            "_theta": Sequence(Value("float64")),
        })
        out = ds.map(
            lambda b: predict_batch(
                model, dictionary, b, "lemma_nostop", "lda_topic_id", "lda_topic_prob",
                "lda_topic_label", topic_topk_col="lda_topic_topk",
                model_name_col="lda_model_name", model_name="m", theta_col="_theta",
                min_vocab_tokens=5, evidence_col="_n", topic_labels={0: "a", 1: "b"},
                custom_collocations=[], fragment_stopwords=set(),
            ),
            batched=True, batch_size=1, features=features,
        )
        assert out["lda_topic_id"] == [None, None, None]
        assert out.features["lda_topic_id"] == Value("int64")
