"""Stopword handling in the LDA pipeline.

The point of these tests is the *ordering* of the two filters. Modeling
stopwords run before gensim's phrase detection so an artefact cannot glue
itself to a real word; fragment stopwords run after it so a compound built
on a fragment survives. Get the order backwards and you either keep
"scanned_camscanner_organisation_pionnier" or lose "al_azhar".
"""

from lda_topic_modeling.constants import (
    ARTIFACT_LABEL_STOPWORDS,
    DOMAIN_STOPWORDS,
    FRAGMENT_STOPWORDS,
    JUNK_COMPOUND_STOPWORDS,
    POST_PHRASE_STOPWORDS,
)
from lda_topic_modeling.modeling import (
    _contains_artifact,
    _normalize_token,
    drop_fragments,
    tokenize_documents,
    tokenize_for_prediction,
)


class TestDigitisationArtifacts:
    def test_camscanner_watermark_is_a_modeling_stopword(self):
        # Pre-phrase, so "Scanned by CamScanner" cannot form a compound with
        # the words its page footer happens to sit next to.
        assert "camscanner" in DOMAIN_STOPWORDS
        assert "scanned" in DOMAIN_STOPWORDS

    def test_watermark_dropped_before_phrase_detection(self):
        docs = ["mémoire maîtrise scanned camscanner islam"] * 30
        tokenized, _ = tokenize_documents(docs, stopwords=set(DOMAIN_STOPWORDS))
        flat = {t for doc in tokenized for t in doc}
        assert not any("camscanner" in t or "scanned" in t for t in flat)
        assert "islam" in flat

    def test_scanner_verb_is_not_a_stopword(self):
        # French "scanner" (medical) and "scandale" must not be collateral.
        assert "scanner" not in DOMAIN_STOPWORDS
        assert "scan" not in DOMAIN_STOPWORDS


class TestFragmentStopwords:
    def test_bare_fragment_dropped_compound_kept(self):
        tokens = ["al", "al_azhar", "el", "el_hadj", "page", "page_facebook"]
        assert drop_fragments(tokens) == ["al_azhar", "el_hadj", "page_facebook"]

    def test_fragments_are_not_modeling_stopwords(self):
        # In DOMAIN_STOPWORDS they would be stripped pre-phrase and the
        # compounds could never form in the first place.
        for frag in FRAGMENT_STOPWORDS:
            assert frag not in DOMAIN_STOPWORDS

    # Joining is driven by an explicit collocation rather than gensim's
    # statistics: the invariant under test is the ORDER of the passes, and a
    # toy corpus is too small to make Phrases fire predictably.
    COLLOC = [("al", "azhar")]

    def test_compound_survives_the_full_training_tokenizer(self):
        docs = ["al azhar université al islam"]
        tokenized, _ = tokenize_documents(
            docs,
            stopwords=set(DOMAIN_STOPWORDS),
            detect_phrases=False,
            custom_collocations=self.COLLOC,
        )
        assert tokenized[0] == ["al_azhar", "université", "islam"]

    def test_prediction_mirrors_training(self):
        docs = ["al azhar université al islam"]
        tokenized, _ = tokenize_documents(
            docs,
            stopwords=set(DOMAIN_STOPWORDS),
            detect_phrases=False,
            custom_collocations=self.COLLOC,
        )
        predicted = tokenize_for_prediction(
            docs[0], set(DOMAIN_STOPWORDS), None, custom_collocations=self.COLLOC
        )
        assert predicted == tokenized[0]


class TestJunkCompounds:
    def test_apparatus_compound_dropped_real_one_kept(self):
        tokens = [
            "university_press", "indiana_university_press",
            "university_medina", "university_abomey_calavi", "university",
        ]
        assert drop_fragments(tokens) == [
            "university_medina", "university_abomey_calavi", "university",
        ]

    def test_licence_residue_dropped(self):
        # "frédérick"/"madore" go pre-phrase; this catches the phraser
        # re-joining what is left of "licence accordée … …".
        assert drop_fragments(["licence_accorder", "licence", "obtenir_licence"]) == [
            "licence", "obtenir_licence",
        ]

    def test_head_words_kept_out_of_the_pre_phrase_set(self):
        # Stripping these would cost a real French noun and the religious
        # formulae allah_accorder / dieu_accorder.
        for word in ("licence", "accorder", "university"):
            assert word not in DOMAIN_STOPWORDS

    def test_post_phrase_set_is_the_union(self):
        assert POST_PHRASE_STOPWORDS == FRAGMENT_STOPWORDS | JUNK_COMPOUND_STOPWORDS

    def test_junk_compounds_are_compounds(self):
        # A bare word here would silently veto every compound built on it;
        # that job belongs to DOMAIN_STOPWORDS.
        assert all("_" in t for t in JUNK_COMPOUND_STOPWORDS)


class TestLabelNormalisation:
    def test_underscore_is_a_word_separator(self):
        # Without this, every phrase counted as a unigram and the ngram
        # subsumption in get_topic_label never fired.
        assert _normalize_token("fête_tabaski") == "fete tabaski"

    def test_accents_stripped(self):
        assert _normalize_token("Côte_Ivoire") == "cote ivoire"

    def test_artifact_detected_inside_a_compound(self):
        assert _contains_artifact(_normalize_token("scanned_camscanner_organisation"))
        assert _contains_artifact(_normalize_token("فايروسات_كورونا_تصيب_الفايروسات"))

    def test_domain_compound_is_not_an_artifact(self):
        for word in ("al_azhar", "el_hadj", "nuit_destin", "école_coranique"):
            assert not _contains_artifact(_normalize_token(word))

    def test_artifact_set_stays_narrow(self):
        # A component-wise veto is blunt: anything in here kills every
        # compound containing it, so it must hold only digitisation damage.
        assert ARTIFACT_LABEL_STOPWORDS.isdisjoint(FRAGMENT_STOPWORDS)
