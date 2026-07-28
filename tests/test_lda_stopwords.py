"""Stopword handling in the LDA pipeline.

The point of these tests is the *ordering* of the two filters. Modeling
stopwords run before gensim's phrase detection so an artefact cannot glue
itself to a real word; fragment stopwords run after it so a compound built
on a fragment survives. Get the order backwards and you either keep
"scanned_camscanner_organisation_pionnier" or lose "al_azhar".
"""

from lda_topic_modeling.constants import (
    ARTIFACT_LABEL_STOPWORDS,
    CONFIG_PRESETS,
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

    def test_journal_title_dropped_but_its_words_survive(self):
        # "religion", "journal" and "africa" are core vocabulary, so the
        # title can only be caught whole.
        assert drop_fragments(
            ["journal_religion_africa", "religion", "journal_officiel", "africa"]
        ) == ["religion", "journal_officiel", "africa"]
        for word in ("religion", "journal", "africa"):
            assert word not in DOMAIN_STOPWORDS

    def test_cited_scholars_dissolve_their_coauthor_strings(self):
        # Removing gomez/perez is what kills leblanc_/savadogo_gomez_perez,
        # so those everyday West African surnames stay out of the set.
        assert {"muriel", "gomez", "perez", "hanretta", "weiss"} <= DOMAIN_STOPWORDS
        for surname in ("leblanc", "savadogo", "souza"):
            assert surname not in DOMAIN_STOPWORDS

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


class TestConfigPresets:
    """Preset resolution for the per-language references models.

    k is part of what lda_topic_id means, and model_path decides which model
    a fit overwrites, so both must resolve correctly per language.
    """

    def _resolve(self, config: str, language: str | None = None) -> dict:
        # Mirrors the resolution in lda_topic_modeling.main().
        preset = CONFIG_PRESETS.get(config, {})
        lang = language or preset.get("language", "Français")
        return {**preset, **preset.get("language_overrides", {}).get(lang, {})}

    def test_french_and_english_resolve_to_different_models(self):
        fr = self._resolve("references")
        en = self._resolve("references", "Anglais")
        assert fr["model_path"] == "lda_model_references"
        assert en["model_path"] == "lda_model_references_en"

    def test_k_is_pinned_per_language(self):
        assert self._resolve("references")["num_topics"] == 16
        assert self._resolve("references", "Anglais")["num_topics"] == 8

    def test_references_no_longer_sweeps_k_by_default(self):
        # C_v cannot separate k on this corpus, and an auto-sweep renumbers
        # every topic on each re-fit.
        for lang in (None, "Anglais"):
            assert not self._resolve("references", lang).get("optimize_topics", False)

    def test_articles_k_does_not_ride_on_the_global_default(self):
        assert CONFIG_PRESETS["articles"]["num_topics"] == 30

    def test_overrides_do_not_mutate_the_base_preset(self):
        self._resolve("references", "Anglais")
        assert CONFIG_PRESETS["references"]["model_path"] == "lda_model_references"
        assert CONFIG_PRESETS["references"]["num_topics"] == 16
