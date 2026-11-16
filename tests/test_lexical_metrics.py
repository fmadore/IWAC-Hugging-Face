"""Tests for the lexical metric functions (MATTR, word count)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lex = _load("post-processing/calculate_lexical_richness.py", "lex_under_test")
wc = _load("post-processing/calculate_word_count.py", "wc_under_test")


class TestMattr:
    def test_too_short_returns_none_not_ttr(self):
        # 5 tokens < window 50 → None (never a plain TTR fallback).
        assert lex.calculate_mattr("un deux trois quatre cinq") is None

    def test_exactly_window_size_computes(self):
        text = " ".join(f"mot{i}" for i in range(50))
        assert lex.calculate_mattr(text) == 1.0  # all types unique

    def test_repetition_lowers_mattr(self):
        varied = " ".join(f"mot{i}" for i in range(100))
        repetitive = "islam " * 100
        assert lex.calculate_mattr(varied) > lex.calculate_mattr(repetitive)

    def test_range_is_zero_one(self):
        text = ("le chat mange la souris " * 30).strip()
        v = lex.calculate_mattr(text)
        assert 0.0 < v <= 1.0

    def test_elided_clitics_are_tokens_like_articles(self):
        # ``l'`` is a token exactly as ``la`` is, so the two texts score alike.
        elided = " ".join(f"l'objet{i}" for i in range(50))
        article = " ".join(f"la objet{i}" for i in range(50))
        assert lex.calculate_mattr(elided) == lex.calculate_mattr(article)

    def test_language_reaches_the_tokenizer(self):
        # 30 elided words: 60 tokens in French (>= window), 30 elsewhere.
        text = " ".join(f"l'objet{i}" for i in range(30))
        assert lex.calculate_mattr(text, language="Français") is not None
        assert lex.calculate_mattr(text, language="Haoussa") is None

    def test_empty_and_none(self):
        assert lex.calculate_mattr("") is None
        assert lex.calculate_mattr(None) is None


class TestCountWords:
    def test_basic(self):
        assert wc.count_words("le chat mange") == 3

    def test_elided_clitic_counts_as_a_word(self):
        assert wc.count_words("l'islam") == 2
        assert wc.count_words("qu'il d'abord") == 4

    def test_empty(self):
        assert wc.count_words("") == 0

    def test_empty_batch_guard(self):
        out = wc.add_word_count_batch({}, text_col="OCR", count_col="nb_mots")
        assert out == {"nb_mots": []}

    def test_update_mode_all_recomputes(self):
        batch = {"OCR": ["un deux", "trois"], "nb_mots": [999, None]}
        out = wc.add_word_count_batch(batch, text_col="OCR", count_col="nb_mots", update_mode="all")
        assert out["nb_mots"] == [2, 1]  # 999 overwritten

    def test_update_mode_missing_recomputes_unproven_counts(self):
        batch = {"OCR": ["un deux", "trois"], "nb_mots": [999, None]}
        out = wc.add_word_count_batch(batch, text_col="OCR", count_col="nb_mots", update_mode="missing")
        assert out["nb_mots"] == [2, 1]  # no provenance: 999 cannot be trusted


class TestReadabilityLanguageGate:
    """French Flesch on a non-French text ranks correct text as unreadable;
    those rows must be null, never a low score."""

    FRENCH = ("Le conseil des imams a tenu sa réunion annuelle à Ouagadougou. " * 12)

    def _run(self, languages, update_mode="all", existing=None):
        n = len(languages)
        batch = {
            "OCR": [self.FRENCH] * n,
            "language": languages,
            "Richesse_Lexicale_OCR": [None] * n,
            "Lisibilite_OCR": existing or [None] * n,
        }
        counter = {"richness_too_short": 0, "readability_failed": 0}
        out = lex.compute_metrics_batch(
            batch, text_col="OCR", richness_col="Richesse_Lexicale_OCR",
            readability_col="Lisibilite_OCR", update_mode=update_mode,
            window_size=50, error_counter=counter,
        )
        return out, counter

    def test_only_monolingual_french_rows_get_a_score(self):
        out, counter = self._run(
            ["Français", "Français|Anglais", "Anglais|Français", "Ewé", "", None]
        )
        scores = out["Lisibilite_OCR"]
        assert scores[0] is not None
        assert scores[1:] == [None] * 5
        assert counter["readability_not_french"] == 5
        # MATTR carries no lexicon and is kept for every row.
        assert all(v is not None for v in out["Richesse_Lexicale_OCR"])

    def test_missing_mode_clears_an_old_score_on_a_non_french_row(self):
        out, _ = self._run(["Kabiyè", "Français"], update_mode="missing",
                           existing=[12.5, 60.0])
        assert out["Lisibilite_OCR"][0] is None
        # The French row has no persisted provenance, so its legacy value is
        # recomputed too, even in missing mode.
        assert out["Lisibilite_OCR"][1] == lex.calculate_readability(self.FRENCH)

    def test_without_a_language_column_nothing_is_scored(self):
        batch = {"OCR": [self.FRENCH], "Richesse_Lexicale_OCR": [None],
                 "Lisibilite_OCR": [None]}
        out = lex.compute_metrics_batch(
            batch, text_col="OCR", richness_col="Richesse_Lexicale_OCR",
            readability_col="Lisibilite_OCR", update_mode="all", window_size=50,
            error_counter={"richness_too_short": 0, "readability_failed": 0},
        )
        assert out["Lisibilite_OCR"] == [None]


class TestReadabilityEvidence:
    SENTENCE = "Le conseil des imams a tenu sa réunion annuelle à Ouagadougou"
    FRENCH = (SENTENCE + ". ") * 12

    def test_a_well_formed_french_text_is_scored(self):
        assert lex.readability_evidence(self.FRENCH) is None
        assert 0 < lex.calculate_readability(self.FRENCH) < 121

    @pytest.mark.parametrize("text, reason", [
        ("...", "no_letters"),
        ("1995 2001 / 12,5", "no_letters"),
        ("Une phrase française.", "too_few_words"),
        # One sentence repeated 60 times without full stops scored -570.
        (" ".join([SENTENCE] * 60), "too_few_sentences"),
        # Enough sentences, but 3 full stops across 660 words.
        (". ".join([" ".join([SENTENCE] * 20)] * 3) + ".", "implausible_sentence_length"),
    ])
    def test_thin_evidence_yields_null(self, text, reason):
        assert lex.readability_evidence(text) == reason
        assert lex.calculate_readability(text) is None

    def test_insufficient_evidence_is_a_settled_null_not_a_failure(self):
        counter = {}
        out = lex.compute_metrics_batch(
            {"OCR": ["Une phrase française."], "language": ["Français"]}, text_col="OCR",
            richness_col="Richesse_Lexicale_OCR", readability_col="Lisibilite_OCR",
            update_mode="missing", window_size=50, error_counter=counter,
        )
        assert out["Lisibilite_OCR"] == [None]
        assert out["Lisibilite_OCR_config_hash"][0]  # not retried on every run
        assert counter == {"richness_too_short": 1, "readability_insufficient": 1}

    def test_thresholds_are_part_of_the_configuration_hash(self, monkeypatch):
        from iwac_common.enrichment import config_fingerprint

        before = config_fingerprint(lex.metric_settings(50)["Lisibilite_OCR"])
        monkeypatch.setattr(lex, "MIN_READABILITY_WORDS", 150)
        after = config_fingerprint(lex.metric_settings(50)["Lisibilite_OCR"])
        assert before != after

    def test_score_does_not_depend_on_textstat_global_language(self):
        import textstat

        expected = lex.calculate_readability(self.FRENCH)
        textstat.set_lang("es")
        try:
            assert lex.calculate_readability(self.FRENCH) == expected
        finally:
            textstat.set_lang("en_US")


class TestStageWiring:
    def test_config_choices_exclude_subsets_without_ocr(self):
        assert "index" not in lex.LEXICAL_SUBSETS and "images" not in lex.LEXICAL_SUBSETS
        with pytest.raises(SystemExit):
            lex.parse_args(["--config", "index"])

    def test_yes_means_missing_and_skips_the_prompt(self):
        from _common import resolve_update_mode

        assert resolve_update_mode(lex.parse_args(["-y"])) == "missing"
        assert lex.parse_args(["--update-mode", "all", "-y"]).update_mode == "all"

    def test_a_run_that_changes_nothing_does_not_push(self, monkeypatch):
        import io

        from datasets import Dataset
        from rich.console import Console

        pushed = []
        state = {"ds": Dataset.from_dict({
            "o:id": ["1"], "OCR": [TestReadabilityLanguageGate.FRENCH], "language": ["Français"],
        })}
        monkeypatch.setattr(lex, "console", Console(file=io.StringIO()))
        monkeypatch.setattr(lex, "ensure_hf_token", lambda **kwargs: "offline-placeholder")
        monkeypatch.setattr(lex, "load_hub_dataset", lambda *a, **k: state["ds"])
        monkeypatch.setattr(lex, "push_dataset", lambda ds, **k: pushed.append(ds) or True)
        argv = ["--config", "articles", "--update-mode", "missing"]
        assert lex.main(argv) == 0
        assert len(pushed) == 1
        state["ds"] = pushed[0]
        assert lex.main(argv) == 0
        assert len(pushed) == 1  # nothing changed: no second push
        assert lex.main(["--config", "articles", "--update-mode", "all", "-y"]) == 0
        assert len(pushed) == 1  # recomputing identical values changes nothing either
