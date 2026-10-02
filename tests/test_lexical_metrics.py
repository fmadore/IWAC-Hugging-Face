"""Tests for the lexical metric functions (MATTR, word count)."""

import importlib.util
import sys
from pathlib import Path

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

    def test_elision_not_counted_as_types(self):
        # l'/d'/qu' fragments must not add types: both texts have the same
        # vocabulary once clitics are stripped.
        a = " ".join(f"l'objet{i}" for i in range(50))
        b = " ".join(f"objet{i}" for i in range(50))
        assert lex.calculate_mattr(a) == lex.calculate_mattr(b)

    def test_empty_and_none(self):
        assert lex.calculate_mattr("") is None
        assert lex.calculate_mattr(None) is None


class TestCountWords:
    def test_basic(self):
        assert wc.count_words("le chat mange") == 3

    def test_elision_counts_one_word(self):
        assert wc.count_words("l'islam") == 1
        assert wc.count_words("qu'il d'abord") == 2

    def test_empty(self):
        assert wc.count_words("") == 0

    def test_empty_batch_guard(self):
        out = wc.add_word_count_batch({}, text_col="OCR", count_col="nb_mots")
        assert out == {"nb_mots": []}

    def test_update_mode_all_recomputes(self):
        batch = {"OCR": ["un deux", "trois"], "nb_mots": [999, None]}
        out = wc.add_word_count_batch(batch, text_col="OCR", count_col="nb_mots", update_mode="all")
        assert out["nb_mots"] == [2, 1]  # 999 overwritten

    def test_update_mode_missing_preserves_existing(self):
        batch = {"OCR": ["un deux", "trois"], "nb_mots": [999, None]}
        out = wc.add_word_count_batch(batch, text_col="OCR", count_col="nb_mots", update_mode="missing")
        assert out["nb_mots"] == [999, 1]  # existing kept, null filled


class TestReadabilityLanguageGate:
    """French Flesch on a non-French text ranks correct text as unreadable;
    those rows must be null, never a low score."""

    FRENCH = ("Le conseil des imams a tenu sa réunion annuelle à Ouagadougou. " * 12)

    def _run(self, languages, update_mode="all", existing=None):
        lex.textstat.set_lang("fr")
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
        lex.textstat.set_lang("fr")
        batch = {"OCR": [self.FRENCH], "Richesse_Lexicale_OCR": [None],
                 "Lisibilite_OCR": [None]}
        out = lex.compute_metrics_batch(
            batch, text_col="OCR", richness_col="Richesse_Lexicale_OCR",
            readability_col="Lisibilite_OCR", update_mode="all", window_size=50,
            error_counter={"richness_too_short": 0, "readability_failed": 0},
        )
        assert out["Lisibilite_OCR"] == [None]

    def test_primary_language_parsing(self):
        assert lex.primary_language(" Français | Anglais") == "Français"
        assert lex.primary_language(None) == ""
