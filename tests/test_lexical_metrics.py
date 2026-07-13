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
