"""Unit tests for the pure statistical functions in analyses/ and the LDA
chunking helper."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kb = _load("analyses/keyness_bursts.py", "kb_under_test")
# modeling.py uses relative imports — load it as a package member
# (conftest.py puts post-processing/ on sys.path).
from lda_topic_modeling import modeling  # noqa: E402


class TestDunningG2:
    def test_equal_rates_score_zero(self):
        # 10/1000 in both corpora → no keyness.
        assert kb.dunning_g2(10, 10, 1000, 1000) == 0.0

    def test_overrepresented_in_a_is_positive(self):
        assert kb.dunning_g2(50, 5, 1000, 1000) > 0

    def test_overrepresented_in_b_is_negative(self):
        assert kb.dunning_g2(5, 50, 1000, 1000) < 0

    def test_zero_count_in_a(self):
        assert kb.dunning_g2(0, 50, 1000, 1000) == 0.0

    def test_matches_textbook_value(self):
        # Rayson & Garside worked formula: a=100,b=50,Na=10000,Nb=10000
        a, b, na, nb = 100, 50, 10_000, 10_000
        e1 = na * (a + b) / (na + nb)
        e2 = nb * (a + b) / (na + nb)
        expected = 2 * (a * math.log(a / e1) + b * math.log(b / e2))
        assert math.isclose(kb.dunning_g2(a, b, na, nb), expected)


class TestKleinbergBursts:
    def test_flat_series_no_burst(self):
        years = np.arange(2000, 2010)
        d = np.full(10, 100)
        r = np.full(10, 5)
        assert kb.kleinberg_bursts(r, d, years) == []

    def test_obvious_spike_detected(self):
        years = np.arange(2000, 2010)
        d = np.full(10, 100)
        r = np.array([2, 2, 2, 2, 40, 45, 2, 2, 2, 2])
        bursts = kb.kleinberg_bursts(r, d, years)
        assert len(bursts) == 1
        assert bursts[0]["start"] == 2004 and bursts[0]["end"] == 2005
        assert bursts[0]["mentions_in_burst"] == 85
        assert bursts[0]["weight"] > 0

    def test_degenerate_inputs(self):
        years = np.array([2000])
        assert kb.kleinberg_bursts(np.array([1]), np.array([10]), years) == []
        years = np.arange(2000, 2005)
        zeros = np.zeros(5)
        assert kb.kleinberg_bursts(zeros, np.full(5, 10), years) == []


class TestChunkTokens:
    def test_short_doc_single_chunk(self):
        assert modeling.chunk_tokens(["a", "b"], 10) == [["a", "b"]]

    def test_even_split(self):
        toks = [str(i) for i in range(200)]
        chunks = modeling.chunk_tokens(toks, 100)
        assert [len(c) for c in chunks] == [100, 100]

    def test_short_tail_merged(self):
        # 210 tokens, chunk 100 → tail of 10 (<25%) merges into chunk 2.
        toks = [str(i) for i in range(210)]
        chunks = modeling.chunk_tokens(toks, 100)
        assert [len(c) for c in chunks] == [100, 110]

    def test_long_tail_kept(self):
        # tail of 30 (>=25%) stays its own chunk.
        toks = [str(i) for i in range(230)]
        chunks = modeling.chunk_tokens(toks, 100)
        assert [len(c) for c in chunks] == [100, 100, 30]

    def test_no_tokens_lost(self):
        toks = [str(i) for i in range(437)]
        chunks = modeling.chunk_tokens(toks, 100)
        assert sum(len(c) for c in chunks) == 437
        assert [t for c in chunks for t in c] == toks

    def test_empty(self):
        assert modeling.chunk_tokens([], 100) == []
