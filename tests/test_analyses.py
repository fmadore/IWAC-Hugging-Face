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


# --- Statistical helpers (analyses/_stats.py) --------------------------------

import importlib.util as _ilu
_stats_spec = _ilu.spec_from_file_location("iwac_stats", REPO_ROOT / "analyses" / "_stats.py")
_stats = _ilu.module_from_spec(_stats_spec)
sys.modules["iwac_stats"] = _stats
_stats_spec.loader.exec_module(_stats)


class TestBHAdjust:
    def test_monotone_and_bounded(self):
        q = _stats.bh_adjust([0.01, 0.02, 0.03, 0.04])
        assert all(0 <= v <= 1 for v in q)

    def test_known_rejections(self):
        # Classic BH example: with these p-values at q<0.05, the smallest
        # survive. Just assert the smallest p gets the smallest q and all
        # q >= p (step-up inflation).
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
        q = _stats.bh_adjust(p)
        assert np.all(q >= p - 1e-12)
        assert np.argmin(q) == np.argmin(p)

    def test_nan_ignored(self):
        q = _stats.bh_adjust([0.01, np.nan, 0.02])
        assert np.isnan(q[1]) and np.isfinite(q[0]) and np.isfinite(q[2])

    def test_all_nan(self):
        q = _stats.bh_adjust([np.nan, np.nan])
        assert np.all(np.isnan(q))


class TestMannKendall:
    def test_monotone_increasing_significant(self):
        s, z, p = _stats.mann_kendall(list(range(12)))
        assert s > 0 and z > 0 and p < 0.05

    def test_monotone_decreasing_negative_s(self):
        s, z, p = _stats.mann_kendall(list(range(12, 0, -1)))
        assert s < 0 and p < 0.05

    def test_flat_not_significant(self):
        s, z, p = _stats.mann_kendall([5.0] * 10)
        assert p == 1.0

    def test_too_short(self):
        assert _stats.mann_kendall([1, 2, 3]) == (0.0, 0.0, 1.0)


class TestWeightedSlope:
    def test_matches_ols_with_equal_weights(self):
        x = [2000, 2001, 2002, 2003]
        y = [0.1, 0.2, 0.3, 0.4]
        w = [1, 1, 1, 1]
        assert abs(_stats.weighted_least_squares_slope(x, y, w) - 0.1) < 1e-9

    def test_weights_shift_slope(self):
        x = [0, 1, 2]
        y = [0.0, 10.0, 0.0]
        heavy_mid = _stats.weighted_least_squares_slope(x, y, [1, 100, 1])
        assert abs(heavy_mid) < 1e-6  # symmetric heavy midpoint → ~flat

    def test_degenerate(self):
        assert np.isnan(_stats.weighted_least_squares_slope([1], [1], [1]))


class TestBootstrapCI:
    def test_matrix_ci_brackets_mean(self):
        rng = np.random.default_rng(0)
        m = rng.normal(0.5, 0.1, size=(500, 3))
        lo, hi = _stats.bootstrap_mean_ci(m, n_boot=200, seed=1)
        mean = m.mean(axis=0)
        assert np.all(lo <= mean) and np.all(mean <= hi)
        assert lo.shape == (3,)

    def test_disabled(self):
        lo, hi = _stats.bootstrap_mean_ci(np.zeros((5, 2)), n_boot=0, seed=1)
        assert np.all(np.isnan(lo)) and np.all(np.isnan(hi))


# --- keyness pure helpers ----------------------------------------------------

class TestLogRatio:
    def test_antisymmetry(self):
        a, b, ta, tb = 40, 10, 1000, 1000
        assert abs(kb.log_ratio(a, b, ta, tb) + kb.log_ratio(b, a, tb, ta)) < 1e-9

    def test_sign(self):
        assert kb.log_ratio(40, 10, 1000, 1000) > 0
        assert kb.log_ratio(10, 40, 1000, 1000) < 0

    def test_zero_count_finite(self):
        assert np.isfinite(kb.log_ratio(0, 10, 1000, 1000))


class TestUniqueSubjects:
    def test_dedup_and_split(self):
        assert kb.unique_subjects("Islam|Islam|Paix") == {"Islam", "Paix"}

    def test_empty_and_nan(self):
        assert kb.unique_subjects("") == set()
        assert kb.unique_subjects(None) == set()


class TestContiguousYearIndex:
    def test_fills_gaps(self):
        idx = kb.contiguous_year_index([2000, 2002, 2005])
        assert list(idx) == list(range(2000, 2006))

    def test_single_year(self):
        assert list(kb.contiguous_year_index([2010])) == [2010]


class TestBurstsWithCalendarGap:
    def test_spike_with_missing_year_no_crash(self):
        # contiguous index over 2000..2009; year 2004 has zero docs.
        years = np.arange(2000, 2010)
        d = np.array([100, 100, 100, 100, 0, 100, 40, 45, 100, 100])
        r = np.array([2, 2, 2, 2, 0, 2, 38, 43, 2, 2])
        bursts = kb.kleinberg_bursts(r, d, years)
        assert isinstance(bursts, list)
        # the 2006-2007 spike should surface
        assert any(b["start"] <= 2007 <= b["end"] for b in bursts)
