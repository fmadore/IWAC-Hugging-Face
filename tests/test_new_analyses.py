"""Tests for pure helpers in the new analyses scripts (topic_sentiment,
entity_networks). Both load by file path (hyphenated / sibling-import scripts).
"""

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "post-processing"))
sys.path.insert(0, str(REPO_ROOT / "analyses"))


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ts = _load("analyses/topic_sentiment.py", "topic_sentiment_ut")
en = _load("analyses/entity_networks.py", "entity_networks_ut")


class TestTopicSentimentHelpers:
    def test_slug_ascii(self):
        assert ts.slug("Très négatif") == "tres_negatif"
        assert ts.slug("Plutôt Objectif") == "plutot_objectif"

    # Column names come from the live panel (iwac_common.sentiment_panel), so
    # these stay correct when the panel is rotated or resized. Votes are built
    # from the panel size rather than hardcoded, because the majority threshold
    # is now "> half the voters" rather than a fixed 2.
    def test_consensus_label_majority(self):
        cols = ts.POLARITY_COLS
        n_majority = len(cols) // 2 + 1
        votes = ["Négatif"] * n_majority + ["Positif"] * (len(cols) - n_majority)
        df = pd.DataFrame([votes], columns=cols)
        assert ts.consensus_label_series(df, cols).iloc[0] == "Négatif"

    def test_consensus_label_no_majority(self):
        cols = ts.POLARITY_COLS
        # Spread votes across distinct labels so no label clears half.
        labels = list(ts.POLARITY_ORDER)
        votes = [labels[i % len(labels)] for i in range(len(cols))]
        df = pd.DataFrame([votes], columns=cols)
        assert ts.consensus_label_series(df, cols).iloc[0] == ""

    def test_consensus_label_needs_two_voters(self):
        cols = ts.POLARITY_COLS
        votes = ["Négatif"] + [None] * (len(cols) - 1)
        df = pd.DataFrame([votes], columns=cols)
        assert ts.consensus_label_series(df, cols).iloc[0] == ""

    def test_polarity_ordinal_maps(self):
        s = ts.polarity_ordinal(pd.Series(list(ts.POLARITY_ORDER)))
        assert s.notna().all()

    def test_year_from_pub_date(self):
        s = ts.year_from_pub_date(pd.Series(["2015-03-01", "1998", "n/a", None]))
        assert s.iloc[0] == 2015 and s.iloc[1] == 1998
        assert pd.isna(s.iloc[2]) and pd.isna(s.iloc[3])


class TestEntityNetworkHelpers:
    def test_split_subjects_dedup(self):
        assert en.split_subjects("COSIM|COSIM|Ramadan") == {"COSIM", "Ramadan"}

    def test_split_subjects_missing(self):
        assert en.split_subjects("") == set()
        assert en.split_subjects(None) == set()
        assert en.split_subjects(float("nan")) == set()

    def test_parse_year(self):
        assert en.parse_year("2010-05") == 2010
        assert en.parse_year("2010") == 2010
        assert en.parse_year("") is None
        assert en.parse_year(None) is None
        assert en.parse_year("abcd") is None

    def test_pmi_positive_when_associated(self):
        # a and b each in 10 articles, co-occur in all 10, out of 100 → strong +.
        assert en.pmi(10, 10, 10, 100) > 0

    def test_pmi_zero_at_independence(self):
        # p(a,b) == p(a)p(b): a in 50, b in 20, co-occur in 10, of 100.
        assert abs(en.pmi(10, 50, 20, 100)) < 1e-9

    def test_pmi_degenerate(self):
        assert math.isnan(en.pmi(0, 10, 10, 100))

    def test_pmi_negative_when_avoiding(self):
        # co-occur less than chance
        assert en.pmi(1, 50, 20, 100) < 0
