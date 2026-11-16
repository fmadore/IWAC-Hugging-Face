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
POLARITY_COLS = [f"{m}_polarite" for m in ts.models_for(None)]


class TestTopicSentimentHelpers:
    def test_slug_ascii(self):
        assert ts.slug("Très négatif") == "tres_negatif"
        assert ts.slug("Plutôt Objectif") == "plutot_objectif"

    # Column names come from the live panel (iwac_common.sentiment_panel), so
    # these stay correct when the panel is rotated or resized. Votes are built
    # from the panel size rather than hardcoded, because the majority threshold
    # is now "> half the voters" rather than a fixed 2.
    def test_consensus_label_majority(self):
        cols = POLARITY_COLS
        n_majority = len(cols) // 2 + 1
        votes = ["Négatif"] * n_majority + ["Positif"] * (len(cols) - n_majority)
        df = pd.DataFrame([votes], columns=cols)
        assert ts.consensus_label_series(df, cols).iloc[0] == "Négatif"

    def test_consensus_label_no_majority(self):
        cols = POLARITY_COLS
        # Spread votes across distinct labels so no label clears half.
        labels = list(ts.POLARITY_ORDER)
        votes = [labels[i % len(labels)] for i in range(len(cols))]
        df = pd.DataFrame([votes], columns=cols)
        assert ts.consensus_label_series(df, cols).iloc[0] == ""

    def test_consensus_label_needs_two_voters(self):
        cols = POLARITY_COLS
        votes = ["Négatif"] + [None] * (len(cols) - 1)
        df = pd.DataFrame([votes], columns=cols)
        assert ts.consensus_label_series(df, cols).iloc[0] == ""

    def test_polarity_ordinal_maps(self):
        s = ts.polarity_ordinal(pd.Series(list(ts.POLARITY_ORDER)))
        assert s.notna().all()

    def test_point_years_exclude_ranges_and_garbage(self):
        s, precision = ts.point_years(pd.Series(["2015-03-01", "1998", "n/a", None, "1998/1999"]))
        assert s.iloc[0] == 2015 and s.iloc[1] == 1998
        assert s.iloc[2:].isna().all()
        assert precision.tolist() == ["day", "year", "other", "[missing]", "range"]


class TestEntityNetworkHelpers:
    def test_split_subjects_dedup(self):
        assert en.split_subjects("COSIM|COSIM|Ramadan") == {"COSIM", "Ramadan"}

    def test_split_subjects_missing(self):
        assert en.split_subjects("") == set()
        assert en.split_subjects(None) == set()
        assert en.split_subjects(float("nan")) == set()

    def test_year_filter_uses_point_dates_only(self):
        df = pd.DataFrame({"pub_date": ["2010-05", "2010", "", None, "abcd", "2010/2011"],
                           "country": ["Bénin"] * 6})
        args = type("Args", (), {"country": None, "year_from": 2010, "year_to": 2010})()
        kept, years, counts = en.filter_articles(df, args)
        assert kept.index.tolist() == [0, 1]
        assert counts["dates"]["by_precision"] == {"[missing]": 2, "other": 1, "range": 1}
        args.year_from = args.year_to = None
        assert len(en.filter_articles(df, args)[0]) == 6

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


class TestEntityResolution:
    ID_TO_TITLE = {"10": "Moussa Traoré", "11": "Moussa Traoré", "12": "Ramadan"}
    TITLE_TO_ID = {"Moussa Traoré": "10", "Ramadan": "12"}

    def test_ids_keep_homonyms_apart(self):
        entities, considered = en.resolve_entities(
            {"subject": "Moussa Traoré", "subject_ids": "11"},
            self.ID_TO_TITLE, self.TITLE_TO_ID,
        )
        assert entities == {"11"} and considered == 1

    def test_labels_are_the_fallback(self):
        entities, considered = en.resolve_entities(
            {"subject": "Ramadan|Inconnu"}, self.ID_TO_TITLE, self.TITLE_TO_ID,
        )
        assert entities == {"12"} and considered == 2
