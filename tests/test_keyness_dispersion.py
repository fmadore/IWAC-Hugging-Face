"""Keyness must not be led by near-hapaxes: dispersion gates and a conservative ranking."""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from iwac_pipeline.analyses import keyness_bursts as kb


def test_reference_arithmetic_for_a_near_hapax_and_a_substantive_word():
    # The review's example: 10 vs 0 in a 1M-token slice against 5M tokens elsewhere...
    assert kb.dunning_g2(10, 0, 10**6, 5 * 10**6) == pytest.approx(35.8353, abs=1e-3)
    assert kb.log_ratio(10, 0, 10**6, 5 * 10**6) == pytest.approx(math.log2(105))
    low, high = kb.log_ratio_interval(10, 0, 10**6, 5 * 10**6)
    se = math.sqrt(1 / 10.5 + 1 / 0.5) / math.log(2)
    assert low == pytest.approx(math.log2(105) - 1.959964 * se, abs=1e-5)
    assert high == pytest.approx(math.log2(105) + 1.959964 * se, abs=1e-5)
    # ...against a substantive word at 5,000 vs 10,000.
    assert kb.log_ratio(5000, 10000, 10**6, 5 * 10**6) == pytest.approx(1.3220, abs=1e-4)
    sub_low, sub_high = kb.log_ratio_interval(5000, 10000, 10**6, 5 * 10**6)
    assert sub_low == pytest.approx(1.2730, abs=1e-4) and sub_high == pytest.approx(1.3710, abs=1e-4)
    # The interval alone does not demote a token absent elsewhere (2.62 > 1.27):
    # the dispersion gates are what keep single-article tokens out.
    assert low > sub_low


def _slices(rare_docs, rare_outlets):
    tokens = {"Bénin": Counter({"rare": 12, "frequent": 200, "filler": 5000}),
              "Togo": Counter({"frequent": 50, "filler": 5000})}
    docs = {"Bénin": Counter({"rare": rare_docs, "frequent": 80, "filler": 100}),
            "Togo": Counter({"frequent": 30, "filler": 100})}
    outlets = {"Bénin": Counter({"rare": rare_outlets, "frequent": 3, "filler": 3}),
               "Togo": Counter({"frequent": 2, "filler": 2})}
    return tokens, docs, outlets


def test_single_article_token_never_enters_the_tested_family():
    tokens, docs, outlets = _slices(rare_docs=1, rare_outlets=1)
    frame = kb.keyness_for_slices(tokens, 25, 10, doc_freq=docs, outlet_range=outlets,
                                  min_docs=5, min_outlets=2)
    assert "rare" not in frame["token"].tolist()
    assert frame.loc[frame.token.eq("frequent"), "n_docs"].item() == 80
    # Without dispersion the same near-hapax is significant and reported.
    assert "rare" in kb.keyness_for_slices(tokens, 25, 10)["token"].tolist()


def test_outlet_minimum_applies_only_where_outlets_are_known():
    tokens, docs, outlets = _slices(rare_docs=6, rare_outlets=1)
    with_outlets = kb.keyness_for_slices(tokens, 25, 10, doc_freq=docs, outlet_range=outlets,
                                         min_docs=5, min_outlets=2)
    assert "rare" not in with_outlets["token"].tolist()
    without = kb.keyness_for_slices(tokens, 25, 10, doc_freq=docs, min_docs=5, min_outlets=2)
    assert "rare" in without["token"].tolist()
    assert without["n_outlets"].isna().all()


def test_ranking_uses_the_lower_confidence_bound_and_exports_the_interval():
    tokens, docs, outlets = _slices(rare_docs=6, rare_outlets=2)
    frame = kb.keyness_for_slices(tokens, 25, 10, doc_freq=docs, outlet_range=outlets,
                                  min_docs=5, min_outlets=2)
    benin = frame[frame.slice.eq("Bénin")].set_index("token")
    # rare has the larger point log-ratio but the wider interval: frequent ranks first.
    assert benin.loc["rare", "log_ratio"] > benin.loc["frequent", "log_ratio"]
    assert benin.loc["frequent", "rank"] < benin.loc["rare", "rank"]
    for token, row in benin.iterrows():
        assert row.log_ratio_ci_low <= row.log_ratio <= row.log_ratio_ci_high
    assert {"g2", "q_value", "n_docs", "n_outlets", "other_count"}.issubset(frame.columns)


def test_keyness_rows_stay_french_and_report_exclusions():
    df = pd.DataFrame({"pub_date": ["2000", "2000", "2000/2001", "2001", "2001"],
                       "language": ["Français", "Ewé", "Français", "Français|Arabe", None],
                       "lemma_nostop": ["mosquée"] * 5})
    rows = kb.keyness_rows(df, 1990, 2010)
    assert rows.index.tolist() == [0]
    counts = rows.attrs["eligibility"]
    assert counts["excluded_no_point_year_in_window"] == 1
    assert counts["excluded_language"] == 3
    assert counts["excluded_language_labels"] == {"Ewé": 1, "Français|Arabe": 1, "[missing]": 1}


def test_bursts_can_be_stratified_by_country():
    years = list(range(2000, 2010))
    rows = []
    for year in years:
        # Togo is covered heavily from 2005 on; its subject spikes only relative to Bénin's exposure.
        rows += [{"year": year, "country": "Bénin", "subject_ids": "1" if year in (2004, 2005) else None}
                 for _ in range(20)]
        rows += [{"year": year, "country": "Togo", "subject_ids": "1"} for _ in range(2 if year < 2005 else 40)]
    index = pd.DataFrame({"o:id": ["1"], "Titre": ["Tabaski"]})
    df = pd.DataFrame(rows)
    pooled, pooled_cov, _ = kb.detect_bursts(df, index, min_subject_total=1)
    split, split_cov, counts = kb.detect_bursts(df, index, stratify="country", min_subject_total=1)
    assert set(pooled.stratum) == {"all"}
    assert counts["strata"] == ["Bénin", "Togo"]
    assert set(split_cov.stratum) == {"Bénin", "Togo"}
    benin = split[split.stratum.eq("Bénin")]
    assert (benin.start <= 2004).any() and (benin.end >= 2005).any()
    assert list(split.columns) == kb.BURST_COLUMNS


def test_keyness_run_exports_dispersion_and_manifest(monkeypatch, tmp_path):
    frame = pd.DataFrame({"o:id": [str(i) for i in range(12)],
                          "lemma_nostop": ["mosquée école"] * 6 + ["marché école"] * 6,
                          "pub_date": ["2000"] * 6 + ["2010"] * 6,
                          "country": ["Bénin"] * 6 + ["Togo"] * 6,
                          "newspaper_ids": ["1", "2"] * 6,
                          "language": ["Français"] * 11 + ["Ewé"], "subject_ids": [None] * 12})
    index = pd.DataFrame({"o:id": ["10"], "Titre": ["Islam"]})
    calls, loads = {}, []

    def loader(repo, config, **kwargs):
        # Like the real loader: the pinned revision travels in attrs.
        loads.append((config, kwargs.get("revision")))
        result = index.copy() if config == "index" else frame.copy()
        result.attrs["iwac_source_revision"] = kwargs.get("revision")
        return result

    monkeypatch.setattr(kb, "load_subset_dataframe", loader)
    monkeypatch.setattr(kb, "write_run_manifest", lambda *a, **k: calls.update(k))
    monkeypatch.setattr("sys.argv", ["keyness_bursts.py", "--source", "csv", "--output-dir", str(tmp_path),
                                     "--min-count", "1", "--min-docs", "2", "--revision", "abc"])
    kb.main()
    country = pd.read_csv(tmp_path / "keyness_country.csv")
    assert set(country.token) == {"mosquée", "marché"}
    assert {"n_docs", "n_outlets", "log_ratio_ci_low"}.issubset(country.columns)
    assert set(country.source_revision) == {"abc"}
    summary = json.loads((tmp_path / "keyness_bursts_summary.json").read_text(encoding="utf-8"))
    assert summary["keyness"]["eligibility"]["excluded_language_labels"] == {"Ewé": 1}
    assert calls["revision"] == "abc"
    # The authority file is read at the analysed subset's revision.
    assert loads == [("articles", "abc"), ("index", "abc")]
    assert np.isclose(summary["keyness"]["dispersion"]["min_docs"], 2)
