"""Entity networks: fractional edge weights, tagged-article PMI, type filter, offline provenance."""
from __future__ import annotations

import json
import math
import subprocess
import sys

import pandas as pd
import pytest

from iwac_pipeline.analyses import entity_networks as en


def test_newman_fractional_weight_caps_each_article_contribution():
    # One densely catalogued article (5 entities) and one pair article.
    raw, fractional, nodes = en.cooccurrence([{"a", "b", "c", "d", "e"}, {"a", "b"}, {"a"}])
    assert raw[("a", "b")] == 2
    assert fractional[("a", "b")] == pytest.approx(1 / 4 + 1)
    assert fractional[("c", "d")] == pytest.approx(1 / 4)
    # Each article adds at most 1 to an entity's weighted degree.
    degree_c = sum(w for pair, w in fractional.items() if "c" in pair)
    assert degree_c == pytest.approx(1.0)
    assert nodes["a"] == 3


def test_pmi_is_computed_over_tagged_articles_only():
    sets = [{"a", "b"}, {"a", "b"}, {"c"}, {"c"}, set(), set(), set(), set()]
    nodes, edges, tagged = en.network_tables(sets, [None] * len(sets), {}, {}, min_node_count=1,
                                             min_edge_weight=1)
    assert tagged == 4
    # p(a,b)=2/4, p(a)=p(b)=2/4 -> log2(2); untagged rows would have added log2(8/4).
    assert edges.loc[0, "pmi"] == pytest.approx(round(math.log2(2), 3))
    assert edges.loc[0, "co_articles"] == 2 and edges.loc[0, "Weight"] == pytest.approx(2.0)


def test_edges_rank_by_fractional_weight_not_raw_count():
    # x-y co-occur in 3 heavily catalogued articles; p-q in 2 pair-only articles.
    crowd = [{"x", "y", f"n{i}", f"m{i}", f"o{i}"} for i in range(3)]
    sets = crowd + [{"p", "q"}, {"p", "q"}]
    _, edges, _ = en.network_tables(sets, [None] * len(sets), {}, {}, min_node_count=1, min_edge_weight=1)
    top = edges.iloc[0]
    assert (top.Source, top.Target) == ("p", "q")
    xy = edges[(edges.Source == "x") & (edges.Target == "y")].iloc[0]
    assert xy.co_articles == 3 and xy.Weight == pytest.approx(0.75)


def test_type_filter_and_hoisted_ambiguity():
    index = pd.DataFrame({"o:id": [1, 2, 3, 4], "Titre": ["Ali", "Ali", "COSIM", "Lomé"],
                          "Type": ["Personnes", "Personnes", "Organisations", "Lieux"]})
    titles, kinds, by_title = en.authority_lookups(index)
    ambiguous = en.ambiguous_titles(titles)
    assert ambiguous == {"Ali"}
    assert en.resolve_entities({"subject": "Ali|COSIM"}, titles, by_title, ambiguous) == ({"3"}, 2)
    assert en.entity_type_filter(kinds, ["organisations", "Lieux"]) == {"3", "4"}
    assert en.entity_type_filter(kinds, None) is None


def _write_inputs(tmp_path):
    articles = pd.DataFrame({"o:id": [str(i) for i in range(1, 7)],
                             "subject_ids": ["1|2|3", "1|2", "1|2", "3|4", "4", None],
                             "subject": [None] * 6, "country": ["Bénin"] * 6,
                             "pub_date": ["2001", "2002", "2003/2004", "2004", "bad", "2005"]})
    index = pd.DataFrame({"o:id": [1, 2, 3, 4], "Titre": ["COSIM", "Ramadan", "Lomé", "Ali"],
                          "Type": ["Organisations", "Événements", "Lieux", "Personnes"]})
    articles.to_parquet(tmp_path / "articles.parquet", index=False)
    index.to_parquet(tmp_path / "index.parquet", index=False)
    return tmp_path / "articles.parquet", tmp_path / "index.parquet"


def test_offline_run_records_provenance_and_requires_explicit_index(tmp_path):
    articles, index = _write_inputs(tmp_path)
    output = tmp_path / "out"
    base = [sys.executable, "-m", "iwac_pipeline.analyses.entity_networks", "--input", str(articles),
            "--output-dir", str(output), "--min-node-count", "1", "--min-edge-weight", "1"]
    refused = subprocess.run(base, capture_output=True, encoding="utf-8")
    assert refused.returncode != 0 and "--index-input" in refused.stderr
    done = subprocess.run([*base, "--index-input", str(index), "--revision", "rev-1",
                           "--types", "Organisations", "Événements"], capture_output=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    edges = pd.read_csv(output / "entity_edges.csv", dtype={"Source": str, "Target": str})
    assert edges[["Source", "Target"]].values.tolist() == [["1", "2"]]
    assert set(edges.source_revision) == {"rev-1"}
    summary = json.loads((output / "entity_networks_summary.json").read_text(encoding="utf-8"))
    assert summary["n_with_matched_authorities"] == 3
    assert "cataloguing density" in summary["degree_caveat"]
    manifest = json.loads((output / "entity_networks.manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["revision"] == "rev-1"
    assert len(manifest["inputs"]["index_input"]["sha256"]) == 64


def test_cohort_ids_file_filters_and_is_recorded(tmp_path):
    articles, index = _write_inputs(tmp_path)
    ledger = pd.DataFrame({"o:id": ["1", "2", "3", "4"], "stable_outlet_cohort": ["True", "True", "False", "True"]})
    ledger.to_csv(tmp_path / "ledger.csv", index=False)
    output = tmp_path / "out"
    done = subprocess.run([sys.executable, "-m", "iwac_pipeline.analyses.entity_networks",
                           "--input", str(articles), "--index-input", str(index), "--output-dir", str(output),
                           "--min-node-count", "1", "--min-edge-weight", "1",
                           "--ids-file", str(tmp_path / "ledger.csv"), "--ids-where", "stable_outlet_cohort"],
                          capture_output=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    summary = json.loads((output / "entity_networks_summary.json").read_text(encoding="utf-8"))
    assert summary["ids_filter"]["rows_kept"] == 3 and summary["ids_filter"]["rows_dropped"] == 3
    manifest = json.loads((output / "entity_networks.manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["ids_filter"]["ids_in_file"] == 3
