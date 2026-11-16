"""related_articles.py: only public items are neighbours, provenance is recorded."""
from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from rich.console import Console

import related_articles as ra
from iwac_common.write_policy import PublicationPolicyError

CONFIG = "c" * 64


def frame():
    """Item 3 is private and the closest vector to item 1; item 2 is public but
    its full text is private (it must stay in the pool)."""
    vectors = {"1": [1.0, 0.0], "2": [0.6, 0.8], "3": [0.99, 0.01], "4": [0.0, 1.0], "5": None}
    return pd.DataFrame({
        "o:id": list(vectors),
        "item_is_public": pd.array([True, True, False, True, None], dtype="boolean"),
        "OCR_is_public": [True, False, True, True, True],
        "embedding_OCR": list(vectors.values()),
        "embedding_OCR_config_hash": [CONFIG, CONFIG, CONFIG, CONFIG, None],
        "title": ["Un", "Deux", "Trois", "Quatre", "Cinq"],
        "newspaper": ["A"] * 5, "pub_date": ["2000"] * 5,
    })


def run(monkeypatch, tmp_path, df, argv):
    pushed, manifests = [], []
    monkeypatch.setattr(ra, "console", Console(file=io.StringIO()))
    monkeypatch.setattr(ra, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(ra, "ensure_hf_token", lambda **k: "offline-placeholder")
    monkeypatch.setattr(ra, "load_subset_dataframe", lambda *a, **k: df.copy())
    monkeypatch.setattr(ra, "write_run_manifest", lambda *a, **k: manifests.append(k))
    monkeypatch.setattr(ra, "load_hub_dataset", lambda *a, **k: "dataset")
    monkeypatch.setattr(ra, "add_columns_by_id", lambda ds, values, **k: (values, k))
    monkeypatch.setattr(ra, "push_dataset", lambda ds, **k: pushed.append((ds, k)) or True)
    monkeypatch.setattr("sys.argv", ["related_articles.py", *argv])
    return ra.main(), pushed, manifests


def test_public_mask_fails_closed():
    df = frame()
    assert ra.public_mask(df).tolist() == [True, True, False, True, False]  # null is private
    with pytest.raises(PublicationPolicyError, match="item_is_public"):
        ra.public_mask(df.drop(columns="item_is_public"))
    with pytest.raises(PublicationPolicyError, match="boolean"):
        ra.public_mask(df.assign(item_is_public=["yes"] * 5))


def test_neighbours_come_from_the_public_pool_only(monkeypatch, tmp_path):
    code, _, manifests = run(monkeypatch, tmp_path, frame(), ["--source", "local", "--topk", "2"])
    assert code == 0
    out = pd.read_parquet(tmp_path / "related_articles_articles.parquet").set_index("o:id")["related_articles"]
    # k=2 public neighbours, even though private item 3 is item 1's nearest vector.
    assert out["1"] == "2:0.6000|4:0.0000"
    # The public item whose OCR is private is a neighbour of the others.
    assert out["4"].startswith("2:")
    assert pd.isna(out["3"]) and pd.isna(out["5"])  # private rows get no list
    assert "3" not in "".join(v for v in out.dropna())
    pairs = pd.read_parquet(tmp_path / "related_articles_articles_candidates.parquet")
    assert set(pairs.id_a) | set(pairs.id_b) == {"1", "2", "4"}
    inputs = manifests[0]["inputs"]
    assert inputs["embedding_config_hash"] == CONFIG and inputs["topk"] == 2
    assert inputs["n_private_rows_excluded"] == 2


def test_missing_visibility_column_aborts(monkeypatch, tmp_path):
    code, _, _ = run(monkeypatch, tmp_path, frame().drop(columns="item_is_public"), ["--source", "local"])
    assert code == 1
    assert not (tmp_path / "related_articles_articles.parquet").exists()


def test_csv_remains_an_alias_of_local(monkeypatch, tmp_path):
    seen = {}
    code, _, _ = run(monkeypatch, tmp_path, frame(), ["--source", "csv"])
    monkeypatch.setattr(ra, "load_subset_dataframe", lambda *a, **k: seen.update(k) or frame())
    monkeypatch.setattr("sys.argv", ["related_articles.py", "--source", "csv"])
    assert ra.main() == 0 and code == 0
    assert seen["source"] == "local"
    assert "item_is_public" in seen["columns"]


def test_push_records_vector_space_and_topk(monkeypatch, tmp_path):
    code, pushed, _ = run(monkeypatch, tmp_path, frame(), ["--topk", "2", "--push"])
    assert code == 0
    (values, add_kwargs), push_kwargs = pushed[0]
    assert str(add_kwargs["types"]["related_articles"]) == "string"
    message = push_kwargs["commit_message"]
    assert "top-2" in message and CONFIG in message
    assert pd.isna(values.set_index("o:id").loc["3", "related_articles"])


def test_push_requires_hub_source(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run(monkeypatch, tmp_path, frame(), ["--source", "local", "--push"])


def test_output_is_accepted_by_the_public_projection(monkeypatch, tmp_path):
    from iwac_common.write_policy import prepare_public_projection

    run(monkeypatch, tmp_path, frame(), ["--source", "local", "--topk", "3"])
    out = pd.read_parquet(tmp_path / "related_articles_articles.parquet")
    projected = frame().drop(columns="embedding_OCR").merge(out, on="o:id")
    projected["private_fields"] = [["OCR"] if not p else [] for p in projected["OCR_is_public"]]
    result, _ = prepare_public_projection(projected, "articles")
    assert result.attrs["neighbour_entries_removed"] == 0
    assert json.dumps(result["related_articles"].tolist())
