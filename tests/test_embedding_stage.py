"""Embedding stages: float32 vectors in memory, no-op runs skip the push,
shared update-mode flags."""
from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
import pytest
from datasets import Dataset
from rich.console import Console

import semantic_embedding as text
import semantic_embedding_images as images
from _embedding_columns import build_embedding_array, column_vectors


class FakeEncoder:
    providers = ["FakeExecutionProvider"]

    def __init__(self):
        self.calls = 0

    def token_lengths(self, texts):
        return [len(t) for t in texts]

    def encode(self, texts):
        self.calls += 1
        return np.full((len(texts), 768), 1 / np.sqrt(768), np.float32)

    def encode_image(self, image):
        self.calls += 1
        return np.full(768, 1 / np.sqrt(768), np.float32)


def run(monkeypatch, module, ds, tmp_path, argv):
    pushed, encoder = [], FakeEncoder()
    monkeypatch.setattr(module, "console", Console(file=io.StringIO()))
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(module, "ensure_hf_token", lambda **kwargs: "offline-placeholder")
    monkeypatch.setattr(module, "load_encoder", lambda args, token: encoder)
    monkeypatch.setattr(module, "load_hub_dataset", lambda *args, **kwargs: ds)
    monkeypatch.setattr(module, "push_dataset", lambda d, **kwargs: pushed.append(d) or True)
    monkeypatch.setattr("sys.argv", ["stage", *argv])
    if module is images:
        monkeypatch.setattr(module, "download_image", lambda url: url.encode())
        monkeypatch.setattr(module, "decode_image", lambda raw: raw)
    return module.main(), pushed, encoder


def test_build_embedding_array_matches_the_list_conversion():
    rows = [np.array([0.5, 0.25], np.float32), None, [], [0.0, 0.0], [1e-8, 3.0], np.zeros(0)]
    expected = pa.array([[0.5, 0.25], None, None, None, [1e-8, 3.0], None], type=pa.list_(pa.float32()))
    built = build_embedding_array(rows)
    assert built.type == pa.list_(pa.float32())
    assert built.equals(expected)
    assert build_embedding_array([None, []]).equals(pa.array([None, None], type=pa.list_(pa.float32())))


def test_column_vectors_reads_float32_views_and_nulls():
    ds = Dataset.from_dict({"o:id": ["1", "2", "3"],
                            "embedding_OCR": pa.array([[0.5, 0.25], None, [1.0, 2.0]],
                                                      type=pa.list_(pa.float32()))})
    ds = ds.select([2, 0, 1])  # an indices mapping must be honoured
    vectors = column_vectors(ds, "embedding_OCR")
    assert [None if v is None else v.tolist() for v in vectors] == [[1.0, 2.0], [0.5, 0.25], None]
    assert vectors[0].dtype == np.float32
    assert column_vectors(ds, "absent") == [None, None, None]


def test_unchanged_run_skips_the_push(monkeypatch, tmp_path):
    ds = Dataset.from_dict({"o:id": ["1"], "OCR": ["source text"]})
    code, pushed, _ = run(monkeypatch, text, ds, tmp_path, ["--config", "articles", "--update-mode", "all"])
    assert code == 0 and len(pushed) == 1
    code, again, encoder = run(monkeypatch, text, pushed[0], tmp_path,
                               ["--config", "articles", "--update-mode", "missing"])
    assert code == 0
    assert again == []  # nothing to commit
    assert encoder.calls == 1  # readiness probe only


def test_retired_column_alone_still_pushes(monkeypatch, tmp_path):
    ds = Dataset.from_dict({"o:id": ["1"], "abstract": ["résumé"]})
    code, pushed, _ = run(monkeypatch, text, ds, tmp_path, ["--config", "references", "--update-mode", "all"])
    stale = pushed[0].add_column("embedding_OCR", [[0.1] * 768])
    code, again, _ = run(monkeypatch, text, stale, tmp_path, ["--config", "references", "-y"])
    assert code == 0 and len(again) == 1
    assert "embedding_OCR" not in again[0].column_names


def test_changed_columns_detects_values_and_membership():
    before = Dataset.from_dict({"o:id": ["1"], "a": ["x"], "gone": [1]})
    after = Dataset.from_dict({"o:id": ["1"], "a": ["y"]})
    assert text.changed_columns(before, after, ["a"]) == ["a", "gone"]
    assert text.changed_columns(before, before, ["a"]) == []


def test_text_stage_accepts_shared_yes_flag(monkeypatch, tmp_path):
    ds = Dataset.from_dict({"o:id": ["1"], "OCR": ["texte"]})
    code, pushed, _ = run(monkeypatch, text, ds, tmp_path, ["--config", "articles", "-y"])
    assert code == 0 and pushed[0]["embedding_OCR"][0] is not None


@pytest.mark.parametrize("argv", [[], ["-y"]])
def test_image_stage_defaults_to_missing_when_headless(monkeypatch, tmp_path, argv):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    ds = Dataset.from_dict({"o:id": ["1"], "image_url": ["one"]})
    code, pushed, _ = run(monkeypatch, images, ds, tmp_path, argv)
    assert code == 0
    assert len(pushed[0]["embedding_image"][0]) == 768
    assert pushed[0].column_names.index("embedding_image") == pushed[0].column_names.index("image_url") + 1
