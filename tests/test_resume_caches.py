"""Resume caches are tied to the input they were computed from.

Before, an interrupted run whose source text changed before the re-run
restored the OLD text's lemmas/embeddings as if fresh, and a scratch run
could feed a production one: the caches were keyed by o:id alone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from datasets import Dataset

from _embedding_utils import (
    cached_value,
    input_fingerprint,
    load_cache,
    make_entry,
    repo_slug,
    save_cache,
)
from _gemini_client import restore_from_cache

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_fingerprint_changes_with_text_and_settings():
    assert input_fingerprint("abc", 28_000, 2_000) == input_fingerprint("abc", 28_000, 2_000)
    assert input_fingerprint("abc", 28_000, 2_000) != input_fingerprint("abd", 28_000, 2_000)
    assert input_fingerprint("abc", 28_000, 2_000) != input_fingerprint("abc", 20_000, 2_000)


def test_only_a_matching_entry_is_reused(tmp_path):
    cache = {
        "1": make_entry([0.1], input_fingerprint("same text")),
        "2": make_entry([0.2], input_fingerprint("old text")),
        "3": [0.3],  # legacy entry: cannot prove its input
    }
    path = tmp_path / "c.json.gz"
    save_cache(cache, path)
    cache = load_cache(path)
    assert cached_value(cache, 1, input_fingerprint("same text")) == [0.1]
    assert cached_value(cache, 2, input_fingerprint("new text")) is None
    assert cached_value(cache, 3, input_fingerprint("anything")) is None
    assert cached_value(cache, 4, input_fingerprint("x")) is None


def test_restore_skips_rows_whose_text_changed():
    cache = {
        "1": make_entry([1.0], input_fingerprint("kept")),
        "2": make_entry([2.0], input_fingerprint("edited before the re-run")),
    }
    embeddings = [[], []]
    restored = restore_from_cache(
        embeddings, ["1", "2"], cache,
        [input_fingerprint("kept"), input_fingerprint("corrected OCR")],
    )
    assert restored == 1
    assert embeddings == [[1.0], []]


def test_repo_slug_separates_repositories():
    assert repo_slug("owner/scratch") != repo_slug("owner/prod")
    assert "/" not in repo_slug("owner/prod")


class _Token:
    def __init__(self, text):
        self.lemma_ = text
        self.is_stop = text in {"le", "la"}
        self.is_alpha = text.isalpha()


class _FakeNlp:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return [_Token(t) for t in text.split()]

    def pipe(self, texts, as_tuples=False, batch_size=None, n_process=1):
        for item in texts:
            if as_tuples:
                text, context = item
                yield self(text), context
            else:
                yield self(item)


def _lemmatizer():
    spec = importlib.util.spec_from_file_location("_lemm_ut", REPO_ROOT / "lemmatize_update_hf.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_lemm_ut"] = module
    spec.loader.exec_module(module)
    return module


def test_lemma_cache_is_not_restored_for_changed_text(tmp_path):
    lem = _lemmatizer()
    cache_file = tmp_path / "lemmas.json.gz"
    save_cache({
        "1": make_entry(["stale lemma", "stale"], input_fingerprint("ancien texte")),
        "2": make_entry(["kept", "kept"], input_fingerprint("texte inchangé")),
    }, cache_file)
    ds = Dataset.from_dict({
        "o:id": ["1", "2"],
        "OCR": ["nouveau texte corrigé", "texte inchangé"],
    })
    nlp = _FakeNlp()
    out = lem.lemmatise_dataset(
        ds, nlp, text_col="OCR", lemma_col="lemma_text", clean_col="lemma_nostop",
        process_choice="all", cache_file=cache_file,
    )
    assert out["lemma_text"][:] == ["nouveau texte corrigé", "texte inchangé"]
    # Both legacy entries lack configuration provenance and must be recomputed.
    assert load_cache(cache_file)["1"]["h"] == input_fingerprint(
        input_fingerprint("nouveau texte corrigé"), out["lemma_text_config_hash"][0])


def test_batched_lemmatisation_matches_the_per_text_path(monkeypatch):
    """nlp.pipe batching must not change a lemma, drop an empty text, or
    misalign a text split into several chunks."""
    lem = _lemmatizer()
    monkeypatch.setattr(lem, "SPACY_MAX_CHUNK_CHARS", 12)
    texts = ["", "la mosquée centrale de Ouagadougou", "", "le prêche", ""]
    batched = list(lem.lemmatise_many(_FakeNlp(), iter(texts), batch_size=2))
    one_by_one = [lem.lemmatise_one(_FakeNlp(), t) for t in texts]
    assert batched == one_by_one
    assert len(batched) == len(texts)
