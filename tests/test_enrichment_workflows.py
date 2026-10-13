"""Offline contracts for enrichment migrations and fail-closed orchestration."""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset
from rich.console import Console

import semantic_embedding as text
import semantic_embedding_images as images
from _embedding_utils import load_cache
from _gemma_encoder import document_text, query_text, validate_vectors
from iwac_common.enrichment import compatible_rows, set_provenance
from iwac_common.text_utils import tokenize_words
from related_articles import candidate_pairs, parse_embeddings, topk_neighbors


@pytest.mark.parametrize("vectors", [np.ones((1, 768)), np.ones((2, 767)), np.ones((3, 768)),
    np.array([[float('nan')] * 768, [1.] * 768]), np.zeros((2, 768))])
def test_vector_validation_rejects_entire_malformed_batch(vectors):
    with pytest.raises(ValueError):
        validate_vectors(vectors, 2)


def test_document_and_query_prompts_follow_the_model_card():
    assert document_text('alpha', 'Title A') == 'title: Title A | text: alpha'
    assert document_text('alpha', '') == 'title: none | text: alpha'
    assert query_text('faith') == 'task: search result | query: faith'


class FakeEncoder:
    """Offline stand-in for GemmaEncoder; ``calls`` counts inference calls."""

    def __init__(self, fail_on=None, interrupt_on=None):
        self.calls, self.texts = 0, []
        self.fail_on, self.interrupt_on = fail_on, interrupt_on
        self.providers = ['FakeExecutionProvider']

    def token_lengths(self, texts):
        return [len(t) for t in texts]

    def _call(self):
        self.calls += 1
        if self.calls == self.interrupt_on:
            raise KeyboardInterrupt
        if self.calls == self.fail_on:
            raise ValueError('malformed output')

    def encode(self, texts):
        self._call()
        self.texts.extend(texts)
        return np.full((len(texts), 768), 1 / np.sqrt(768), np.float32)

    def encode_image(self, image):
        self._call()
        return np.full(768, 1 / np.sqrt(768), np.float32)


def configure_main(monkeypatch, module, dataset, tmp_path, argv, encoder=None):
    calls = []
    encoder = encoder or FakeEncoder()
    monkeypatch.setattr(module, 'console', Console(file=io.StringIO()))
    monkeypatch.setattr(module, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(module, 'ensure_hf_token', lambda **kwargs: 'offline-placeholder')
    monkeypatch.setattr(module, 'load_encoder', lambda args, token: encoder)
    monkeypatch.setattr(module, 'load_hub_dataset', lambda *args, **kwargs: dataset)
    monkeypatch.setattr(module, 'push_dataset', lambda ds, **kwargs: calls.append(ds) or True)
    monkeypatch.setattr('sys.argv', ['audit', *argv])
    if module is images:
        monkeypatch.setattr(module, 'download_image', lambda url: url.encode())
        monkeypatch.setattr(module, 'decode_image', lambda raw: raw)
    return calls, encoder


def test_image_main_never_pushes_a_failed_embedding(monkeypatch, tmp_path):
    ds = Dataset.from_dict({'o:id': [1, 2], 'image_url': ['one', 'two']})
    pushed, _ = configure_main(monkeypatch, images, ds, tmp_path, ['--update-mode', 'all'],
                               encoder=FakeEncoder(fail_on=3))  # probe, image 1, then image 2 fails
    assert images.main() == 1
    assert pushed == []
    assert len(load_cache(next(tmp_path.glob('*.sqlite3')))) == 1  # image 1 kept for the retry


@pytest.mark.parametrize('module,source,column,args', [
    (text, 'OCR', 'embedding_OCR', ['--config', 'articles']),
    (images, 'image_url', 'embedding_image', []),
])
def test_all_mode_clears_removed_source(monkeypatch, tmp_path, module, source, column, args):
    ds = Dataset.from_dict({'o:id': [1], source: [''], column: [[1.] * 768]})
    pushed, encoder = configure_main(monkeypatch, module, ds, tmp_path, [*args, '--update-mode', 'all'])
    assert module.main() == 0
    assert pushed[0][column][0] is None
    assert pushed[0][f'{column}_input_hash'][0] is None
    assert encoder.calls == 1  # only readiness probe, no inference on missing input


def test_text_legacy_migration_then_matching_provenance_skips_inference(monkeypatch, tmp_path):
    ds = Dataset.from_dict({'o:id': [1], 'OCR': ['corrected text'], 'embedding_OCR': [[2.] * 768]})
    pushed, encoder = configure_main(monkeypatch, text, ds, tmp_path, ['--config', 'articles', '--update-mode', 'missing'])
    assert text.main() == 0
    assert encoder.calls == 2
    migrated = pushed[-1]
    pushed, encoder = configure_main(monkeypatch, text, migrated, tmp_path, ['--config', 'articles', '--update-mode', 'missing'])
    assert text.main() == 0
    assert encoder.calls == 1
    assert pushed[0]['embedding_OCR_config_hash'][0]


def test_text_chunks_carry_the_document_instruction(monkeypatch, tmp_path):
    ds = Dataset.from_dict({'o:id': [1], 'OCR': ['alpha'], 'title': ['Titre']})
    _, encoder = configure_main(monkeypatch, text, ds, tmp_path, ['--config', 'articles', '--update-mode', 'all'])
    assert text.main() == 0
    assert encoder.texts[-1] == 'title: Titre | text: alpha'


def test_image_same_url_changed_bytes_refreshes(monkeypatch, tmp_path):
    ds = Dataset.from_dict({'o:id': [1], 'image_url': ['one']})
    pushed, _ = configure_main(monkeypatch, images, ds, tmp_path, [])
    assert images.main() == 0
    ds = pushed[-1]
    pushed, encoder = configure_main(monkeypatch, images, ds, tmp_path, [])
    assert images.main() == 0
    assert encoder.calls == 1
    pushed, encoder = configure_main(monkeypatch, images, ds, tmp_path, [])
    monkeypatch.setattr(images, 'download_image', lambda url: b'changed bytes')
    assert images.main() == 0
    assert encoder.calls == 2
    assert pushed[-1]['embedding_image_input_hash'][0] != ds['embedding_image_input_hash'][0]


def test_interrupted_run_resumes_from_its_last_finished_batch(monkeypatch, tmp_path):
    # Six equal-length one-chunk rows and a budget of two rows: three batches.
    ds = Dataset.from_dict({'o:id': list(range(6)), 'OCR': [f'row text {i}' for i in range(6)]})
    budget = 2 * len(document_text('row text 0'))  # FakeEncoder counts one token per character
    argv = ['--config', 'articles', '--update-mode', 'all', '--batch-tokens', str(budget)]
    pushed, encoder = configure_main(monkeypatch, text, ds, tmp_path, argv,
                                     encoder=FakeEncoder(interrupt_on=4))  # probe, batch 1, batch 2, interrupt
    assert text.main() == 130
    assert pushed == []
    assert len(load_cache(next(tmp_path.glob('*.sqlite3')))) == 4  # both finished batches were saved
    pushed, encoder = configure_main(monkeypatch, text, ds, tmp_path, argv)
    assert text.main() == 0
    assert encoder.calls == 2  # probe + the one unfinished batch; nothing is recomputed
    assert len(encoder.texts) == 3  # probe text + 2 chunks
    assert all(v is not None for v in pushed[-1]['embedding_OCR'])
    assert not list(tmp_path.glob('*.sqlite3'))  # cache removed after the verified push


def test_failed_batch_leaves_its_rows_empty_and_blocks_the_push(monkeypatch, tmp_path):
    ds = Dataset.from_dict({'o:id': [1, 2], 'OCR': ['a', 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb']})
    pushed, _ = configure_main(monkeypatch, text, ds, tmp_path,
                               ['--config', 'articles', '--update-mode', 'all', '--batch-tokens', '1'],
                               encoder=FakeEncoder(fail_on=3))
    assert text.main() == 1
    assert pushed == []
    assert len(load_cache(next(tmp_path.glob('*.sqlite3')))) == 1  # the successful batch survives


def test_provenance_rejects_same_dimension_different_configuration():
    ds = Dataset.from_dict({'o:id': [1, 2]})
    ds = set_provenance(ds, 'embedding_OCR', ['a', 'b'], 'old-model', [True, False])
    assert compatible_rows(ds, 'embedding_OCR', ['a', 'b'], 'old-model') == [True, False]
    assert compatible_rows(ds, 'embedding_OCR', ['a', 'b'], 'new-model') == [False, False]


def test_multilingual_byte_chunks_are_bounded_and_unicode_is_not_split():
    original = 'Éwé العربية جَامِعٌ ' * 1000
    chunks = text.chunk_text(original)
    assert len(chunks) > 1
    assert all(len(chunk.encode()) <= text.CHUNK_SIZE for chunk in chunks)
    assert original.startswith(chunks[0]) and original.endswith(chunks[-1])
    with pytest.raises(ValueError):
        text.chunk_text('😀', chunk_size=1, overlap=0)


def test_unicode_words_preserve_combining_marks_and_normalize_equivalent_accents():
    assert tokenize_words('mosque\u0301e') == tokenize_words('mosquée')
    assert tokenize_words('جَامِعٌ', language='Arabe') == ['جَامِعٌ']
    assert tokenize_words("d'amour", language='Anglais') == ['d', 'amour']
    assert tokenize_words("d'amour", language='Français') == ['amour']


@pytest.mark.parametrize('values', [[[1., float('nan')]], [[1., 2.], [1.]], [[[1., 2.]]]])
def test_neighbors_reject_invalid_vectors(values):
    with pytest.raises(ValueError):
        parse_embeddings(pd.Series(values))


def test_neighbors_ties_are_stable_and_candidate_pairs_cover_all_topk():
    indices, scores = topk_neighbors(np.array([[1., 0.]] * 4, dtype=np.float32), 2)
    assert indices.tolist() == [[1, 2], [0, 2], [0, 1], [0, 1]]
    pairs = candidate_pairs(np.array(['a', 'b', 'c', 'd']), indices, scores)
    assert ('a', 'c') in set(zip(pairs.id_a, pairs.id_b))
    assert len(pairs) == 5


def test_readability_failure_remains_retryable(monkeypatch):
    import calculate_lexical_richness as lex
    batch = {'OCR': ['Une phrase française.'], 'language': ['Français']}
    def run():
        return lex.compute_metrics_batch(batch, text_col='OCR', richness_col='Richesse_Lexicale_OCR',
            readability_col='Lisibilite_OCR', update_mode='missing', window_size=50,
            error_counter={'richness_too_short': 0, 'readability_failed': 0})
    monkeypatch.setattr(lex, 'calculate_readability', lambda value: None)
    failed = run()
    assert failed['Lisibilite_OCR_config_hash'] == [None]
    monkeypatch.setattr(lex, 'calculate_readability', lambda value: 42.)
    recovered = run()
    assert recovered['Lisibilite_OCR'] == [42.]
    assert recovered['Lisibilite_OCR_config_hash'][0]


def test_lemma_routing_requires_metadata_and_clears_ambiguous_old_outputs(tmp_path):
    import lemmatize_update_hf as lemma
    class Nlp:
        meta = {'lang': 'fr', 'version': 'test'}
        def pipe(self, *args, **kwargs):
            raise AssertionError('No eligible rows should reach the model')
    kwargs = dict(text_col='OCR', lemma_col='lemma_text', clean_col='lemma_nostop',
                  process_choice='empty', cache_file=tmp_path / 'cache.json.gz', language_filter='Français')
    with pytest.raises(ValueError, match='no language column'):
        lemma.lemmatise_dataset(Dataset.from_dict({'o:id': [1], 'OCR': ['text']}), Nlp(), **kwargs)
    ds = Dataset.from_dict({'o:id': [1, 2], 'OCR': ['text', ''],
                           'language': ['Français|Anglais', 'Français'],
                           'lemma_text': ['stale', 'stale'], 'lemma_nostop': ['stale', 'stale'],
                           'lda_topic_id': [1, 2]})
    out = lemma.lemmatise_dataset(ds, Nlp(), **kwargs)
    assert list(out['lemma_text']) == ['', '']
    assert list(out['lda_topic_id']) == [None, None]


def test_reembedding_invalidates_all_neighbor_rows(monkeypatch, tmp_path):
    ds = Dataset.from_dict({'o:id': [1, 2], 'OCR': ['new text', 'other text'],
                           'embedding_OCR': [[2.] * 768, [3.] * 768],
                           'related_articles': ['2:0.9', '1:0.9']})
    pushed, _ = configure_main(monkeypatch, text, ds, tmp_path,
                               ['--config', 'articles', '--update-mode', 'missing'])
    assert text.main() == 0
    assert list(pushed[-1]['related_articles']) == [None, None]


def test_readable_configuration_matches_persisted_hash(monkeypatch, tmp_path):
    import json
    from iwac_common.enrichment import config_fingerprint
    ds = Dataset.from_dict({'o:id': [1], 'OCR': ['source text']})
    pushed, _ = configure_main(monkeypatch, text, ds, tmp_path, ['--config', 'articles', '--update-mode', 'all'])
    assert text.main() == 0
    row = pushed[-1][0]
    settings = json.loads(row['embedding_OCR_config_json'])
    assert settings['model'] == 'google/embeddinggemma-2'
    assert len(settings['model_revision']) == 40
    assert settings['onnx_export'].startswith('onnx-community/embeddinggemma-2-ONNX@')
    assert config_fingerprint(settings) == row['embedding_OCR_config_hash']


def test_lemma_resume_rejects_changed_processing_configuration(tmp_path):
    import lemmatize_update_hf as lemma
    class Nlp:
        def __init__(self, name):
            self.meta = {'lang': 'fr', 'name': name, 'version': '1'}
            self.calls = 0
        def pipe(self, texts, **kwargs):
            for text_value, idx in texts:
                self.calls += 1
                yield [SimpleNamespace(lemma_=self.meta['name'], is_alpha=True, is_stop=False)], idx
    ds = Dataset.from_dict({'o:id': [1], 'OCR': ['source'], 'language': ['Français']})
    kwargs = dict(text_col='OCR', lemma_col='lemma_text', clean_col='lemma_nostop',
                  process_choice='empty', cache_file=tmp_path / 'cache.json.gz', language_filter='Français')
    old, new = Nlp('old'), Nlp('new')
    ds = lemma.lemmatise_dataset(ds, old, **kwargs)
    out = lemma.lemmatise_dataset(ds, new, **kwargs)
    assert list(out['lemma_text']) == ['new']
    assert new.calls == 1
    assert out['lemma_text_config_hash'][0] != ds['lemma_text_config_hash'][0]


def test_text_embedding_configs_match_schema():
    """Every subset with a text embedding column can be embedded, and into that column."""
    from iwac_common.schema import SUBSETS

    expected = {cfg: column for cfg, spec in SUBSETS.items()
                for column in (spec.embedding_columns or {}) if column != "embedding_image"}
    assert {cfg: emb for cfg, (_, emb, _) in text.CONFIG_SETTINGS.items()} == expected
    stems = [stem for _, _, stem in text.CONFIG_SETTINGS.values()]
    assert len(stems) == len(set(stems))


def test_related_item_configs_read_schema_embeddings_and_are_publishable():
    """Each related-items subset ranks a schema embedding column, invalidates its
    rankings when that embedding's sources change, and may publish the result."""
    import related_articles
    from iwac_common.repos import load_public_columns
    from iwac_common.schema import DERIVED_FROM, SUBSETS

    assert "documents" in related_articles.CONFIG_SETTINGS
    allow = load_public_columns()
    for cfg, (embedding, _) in related_articles.CONFIG_SETTINGS.items():
        assert embedding in (SUBSETS[cfg].embedding_columns or {})
        for source, derived in DERIVED_FROM[cfg].items():
            if embedding in derived:
                assert "related_articles" in derived, (cfg, source)
        if cfg != "publications":  # its rankings are not yet reviewed for publication
            assert "related_articles" in allow[cfg]


def test_references_embed_abstracts_and_retire_full_text_vectors():
    """A superseded vector and its provenance leave with the run that replaces it."""
    assert text.CONFIG_SETTINGS["references"][:2] == ("abstract", "embedding_abstract")
    ds = Dataset.from_dict({
        "o:id": ["1"], "embedding_OCR": [[0.1]], "embedding_OCR_input_hash": ["h"],
        "embedding_OCR_config_hash": ["c"], "embedding_OCR_config_json": ["{}"],
        "embedding_abstract": [[0.2]],
    })
    kept, dropped = text.drop_retired_columns(ds, "references")
    assert kept.column_names == ["o:id", "embedding_abstract"]
    assert set(dropped) == {"embedding_OCR", "embedding_OCR_input_hash",
                            "embedding_OCR_config_hash", "embedding_OCR_config_json"}
    assert text.drop_retired_columns(ds, "articles")[0].column_names == ds.column_names
