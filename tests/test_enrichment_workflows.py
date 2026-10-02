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
from _gemini_client import validate_response
from iwac_common.enrichment import compatible_rows, set_provenance
from iwac_common.text_utils import tokenize_words
from related_articles import candidate_pairs, parse_embeddings, topk_neighbors


def response(vectors):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])


@pytest.mark.parametrize("vectors", [[], [[1.] * 768], [None, [1.] * 768],
    [[1.] * 767, [1.] * 768], [[float('nan')] * 768, [1.] * 768], [[0.] * 768] * 2,
    [[1.] * 768] * 3])
def test_response_validation_rejects_entire_malformed_batch(vectors):
    with pytest.raises(ValueError):
        validate_response(response(vectors), 2, 768)


def test_text_payload_uses_separate_content_and_instruction(monkeypatch):
    captured = []
    def embed(**kwargs):
        captured.append(kwargs)
        return response([[1.] * 768] * len(kwargs['contents']))
    client = SimpleNamespace(models=SimpleNamespace(embed_content=embed))
    output = text.embed_texts_with_retry(client, ['alpha', 'beta'], 'RETRIEVAL_DOCUMENT', 768,
                                         titles=['Title A', 'Title B'])
    assert len(output) == 2
    kwargs = captured[0]
    assert kwargs['config'].task_type is None
    assert [item.parts[0].text for item in kwargs['contents']] == [
        'title: Title A | text: alpha', 'title: Title B | text: beta']
    assert text.format_embedding_text('faith', 'RETRIEVAL_QUERY') == 'task: search result | query: faith'


def configure_main(monkeypatch, module, dataset, tmp_path, argv, *, short_images=False):
    calls = []
    api_calls = []
    def embed(**kwargs):
        api_calls.append(kwargs)
        n = len(kwargs['contents'])
        if short_images and api_calls and len(api_calls) > 1:
            n = 1
        return response([[1.] * 768] * n)
    monkeypatch.setenv('GOOGLE_API_KEY', 'offline-placeholder')
    monkeypatch.setattr(module, 'console', Console(file=io.StringIO()))
    monkeypatch.setattr(module, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(module, 'ensure_hf_token', lambda **kwargs: 'offline-placeholder')
    monkeypatch.setattr(module.genai, 'Client', lambda **kwargs: SimpleNamespace(models=SimpleNamespace(embed_content=embed)))
    monkeypatch.setattr(module, 'load_hub_dataset', lambda *args, **kwargs: dataset)
    monkeypatch.setattr(module, 'push_dataset', lambda ds, **kwargs: calls.append(ds) or True)
    monkeypatch.setattr(module, 'call_with_retry', lambda callback: callback())
    monkeypatch.setattr('sys.argv', ['audit', '--delay', '0', *argv])
    if module is images:
        monkeypatch.setattr(module, 'download_image_bytes', lambda url, side: url.encode())
    return calls, api_calls


def test_image_main_never_pushes_short_response(monkeypatch, tmp_path):
    ds = Dataset.from_dict({'o:id': [1, 2], 'image_url': ['one', 'two']})
    pushed, _ = configure_main(monkeypatch, images, ds, tmp_path, ['--update-mode', 'all'], short_images=True)
    assert images.main() == 1
    assert pushed == []


@pytest.mark.parametrize('module,source,column,args', [
    (text, 'OCR', 'embedding_OCR', ['--config', 'articles']),
    (images, 'image_url', 'embedding_image', []),
])
def test_all_mode_clears_removed_source(monkeypatch, tmp_path, module, source, column, args):
    ds = Dataset.from_dict({'o:id': [1], source: [''], column: [[1.] * 768]})
    pushed, api_calls = configure_main(monkeypatch, module, ds, tmp_path, [*args, '--update-mode', 'all'])
    assert module.main() == 0
    assert pushed[0][column][0] is None
    assert pushed[0][f'{column}_input_hash'][0] is None
    assert len(api_calls) == 1  # only readiness probe, no inference on missing input


def test_text_legacy_migration_then_matching_provenance_skips_inference(monkeypatch, tmp_path):
    ds = Dataset.from_dict({'o:id': [1], 'OCR': ['corrected text'], 'embedding_OCR': [[2.] * 768]})
    pushed, api_calls = configure_main(monkeypatch, text, ds, tmp_path, ['--config', 'articles', '--update-mode', 'missing'])
    assert text.main() == 0
    assert len(api_calls) == 2
    migrated = pushed[-1]
    pushed, api_calls = configure_main(monkeypatch, text, migrated, tmp_path, ['--config', 'articles', '--update-mode', 'missing'])
    assert text.main() == 0
    assert len(api_calls) == 1
    assert pushed[0]['embedding_OCR_config_hash'][0]


def test_image_same_url_changed_bytes_refreshes(monkeypatch, tmp_path):
    ds = Dataset.from_dict({'o:id': [1], 'image_url': ['one']})
    pushed, _ = configure_main(monkeypatch, images, ds, tmp_path, [])
    assert images.main() == 0
    ds = pushed[-1]
    pushed, api_calls = configure_main(monkeypatch, images, ds, tmp_path, [])
    assert images.main() == 0
    assert len(api_calls) == 1
    pushed, api_calls = configure_main(monkeypatch, images, ds, tmp_path, [])
    monkeypatch.setattr(images, 'download_image_bytes', lambda url, side: b'changed bytes')
    assert images.main() == 0
    assert len(api_calls) == 2
    assert pushed[-1]['embedding_image_input_hash'][0] != ds['embedding_image_input_hash'][0]


def test_checkpoint_pools_each_completed_row_once_and_keeps_failed_rows_retryable(monkeypatch, tmp_path):
    pooled = []
    monkeypatch.setattr(text, 'average_embeddings', lambda vectors, weights: pooled.append(vectors) or [1.])
    n = 50
    flat = [None] * n
    rows = [(i, ['abc']) for i in range(n)]
    cache, state, output = {}, {}, [[] for _ in range(n)]
    path = tmp_path / 'cache.sqlite3'
    for end in range(5, n + 1, 5):
        flat[end-5:end] = [[1.]] * 5
        flat[2] = None  # a permanently failed earlier request must not stall later rows
        text._save_completed_to_cache(cache, rows, flat, list(range(n)), output, path,
                                      ['hash'] * n, state, end)
    assert len(pooled) == n - 1
    assert len(load_cache(path)) == n - 1
    assert output[2] == []
    assert state == {'offset': n, 'row': n}


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
    assert settings['model'] == 'gemini-embedding-2'
    assert settings['google_genai']
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
