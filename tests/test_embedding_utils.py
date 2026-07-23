import math

import pytest

from _embedding_utils import (
    average_embeddings,
    cache_fingerprint,
    chunk_text,
    delete_cache,
    is_empty_embedding,
    load_cache,
    save_cache,
)


class TestAverageEmbeddings:
    def test_single_vector_passthrough(self):
        assert average_embeddings([[1.0, 2.0]]) == [1.0, 2.0]

    def test_unweighted_mean(self):
        out = average_embeddings([[1.0, 0.0], [0.0, 1.0]])
        assert out == [0.5, 0.5]

    def test_length_weighted_mean(self):
        # A "chunk" 3x longer dominates 3:1.
        out = average_embeddings([[1.0, 0.0], [0.0, 1.0]], weights=[3, 1])
        assert math.isclose(out[0], 0.75) and math.isclose(out[1], 0.25)

    def test_zero_weights_fall_back_to_mean(self):
        out = average_embeddings([[2.0], [4.0]], weights=[0, 0])
        assert out == [3.0]

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            average_embeddings([[1.0], [2.0]], weights=[1.0])


class TestCacheFingerprint:
    def test_shape(self):
        assert (
            cache_fingerprint("gemini-embedding-2", 768, "RETRIEVAL_DOCUMENT")
            == "gemini-embedding-2_768d_retrieval_document"
        )

    def test_distinct_params_distinct_fingerprints(self):
        a = cache_fingerprint("m", 768, "SEMANTIC_SIMILARITY")
        b = cache_fingerprint("m", 1536, "SEMANTIC_SIMILARITY")
        c = cache_fingerprint("m", 768, "RETRIEVAL_DOCUMENT")
        assert len({a, b, c}) == 3

    def test_slash_in_model_is_path_safe(self):
        assert "/" not in cache_fingerprint("org/model", 8, "X")


class TestCacheRoundTrip:
    def test_save_load_delete(self, tmp_path):
        f = tmp_path / "cache.json.gz"
        save_cache({"1": [0.1, 0.2], "2": ["lemma", "clean"]}, f)
        assert load_cache(f) == {"1": [0.1, 0.2], "2": ["lemma", "clean"]}
        delete_cache(f)
        assert not f.exists()
        assert load_cache(f) == {}


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("abc", chunk_size=10, overlap=2) == ["abc"]

    def test_chunks_cover_text_with_overlap(self):
        text = "abcdefghij" * 10  # 100 chars
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        assert chunks[0] == text[:40]
        assert chunks[1][:10] == chunks[0][-10:]  # overlap preserved
        # Reconstruct: stripping the overlap from every chunk after the first
        # must give back the original text.
        rebuilt = chunks[0] + "".join(c[10:] for c in chunks[1:])
        assert rebuilt == text


class TestIsEmptyEmbedding:
    def test_cases(self):
        assert is_empty_embedding(None)
        assert is_empty_embedding([])
        assert is_empty_embedding([0.0, 0.0])
        assert not is_empty_embedding([0.0, 0.1])
