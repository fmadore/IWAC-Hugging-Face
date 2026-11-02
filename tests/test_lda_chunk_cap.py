"""Per-document chunk cap for LDA training.

A long book chunked into hundreds of training units outvotes the rest of a
small corpus. The cap keeps at most N chunks per document, evenly spaced, at
training time only, and is part of the model's frozen identity.
"""

from lda_topic_modeling.artifacts import frozen_preprocessing, preprocessing_fingerprint
from lda_topic_modeling.lda_topic_modeling import training_chunks
from lda_topic_modeling.modeling import cap_chunks, chunk_tokens


def _chunks(n):
    return [[f"c{i}"] for i in range(n)]


class TestCapChunks:
    def test_short_document_untouched(self):
        assert cap_chunks(_chunks(3), 5) == _chunks(3)

    def test_no_cap_is_identity(self):
        assert cap_chunks(_chunks(40), None) == _chunks(40)

    def test_spans_whole_document(self):
        # First and last chunks are kept: introduction and conclusion both count.
        kept = cap_chunks(_chunks(100), 5)
        assert [c[0] for c in kept] == ["c0", "c25", "c50", "c74", "c99"]

    def test_never_exceeds_cap_and_keeps_order(self):
        for n in range(1, 60):
            for cap in range(1, 12):
                kept = [int(c[0][1:]) for c in cap_chunks(_chunks(n), cap)]
                assert len(kept) == min(n, cap)
                assert kept == sorted(set(kept))

    def test_cap_of_one_takes_the_middle(self):
        assert cap_chunks(_chunks(9), 1) == [["c4"]]


class TestTrainingChunks:
    def test_cap_applies_per_document(self):
        long_doc = [f"w{i}" for i in range(10_000)]
        short_doc = [f"s{i}" for i in range(1_500)]
        prep = frozen_preprocessing(set(), "Français", 1000, max_chunks_per_doc=4)
        parts = training_chunks([long_doc, short_doc], prep)
        assert len(parts) == 4 + len(chunk_tokens(short_doc, 1000))

    def test_legacy_preprocessing_without_key_is_uncapped(self):
        doc = [f"w{i}" for i in range(10_000)]
        prep = frozen_preprocessing(set(), "Français", 1000)
        assert "max_chunks_per_doc" not in prep
        assert len(training_chunks([doc], prep)) == 10


class TestIdentity:
    def test_uncapped_fingerprint_unchanged_by_the_new_parameter(self):
        a = frozen_preprocessing({"x"}, "Français", 1000)
        b = frozen_preprocessing({"x"}, "Français", 1000, max_chunks_per_doc=None)
        assert preprocessing_fingerprint(a) == preprocessing_fingerprint(b)

    def test_cap_changes_the_fingerprint(self):
        a = frozen_preprocessing({"x"}, "Français", 1000)
        b = frozen_preprocessing({"x"}, "Français", 1000, max_chunks_per_doc=8)
        assert preprocessing_fingerprint(a) != preprocessing_fingerprint(b)
