import pandas as pd
import pytest

import iwac_common.hub_merge as hub_merge
from iwac_common.hub_merge import (
    DuplicateIdError,
    HubBaselineUnavailableError,
    ShrinkGuardError,
    merge_with_hub_dataset,
)


class FakeDataset:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


@pytest.fixture
def hub(monkeypatch):
    """Patch load_dataset inside hub_merge to serve a canned existing frame."""

    def _install(existing_df):
        monkeypatch.setattr(
            hub_merge, "load_dataset", lambda *a, **k: FakeDataset(existing_df)
        )

    return _install


def _existing(n=4, computed=True):
    df = pd.DataFrame(
        {
            "o:id": [str(i) for i in range(1, n + 1)],
            "title": [f"t{i}" for i in range(1, n + 1)],
        }
    )
    if computed:
        df["embedding_OCR"] = [[0.1] * 3] * n
    return df


def _new(n=4):
    return pd.DataFrame(
        {
            "o:id": [str(i) for i in range(1, n + 1)],
            "title": [f"T{i}" for i in range(1, n + 1)],
        }
    )


class TestComputedColumnPreservation:
    def test_left_merge_preserves_hub_only_columns(self, hub):
        hub(_existing())
        out = merge_with_hub_dataset(_new(), "repo", "articles")
        assert "embedding_OCR" in out.columns
        assert len(out) == 4
        assert list(out["title"]) == ["T1", "T2", "T3", "T4"]  # fresh Omeka wins

    def test_new_item_gets_null_computed_column(self, hub):
        hub(_existing(3))
        out = merge_with_hub_dataset(_new(4), "repo", "articles")
        assert out.loc[out["o:id"] == "4", "embedding_OCR"].isna().all()

    def test_failed_media_field_keeps_existing_value(self, hub):
        existing = _existing()
        existing["thumbnail"] = [f"old-{i}" for i in range(4)]
        fresh = _new()
        fresh["thumbnail"] = ["", "new-2", "new-3", "new-4"]
        hub(existing)
        out = merge_with_hub_dataset(
            fresh,
            "repo",
            "articles",
            preserve_fields_by_id={"1": {"thumbnail"}},
        )
        assert out.loc[out["o:id"] == "1", "thumbnail"].item() == "old-0"
        assert out.loc[out["o:id"] == "2", "thumbnail"].item() == "new-2"

    def test_failed_mapper_row_does_not_blank_other_computed_values(self, hub):
        existing = _existing(2)
        fresh = _new(1)
        hub(existing)
        out = merge_with_hub_dataset(
            fresh,
            "repo",
            "articles",
            preserve_existing_ids=["2"],
        ).sort_values("o:id")
        assert out["title"].tolist() == ["T1", "t2"]
        assert out["embedding_OCR"].notna().all()


class TestFailClosedBaseline:
    def test_load_error_is_not_treated_as_first_run(self, monkeypatch):
        def fail(*args, **kwargs):
            raise OSError("offline")

        monkeypatch.setattr(hub_merge, "load_dataset", fail)
        with pytest.raises(HubBaselineUnavailableError, match="Refusing to assume"):
            merge_with_hub_dataset(_new(), "repo", "articles")

    def test_explicit_initialize_requires_absent_config(self, monkeypatch):
        monkeypatch.setattr(
            hub_merge, "load_dataset", lambda *a, **k: (_ for _ in ()).throw(OSError("404"))
        )
        monkeypatch.setattr(hub_merge, "get_repo_configs", lambda *a, **k: set())
        out = merge_with_hub_dataset(
            _new(), "repo", "articles", allow_initialize=True
        )
        assert list(out["o:id"]) == ["1", "2", "3", "4"]


class TestSentimentColumnRename:
    """The 2026-07 vendor→model rename of the sentiment columns.

    The rename only works because the mapper emits the NEW names while the OLD
    ones are passed as ``columns_to_exclude``. Without the exclusion the merge
    would preserve the old names as Hub-only columns and the dataset would carry
    both sets; with it, the rename happens in place.
    """

    # Names come from the live panel, so a future rotation cannot leave this
    # test asserting against a column prefix that no longer exists.
    LEGACY_COL = "gemini_polarite"

    @staticmethod
    def _current_col():
        # An *active* member: a frozen one's columns are deliberately absent from
        # the mapper's output, so it could not stand in for a renamed column.
        from iwac_common.sentiment_panel import active_models

        return active_models()[0].column("polarite")

    def _hub_with_legacy(self):
        df = _existing()
        df[self.LEGACY_COL] = ["Neutre"] * 4
        df["embedding_OCR"] = [[0.1] * 3] * 4
        return df

    def _new_with_renamed(self):
        df = _new()
        df[self._current_col()] = ["Neutre"] * 4
        return df

    def test_legacy_column_dropped_when_excluded(self, hub):
        hub(self._hub_with_legacy())
        out = merge_with_hub_dataset(
            self._new_with_renamed(), "repo", "articles",
            columns_to_exclude=[self.LEGACY_COL],
        )
        assert self.LEGACY_COL not in out.columns
        assert self._current_col() in out.columns
        # Genuinely computed columns are still preserved.
        assert "embedding_OCR" in out.columns

    def test_without_exclusion_both_names_survive(self, hub):
        """Guards the failure mode: forgetting columns_to_exclude duplicates."""
        hub(self._hub_with_legacy())
        out = merge_with_hub_dataset(self._new_with_renamed(), "repo", "articles")
        assert self.LEGACY_COL in out.columns
        assert self._current_col() in out.columns

    def test_panel_columns_match_uploader_spec(self):
        """The uploader's exclusion list must be exactly the pre-rename names."""
        from iwac_common.sentiment_panel import LEGACY_VENDOR_COLUMNS, all_columns

        assert len(LEGACY_VENDOR_COLUMNS) == 18
        # No overlap: an old name that is also a new name would be dropped and
        # re-added in the same merge, which is not a rename.
        assert not set(LEGACY_VENDOR_COLUMNS) & set(all_columns())


class TestShrinkGuard:
    def test_truncated_fetch_raises(self, hub):
        hub(_existing(100))
        with pytest.raises(ShrinkGuardError):
            merge_with_hub_dataset(_new(50), "repo", "articles")

    def test_small_shrink_within_threshold_passes(self, hub):
        hub(_existing(100))
        out = merge_with_hub_dataset(_new(97), "repo", "articles")
        assert len(out) == 97

    def test_allow_shrink_overrides(self, hub):
        hub(_existing(100))
        out = merge_with_hub_dataset(_new(50), "repo", "articles", allow_shrink=True)
        assert len(out) == 50

    def test_no_existing_data_never_trips(self, hub):
        hub(pd.DataFrame())
        out = merge_with_hub_dataset(_new(2), "repo", "articles")
        assert len(out) == 2


class TestDuplicateIds:
    @pytest.mark.parametrize("bad", [
        pd.DataFrame({"title": ["title"]}),
        pd.DataFrame({"o:id": [None]}),
        pd.DataFrame({"o:id": ["  "]}),
    ])
    def test_invalid_fresh_ids_fail_before_hub_read(self, monkeypatch, bad):
        def unexpected(*args, **kwargs):
            pytest.fail("Invalid input must fail before reading the Hub")
        monkeypatch.setattr(hub_merge, "load_dataset", unexpected)
        with pytest.raises(ValueError, match="o:id"):
            merge_with_hub_dataset(bad, "repo", "articles")

    def test_duplicate_new_ids_raise(self, hub):
        hub(_existing())
        bad = _new()
        bad.loc[1, "o:id"] = "1"
        with pytest.raises(DuplicateIdError):
            merge_with_hub_dataset(bad, "repo", "articles")

    def test_duplicate_hub_ids_raise(self, hub):
        dupes = _existing()
        dupes.loc[1, "o:id"] = "1"
        hub(dupes)
        with pytest.raises(DuplicateIdError):
            merge_with_hub_dataset(_new(), "repo", "articles")


class TestOuterMergeStaleRows:
    def test_stale_rows_kept_by_default(self, hub):
        hub(_existing(20))
        out = merge_with_hub_dataset(_new(20).iloc[:19], "repo", "references", how="outer")
        assert len(out) == 20  # deleted Omeka item survives
        assert "20" in set(out["o:id"])

    def test_stale_rows_dropped_on_request(self, hub):
        hub(_existing(20))
        out = merge_with_hub_dataset(
            _new(20).iloc[:19], "repo", "references", how="outer", stale_rows="drop"
        )
        assert len(out) == 19
        assert "20" not in set(out["o:id"])

    def test_invalid_stale_rows_value(self, hub):
        hub(_existing())
        with pytest.raises(ValueError):
            merge_with_hub_dataset(_new(), "repo", "references", stale_rows="maybe")


class TestSchemaStability:
    def test_hub_column_order_survives_an_upload(self, hub):
        existing = pd.DataFrame({
            "o:id": ["1", "2"],
            "pub_date": ["2001-01-01", "2002"],
            "hijri_year": pd.array([1421, None], dtype="Int64"),
            "title": ["a", "b"],
        })
        hub(existing)
        fresh = pd.DataFrame({
            "o:id": ["1", "2"], "title": ["A", "B"],
            "pub_date": ["2001-01-01", "2002"], "descriptionAI": ["x", "y"],
        })
        out = merge_with_hub_dataset(fresh, "repo", "articles")
        # hijri stays beside pub_date; the brand-new column follows its
        # mapper predecessor instead of the Hub-only block moving to the end.
        assert list(out.columns) == [
            "o:id", "pub_date", "descriptionAI", "hijri_year", "title"
        ]

    def test_stable_order_puts_a_leading_new_column_first(self):
        from iwac_common.hub_merge import stable_column_order

        assert stable_column_order(
            ["new", "o:id", "x"], ["o:id", "x"], ["new", "o:id"]
        ) == ["new", "o:id", "x"]

    def test_nullable_int_hub_column_is_not_turned_into_float(self, monkeypatch):
        import pyarrow as pa
        from datasets import Dataset

        table = pa.table({
            "o:id": ["1", "2"],
            "title": ["a", "b"],
            "lda_topic_id": pa.array([3, None], pa.int64()),
        })
        monkeypatch.setattr(hub_merge, "load_dataset", lambda *a, **k: Dataset(table))
        out = merge_with_hub_dataset(
            pd.DataFrame({"o:id": ["1", "2", "3"], "title": ["A", "B", "C"]}),
            "repo", "articles",
        )
        assert str(out["lda_topic_id"].dtype) == "Int64"
        assert out["lda_topic_id"].tolist()[0] == 3
        assert Dataset.from_pandas(out, preserve_index=False).features[
            "lda_topic_id"
        ].dtype == "int64"

    def test_pinned_revision_reuses_the_cache(self, monkeypatch):
        calls = []

        def fake_load(*args, **kwargs):
            calls.append(kwargs)
            return FakeDataset(_existing())

        monkeypatch.setattr(hub_merge, "load_dataset", fake_load)
        monkeypatch.setattr(hub_merge, "get_repo_revision", lambda *a, **k: "abc123")
        merge_with_hub_dataset(_new(), "repo", "articles", revision_out={})
        assert calls[0]["revision"] == "abc123"
        assert calls[0]["download_mode"] == "reuse_dataset_if_exists"


class TestStaleDerivedValues:
    DERIVED = {"OCR": ("embedding_OCR", "lemma_text", "nb_mots")}

    def _existing(self):
        return pd.DataFrame({
            "o:id": ["1", "2", "3"],
            "OCR": ["texte un", "texte deux", "texte  trois"],
            "embedding_OCR": [[0.1], [0.2], [0.3]],
            "lemma_text": ["un", "deux", "trois"],
        })

    def _fresh(self):
        return pd.DataFrame({
            "o:id": ["1", "2", "3", "4"],
            # 2 corrected; 3 only re-wrapped; 4 is new.
            "OCR": ["texte un", "texte deux corrigé", "texte trois", "nouveau"],
            "nb_mots": [2, 3, 2, 1],  # mapper-produced: fresh, never "stale"
        })

    def test_changed_text_is_reported_not_whitespace(self, hub):
        hub(self._existing())
        stale = {}
        out = merge_with_hub_dataset(
            self._fresh(), "repo", "articles", derived_from=self.DERIVED, stale_out=stale,
        )
        assert stale == {"OCR": {"ids": ["2"], "derived": ["embedding_OCR", "lemma_text"]}}
        # Default keeps the values (reported only).
        assert out.set_index("o:id").loc["2", "lemma_text"] == "deux"

    def test_invalidation_clears_only_the_changed_rows(self, hub):
        hub(self._existing())
        out = merge_with_hub_dataset(
            self._fresh(), "repo", "articles", derived_from=self.DERIVED,
            invalidate_derived=True,
        ).set_index("o:id")
        assert pd.isna(out.loc["2", "lemma_text"])
        assert out.loc["2", "embedding_OCR"] is None
        assert out.loc["1", "lemma_text"] == "un"
        assert list(out.loc["3", "embedding_OCR"]) == [0.3]
        assert out.loc["2", "nb_mots"] == 3

    def test_worklist_accumulates_across_runs(self, tmp_path, monkeypatch):
        import json

        from iwac_common.upload_runner import record_stale_derived

        monkeypatch.setenv("IWAC_STATE_DIR", str(tmp_path))
        record_stale_derived("o/r", "articles",
                             {"OCR": {"ids": ["2"], "derived": ["lemma_text"]}}, "sha1")
        path = record_stale_derived("o/r", "articles",
                                    {"OCR": {"ids": ["10"], "derived": ["lemma_text"]}}, "sha2")
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["columns"]["lemma_text"] == ["2", "10"]
        assert record["last_revision_before_push"] == "sha2"

    def test_every_derived_column_is_hub_only_somewhere(self):
        """A dependency on a mapper column would be meaningless (it is always
        fresh); the declared sources must be real mapper/Hub columns."""
        from iwac_common.schema import DERIVED_FROM, SUBSETS

        assert set(DERIVED_FROM) == set(SUBSETS)
        for config, mapping in DERIVED_FROM.items():
            for source, derived in mapping.items():
                assert source not in derived, (config, source)
