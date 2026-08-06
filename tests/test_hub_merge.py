import pandas as pd
import pytest

import iwac_common.hub_merge as hub_merge
from iwac_common.hub_merge import (
    DuplicateIdError,
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
