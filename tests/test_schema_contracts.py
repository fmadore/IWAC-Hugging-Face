"""Contracts around embedding columns, especially on newly added items."""

import numpy as np
import pandas as pd
import pytest

from iwac_common.schema import (
    DataContractError,
    normalize_embedding_nulls,
    validate_embedding_dimensions,
)


def _frame(values):
    return pd.DataFrame({"o:id": [str(i) for i in range(len(values))],
                         "embedding_OCR": values})


def test_merge_nan_is_treated_as_missing_not_as_a_broken_vector():
    """A left-merge fills a brand-new item's embedding with NaN, not None."""
    df = _frame([[0.1] * 768, np.nan, None])
    validate_embedding_dimensions(df, "articles")


def test_wrong_dimension_still_fails():
    df = _frame([[0.1] * 12])
    with pytest.raises(DataContractError):
        validate_embedding_dimensions(df, "articles")


def test_normalize_replaces_nan_with_none_and_keeps_vectors():
    vector = [0.1] * 768
    df = normalize_embedding_nulls(_frame([vector, np.nan, None]), "articles")
    assert df["embedding_OCR"].tolist() == [vector, None, None]


def test_normalize_is_a_no_op_without_the_column():
    df = pd.DataFrame({"o:id": ["1"]})
    assert normalize_embedding_nulls(df, "articles").columns.tolist() == ["o:id"]


# ---------------------------------------------------------------------------
# Canonical types at the write gateway
# ---------------------------------------------------------------------------

import pyarrow as pa  # noqa: E402
from datasets import Dataset  # noqa: E402

from iwac_common.schema import (  # noqa: E402
    conform_dataset,
    dataset_to_pandas,
    validate_dataset,
    validate_ids,
)


def _arrow_ds(**columns):
    n = len(next(iter(columns.values())))
    return Dataset(pa.table({"o:id": [str(i) for i in range(n)], **columns}))


class TestConformDataset:
    def test_round_tripped_float_ints_become_int64_again(self):
        """The exact drift seen on the public card: nullable ints as float64."""
        ds = _arrow_ds(
            lda_topic_id=pa.array([3.0, None, 7.0]),
            hijri_year=pa.array([1421.0, float("nan"), 1446.0]),
            title=pa.array(["a", "b", "c"]),
        )
        out = conform_dataset(ds, "articles")
        assert out.features["lda_topic_id"].dtype == "int64"
        assert out.features["hijri_year"].dtype == "int64"
        assert out["lda_topic_id"][:] == [3, None, 7]
        assert out["hijri_year"][:] == [1421, None, 1446]
        assert out.features["title"].dtype == "string"

    def test_fractional_value_in_declared_int_column_fails(self):
        ds = _arrow_ds(nb_pages=pa.array([1.5, 2.0]))
        with pytest.raises(DataContractError, match="fractional"):
            conform_dataset(ds, "articles")

    def test_all_null_int_column_gets_a_real_type(self):
        ds = _arrow_ds(hijri_day=pa.array([None, None], pa.null()))
        assert conform_dataset(ds, "images").features["hijri_day"].dtype == "int64"

    def test_embeddings_are_stored_as_float32(self):
        ds = _arrow_ds(embedding_OCR=pa.array([[0.5] * 768, None], pa.list_(pa.float64())))
        out = conform_dataset(ds, "articles")
        assert out.features["embedding_OCR"].feature.dtype == "float32"
        assert out["embedding_OCR"][1] is None

    def test_undeclared_columns_and_subsets_are_untouched(self):
        ds = _arrow_ds(consensus_subjectivite_score=pa.array([2.5, None]))
        out = conform_dataset(ds, "articles")
        assert out is ds

    def test_source_revision_rides_along(self):
        ds = _arrow_ds(nb_mots=pa.array([1.0, 2.0]))
        ds._iwac_source_revision = "abc"
        assert conform_dataset(ds, "articles")._iwac_source_revision == "abc"

    def test_pandas_conversion_keeps_nullable_ints(self):
        ds = _arrow_ds(frequency=pa.array([1, None], pa.int64()))
        assert str(dataset_to_pandas(ds)["frequency"].dtype) == "Int64"


class TestStricterContracts:
    @pytest.mark.parametrize(
        "value, message",
        [
            ("x" * 768, "not a vector"),
            ([[0.1] * 2] * 768, "one-dimensional"),
            ([float("nan")] + [0.1] * 767, "NaN or infinity"),
            ([float("inf")] + [0.1] * 767, "NaN or infinity"),
        ],
    )
    def test_frame_rejects_malformed_vectors(self, value, message):
        with pytest.raises(DataContractError, match=message):
            validate_embedding_dimensions(_frame([value]), "articles")

    def test_arrow_dataset_rejects_nonfinite_vector(self):
        ds = _arrow_ds(embedding_OCR=pa.array(
            [[0.1] * 768, [float("nan")] + [0.1] * 767], pa.list_(pa.float32())
        ))
        with pytest.raises(DataContractError, match="o:id=1 contains NaN"):
            validate_dataset(ds, "articles")

    def test_arrow_dataset_rejects_string_embeddings(self):
        ds = _arrow_ds(embedding_OCR=pa.array(["x" * 768]))
        with pytest.raises(DataContractError, match="not a list"):
            validate_dataset(ds, "articles")

    def test_arrow_dataset_rejects_nested_vectors(self):
        ds = _arrow_ds(embedding_OCR=pa.array([[[0.1]] * 768]))
        with pytest.raises(DataContractError, match="not numbers"):
            validate_dataset(ds, "articles")

    def test_arrow_dataset_accepts_nulls_and_empties(self):
        ds = _arrow_ds(embedding_OCR=pa.array(
            [[0.1] * 768, None, []], pa.list_(pa.float32())
        ))
        validate_dataset(ds, "articles")

    def test_blank_ids_are_rejected_everywhere(self):
        with pytest.raises(DataContractError, match="blank"):
            validate_ids(pd.DataFrame({"o:id": ["1", "  "]}))
        with pytest.raises(DataContractError, match="blank"):
            validate_dataset(Dataset(pa.table({"o:id": ["1", ""]})), "articles")
