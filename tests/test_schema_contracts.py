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
