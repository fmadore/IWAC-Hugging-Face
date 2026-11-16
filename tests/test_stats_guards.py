"""Guards in analyses/_stats.py: undefined tests, cluster floor, one language rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from iwac_pipeline.analyses import _stats


def test_cluster_bootstrap_floor_returns_nan_below_the_constant():
    values = np.arange(90, dtype=float)
    nine = [f"c{i % 9}" for i in range(90)]
    ten = [f"c{i % 10}" for i in range(90)]
    assert _stats.MIN_BOOTSTRAP_CLUSTERS == 10
    lo, hi = _stats.bootstrap_mean_ci(values, 200, 1, clusters=nine)
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = _stats.bootstrap_mean_ci(values, 200, 1, clusters=ten)
    assert lo < values.mean() < hi
    matrix = np.c_[values, values]
    lo, hi = _stats.bootstrap_mean_ci(matrix, 50, 1, clusters=nine)
    assert lo.shape == (2,) and np.isnan(lo).all()


def test_language_mask_is_shared_and_alias_aware():
    labels = pd.Series(["Français", " french ", "fr", "Français|Arabe", "Ewé", "", None, "English", "en"])
    assert _stats.french_language_mask(labels).tolist() == [True, True, True, False, False, False, False, False, False]
    assert _stats.language_mask(labels, "English").tolist()[-2:] == [True, True]
    assert _stats.language_mask(labels, "Ewé").sum() == 1
    with_unknown = _stats.language_mask(labels, "Français", include_unknown=True)
    assert with_unknown.tolist()[5:7] == [True, True]
    counts = _stats.excluded_language_counts(labels, _stats.french_language_mask(labels))
    assert counts == {"English": 1, "en": 1, "Ewé": 1, "Français|Arabe": 1, "[missing]": 2}
