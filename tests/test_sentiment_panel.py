"""Panel invariants, and the mechanism that keeps a retired model's columns.

The expensive failure this file guards is not a crash: it is a *silent* one.
Generation 1's Omeka properties were deleted in 2026-08, and its 18 Hub columns
survive only because the uploader stops emitting them and ``hub_merge``
preserves what it does not overwrite. Break that pair and a single upload
overwrites four years of annotation with empty strings — with the row count
unchanged, so no shrink guard fires.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

import iwac_common.hub_merge as hub_merge
from iwac_common.hub_merge import merge_with_hub_dataset
from iwac_common.sentiment_panel import (
    DIMENSION_FIELDS,
    SUBJECTIVITE_ORDER,
    PANEL,
    active_models,
    all_columns,
    frozen_models,
    generation,
    label_subjectivite_columns,
    latest_generation,
    numeric_subjectivite_columns,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_uploader():
    """Import the articles uploader by path (``articles/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "upload_newspaper_ut", REPO_ROOT / "articles" / "upload_newspaper_hf.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


uploader = _load_uploader()


def _omeka_item(prefix: str, subjectivite: str = "Plutôt objectif") -> dict:
    """A minimal Omeka item carrying one model's six values, shaped as the API
    returns them: every scale is a ``resource:item`` link read via
    ``display_title``, never a literal."""
    link = lambda title: [{"type": "resource:item", "display_title": title}]  # noqa: E731
    text = lambda s: [{"type": "literal", "@value": s}]  # noqa: E731
    item = {
        f"{prefix}Centralite": link("Central"),
        f"{prefix}Polarite": link("Neutre"),
        f"{prefix}CentraliteJustification": text("parce que."),
        f"{prefix}PolariteJustification": text("parce que."),
        f"{prefix}SubjectiviteJustification": text("parce que."),
    }
    if subjectivite:
        item[f"{prefix}SubjectiviteScore"] = link(subjectivite)
    return item


class TestPanelShape:
    def test_generation_1_is_frozen_and_generation_2_is_live(self):
        assert [m.prefix for m in frozen_models()] == [m.prefix for m in generation(1)]
        assert [m.prefix for m in active_models()] == [m.prefix for m in generation(2)]
        assert latest_generation() == generation(2)

    def test_frozen_member_refuses_to_name_an_omeka_property(self):
        """The uploader must not be able to read a retired property by accident."""
        for model in frozen_models():
            with pytest.raises(ValueError, match="frozen"):
                model.omeka_property("Centralite")

    def test_active_members_declare_a_prompt_fingerprint(self):
        """A generation-2 column without a fingerprint cannot be attributed to a
        prompt, which is the whole point of the generation-2 rename."""
        for model in active_models():
            assert model.prompt_fingerprint, model.prefix

    def test_prefixes_are_unique(self):
        prefixes = [m.prefix for m in PANEL]
        assert len(prefixes) == len(set(prefixes))
        assert len(all_columns()) == len(set(all_columns())) == 6 * len(PANEL)

    def test_subjectivite_representation_splits_by_generation(self):
        """Type, not naming, is what varies: the column is ``_subjectivite_score``
        in both generations but holds a float in one and a label in the other."""
        assert set(numeric_subjectivite_columns()) == {
            m.subjectivite_column for m in generation(1)
        }
        assert set(label_subjectivite_columns()) == {
            m.subjectivite_column for m in generation(2)
        }
        assert not set(numeric_subjectivite_columns()) & set(label_subjectivite_columns())


class TestUploaderReadsOnlyLiveProperties:
    def test_mapper_emits_active_columns_only(self):
        item = {}
        for model in active_models():
            item.update(_omeka_item(model.omeka_prefix))
        cols = uploader._sentiment_columns(item)

        for model in active_models():
            for suffix, _ in DIMENSION_FIELDS:
                assert model.column(suffix) in cols
        for model in frozen_models():
            for suffix, _ in DIMENSION_FIELDS:
                assert model.column(suffix) not in cols, (
                    "a frozen model's column must not be produced — the mapper "
                    "emitting it empty is what would overwrite the Hub values"
                )

    def test_generation_2_subjectivite_keeps_the_label(self):
        model = active_models()[0]
        cols = uploader._sentiment_columns(_omeka_item(model.omeka_prefix, "Mixte"))
        assert cols[model.subjectivite_column] == "Mixte"

    def test_unscored_subjectivite_is_empty_not_none(self):
        """``""`` matches every other string column in the subset; the rows are
        real (the model declined to score), so they must not read as absent."""
        model = active_models()[0]
        cols = uploader._sentiment_columns(_omeka_item(model.omeka_prefix, ""))
        assert cols[model.subjectivite_column] == ""

    def test_label_columns_are_never_cast_to_int(self):
        """``astype("Int64")`` on a label column blanks it outright."""
        assert not set(uploader.SPEC.int_columns) & set(label_subjectivite_columns())
        assert set(numeric_subjectivite_columns()) <= set(uploader.SPEC.int_columns)

    def test_score_reader_ranks_the_label_it_reads(self):
        """The generation-1 reader is unexercised by the live panel but is the
        other half of ``subjectivite_is_label``; keep it honest."""
        item = _omeka_item("iwac:whatever", "Très subjectif")
        assert uploader._get_subjectivity_score(item, "iwac:whateverSubjectiviteScore") == 5
        assert uploader._get_subjectivity_score({}, "iwac:missing") is None
        assert SUBJECTIVITE_ORDER["Très subjectif"] == 5


class FakeDataset:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


class TestFrozenColumnsSurviveAnUpload:
    """End-to-end on the merge: the frozen block must come back untouched."""

    @staticmethod
    def _hub_frame():
        df = pd.DataFrame({"o:id": ["1", "2"], "title": ["old", "old"]})
        for model in frozen_models():
            df[model.column("polarite")] = ["Négatif", "Neutre"]
            df[model.subjectivite_column] = [4.0, 2.0]
        return df

    @staticmethod
    def _fresh_frame():
        df = pd.DataFrame({"o:id": ["1", "2"], "title": ["new", "new"]})
        for model in active_models():
            df[model.column("polarite")] = ["Positif", "Neutre"]
            df[model.subjectivite_column] = ["Plutôt objectif", "Mixte"]
        return df

    def test_retired_generation_is_preserved_verbatim(self, monkeypatch):
        monkeypatch.setattr(
            hub_merge, "load_dataset", lambda *a, **k: FakeDataset(self._hub_frame())
        )
        out = merge_with_hub_dataset(
            self._fresh_frame(), "repo", "articles", min_row_ratio=0.0
        )
        for model in frozen_models():
            assert list(out[model.column("polarite")]) == ["Négatif", "Neutre"]
            assert list(out[model.subjectivite_column]) == [4.0, 2.0]
        for model in active_models():
            assert list(out[model.column("polarite")]) == ["Positif", "Neutre"]
        assert list(out["title"]) == ["new", "new"]  # fresh Omeka still wins


class TestCrossGenerationOrdinal:
    """``subjectivite_ordinal`` is the only bridge across the generation break."""

    @staticmethod
    def _fn():
        from sentiment_agreement import subjectivite_ordinal

        return subjectivite_ordinal

    def test_reads_generation_2_labels(self):
        s = pd.Series(["Très objectif", "Mixte", "Très subjectif", "", None])
        assert list(self._fn()(s)[:3]) == [1.0, 3.0, 5.0]
        assert self._fn()(s)[3:].isna().all()

    def test_reads_generation_1_integers(self):
        s = pd.Series([1.0, 3.0, 5.0, None])
        out = self._fn()(s)
        assert list(out[:3]) == [1.0, 3.0, 5.0]
        assert out[3:].isna().all()

    def test_unknown_values_are_missing_not_zero(self):
        out = self._fn()(pd.Series(["Non applicable", "objectif?", "7"]))
        # 'Non applicable' is not a point on the subjectivité scale; a stray
        # numeric string is out of range but parses — neither may become a 0.
        assert out.isna().tolist() == [True, True, False]
        assert out.iloc[2] == 7.0

    def test_agreement_script_defaults_to_one_generation(self):
        import sentiment_agreement as sa

        assert sa.MODELS == [m.prefix for m in latest_generation()]
        assert sa.models_for(1) == [m.prefix for m in generation(1)]
        assert sa.models_for(None) == [m.prefix for m in PANEL]


class TestPublicAllowlistTracksThePanel:
    def test_every_panel_column_is_allowlisted(self):
        """``publish_public.py`` aborts on an unlisted column, so a rotation that
        forgets this file blocks the public push instead of shipping."""
        from iwac_common.repos import load_public_columns

        allow = set(load_public_columns()["articles"])
        missing = [c for c in all_columns() if c not in allow]
        assert not missing, f"add to public_columns.json['articles']: {missing}"

    def test_justifications_clear_the_prose_guard(self):
        """Justifications are prose and would otherwise look like leaked full
        text. Derived from the panel, so a new member is covered automatically —
        this asserts the derivation actually reaches the publisher."""
        import publish_public as pp

        from iwac_common.sentiment_panel import all_justification_columns

        for col in all_justification_columns():
            assert col in pp.PUBLIC_TEXT_ALLOWLIST
        # And a genuinely long value in one of them must not raise a suspect.
        col = active_models()[0].column("polarite_justification")
        df = pd.DataFrame({col: ["x" * 5_000, "x" * 5_000]})
        assert pp.find_suspect_columns(df, handled=set()) == []
