"""Tests for the privacy boundary: publish_public.py masking + guards.

These are the most important tests in the repo — a regression here leaks
private full text to the public dataset.
"""

import json

import pandas as pd
import pytest

import publish_public as pp


LONG = "x" * 5_000  # > SUSPECT_MEAN_CHARS
SHORT = "hello"


class TestMaskContentColumns:
    def _articles_df(self, flags):
        n = len(flags)
        return pd.DataFrame(
            {
                "o:id": [str(i) for i in range(n)],
                "OCR": [f"text{i}" for i in range(n)],
                "lemma_text": [f"lemma{i}" for i in range(n)],
                "lemma_nostop": [f"nostop{i}" for i in range(n)],
                "OCR_is_public": flags,
            }
        )

    def test_private_rows_blanked_public_rows_kept(self):
        df = self._articles_df([True, False])
        cols, kept, blanked = pp.mask_content_columns(df, "articles")
        assert set(cols) == {"OCR", "lemma_text", "lemma_nostop"}
        assert kept == 1 and blanked == 1
        assert df.loc[0, "OCR"] == "text0"
        assert df.loc[1, "OCR"] == ""
        assert df.loc[1, "lemma_text"] == "" and df.loc[1, "lemma_nostop"] == ""

    def test_null_flag_is_private(self):
        # A row with no flag must NEVER keep its text.
        df = self._articles_df([True, None])
        _, kept, blanked = pp.mask_content_columns(df, "articles")
        assert kept == 1 and blanked == 1
        assert df.loc[1, "OCR"] == ""

    @pytest.mark.parametrize("flag", ["False", "true", 0, 1])
    def test_non_boolean_flag_is_rejected(self, flag):
        df = self._articles_df([flag])
        with pytest.raises(pp.InvalidFlagError):
            pp.mask_content_columns(df, "articles")

    def test_missing_flag_column_raises(self):
        df = self._articles_df([True, False]).drop(columns=["OCR_is_public"])
        with pytest.raises(pp.MissingFlagError):
            pp.mask_content_columns(df, "articles")

    def test_subset_without_content_columns_is_noop(self):
        df = pd.DataFrame({"o:id": ["1"], "title": ["photo"]})
        cols, kept, blanked = pp.mask_content_columns(df, "images")
        assert cols == [] and kept == 0 and blanked == 0
        assert df.loc[0, "title"] == "photo"


class TestFindSuspectColumns:
    def test_long_object_column_flagged(self):
        df = pd.DataFrame({"new_text": [LONG, LONG]})
        assert [s[0] for s in pp.find_suspect_columns(df, handled=set())] == ["new_text"]

    def test_list_of_str_column_flagged(self):
        # A full text chunked into list[str] must not evade the guard.
        df = pd.DataFrame({"chunks": [[LONG[:3000], LONG[:3000]], [LONG]]})
        assert [s[0] for s in pp.find_suspect_columns(df, handled=set())] == ["chunks"]

    def test_pandas_string_dtype_flagged(self):
        df = pd.DataFrame({"typed_text": pd.array([LONG, LONG], dtype="string")})
        assert [s[0] for s in pp.find_suspect_columns(df, handled=set())] == ["typed_text"]

    def test_short_and_numeric_columns_pass(self):
        df = pd.DataFrame({"title": [SHORT, SHORT], "n": [1, 2], "f": [0.1, None]})
        assert pp.find_suspect_columns(df, handled=set()) == []

    def test_handled_and_allowlisted_columns_skipped(self):
        df = pd.DataFrame({"OCR": [LONG], "descriptionAI": [LONG]})
        assert pp.find_suspect_columns(df, handled={"OCR"}) == []

    def test_max_length_trigger(self):
        # mean below threshold but one extreme value above the hard ceiling
        df = pd.DataFrame({"c": [SHORT] * 99 + ["y" * 40_000]})
        assert [s[0] for s in pp.find_suspect_columns(df, handled=set())] == ["c"]


class TestColumnAllowlist:
    def _patch_allowlist(self, monkeypatch, tmp_path, data):
        f = tmp_path / "public_columns.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        import iwac_common.repos as repos

        monkeypatch.setattr(repos, "PUBLIC_COLUMNS_FILE", str(f))
        monkeypatch.setattr(pp, "PUBLIC_COLUMNS_FILE", str(f))
        return f

    def test_known_columns_pass(self, monkeypatch, tmp_path):
        self._patch_allowlist(monkeypatch, tmp_path, {"articles": ["o:id", "title"]})
        df = pd.DataFrame({"o:id": ["1"], "title": ["t"]})
        assert pp.check_column_allowlist("articles", df, approve=set()) == []

    def test_unknown_column_aborts(self, monkeypatch, tmp_path):
        self._patch_allowlist(monkeypatch, tmp_path, {"articles": ["o:id"]})
        df = pd.DataFrame({"o:id": ["1"], "surprise": [LONG]})
        with pytest.raises(SystemExit):
            pp.check_column_allowlist("articles", df, approve=set())

    def test_unknown_subset_aborts(self, monkeypatch, tmp_path):
        self._patch_allowlist(monkeypatch, tmp_path, {"articles": ["o:id"]})
        df = pd.DataFrame({"o:id": ["1"]})
        with pytest.raises(SystemExit):
            pp.check_column_allowlist("mystery", df, approve=set())

    def test_approved_column_is_deferred_until_publish(self, monkeypatch, tmp_path):
        f = self._patch_allowlist(monkeypatch, tmp_path, {"articles": ["o:id"]})
        df = pd.DataFrame({"o:id": ["1"], "new_metric": [0.9]})
        approved = pp.check_column_allowlist("articles", df, approve={"new_metric"})
        assert approved == ["new_metric"]
        on_disk = json.loads(f.read_text(encoding="utf-8"))
        assert "new_metric" not in on_disk["articles"]
        pp.persist_public_column_approvals({"articles": approved})
        on_disk = json.loads(f.read_text(encoding="utf-8"))
        assert "new_metric" in on_disk["articles"]

    def test_partial_approval_still_aborts(self, monkeypatch, tmp_path):
        self._patch_allowlist(monkeypatch, tmp_path, {"articles": ["o:id"]})
        df = pd.DataFrame({"o:id": ["1"], "a": [1], "b": [2]})
        with pytest.raises(SystemExit):
            pp.check_column_allowlist("articles", df, approve={"a"})


class TestLiveAllowlistFile:
    def test_repo_allowlist_covers_content_columns(self):
        """Every content column must be in the allowlist (they ARE projected,
        masked per row) and every subset with content columns must be listed."""
        from iwac_common.repos import CONTENT_COLUMNS, load_public_columns

        allow = load_public_columns()
        for cfg, cols in CONTENT_COLUMNS.items():
            assert cfg in allow
            for c in cols:
                assert c in allow[cfg], f"{c} missing from allowlist[{cfg}]"
            assert "OCR_is_public" in allow[cfg]

    def test_bilingual_summary_columns_are_paired(self):
        """`descriptionAI_en` must be allow-listed wherever `descriptionAI` is.

        The mappers emit the two together, so a subset that lists only the
        French half would abort publish_public.py on the first bilingual push —
        after the private mirror had already taken the column.
        """
        from iwac_common.repos import load_public_columns

        allow = load_public_columns()
        for cfg, cols in allow.items():
            if cfg.startswith("_") or "descriptionAI" not in cols:
                continue
            assert "descriptionAI_en" in cols, (
                f"allowlist[{cfg}] has descriptionAI but not descriptionAI_en"
            )


class TestSquashGuard:
    """A DOI cites one revision; squashing the history would orphan it."""

    class _Api:
        def __init__(self, tags=None, error=None):
            self._tags, self._error = tags, error

        def dataset_info(self, **kwargs):
            if self._error:
                raise self._error

            class Info:
                pass

            info = Info()
            info.tags = self._tags
            return info

    def test_repo_with_doi_is_refused(self):
        api = self._Api(tags=["language:fr", "doi:10.57967/hf/9857"])
        with pytest.raises(pp.SquashRefusedError, match="10.57967/hf/9857"):
            pp.assert_squash_allowed("owner/public", token="t", api=api)

    def test_unreadable_tags_fail_closed(self):
        api = self._Api(error=RuntimeError("401"))
        with pytest.raises(pp.SquashRefusedError, match="Cannot read"):
            pp.assert_squash_allowed("owner/public", token="t", api=api)

    def test_scratch_repo_without_doi_is_allowed(self):
        pp.assert_squash_allowed(
            "owner/scratch", token="t", api=self._Api(tags=["language:fr"])
        )


class TestItemAndPropertyProjection:
    def _frame(self):
        return pd.DataFrame({
            "o:id": ["1", "2"], "item_is_public": [True, False],
            "private_fields": [["author", "author_ids", "descriptionAI", "OCR"], []],
            "OCR_is_public": [False, True], "OCR": ["private text", "private parent text"],
            "lemma_text": ["private lemma", "private parent lemma"],
            "author": ["restricted name", "other"], "author_ids": ["99", "100"],
            "descriptionAI": ["restricted summary", "summary"],
            "embedding_OCR": [[0.1] * 768, [0.2] * 768], "nb_mots": [2, 3],
        })

    def test_private_parent_is_excluded_even_if_content_value_public(self):
        from iwac_common.write_policy import prepare_public_projection
        result, omitted = prepare_public_projection(self._frame(), "articles")
        assert omitted == 1
        assert result["o:id"].tolist() == ["1"]

    def test_source_restrictions_and_lemmas_masked_but_reviewed_derivatives_remain(self):
        from iwac_common.write_policy import prepare_public_projection
        result, _ = prepare_public_projection(self._frame(), "articles")
        for column in ("OCR", "lemma_text", "author", "author_ids", "descriptionAI"):
            assert result.iloc[0][column] == ""
        assert len(result.iloc[0]["embedding_OCR"]) == 768
        assert result.iloc[0]["nb_mots"] == 2

    def test_direct_restriction_overrides_reviewed_derived_exception(self):
        from iwac_common.write_policy import prepare_public_projection
        frame = self._frame()
        frame.at[0, "private_fields"].append("embedding_OCR")
        result, _ = prepare_public_projection(frame, "articles")
        assert result.iloc[0]["embedding_OCR"] is None

    @pytest.mark.parametrize("column", ["item_is_public", "private_fields"])
    def test_legacy_mirror_needs_visibility_backfill(self, column):
        from iwac_common.write_policy import PublicationPolicyError, prepare_public_projection
        with pytest.raises(PublicationPolicyError, match="refresh"):
            prepare_public_projection(self._frame().drop(columns=column), "articles")

    @pytest.mark.parametrize("bad", ["[]", None, [42]])
    def test_malformed_restriction_metadata_is_refused(self, bad):
        from iwac_common.write_policy import PublicationPolicyError, prepare_public_projection
        frame = self._frame()
        frame.at[0, "private_fields"] = bad
        with pytest.raises(PublicationPolicyError, match="list"):
            prepare_public_projection(frame, "articles")

    def test_known_private_provenance_removed_unknown_column_still_guarded(self):
        from iwac_common.write_policy import prepare_public_projection
        frame = self._frame()
        frame["embedding_OCR_input_hash"] = ["hash", "hash"]
        frame["new_secret_hash"] = ["unknown", "unknown"]
        result, _ = prepare_public_projection(frame, "articles")
        assert "embedding_OCR_input_hash" not in result.columns
        assert "new_secret_hash" in result.columns
        with pytest.raises(SystemExit):
            pp.check_column_allowlist("articles", result, approve=set())


def test_main_publishes_all_selected_subsets_in_one_batch(monkeypatch):
    import sys
    from types import SimpleNamespace
    from datasets import Dataset
    datasets = {
        config: Dataset.from_dict({"o:id": ["1"], "title": ["example"],
                                   "item_is_public": [True], "private_fields": [[]]})
        for config in ("articles", "images")
    }
    calls = []
    monkeypatch.setattr(sys, "argv", ["publish_public.py", "--config", "articles,images", "-y"])
    monkeypatch.setattr(pp, "ensure_hf_token", lambda **k: "fake")
    monkeypatch.setattr(pp, "get_repo_revision", lambda *a, **k: "before")
    monkeypatch.setattr(pp, "load_dataset", lambda *a, **k: datasets[k["name"]])
    monkeypatch.setattr(pp, "push_datasets_verified", lambda prepared, **kwargs:
                        calls.append((prepared, kwargs)) or SimpleNamespace(after_revision="after"))
    pp.main()
    assert len(calls) == 1
    assert set(calls[0][0]) == {"articles", "images"}
    assert calls[0][1]["mode"] == "public_projection"


def test_main_later_invalid_subset_cannot_publish_earlier_subset(monkeypatch):
    import sys
    from datasets import Dataset
    good = Dataset.from_dict({"o:id": ["1"], "item_is_public": [True], "private_fields": [[]]})
    bad = Dataset.from_dict({"o:id": ["2"], "title": ["legacy visibility missing"]})
    monkeypatch.setattr(sys, "argv", ["publish_public.py", "--config", "articles,images", "-y"])
    monkeypatch.setattr(pp, "ensure_hf_token", lambda **k: "fake")
    monkeypatch.setattr(pp, "get_repo_revision", lambda *a, **k: "before")
    monkeypatch.setattr(pp, "load_dataset", lambda *a, **k: good if k["name"] == "articles" else bad)
    calls = []
    monkeypatch.setattr(pp, "push_datasets_verified", lambda *a, **k: calls.append(k))
    with pytest.raises(SystemExit):
        pp.main()
    assert not calls


@pytest.mark.parametrize("generation_number", [1, 2])
def test_private_annotation_masks_only_its_generation_and_dimension(generation_number):
    from iwac_common.sentiment_panel import generation, consensus_columns
    from iwac_common.write_policy import prepare_public_projection

    model = generation(generation_number)[0]
    names = consensus_columns(generation_number)
    other = consensus_columns(1 if generation_number == 2 else 2)
    frame = pd.DataFrame({
        "o:id": ["1"], "item_is_public": [True], "private_fields": [[model.column("polarite")]],
        model.column("polarite"): ["private vote"],
        names["consensus_polarite"]: ["private-derived consensus"],
        names["sentiment_disagreement"]: ["polarite"],
        names["instrument_id"]: ["instrument"],
        names["consensus_centralite"]: ["public centrality"],
        other["consensus_polarite"]: ["independent historical consensus"],
        "consensus_polarite": ["legacy unknown generation"],
        "sentiment_disagreement": ["polarite"],
    })
    result, _ = prepare_public_projection(frame, "articles")
    for column in (model.column("polarite"), names["consensus_polarite"],
                   names["sentiment_disagreement"], names["instrument_id"],
                   "consensus_polarite", "sentiment_disagreement"):
        assert result.iloc[0][column] == ""
    assert result.iloc[0][names["consensus_centralite"]] == "public centrality"
    assert result.iloc[0][other["consensus_polarite"]] == "independent historical consensus"


def test_gateway_policy_rejects_unmasked_consensus_of_private_votes():
    from datasets import Dataset
    from iwac_common.sentiment_panel import generation, consensus_columns
    from iwac_common.write_policy import PublicationPolicyError, validate_public_projection

    source = generation(2)[0].column("polarite")
    consensus = consensus_columns(2)["consensus_polarite"]
    ds = Dataset.from_dict({
        "o:id": ["1"], "item_is_public": [True], "private_fields": [[source]],
        source: [""], consensus: ["still discloses private votes"],
    })
    with pytest.raises(PublicationPolicyError, match="Private value"):
        validate_public_projection(ds, "articles")
