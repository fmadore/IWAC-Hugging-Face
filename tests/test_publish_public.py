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

    def test_approved_column_persisted(self, monkeypatch, tmp_path):
        f = self._patch_allowlist(monkeypatch, tmp_path, {"articles": ["o:id"]})
        df = pd.DataFrame({"o:id": ["1"], "new_metric": [0.9]})
        approved = pp.check_column_allowlist("articles", df, approve={"new_metric"})
        assert approved == ["new_metric"]
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
