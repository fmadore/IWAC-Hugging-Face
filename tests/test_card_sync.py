"""The read-only post-commit card check.

What this protects: a card whose declared ``features`` disagree with the
parquet makes ``load_dataset`` raise ``CastError`` — the subset becomes
unloadable for every consumer, including this pipeline's own next run. The
gateway commits card and parquet atomically; this check verifies the result at
the committed revision and raises (never writes) on any mismatch.
"""

import pyarrow as pa
import pytest
from huggingface_hub import DatasetCard

import iwac_common.card_sync as card_sync
from iwac_common.card_sync import CardSchemaError, sync_card_features

SENTIMENT = "gpt_5_6_luna_polarite"


def _card(*, articles_cols, with_info=True) -> str:
    """A card in the shape these repos really use: multi-config dataset_info."""
    if not with_info:
        return "---\nlicense: cc-by-nc-sa-4.0\nviewer: false\n---\n\n# Title\n\nProse.\n"
    feats = "\n".join(f"  - name: {c}\n    dtype: string" for c in articles_cols)
    return (
        "---\n"
        "license: cc-by-nc-sa-4.0\n"
        "dataset_info:\n"
        "- config_name: articles\n"
        "  features:\n"
        f"{feats}\n"
        "  splits:\n"
        "  - name: train\n"
        "    num_bytes: 100\n"
        "    num_examples: 2\n"
        "  download_size: 50\n"
        "  dataset_size: 100\n"
        "- config_name: index\n"
        "  features:\n"
        "  - name: o:id\n"
        "    dtype: int64\n"
        "  splits:\n"
        "  - name: train\n"
        "    num_bytes: 10\n"
        "    num_examples: 1\n"
        "  download_size: 5\n"
        "  dataset_size: 10\n"
        "---\n"
        "\n# Islam West Africa Collection\n\nProse that must survive.\n"
    )


@pytest.fixture
def hub(monkeypatch):
    """Serve a canned card + parquet schema."""
    state = {"card": None, "schema": None}

    def _install(card_text: str, schema: pa.Schema):
        state["card"] = card_text
        state["schema"] = card_sync._normalize_schema(schema)
        monkeypatch.setattr(
            card_sync, "_load_card",
            lambda *a, **k: DatasetCard(state["card"]),
        )
        monkeypatch.setattr(
            card_sync, "_parquet_schema", lambda *a, **k: state["schema"]
        )
        monkeypatch.setattr(card_sync, "resolve_hf_token", lambda *a, **k: "tok")
        monkeypatch.setattr(card_sync, "get_repo_revision", lambda *a, **k: "before")
        return state

    return _install


def _schema(cols, string_type=pa.string()):
    return pa.schema([pa.field(c, string_type) for c in cols])


class TestNarrow:
    """large_string must normalise to string, or the guard rewrites the card on
    every push forever — a silent commit loop."""

    def test_large_string_becomes_string(self):
        assert card_sync._narrow(pa.large_string()) == pa.string()

    def test_recurses_through_lists(self):
        assert card_sync._narrow(pa.large_list(pa.large_string())) == pa.list_(pa.string())
        assert card_sync._narrow(pa.list_(pa.float64())) == pa.list_(pa.float64())

    def test_leaves_other_types_alone(self):
        for t in (pa.int64(), pa.float64(), pa.bool_(), pa.string()):
            assert card_sync._narrow(t) == t


class TestMatchingCard:
    def test_passes_when_already_correct(self, hub):
        hub(_card(articles_cols=["o:id", "title"]), _schema(["o:id", "title"]))
        sync_card_features("repo", "articles")

    def test_large_string_parquet_is_not_a_mismatch(self, hub):
        """The real parquet is large_string while the card says string; that must
        read as a match, not fail verification on every single push."""
        hub(_card(articles_cols=["o:id", "title"]),
            _schema(["o:id", "title"], pa.large_string()))
        sync_card_features("repo", "articles")

    def test_card_without_dataset_info_passes(self, hub):
        hub(_card(articles_cols=[], with_info=False), _schema(["o:id"]))
        sync_card_features("repo", "articles")

    def test_supplied_footer_schema_skips_the_shard_read(self, hub, monkeypatch):
        hub(_card(articles_cols=["o:id", "title"]), _schema(["o:id"]))

        def unexpected(*args, **kwargs):
            pytest.fail("the caller already read the footers")

        monkeypatch.setattr(card_sync, "_parquet_schema", unexpected)
        sync_card_features("repo", "articles",
                           parquet_schema=_schema(["o:id", "title"], pa.large_string()))


class TestMismatchFailsClosed:
    """The check never writes: any disagreement raises, card left untouched."""

    @pytest.mark.parametrize("card_cols, parquet", [
        (["o:id", "title"], _schema(["o:id", "title", SENTIMENT])),
        (["o:id", "gone"], _schema(["o:id"])),
        (["title", "o:id"], _schema(["o:id", "title"])),  # order is schema too
        (["o:id", "nb_mots"], pa.schema([("o:id", pa.string()), ("nb_mots", pa.int64())])),
        (["o:id", "embedding_OCR"],
         pa.schema([("o:id", pa.string()), ("embedding_OCR", pa.list_(pa.float32()))])),
    ])
    def test_mismatch_raises_without_writing(self, hub, card_cols, parquet):
        card = _card(articles_cols=card_cols)
        state = hub(card, parquet)
        with pytest.raises(CardSchemaError, match=f"declares {len(card_cols)} columns"):
            sync_card_features("repo", "articles")
        assert state["card"] == card

    def test_check_has_no_repair_parameter(self):
        import inspect

        assert "repair" not in inspect.signature(sync_card_features).parameters
        assert not hasattr(card_sync, "HfApi")


class TestFailureModes:
    def test_expected_columns_guard_catches_the_wrong_frame(self, hub):
        """The parquet on the Hub not matching what the caller pushed means
        something else wrote the repo."""
        hub(_card(articles_cols=["o:id"]), _schema(["o:id", "surprise"]))
        with pytest.raises(CardSchemaError, match="does not match the frame"):
            sync_card_features("repo", "articles", expected_columns=["o:id"])

    def test_missing_parquet_raises(self, monkeypatch):
        import iwac_common.hub as hub_module

        monkeypatch.setattr(card_sync, "resolve_hf_token", lambda *a, **k: "tok")
        monkeypatch.setattr(card_sync, "get_repo_revision", lambda *a, **k: "before")

        class EmptyFs:
            def __init__(self, *a, **k):
                pass

            def glob(self, *a, **k):
                return []

        monkeypatch.setattr(hub_module, "HfFileSystem", EmptyFs)
        with pytest.raises(CardSchemaError, match="No parquet found"):
            sync_card_features("repo", "articles")


class TestWiredIntoSingleWriter:
    """All scripts delegate to the one gateway that commits and verifies cards."""

    def test_verified_writer_calls_guard(self):
        import inspect

        import iwac_common.hub as hub

        src = inspect.getsource(hub._verify_committed_dataset)
        assert "sync_card_features(" in src
        assert "expected_schema=" in src
        assert "revision=revision" in src

    def test_no_script_bypasses_verified_writer(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        offenders = []
        code_dirs = [
            "iwac_common", "articles", "audiovisual", "document", "images",
            "index", "islamic-publications", "reference", "post-processing",
            "analyses", "data",
        ]
        candidates = list(root.glob("*.py"))
        for directory in code_dirs:
            candidates.extend((root / directory).rglob("*.py"))
        for path in candidates:
            if path == root / "iwac_common" / "hub.py":
                continue
            if ".push_to_hub(" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(root)))
        assert offenders == []


def test_expected_types_are_checked_even_if_names_match(hub):
    hub(_card(articles_cols=["o:id", "nb_mots"]),
        pa.schema([("o:id", pa.string()), ("nb_mots", pa.string())]))
    with pytest.raises(CardSchemaError, match="schema differs"):
        sync_card_features("repo", "articles", expected_schema=pa.schema([
            ("o:id", pa.string()), ("nb_mots", pa.int64())]))


def test_parquet_verification_reads_every_shard_at_pinned_revision(monkeypatch):
    import io
    import pyarrow.parquet as pq

    bodies = {}
    for name, values in (("one", [1]), ("two", ["changed type"])):
        output = io.BytesIO()
        pq.write_table(pa.table({"o:id": ["1"], "value": values}), output)
        bodies[name] = output.getvalue()

    class Files:
        def __init__(self, **kwargs):
            pass
        def glob(self, pattern):
            assert "repo@revision/articles" in pattern
            return ["one", "two"]
        def open(self, path, mode):
            return io.BytesIO(bodies[path])

    import iwac_common.hub as hub_module

    monkeypatch.setattr(hub_module, "HfFileSystem", Files)
    with pytest.raises(CardSchemaError, match="Inconsistent"):
        card_sync._parquet_schema("repo", "articles", "token", "revision")
