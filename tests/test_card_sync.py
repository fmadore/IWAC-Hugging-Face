"""The post-push card guard.

What this protects: ``push_to_hub`` refreshes a config's byte sizes but not its
``features`` list, so a schema change leaves the card declaring the old columns
and ``load_dataset`` raises ``CastError`` — the subset becomes unloadable for
every consumer, including this pipeline's own next run. It happened twice on
2026-08-06, on the private mirror and then on the public citable dataset.

The guard has to *repair*, not merely abort: by the time it runs the push has
already landed, so refusing to continue would leave the dataset broken.
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
    """Serve a canned card + parquet schema; capture any card upload."""
    state = {"uploads": [], "card": None, "schema": None}

    def _install(card_text: str, schema: pa.Schema, *, repair_writes=True):
        state["card"] = card_text
        state["schema"] = schema

        class FakeApi:
            def __init__(self, *a, **k):
                pass

            def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id,
                            repo_type, commit_message):
                text = path_or_fileobj.decode("utf-8")
                state["uploads"].append(text)
                if repair_writes:  # the Hub now serves what we wrote
                    state["card"] = text

        monkeypatch.setattr(card_sync, "HfApi", FakeApi)
        monkeypatch.setattr(
            card_sync.DatasetCard, "load",
            classmethod(lambda cls, *a, **k: DatasetCard(state["card"])),
        )
        monkeypatch.setattr(
            card_sync, "_parquet_schema", lambda *a, **k: state["schema"]
        )
        monkeypatch.setattr(card_sync, "resolve_hf_token", lambda *a, **k: "tok")
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
    def test_no_upload_when_already_correct(self, hub):
        st = hub(_card(articles_cols=["o:id", "title"]), _schema(["o:id", "title"]))
        assert sync_card_features("repo", "articles") is True
        assert st["uploads"] == []

    def test_large_string_parquet_is_not_a_mismatch(self, hub):
        """The real parquet is large_string while the card says string; that must
        read as a match, not trigger a repair on every single push."""
        st = hub(_card(articles_cols=["o:id", "title"]),
                 _schema(["o:id", "title"], pa.large_string()))
        assert sync_card_features("repo", "articles") is True
        assert st["uploads"] == []

    def test_card_without_dataset_info_passes(self, hub):
        st = hub(_card(articles_cols=[], with_info=False), _schema(["o:id"]))
        assert sync_card_features("repo", "articles") is True
        assert st["uploads"] == []


class TestRepair:
    def test_stale_card_is_repaired_and_reports_false(self, hub):
        st = hub(_card(articles_cols=["o:id", "title"]),
                 _schema(["o:id", "title", SENTIMENT]))
        assert sync_card_features("repo", "articles") is False
        assert len(st["uploads"]) == 1
        declared = [f["name"] for f in
                    DatasetCard(st["uploads"][0]).data["dataset_info"][0]["features"]]
        assert declared == ["o:id", "title", SENTIMENT]

    def test_repair_is_idempotent(self, hub):
        """A second run right after must be a no-op, not another commit."""
        st = hub(_card(articles_cols=["o:id"]), _schema(["o:id", SENTIMENT]))
        assert sync_card_features("repo", "articles") is False
        assert sync_card_features("repo", "articles") is True
        assert len(st["uploads"]) == 1

    def test_other_configs_and_prose_survive(self, hub):
        st = hub(_card(articles_cols=["o:id"]), _schema(["o:id", SENTIMENT]))
        sync_card_features("repo", "articles")
        written = DatasetCard(st["uploads"][0])
        index_entry = written.data["dataset_info"][1]
        assert index_entry["config_name"] == "index"
        assert [f["name"] for f in index_entry["features"]] == ["o:id"]
        assert "Prose that must survive." in written.text
        # Sizes belong to push_to_hub; the guard must not invent new ones.
        articles = written.data["dataset_info"][0]
        assert articles["splits"][0]["num_examples"] == 2
        assert articles["download_size"] == 50

    def test_dropped_column_is_also_repaired(self, hub):
        st = hub(_card(articles_cols=["o:id", "gone"]), _schema(["o:id"]))
        assert sync_card_features("repo", "articles") is False
        declared = [f["name"] for f in
                    DatasetCard(st["uploads"][0]).data["dataset_info"][0]["features"]]
        assert declared == ["o:id"]

    def test_reordering_is_repaired(self, hub):
        """Column order is part of the declared schema; a silent reorder still
        needs the card rewritten so declared and actual agree exactly."""
        st = hub(_card(articles_cols=["title", "o:id"]), _schema(["o:id", "title"]))
        assert sync_card_features("repo", "articles") is False
        declared = [f["name"] for f in
                    DatasetCard(st["uploads"][0]).data["dataset_info"][0]["features"]]
        assert declared == ["o:id", "title"]


class TestFailureModes:
    def test_repair_false_raises_without_writing(self, hub):
        st = hub(_card(articles_cols=["o:id"]), _schema(["o:id", SENTIMENT]))
        with pytest.raises(CardSchemaError, match="declares 1 columns"):
            sync_card_features("repo", "articles", repair=False)
        assert st["uploads"] == []

    def test_raises_when_the_rewrite_does_not_take(self, hub):
        """If the Hub still serves the stale card after the write, the subset is
        still unloadable — that must not be reported as success."""
        st = hub(_card(articles_cols=["o:id"]), _schema(["o:id", SENTIMENT]),
                 repair_writes=False)
        with pytest.raises(CardSchemaError, match="still does not match"):
            sync_card_features("repo", "articles")
        assert len(st["uploads"]) == 1

    def test_expected_columns_guard_catches_the_wrong_frame(self, hub):
        """The parquet on the Hub not matching what the caller pushed means
        something else wrote the repo; never paper over that with a card edit."""
        st = hub(_card(articles_cols=["o:id"]), _schema(["o:id", "surprise"]))
        with pytest.raises(CardSchemaError, match="does not match the frame"):
            sync_card_features("repo", "articles", expected_columns=["o:id"])
        assert st["uploads"] == []

    def test_missing_parquet_raises(self, monkeypatch):
        monkeypatch.setattr(card_sync, "resolve_hf_token", lambda *a, **k: "tok")

        class EmptyFs:
            def __init__(self, *a, **k):
                pass

            def glob(self, *a, **k):
                return []

        monkeypatch.setattr(card_sync, "HfFileSystem", EmptyFs)
        with pytest.raises(CardSchemaError, match="No parquet found"):
            sync_card_features("repo", "articles")


class TestWiredIntoBothPushers:
    """Both writers to the Hub must call the guard, or the hazard returns."""

    def test_upload_runner_calls_it(self):
        import inspect

        import iwac_common.upload_runner as runner

        src = inspect.getsource(runner)
        assert "sync_card_features(" in src
        assert "CardSchemaError" in src

    def test_publish_public_calls_it(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent
               / "post-processing" / "publish_public.py").read_text(encoding="utf-8")
        assert "sync_card_features(" in src
        assert "CardSchemaError" in src
