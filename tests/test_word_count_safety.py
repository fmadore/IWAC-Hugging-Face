"""nb_mots: one definition, one code path, no type drift."""

from datasets import Dataset, Value

import calculate_word_count as wc
from _common import map_with_progress
from iwac_common.text_utils import count_words


def _count(ds, update_mode):
    return map_with_progress(
        ds,
        lambda batch: wc.add_word_count_batch(
            batch, text_col="OCR", count_col="nb_mots", update_mode=update_mode
        ),
        output_types=wc.word_count_output_types(),
    )


def test_references_are_counted_from_ocr_like_every_subset():
    """The private repo holds references' full text; no Omeka fetch needed."""
    assert "references" in wc.WORD_COUNT_SUBSETS
    assert not hasattr(wc, "ReferenceContentClient")


def test_missing_mode_recomputes_counts_without_provenance():
    """A legacy count cannot prove which tokenizer produced it."""
    ds = Dataset.from_dict({
        "o:id": ["1", "2"], "OCR": ["un deux trois", "l'islam et"],
        "language": ["Français", "Français"], "nb_mots": [7, None],
    })
    out = _count(ds, "missing")
    assert out["nb_mots"][:] == [3, 3]
    assert out["nb_mots_config_hash"][:] == [wc.WORD_COUNT_CONFIG] * 2


def test_missing_mode_keeps_a_count_with_current_provenance():
    ds = Dataset.from_dict({
        "o:id": ["1"], "OCR": ["un deux trois"], "language": ["Français"], "nb_mots": [7],
    })
    first = _count(ds, "all")
    tampered = first.remove_columns("nb_mots").add_column("nb_mots", [99])
    assert _count(tampered, "missing")["nb_mots"][:] == [99]  # provenance matches: kept
    assert _count(tampered, "all")["nb_mots"][:] == [3]


def test_missing_mode_recounts_when_the_language_changes():
    ds = Dataset.from_dict({
        "o:id": ["1"], "OCR": ["l'église"], "language": ["Haoussa"],
    })
    first = _count(ds, "missing")
    assert first["nb_mots"][:] == [1]
    relabelled = first.remove_columns("language").add_column("language", ["Français"])
    assert _count(relabelled, "missing")["nb_mots"][:] == [2]


def test_missing_mode_recounts_after_a_tokenizer_bump(monkeypatch):
    ds = Dataset.from_dict({"o:id": ["1"], "OCR": ["un deux"], "language": ["Français"]})
    first = _count(ds, "missing")
    tampered = first.remove_columns("nb_mots").add_column("nb_mots", [99])
    monkeypatch.setattr(wc, "WORD_COUNT_CONFIG", "a-newer-tokenizer")
    assert _count(tampered, "missing")["nb_mots"][:] == [2]


def test_main_skips_a_push_that_changes_nothing(monkeypatch):
    import io

    from rich.console import Console

    pushed = []
    state = {"ds": Dataset.from_dict({"o:id": ["1"], "OCR": ["l'imam"], "language": ["Français"]})}
    monkeypatch.setattr(wc, "console", Console(file=io.StringIO()))
    monkeypatch.setattr(wc, "ensure_hf_token", lambda **kwargs: "offline-placeholder")
    monkeypatch.setattr(wc, "load_hub_dataset", lambda *a, **k: state["ds"])
    monkeypatch.setattr(wc, "push_dataset", lambda ds, **k: pushed.append((ds, k)) or True)
    monkeypatch.setattr("sys.argv", ["wc", "--config", "articles", "-y"])
    assert wc.main() == 0
    ds, kwargs = pushed[0]
    assert ds["nb_mots"][:] == [2]
    assert f"tokenizer v{wc.TOKENIZER_VERSION}" in kwargs["commit_message"]
    state["ds"] = ds
    assert wc.main() == 0
    assert len(pushed) == 1


def test_word_count_provenance_is_private():
    from iwac_common.enrichment import PRIVATE_PROVENANCE_COLUMNS

    assert set(wc.OUTPUT_COLUMNS[1:]) <= PRIVATE_PROVENANCE_COLUMNS


def test_count_agrees_with_the_mattr_token_stream(monkeypatch):
    """nb_mots and the MATTR stage tokenize the same row the same way."""
    import calculate_lexical_richness as lex

    texts = ["l'imam d'Abidjan qu'on attendait", "Jama'at da'wa l'umma"]
    languages = ["Français|Anglais", "Haoussa"]
    batch = wc.add_word_count_batch(
        {"OCR": list(texts), "language": list(languages)}, text_col="OCR", count_col="nb_mots",
    )
    seen = []
    original = lex.tokenize_words

    def spy(text, language):
        tokens = original(text, language=language)
        seen.append(len(tokens))
        return tokens

    monkeypatch.setattr(lex, "tokenize_words", spy)
    lex.compute_metrics_batch(
        {"OCR": list(texts), "language": list(languages)}, text_col="OCR",
        richness_col="Richesse_Lexicale_OCR", readability_col="Lisibilite_OCR",
        update_mode="all", window_size=1, error_counter={},
    )
    assert batch["nb_mots"] == seen == [7, 3]


def test_count_column_stays_int64_with_nulls_elsewhere():
    ds = Dataset.from_dict({
        "o:id": ["1", "2"], "OCR": ["a b", ""], "nb_pages": [3, None],
    })
    out = _count(ds, "all")
    assert out.features["nb_mots"] == Value("int64")
    assert out.features["nb_pages"] == Value("int64")


def test_mapper_and_post_processing_share_one_definition():
    """The upload mappers for references and audiovisual import the same
    counter the post-processing script uses, so the value cannot flip."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for rel in ("reference/upload_reference_hf.py", "audiovisual/upload_audiovisual_hf.py"):
        spec = importlib.util.spec_from_file_location(f"_wc_{Path(rel).stem}", root / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assert module.count_words is count_words
    assert wc.count_words is count_words
    assert count_words("l'islam") == 2
