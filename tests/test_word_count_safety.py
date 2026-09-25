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
        output_types={"nb_mots": Value("int64")},
    )


def test_references_are_counted_from_ocr_like_every_subset():
    """The private repo holds references' full text; no Omeka fetch needed."""
    assert "references" in wc.WORD_COUNT_SUBSETS
    assert not hasattr(wc, "ReferenceContentClient")


def test_missing_mode_preserves_existing_counts():
    ds = Dataset.from_dict({
        "o:id": ["1", "2"], "OCR": ["un deux trois", "l'islam et"], "nb_mots": [7, None],
    })
    assert _count(ds, "missing")["nb_mots"][:] == [7, 2]


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
    assert wc.count_words("l'islam") == count_words("l'islam") == 1
