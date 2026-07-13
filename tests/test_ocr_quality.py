"""Tests for the OCR quality metric (dictionary hit-rate via wordfreq)."""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oq = _load("post-processing/calculate_ocr_quality.py", "oq_under_test")

CLEAN_FR = (
    "la communauté musulmane organise chaque année une grande fête pour "
    "célébrer la fin du mois sacré et les fidèles se rassemblent dans la "
    "grande mosquée pour écouter le sermon avant de partager un repas"
)
CLEAN_EN = (
    "the muslim community organizes every single year a large festival to "
    "celebrate the end of the sacred month and the faithful gather together "
    "inside the great mosque to listen quietly before sharing their meal"
)
# Alphabetic garbage (isalpha, len >= 3, lowercase) mixed with a few real
# words — enough scoreable tokens, but mostly lexicon misses.
GARBLED = (
    "l3 pr0ph3te mxqzt vwxrq zzqpt bnkrw qwzxv trvkp mmzqr xkzvb wqrtz "
    "pzvxk rqwzt vbxqm zktrw qxvzp wrtkz mosquée xqzwv tkrpz vmxqw zrqtx "
    "pwvkz imam qztrx vkwzp"
)


class TestScoreOcrQuality:
    def test_clean_french_scores_high(self):
        assert oq.score_ocr_quality(CLEAN_FR, "fr") > 0.9

    def test_garbled_ocr_scores_low(self):
        score = oq.score_ocr_quality(GARBLED, "fr")
        assert score is not None
        assert score < 0.5

    def test_clean_beats_garbled(self):
        assert oq.score_ocr_quality(CLEAN_FR, "fr") > oq.score_ocr_quality(GARBLED, "fr")

    def test_short_text_returns_none(self):
        assert oq.score_ocr_quality("le chat mange la souris", "fr") is None

    def test_empty_and_none(self):
        assert oq.score_ocr_quality("", "fr") is None
        assert oq.score_ocr_quality("   ", "fr") is None
        assert oq.score_ocr_quality(None, "fr") is None

    def test_english_row_scored_with_en_lexicon(self):
        assert oq.score_ocr_quality(CLEAN_EN, "en") > 0.9

    def test_digits_and_short_tokens_not_counted(self):
        # Same scoreable tokens → same score, regardless of digit noise.
        noisy = CLEAN_FR + " 12 345 1987 ab le du 9x"
        assert oq.score_ocr_quality(noisy, "fr") == oq.score_ocr_quality(CLEAN_FR, "fr")

    def test_range_is_zero_one(self):
        score = oq.score_ocr_quality(CLEAN_FR, "fr")
        assert 0.0 <= score <= 1.0


class TestProperNounHeuristic:
    def test_mid_sentence_capitalized_name_skipped(self):
        # "Zorbatello" is not in any lexicon, always capitalized, appears
        # mid-sentence → skipped, so it must not lower the score.
        with_name = CLEAN_FR + " et le président Zorbatello salue Zorbatello"
        assert oq.score_ocr_quality(with_name, "fr") == oq.score_ocr_quality(CLEAN_FR, "fr")

    def test_sentence_initial_capital_not_proper_noun(self):
        # "Mosquée" capitalized only at sentence start → not flagged.
        assert "mosquée" not in oq.proper_noun_candidates("Il prie. Mosquée ouverte.")

    def test_lowercase_occurrence_disqualifies(self):
        # Seen lowercase elsewhere → not a proper noun even when capitalized
        # mid-sentence.
        text = "la grande Mosquée et la mosquée du vendredi"
        assert "mosquée" not in oq.proper_noun_candidates(text)

    def test_candidates_detected(self):
        text = "le président Ouattara visite Ouattara encore"
        assert "ouattara" in oq.proper_noun_candidates(text)


class TestResolveLang:
    def test_default_fr(self):
        assert oq.resolve_lang(None) == "fr"
        assert oq.resolve_lang("") == "fr"
        assert oq.resolve_lang("Français") == "fr"

    def test_english_primary(self):
        assert oq.resolve_lang("Anglais") == "en"
        assert oq.resolve_lang("Anglais | Français") == "en"

    def test_first_listed_wins(self):
        assert oq.resolve_lang("Français | Anglais") == "fr"


class TestBatchFn:
    def test_update_mode_missing_keeps_existing(self):
        batch = {
            "OCR": [CLEAN_FR, CLEAN_FR],
            "language": ["Français", "Français"],
            "ocr_quality": [0.123, None],
        }
        out = oq.add_ocr_quality_batch(dict(batch), update_mode="missing")
        assert out["ocr_quality"][0] == 0.123  # untouched
        assert out["ocr_quality"][1] is not None and out["ocr_quality"][1] > 0.9

    def test_update_mode_all_recomputes(self):
        batch = {
            "OCR": [CLEAN_FR],
            "language": ["Français"],
            "ocr_quality": [0.123],
        }
        out = oq.add_ocr_quality_batch(dict(batch), update_mode="all")
        assert out["ocr_quality"][0] > 0.9

    def test_empty_batch_guard(self):
        out = oq.add_ocr_quality_batch({})
        assert out == {"ocr_quality": []}
