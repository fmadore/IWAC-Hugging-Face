"""Lemmatiser vocabulary policy, cache checkpoints and stage wiring (offline).

Uses ``spacy.blank`` pipelines: they share the language's tokenizer rules
with the ``*_core_news_lg`` / ``*_core_web_lg`` models, have no lemmatizer
(so the lemma falls back to the surface form) and need no download.
"""
from __future__ import annotations

import io
import sqlite3

import pytest
import spacy
from datasets import Dataset
from rich.console import Console

import lemmatize_update_hf as lemma


@pytest.fixture(scope="module")
def fr():
    return spacy.blank("fr")


@pytest.fixture(scope="module")
def en():
    return spacy.blank("en")


def lemmas(nlp, text):
    return [value for value, _ in lemma.lemmatise_one(nlp, lemma.normalize(text))]


@pytest.mark.parametrize("text, expected", [
    ("Côte-d'Ivoire", ["côte-d'ivoire"]),
    ("sous-préfet", ["sous-préfet"]),
    ("peut-être", ["peut-être"]),
    ("aujourd'hui", ["aujourd'hui"]),
    ("Jama'at", ["jama'at"]),
    ("da'wa", ["da'wa"]),
    ("Qur'an", ["qur'an"]),
    ("Jama’at Qur’an", ["jama'at", "qur'an"]),  # typographic apostrophes
    ("El-Hadj", ["el", "hadj"]),
    ("Al-Azhar", ["al", "azhar"]),
    ("l'islam", ["islam"]),
    ("L’islam", ["islam"]),
    ("qu'il", ["il"]),
    ("jusqu'à", ["à"]),
    ("Aïd-el-Kébir", ["aïd", "el", "kébir"]),
    ("Ahl-us-Sunna", ["ahl", "us", "sunna"]),
    ("Shi'a 'Ulama'", ["shi'a", "ulama"]),
])
def test_french_domain_vocabulary_survives(fr, text, expected):
    assert lemmas(fr, text) == expected


@pytest.mark.parametrize("text", ["1995", "12,5", "3e", "...", "--", "M.", "U.S.A.", "« »", "l' d'"])
def test_numbers_punctuation_abbreviations_and_bare_clitics_are_dropped(fr, text):
    assert lemmas(fr, text) == []


def test_english_keeps_transliterations_and_drops_clitics(en):
    assert lemmas(en, "Jama'at da'wa Qur'an") == ["jama'at", "da'wa", "qur'an"]
    assert lemmas(en, "the imam's sermon wasn't long") == ["the", "imam", "sermon", "was", "long"]


def test_no_lemma_contains_whitespace_and_stop_flags_survive(fr):
    pairs = lemma.lemmatise_one(fr, lemma.normalize("Il est à Côte-d'Ivoire avec la Jama'at"))
    assert all(value.split() == [value] for value, _ in pairs)
    assert dict(pairs)["il"] is True and dict(pairs)["jama'at"] is False


def test_batched_path_matches_the_reference_on_real_tokens(fr, monkeypatch):
    monkeypatch.setattr(lemma, "SPACY_MAX_CHUNK_CHARS", 20)
    texts = [lemma.normalize(t) for t in ["", "La Jama'at de Côte-d'Ivoire et le da'wa", "", "qu'il vienne"]]
    assert list(lemma.lemmatise_many(fr, iter(texts), batch_size=2)) == [
        lemma.lemmatise_one(fr, text) for text in texts
    ]


def test_logic_version_bumped():
    assert lemma.LEMMA_LOGIC_VERSION == 3
    assert "_v3" in lemma.cache_fingerprint("fr_core_news_lg", "3.8.0")


# --- lemmatise_dataset ----------------------------------------------------

def _dataset(texts, languages=None):
    return Dataset.from_dict({
        "o:id": [str(i) for i in range(len(texts))], "OCR": list(texts),
        "language": languages or ["Français"] * len(texts),
    })


def _run(ds, nlp, cache_file, **kwargs):
    kwargs.setdefault("process_choice", "missing")
    return lemma.lemmatise_dataset(
        ds, nlp, text_col="OCR", lemma_col="lemma_text", clean_col="lemma_nostop",
        cache_file=cache_file, language_filter="Français", **kwargs,
    )


def test_returns_none_when_nothing_changes(fr, tmp_path, monkeypatch):
    monkeypatch.setattr(lemma, "console", Console(file=io.StringIO()))
    first = _run(_dataset(["La Jama'at prêche", ""]), fr, tmp_path / "c.sqlite3")
    assert first["lemma_text"][:] == ["la jama'at prêche", ""]
    assert _run(first, fr, tmp_path / "d.sqlite3") is None
    assert _run(first, fr, tmp_path / "e.sqlite3", process_choice="all") is None
    assert _run(first, fr, tmp_path / "f.sqlite3", process_choice="empty") is None  # deprecated alias


def test_unknown_process_choice_is_rejected(fr, tmp_path, monkeypatch):
    monkeypatch.setattr(lemma, "console", Console(file=io.StringIO()))
    with pytest.raises(ValueError, match="process_choice"):
        _run(_dataset(["texte"]), fr, tmp_path / "c.sqlite3", process_choice="sometimes")


def test_dry_run_reports_without_running_spacy_or_writing_the_cache(tmp_path, monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(lemma, "console", Console(file=buffer, width=120))

    class Nlp:
        meta = {"lang": "fr", "name": "test", "version": "1"}

        def pipe(self, *args, **kwargs):
            raise AssertionError("a dry run must not lemmatise")

    ds = _dataset(["un", "deux", "", "three"], ["Français", "Français", "Français", "Anglais"])
    assert _run(ds, Nlp(), tmp_path / "c.sqlite3", dry_run=True) is None
    assert not (tmp_path / "c.sqlite3").exists()
    rows = {line.split("│")[1].strip(): line.split("│")[2].strip()
            for line in buffer.getvalue().splitlines() if line.count("│") == 3}
    assert rows["Would be lemmatised"] == "2"
    assert rows["Blank text (empty lemmas)"] == "1"
    assert rows["Other language (kept as-is)"] == "1"


def test_sqlite_checkpoints_write_only_new_entries(fr, tmp_path, monkeypatch):
    monkeypatch.setattr(lemma, "console", Console(file=io.StringIO()))
    monkeypatch.setattr(lemma, "CHECKPOINT_EVERY", 2)
    written = []
    real_save = lemma.save_cache

    def spy(entries, path):
        written.append(sorted(entries))
        real_save(entries, path)

    monkeypatch.setattr(lemma, "save_cache", spy)
    cache_file = tmp_path / "lemmas.sqlite3"
    _run(_dataset(["un", "deux", "trois", "quatre", "cinq"]), fr, cache_file)
    assert written == [["0", "1"], ["2", "3"], ["4"]]
    with sqlite3.connect(cache_file) as connection:
        assert connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 5


def test_an_interrupted_run_resumes_from_the_sqlite_cache(fr, tmp_path, monkeypatch):
    monkeypatch.setattr(lemma, "console", Console(file=io.StringIO()))
    cache_file = tmp_path / "lemmas.sqlite3"
    ds = _dataset(["La da'wa", "Le Qur'an"])
    _run(ds, fr, cache_file)  # leaves the cache behind, as a failed push would

    class Exploding:
        meta = fr.meta

        def pipe(self, *args, **kwargs):
            raise AssertionError("every row should come from the cache")

    out = _run(ds, Exploding(), cache_file)
    assert out["lemma_text"][:] == ["la da'wa", "le qur'an"]


# --- main() wiring ----------------------------------------------------------

def _configure_main(monkeypatch, tmp_path, ds, argv):
    pushed = []
    monkeypatch.setattr(lemma, "console", Console(file=io.StringIO()))
    monkeypatch.setattr(lemma, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lemma, "ensure_hf_token", lambda **kwargs: "offline-placeholder")
    monkeypatch.setattr(lemma, "load_hub_dataset", lambda *a, **k: ds)
    monkeypatch.setattr(lemma, "load_spacy_model", lambda name: spacy.blank(name[:2]))
    monkeypatch.setattr(lemma, "push_dataset", lambda data, **k: pushed.append((data, k)) or True)
    monkeypatch.setattr("sys.argv", ["lemma", *argv])
    return pushed


def test_commit_message_names_language_and_model_and_cache_is_removed(monkeypatch, tmp_path):
    ds = _dataset(["The Jama'at met", "La réunion"], ["Anglais", "Français"])
    pushed = _configure_main(monkeypatch, tmp_path, ds, ["--config", "references", "--language", "Anglais", "-y"])
    assert lemma.main() == 0
    data, kwargs = pushed[0]
    message = kwargs["commit_message"]
    assert "Anglais" in message and "en_core_web_lg" in message and "French" not in message
    assert f"logic v{lemma.LEMMA_LOGIC_VERSION}" in message
    assert data["lemma_text"][:] == ["the jama'at met", ""]
    assert not list(tmp_path.glob("*.sqlite3"))


def test_main_skips_the_push_when_nothing_changes(monkeypatch, tmp_path):
    ds = _dataset(["La Jama'at"])
    pushed = _configure_main(monkeypatch, tmp_path, ds, ["--config", "articles", "-y"])
    assert lemma.main() == 0
    pushed_again = _configure_main(monkeypatch, tmp_path, pushed[0][0], ["--config", "articles", "-y"])
    assert lemma.main() == 0
    assert pushed_again == []


def test_dry_run_never_pushes(monkeypatch, tmp_path):
    pushed = _configure_main(monkeypatch, tmp_path, _dataset(["texte"]), ["--config", "articles", "--dry-run", "-y"])
    assert lemma.main() == 0
    assert pushed == []
    assert not list(tmp_path.iterdir())


def test_deprecated_mode_alias_and_config_choices(monkeypatch, tmp_path):
    pushed = _configure_main(monkeypatch, tmp_path, _dataset(["texte"]), ["--config", "articles", "--mode", "empty"])
    assert lemma.main() == 0
    assert "mode: missing" in pushed[0][1]["commit_message"]
    _configure_main(monkeypatch, tmp_path, _dataset(["texte"]), ["--config", "index", "-y"])
    with pytest.raises(SystemExit):
        lemma.main()


def test_capital_n_m_names_are_not_clitics():
    import spacy

    from lemmatize_update_hf import token_lemmas

    nlp = spacy.blank("fr")
    lemmas = [lemma for lemma, _ in token_lemmas(nlp("M. N'Guessan arrive à N'Djamena, il n'est pas M'Bow."))]
    assert "n'guessan" in lemmas and "n'djamena" in lemmas and "m'bow" in lemmas
    assert "n'est" not in lemmas and "est" in lemmas
