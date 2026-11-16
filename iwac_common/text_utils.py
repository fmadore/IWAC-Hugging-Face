"""Shared tokenization helpers for the IWAC pipeline.

Two tokenizers with distinct purposes:

- :func:`simple_tokenize` — the canonical whitespace tokenizer for text that
  is ALREADY lemmatized (``lemma_nostop``). Used by LDA training, LDA
  prediction, ``analyses/topic_prevalence.py`` and ``analyses/keyness_bursts.py``
  so all four stay in lockstep (they had drifted: keyness skipped
  lowercasing and case-split tokens the LDA side merged).

- :func:`tokenize_words` — a language-aware word tokenizer for RAW text
  (OCR), used by ``nb_mots`` (:func:`count_words`) and by the MATTR token
  stream, which therefore agree by construction for the same row language.
"""

from __future__ import annotations

import re
import unicodedata
from typing import AbstractSet, Iterable, List

# Bump whenever tokenize_words() output changes for identical input: it is
# part of the persisted provenance of nb_mots and Richesse_Lexicale_OCR.
# 3 = elided clitics are tokens of their own (``l'église`` = 2, like
# ``la mosquée``), the split applies only to rows whose language includes
# French, and apostrophes inside a word keep it whole (``da'wa``).
TOKENIZER_VERSION = 3

#: Language labels / codes under which the French elision split applies.
FRENCH_LANGUAGE_LABELS = frozenset({"Français", "fr", "fra"})

# Typographic apostrophes normalised to ASCII. The transliteration letters
# U+02BC, U+02BE and U+02BF are letters and stay inside the word as-is.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "‛": "'",
                              "′": "'", "＇": "'"})
_ELIDED_CLITIC_RE = re.compile(r"(qu|jusqu|lorsqu|puisqu|quoiqu|quelqu|[ldjmnstc])'(?=.)")
# Lexicalised forms whose apostrophe follows a clitic-shaped head but which
# are one word (``aujourd'hui`` never matches: its head is ``aujourd``).
_LEXICALISED_APOSTROPHE_WORDS = frozenset({"quelqu'un", "quelqu'une"})


def simple_tokenize(
    text: str,
    stopwords: AbstractSet[str] = frozenset(),
    min_len: int = 2,
) -> List[str]:
    """Lowercase, whitespace-split, drop tokens shorter than ``min_len`` or
    in ``stopwords``. The single source of truth for tokenizing
    ``lemma_nostop`` across LDA training/prediction and the analyses.
    """
    if not text:
        return []
    return [
        t
        for t in str(text).lower().split()
        if len(t) >= min_len and t not in stopwords
    ]


def language_labels(value: object) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in str(value or "").split("|") if part.strip()))


def applies_french_elision(language: object) -> bool:
    """Whether the French elision split applies to a row.

    ``language`` is the row's ``language`` value (pipe-joined labels), a list
    of labels, or ``None``. The rule is "the row's languages include French",
    so a ``Français|Anglais`` row is split like a French one and a Hausa,
    Ewé or unlabelled row is not.
    """
    labels = language if isinstance(language, (list, tuple)) else language_labels(language)
    return any(label in FRENCH_LANGUAGE_LABELS for label in labels)


def _iter_words(text: str) -> Iterable[str]:
    """Yield the words of lowercased, apostrophe-normalised ``text``.

    A word is a run of letters, digits, ``_`` and (inside a word) combining
    marks — Python's ``\\w`` excludes those marks, which would split Arabic and
    African-language words at every vowel or tone mark. An apostrophe stays
    inside a word when letters follow it (``da'wa``, ``jama'at``,
    ``aujourd'hui``); one at a word edge is punctuation. Hyphens separate.
    """
    current: list[str] = []
    for index, char in enumerate(text):
        if char.isalnum() or char == "_" or (current and unicodedata.category(char).startswith("M")):
            current.append(char)
            continue
        following = text[index + 1] if index + 1 < len(text) else ""
        if char == "'" and current and following.isalnum():
            current.append(char)
            continue
        if current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


_NAME_PREFIX_RE = re.compile(r"(?<!\w)[NM]'[^\W\d_a-zà-ÿ][^\W\d_]*")


def _split_elision(word: str) -> List[str]:
    """``l'église`` → ``["l'", "église"]``; ``jusqu'à`` → ``["jusqu'", "à"]``."""
    parts: List[str] = []
    while word not in _LEXICALISED_APOSTROPHE_WORDS:
        match = _ELIDED_CLITIC_RE.match(word)
        if not match:
            break
        parts.append(word[:match.end()])
        word = word[match.end():]
    parts.append(word)
    return parts


def tokenize_words(text: str, language: object = "Français") -> List[str]:
    """Tokenize raw text into lowercase words (``TOKENIZER_VERSION`` 3).

    For a row whose language includes French (:func:`applies_french_elision`)
    an elided clitic is a token of its own, so ``l'église`` → ``["l'",
    "église"]`` counts two words exactly like ``la mosquée``, and ``qu'il`` →
    ``["qu'", "il"]``. ``aujourd'hui`` and ``quelqu'un`` stay whole. In every
    other language an apostrophe inside a word keeps it whole (``da'wa``).
    Transliterations such as ``Jama'at`` / ``Qur'an`` are one token in every
    language. Digits are kept, punctuation dropped, hyphens separate.
    ``language`` is the row's raw ``language`` value; pass it explicitly
    (``None`` means "no language", not French).
    """
    if not text:
        return []
    normalised = unicodedata.normalize("NFC", str(text)).translate(_APOSTROPHES)
    words = _iter_words(normalised.lower())
    if not applies_french_elision(language):
        return list(words)
    # ``N'Guessan``, ``N'Djamena``, ``M'Bow`` are names, not ne/me + word.
    names = {match.lower() for match in _NAME_PREFIX_RE.findall(normalised)}
    return [
        part
        for word in words
        for part in ([word] if word in names else _split_elision(word))
    ]


def count_words(text: object, language: object = "Français") -> int:
    """``nb_mots``: the number of :func:`tokenize_words` tokens (0 if empty).

    The single definition of the word count, shared by
    ``calculate_word_count.py`` and the references/audiovisual upload mappers.
    Pass the row's ``language`` value so the count matches the MATTR token
    stream of ``calculate_lexical_richness.py``; the French default exists for
    backward compatibility with callers that pass none.
    """
    if not text:
        return 0
    return len(tokenize_words(str(text), language=language))


__all__ = [
    "simple_tokenize", "tokenize_words", "count_words", "applies_french_elision",
    "language_labels", "TOKENIZER_VERSION",
]
