"""Shared tokenization helpers for the IWAC pipeline.

Two tokenizers with distinct purposes:

- :func:`simple_tokenize` — the canonical whitespace tokenizer for text that
  is ALREADY lemmatized (``lemma_nostop``). Used by LDA training, LDA
  prediction, ``analyses/topic_prevalence.py`` and ``analyses/keyness_bursts.py``
  so all four stay in lockstep (they had drifted: keyness skipped
  lowercasing and case-split tokens the LDA side merged).

- :func:`tokenize_words` — a French-aware word tokenizer for RAW text
  (OCR). Splits off elided clitics (``l'islam`` → ``islam``,
  ``aujourd'hui`` stays whole via the exception list) so word counts and
  type counts aren't inflated by ``l``/``d``/``qu`` fragments. Used by the
  word-count and lexical-richness metrics.
"""

from __future__ import annotations

import re
from typing import AbstractSet, List

# Elided clitics that should be separated from the following word.
# ``aujourd'hui`` is a lexicalized exception handled before splitting.
_ELISION_RE = re.compile(
    r"\b(?:qu|jusqu|lorsqu|puisqu|quoiqu|quelqu|[ldjmnstc])[’']", re.IGNORECASE
)
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_APOSTROPHE_WORDS = {
    "aujourd'hui": "aujourdhui",
    "aujourd’hui": "aujourdhui",
}


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


def tokenize_words(text: str) -> List[str]:
    """Tokenize raw (French) text into words, handling elision.

    ``l'islam`` → ``["islam"]``; ``qu'il`` → ``["il"]``;
    ``aujourd'hui`` → ``["aujourdhui"]`` (kept as one token).
    Digits are kept, punctuation dropped. Output is lowercase.
    """
    if not text:
        return []
    lowered = str(text).lower()
    for src, repl in _APOSTROPHE_WORDS.items():
        lowered = lowered.replace(src, repl)
    lowered = _ELISION_RE.sub(" ", lowered)
    return _WORD_RE.findall(lowered)


__all__ = ["simple_tokenize", "tokenize_words"]
