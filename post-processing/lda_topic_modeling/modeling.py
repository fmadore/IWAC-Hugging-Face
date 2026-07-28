"""
modeling.py
-----------
LDA model creation, training, loading, and inference utilities using gensim.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm

import unicodedata

try:
    from iwac_common.text_utils import simple_tokenize
except ImportError:  # venv without the editable install
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from iwac_common.text_utils import simple_tokenize

from .constants import (
    ARTIFACT_LABEL_STOPWORDS,
    DOMAIN_STOPWORDS,
    FRAGMENT_STOPWORDS,
    POST_PHRASE_STOPWORDS,
    LABEL_ONLY_STOPWORDS,
    LDA_GEO_STOPWORDS,
    LDA_GENERIC_STOPWORDS,
    CUSTOM_COLLOCATIONS,
    DEFAULT_NUM_TOPICS,
    DEFAULT_PASSES,
    DEFAULT_ITERATIONS,
    DEFAULT_CHUNKSIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_MINIMUM_PROBABILITY,
    DEFAULT_NO_BELOW,
    DEFAULT_NO_ABOVE,
    DEFAULT_TOPIC_RANGE_START,
    DEFAULT_TOPIC_RANGE_END,
    DEFAULT_TOPIC_RANGE_STEP,
    DEFAULT_SWEEP_PASSES,
    DEFAULT_SWEEP_ITERATIONS,
    DEFAULT_TOPIC_TOPK,
    STABILITY_TOPN_WORDS,
)

# Combined stopword set for label filtering (module-level to avoid
# rebuilding on every call).  Includes both accented and unaccented
# forms so the accent-stripped w_norm always finds a match.
_ALL_LABEL_STOPWORDS = (
    LABEL_ONLY_STOPWORDS | DOMAIN_STOPWORDS | FRAGMENT_STOPWORDS
    | LDA_GEO_STOPWORDS | LDA_GENERIC_STOPWORDS
)


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder tolerant of NumPy scalars/arrays in metrics dicts.

    Replaces the former global ``json`` monkey-patch from
    ``topic_modeling/patches.py`` — encoding is now opt-in at the one
    call site that needs it (``save_model_parameters``).
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def tokenize_documents(
    docs: List[str],
    stopwords: set[str] | None = None,
    min_token_length: int = 2,
    detect_phrases: bool = True,
    phrase_min_count: int = 20,
    phrase_threshold: float = 10.0,
    custom_collocations: List[Tuple[str, ...]] | None = None,
    fragment_stopwords: set[str] | None = None,
) -> Tuple[List[List[str]], Tuple[Phraser, Phraser] | None]:
    """Tokenize documents for LDA.

    Since ``lemma_nostop`` is already lemmatized and partially cleaned,
    we only need to split on whitespace and apply minimal filtering
    (via the shared :func:`simple_tokenize`).

    When *detect_phrases* is True, gensim ``Phrases`` is used to join
    frequent collocations (e.g. "el" + "hadj" → "el_hadj", "nuit" +
    "destin" → "nuit_destin").

    *fragment_stopwords* (default :data:`FRAGMENT_STOPWORDS`) are dropped
    only once phrase detection and custom collocations have run, so a
    fragment survives inside its compounds and disappears everywhere else.
    Pass an empty set to disable.

    Returns ``(tokenized_docs, phraser)`` where *phraser* is the
    ``(bigram_phraser, trigram_phraser)`` pair when phrase detection ran,
    else None.
    """
    if stopwords is None:
        stopwords = set()
    tokenized: List[List[str]] = []
    for doc in docs:
        if not doc or not str(doc).strip():
            tokenized.append([])
            continue
        tokenized.append(simple_tokenize(doc, stopwords, min_token_length))

    phraser = None
    if detect_phrases and tokenized:
        # Bigrams (côte_ivoire, burkina_faso, el_hadj, ...)
        bigram_model = Phrases(tokenized, min_count=phrase_min_count, threshold=phrase_threshold)
        bigram_phraser = Phraser(bigram_model)
        tokenized = [bigram_phraser[doc] for doc in tokenized]
        # Trigrams (e.g. chained bigrams)
        trigram_model = Phrases(tokenized, min_count=phrase_min_count, threshold=phrase_threshold)
        trigram_phraser = Phraser(trigram_model)
        tokenized = [trigram_phraser[doc] for doc in tokenized]
        phraser = (bigram_phraser, trigram_phraser)

    # Apply custom collocations as a safety net after phrase detection
    collocations = custom_collocations if custom_collocations is not None else CUSTOM_COLLOCATIONS
    if collocations:
        tokenized = [apply_custom_collocations(doc, collocations) for doc in tokenized]

    fragments = POST_PHRASE_STOPWORDS if fragment_stopwords is None else fragment_stopwords
    if fragments:
        tokenized = [drop_fragments(doc, fragments) for doc in tokenized]

    return tokenized, phraser


def drop_fragments(tokens: List[str], fragments: set[str] | None = None) -> List[str]:
    """Drop bare fragments and junk compounds, keeping everything else.

    Runs after phrase detection, so ``["al", "al_azhar"]`` → ``["al_azhar"]``
    while ``"university_press"`` goes and ``"university_medina"`` stays.
    Matching is on the whole token — ``al_azhar`` is never split — so only an
    exact listed token is removed.
    """
    frags = POST_PHRASE_STOPWORDS if fragments is None else fragments
    if not frags or not tokens:
        return tokens
    return [t for t in tokens if t not in frags]


def apply_custom_collocations(
    tokens: List[str],
    collocations: List[Tuple[str, ...]] | None = None,
) -> List[str]:
    """Join custom collocations in a token list.

    Scans for consecutive tokens matching predefined multi-word
    expressions and joins them with underscores.  Supports bigrams,
    trigrams, and longer patterns.

    This is applied *after* gensim phrase detection as a safety net
    for domain-specific collocations the statistics miss.
    """
    if not collocations or not tokens:
        return tokens

    # Build lookup keyed on first token for O(n) scanning
    by_first: Dict[str, List[Tuple[str, ...]]] = {}
    for colloc in collocations:
        by_first.setdefault(colloc[0], []).append(colloc)

    # Sort each group longest-first so longer patterns match before shorter
    for first in by_first:
        by_first[first].sort(key=len, reverse=True)

    result: List[str] = []
    i = 0
    while i < len(tokens):
        matched = False
        candidates = by_first.get(tokens[i])
        if candidates:
            for colloc in candidates:
                n = len(colloc)
                if i + n <= len(tokens) and tuple(tokens[i : i + n]) == colloc:
                    result.append("_".join(colloc))
                    i += n
                    matched = True
                    break
        if not matched:
            result.append(tokens[i])
            i += 1

    return result


def chunk_tokens(tokens: List[str], chunk_size: int) -> List[List[str]]:
    """Split a token list into consecutive ``chunk_size``-token chunks.

    Long documents (periodical issues, books, theses) swamp LDA when
    modeled whole: one doc contributes one topic mixture regardless of
    length, and its vocabulary dominates the dictionary. Training on
    fixed-size chunks instead is the standard DH remedy; a short tail
    (< 25% of chunk_size) is merged into the previous chunk rather than
    kept as a noisy fragment.
    """
    if not tokens:
        return []
    if chunk_size <= 0 or len(tokens) <= chunk_size:
        return [tokens]
    chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size // 4:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks


def build_dictionary(
    tokenized_docs: List[List[str]],
    no_below: int = DEFAULT_NO_BELOW,
    no_above: float = DEFAULT_NO_ABOVE,
) -> Dictionary:
    """Build a gensim Dictionary with frequency filtering."""
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    return dictionary


def build_corpus(
    dictionary: Dictionary,
    tokenized_docs: List[List[str]],
) -> List[List[Tuple[int, int]]]:
    """Convert tokenized documents to bag-of-words corpus."""
    return [dictionary.doc2bow(doc) for doc in tokenized_docs]


def create_lda_model(
    corpus: List[List[Tuple[int, int]]],
    dictionary: Dictionary,
    num_topics: int = DEFAULT_NUM_TOPICS,
    passes: int = DEFAULT_PASSES,
    iterations: int = DEFAULT_ITERATIONS,
    chunksize: int = DEFAULT_CHUNKSIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    minimum_probability: float = DEFAULT_MINIMUM_PROBABILITY,
    workers: int | None = None,
    logger: logging.Logger | None = None,
) -> LdaModel:
    """Train an LDA model.

    Uses LdaMulticore when *workers* > 1, otherwise single-core LdaModel
    for strict reproducibility.
    """
    log = logger or logging.getLogger(__name__)

    log.info(f"Training LDA: {num_topics} topics, {passes} passes, "
             f"{iterations} iterations, chunksize={chunksize}")

    common_kwargs: Dict[str, Any] = dict(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        chunksize=chunksize,
        random_state=random_state,
        minimum_probability=minimum_probability,
        alpha="auto",
        eta="auto",
        per_word_topics=True,
    )

    if workers and workers > 1:
        # LdaMulticore raises NotImplementedError for alpha="auto"; the
        # closest supported prior is a fixed asymmetric one.
        common_kwargs["alpha"] = "asymmetric"
        log.info(
            f"Using LdaMulticore with {workers} workers "
            "(alpha='asymmetric'; learned 'auto' alpha needs the single-core path)"
        )
        model = LdaMulticore(workers=workers, **common_kwargs)
    else:
        log.info("Using single-core LdaModel (reproducible)")
        model = LdaModel(**common_kwargs)

    log.info("LDA training complete.")
    return model


def save_lda_model(
    model: LdaModel,
    dictionary: Dictionary,
    model_dir: Path,
    logger: logging.Logger | None = None,
    phraser: Tuple[Phraser, Phraser] | None = None,
) -> None:
    """Persist model, dictionary and optional phrasers to disk."""
    log = logger or logging.getLogger(__name__)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "lda_model"
    dict_path = model_dir / "dictionary"

    model.save(str(model_path))
    dictionary.save(str(dict_path))

    if phraser is not None:
        bigram_phraser, trigram_phraser = phraser
        bigram_phraser.save(str(model_dir / "bigram_phraser"))
        trigram_phraser.save(str(model_dir / "trigram_phraser"))
        log.info("Bigram and trigram phrasers saved")

    log.info(f"LDA model saved to {model_dir}")


def load_lda_model(
    model_dir: Path,
    logger: logging.Logger | None = None,
) -> Tuple[LdaModel, Dictionary, Tuple[Phraser, Phraser] | None]:
    """Load a previously saved LDA model, dictionary and optional phrasers."""
    log = logger or logging.getLogger(__name__)
    model_path = model_dir / "lda_model"
    dict_path = model_dir / "dictionary"
    bigram_path = model_dir / "bigram_phraser"
    trigram_path = model_dir / "trigram_phraser"

    if not model_path.exists():
        raise FileNotFoundError(f"LDA model not found: {model_path}")
    if not dict_path.exists():
        raise FileNotFoundError(f"Dictionary not found: {dict_path}")

    model = LdaModel.load(str(model_path))
    dictionary = Dictionary.load(str(dict_path))

    phraser = None
    if bigram_path.exists() and trigram_path.exists():
        bigram_phraser = Phraser.load(str(bigram_path))
        trigram_phraser = Phraser.load(str(trigram_path))
        phraser = (bigram_phraser, trigram_phraser)
        log.info("Bigram and trigram phrasers loaded")

    log.info(f"LDA model loaded from {model_dir} ({model.num_topics} topics)")
    return model, dictionary, phraser


def _normalize_token(token: str) -> str:
    """Lowercase, strip accents, and treat ``_`` as a word separator.

    Phrase tokens arrive underscore-joined (``fête_tabaski``). Normalising
    the underscore to a space is what lets the ngram-preference and
    subsumption logic in :func:`get_topic_label` recognise them as
    multi-word — while it kept the underscore, every phrase counted as a
    unigram and labels wasted slots on redundant pairs like
    "fête - tabaski - … - fête_tabaski".
    """
    if not token:
        return ""
    t = token.lower().replace("_", " ")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = "".join(ch for ch in t if ch.isalnum() or ch in {" ", "-"})
    return " ".join(t.split())


# Stopwords normalised the same way, so underscore-joined entries
# ("op_cit", "côte_ivoire") and accented ones still match.
_ALL_LABEL_STOPWORDS_NORM = {
    n for n in (_normalize_token(w) for w in _ALL_LABEL_STOPWORDS) if n
}

_ARTIFACT_LABEL_STOPWORDS_NORM = {
    n for n in (_normalize_token(w) for w in ARTIFACT_LABEL_STOPWORDS) if n
}


def _contains_artifact(token_norm: str) -> bool:
    """True when any word of a (possibly compound) label candidate is junk.

    Whole-token matching is deliberate everywhere else — this is the one
    place a component-wise test is right, because a compound built around
    a digitisation artefact is junk however many real words it swallowed.
    """
    return any(part in _ARTIFACT_LABEL_STOPWORDS_NORM for part in token_norm.split())


def _is_subsumed_by_ngram(token_norm: str, selected_norms: List[str]) -> bool:
    """Check if a unigram is already contained within a selected multi-word ngram.

    For example, "cote" is subsumed by "cote ivoire", and "ivoire" is subsumed
    by "cote ivoire".  This prevents labels like "Ivoire - Cote Ivoire - Cote".
    """
    if " " in token_norm:
        return False
    for selected in selected_norms:
        if " " not in selected:
            continue
        if token_norm in selected.split():
            return True
    return False


def compute_corpus_word_probs(
    dictionary: Dictionary,
    corpus: List[List[Tuple[int, int]]] | None = None,
) -> np.ndarray | None:
    """Corpus-wide word probability p(w) over the dictionary vocabulary.

    Counted once from the bag-of-words *corpus* when given; otherwise from
    ``dictionary.cfs`` (collection frequencies persisted with the trained
    dictionary — available at predict time without the training corpus).
    Returns None when no counts are available (relevance re-ranking then
    falls back to pure top-probability labels).
    """
    probs = np.zeros(len(dictionary), dtype=np.float64)
    if corpus is not None:
        for doc in corpus:
            for tid, cnt in doc:
                probs[int(tid)] += cnt
    else:
        for tid, cnt in dictionary.cfs.items():
            if 0 <= int(tid) < len(probs):
                probs[int(tid)] = cnt
    total = probs.sum()
    if total <= 0:
        return None
    return probs / total


def get_topic_label(
    model: LdaModel,
    topic_id: int,
    top_n: int = 6,
    lambda_relevance: float | None = None,
    word_probs: np.ndarray | None = None,
) -> str:
    """Return a human-readable label for a topic.

    Applies stopword removal and substring-aware deduplication so that
    e.g. "Cote", "Ivoire", "Cote Ivoire" collapse to just "Cote Ivoire".

    When *lambda_relevance* and *word_probs* (from
    :func:`compute_corpus_word_probs`) are given, candidate words are
    re-ranked by LDAvis-style relevance (Sievert & Shirley 2014)::

        relevance(w, k) = λ·log p(w|k) + (1−λ)·log(p(w|k) / p(w))

    so corpus-common words no longer dominate several labels. Re-ranking
    is restricted to the topic's top words by probability (a standard
    practical guard against surfacing ultra-rare noise words), then the
    existing stopword / dedup / ngram-preference logic applies unchanged.
    ``lambda_relevance=None`` (the default) is the escape hatch: pure
    top-probability candidates, exactly the old behavior.
    """
    n_candidates = top_n * 3
    if (
        lambda_relevance is not None
        and word_probs is not None
        and len(word_probs) == model.num_terms
    ):
        eps = 1e-12
        topic_dist = model.get_topics()[int(topic_id)].astype(np.float64)
        # Candidate pool: the topic's own top words by p(w|k), so relevance
        # re-ranks plausible words instead of dredging the whole vocabulary.
        pool = np.argsort(topic_dist)[::-1][: max(50, top_n * 8)]
        p_wk = topic_dist[pool] + eps
        p_w = word_probs[pool] + eps
        relevance = lambda_relevance * np.log(p_wk) + (1.0 - lambda_relevance) * np.log(p_wk / p_w)
        order = pool[np.argsort(relevance)[::-1]][:n_candidates]
        raw_words = [(model.id2word[int(i)], float(topic_dist[int(i)])) for i in order]
    else:
        # Fetch more candidates than needed so we can filter and still fill top_n slots
        raw_words = model.show_topic(int(topic_id), topn=n_candidates)

    seen_norms: set[str] = set()
    candidates: List[Tuple[str, str]] = []  # (original, normalized)
    for word, _ in raw_words:
        w_norm = _normalize_token(word)
        if not w_norm:
            continue
        if w_norm in _ALL_LABEL_STOPWORDS_NORM or _contains_artifact(w_norm):
            continue
        if w_norm in seen_norms:
            continue
        seen_norms.add(w_norm)
        candidates.append((word, w_norm))

    # Prefer multi-word ngrams; drop unigrams subsumed by them
    candidates.sort(key=lambda c: c[1].count(" "), reverse=True)

    selected_words: List[str] = []
    selected_norms: List[str] = []
    for word, w_norm in candidates:
        if _is_subsumed_by_ngram(w_norm, selected_norms):
            continue
        selected_words.append(word)
        selected_norms.append(w_norm)
        if len(selected_words) >= top_n:
            break

    return " - ".join(selected_words) if selected_words else f"Topic_{topic_id}"


def predict_document(
    model: LdaModel,
    dictionary: Dictionary,
    tokens: List[str],
    minimum_probability: float = DEFAULT_MINIMUM_PROBABILITY,
    topk: int = DEFAULT_TOPIC_TOPK,
    chunk_words: int | None = None,
    lambda_relevance: float | None = None,
    word_probs: np.ndarray | None = None,
    return_distribution: bool = False,
) -> Tuple[Any, ...]:
    """Predict topics for a single tokenized document.

    Returns (topic_id, probability, label, topk_str) — the dominant topic
    plus a compact top-k distribution string ``"id:prob|id:prob|..."``
    (descending probability, entries below *minimum_probability* dropped).
    Returns (None, None, None, None) for empty docs.

    With *return_distribution* set, a fifth element is appended: the full
    document-topic distribution as a list of ``num_topics`` floats (theta),
    or None for empty docs.

    *lambda_relevance* / *word_probs* are forwarded to
    :func:`get_topic_label` for relevance-weighted labels.

    With *chunk_words* set (models trained on chunks), the document is
    split into chunks, each chunk is inferred separately, and the
    distributions are averaged weighted by chunk length — so a book and
    an article both yield one comparable document-level mixture.
    """
    empty = (None, None, None, None, None) if return_distribution else (None, None, None, None)
    if not tokens:
        return empty

    if chunk_words:
        dist = np.zeros(model.num_topics)
        total_weight = 0
        for chunk in chunk_tokens(tokens, chunk_words):
            bow = dictionary.doc2bow(chunk)
            if not bow:
                continue
            for tid, prob in model.get_document_topics(bow, minimum_probability=0.0):
                dist[int(tid)] += float(prob) * len(chunk)
            total_weight += len(chunk)
        if total_weight == 0:
            return empty
        dist /= total_weight
        ranked = [(int(tid), float(dist[tid])) for tid in np.argsort(dist)[::-1]]
    else:
        bow = dictionary.doc2bow(tokens)
        if not bow:
            return empty
        topic_distribution = model.get_document_topics(bow, minimum_probability=0.0)
        if not topic_distribution:
            return empty
        dist = np.zeros(model.num_topics)
        for tid, prob in topic_distribution:
            dist[int(tid)] = float(prob)
        ranked = sorted(topic_distribution, key=lambda x: x[1], reverse=True)

    best_topic_id, best_prob = ranked[0]
    label = get_topic_label(
        model, best_topic_id,
        lambda_relevance=lambda_relevance, word_probs=word_probs,
    )
    topk_str = "|".join(
        f"{int(tid)}:{prob:.4f}"
        for tid, prob in ranked[:topk]
        if prob >= minimum_probability
    )
    result = (int(best_topic_id), float(best_prob), label, topk_str or None)
    if return_distribution:
        return result + ([float(p) for p in dist],)
    return result


def apply_phraser(tokens: List[str], phraser: Tuple[Phraser, Phraser] | None) -> List[str]:
    """Apply bigram+trigram phrasers to a token list."""
    if phraser is None:
        return tokens
    bigram_phraser, trigram_phraser = phraser
    tokens = bigram_phraser[tokens]
    tokens = trigram_phraser[tokens]
    return list(tokens)


def tokenize_for_prediction(
    text: str,
    stopwords: set[str] | None,
    phraser: Tuple[Phraser, Phraser] | None,
    min_token_length: int = 2,
    custom_collocations: List[Tuple[str, ...]] | None = None,
    fragment_stopwords: set[str] | None = None,
) -> List[str]:
    """Reproduce training-time tokenization for a single document.

    The inference-side mirror of :func:`tokenize_documents`: same stopword
    pass, same phrasers, same collocations, same post-phrase fragment
    filter. Every caller that scores documents against a trained model must
    go through here — training and prediction drifting apart silently
    mis-assigns topics.
    """
    tokens = simple_tokenize(text, stopwords or set(), min_token_length)
    tokens = apply_phraser(tokens, phraser)
    collocations = custom_collocations if custom_collocations is not None else CUSTOM_COLLOCATIONS
    tokens = apply_custom_collocations(tokens, collocations)
    return drop_fragments(tokens, fragment_stopwords)


def predict_batch(
    model: LdaModel,
    dictionary: Dictionary,
    batch: Dict[str, List[Any]],
    text_col: str,
    topic_id_col: str,
    topic_prob_col: str,
    topic_label_col: str,
    stopwords: set[str] | None = None,
    min_token_length: int = 2,
    phraser: Tuple[Phraser, Phraser] | None = None,
    topic_topk_col: str | None = None,
    topk: int = DEFAULT_TOPIC_TOPK,
    chunk_words: int | None = None,
    language: str = "Français",
    model_name_col: str | None = None,
    model_name: str | None = None,
    theta_col: str | None = None,
    lambda_relevance: float | None = None,
    word_probs: np.ndarray | None = None,
) -> Dict[str, List[Any]]:
    """Predict topics for a HuggingFace dataset batch (batched map function).

    Only rows whose ``language`` equals *language* are processed. Skipped
    rows KEEP whatever topic values the batch already carries (None on a
    first run) — so per-language models compose: the French model fills
    French rows, then an English model fills English rows without erasing
    the French ones. When *topic_topk_col* is set, a compact top-k
    distribution string (``"id:prob|id:prob|..."``) is stored alongside
    the dominant topic.

    When *model_name_col* is set, *model_name* (the model directory's
    basename) is written for every row this pass computes, so FR/EN models
    sharing the lda_* columns stay disambiguated; skipped rows keep their
    existing value (same preserve pattern as the topic columns).

    When *theta_col* is set, the full document-topic distribution (list of
    ``num_topics`` floats) is stored for computed rows (None elsewhere) —
    the caller extracts it into ``doc_topics.parquet`` and drops the
    column before pushing.

    *lambda_relevance* / *word_probs* enable relevance-weighted labels
    (see :func:`get_topic_label`); both default to the legacy behavior.
    """
    texts = batch[text_col]
    languages = batch.get("language", [None] * len(texts))

    def _existing(col: str | None) -> List[Any]:
        vals = batch.get(col) if col else None
        return list(vals) if vals is not None else [None] * len(texts)

    topics: List[int | None] = _existing(topic_id_col)
    probabilities: List[float | None] = _existing(topic_prob_col)
    labels: List[str | None] = _existing(topic_label_col)
    topks: List[str | None] = _existing(topic_topk_col)
    model_names: List[str | None] = _existing(model_name_col)
    # Theta is per-pass output, never merged from existing data.
    thetas: List[List[float] | None] = [None] * len(texts)

    sw = stopwords or set()

    for i, text in enumerate(texts):
        lang = languages[i] if i < len(languages) else None
        if lang is not None and lang != language:
            continue
        if not text or not str(text).strip():
            continue

        tokens = tokenize_for_prediction(text, sw, phraser, min_token_length)
        tid, prob, label, topk_str, theta = predict_document(
            model, dictionary, tokens, topk=topk, chunk_words=chunk_words,
            lambda_relevance=lambda_relevance, word_probs=word_probs,
            return_distribution=True,
        )
        topics[i] = tid
        probabilities[i] = prob
        labels[i] = label
        topks[i] = topk_str
        model_names[i] = model_name
        thetas[i] = theta

    batch[topic_id_col] = topics
    batch[topic_prob_col] = probabilities
    batch[topic_label_col] = labels
    if topic_topk_col is not None:
        batch[topic_topk_col] = topks
    if model_name_col is not None:
        batch[model_name_col] = model_names
    if theta_col is not None:
        batch[theta_col] = thetas
    return batch


def compute_coherence(
    model: LdaModel,
    tokenized_docs: List[List[str]],
    dictionary: Dictionary,
    corpus: List[List[Tuple[int, int]]],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Compute coherence metrics (C_v, NPMI, U_Mass) and topic diversity."""
    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "num_topics": model.num_topics,
        "num_documents": len(tokenized_docs),
    }

    for coherence_type, needs_corpus in [("c_v", False), ("c_npmi", False), ("u_mass", True)]:
        try:
            kwargs: Dict[str, Any] = dict(
                model=model,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence=coherence_type,
            )
            if needs_corpus:
                kwargs["corpus"] = corpus
            cm = CoherenceModel(**kwargs)
            score = cm.get_coherence()
            per_topic = cm.get_coherence_per_topic()
            metrics[coherence_type] = {
                "score": float(score),
                "per_topic": [float(s) for s in per_topic],
            }
            logger.info(f"Coherence {coherence_type}: {score:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute {coherence_type}: {e}")
            metrics[coherence_type] = {"error": str(e)}

    # Topic diversity
    all_words: List[str] = []
    for tid in range(model.num_topics):
        all_words.extend(w for w, _ in model.show_topic(tid, topn=10))
    unique = set(all_words)
    diversity = len(unique) / len(all_words) if all_words else 0
    metrics["topic_diversity"] = {"score": float(diversity)}
    logger.info(f"Topic diversity: {diversity:.4f}")

    return metrics


def save_model_parameters(
    model_dir: Path,
    num_topics: int,
    passes: int,
    iterations: int,
    chunksize: int,
    no_below: int,
    no_above: float,
    stopwords_used: List[str],
    fragment_stopwords: List[str] | None = None,
    coherence_metrics: Dict[str, Any] | None = None,
    extra_info: Dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    alpha: str = "auto",
    lambda_relevance: float | None = None,
    evaluation: Dict[str, Any] | None = None,
) -> Path:
    """Save training parameters to JSON for reproducibility.

    *lambda_relevance* records the LDAvis-style label re-ranking weight
    (None = pure top-probability labels). *evaluation* records sweep
    robustness settings (``sweep_n_seeds``, ``holdout_fraction``) under a
    top-level ``evaluation`` key; the per-k sweep values live in
    ``extra.topic_optimization.results``.
    """
    params: Dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "method": "LDA (gensim)",
            "pipeline_version": "1.1.0",
        },
        "lda": {
            "num_topics": num_topics,
            "passes": passes,
            "iterations": iterations,
            "chunksize": chunksize,
            "alpha": alpha,
            "eta": "auto",
            "random_state": DEFAULT_RANDOM_STATE,
            "label_lambda_relevance": lambda_relevance,
        },
        "dictionary_filter": {
            "no_below": no_below,
            "no_above": no_above,
        },
        "stopwords": {
            "count": len(stopwords_used),
            "words": sorted(stopwords_used),
            # Removed after phrase detection: bare fragments whose compounds
            # must survive, plus whole compounds that are apparatus.
            "fragments": sorted(
                POST_PHRASE_STOPWORDS if fragment_stopwords is None else fragment_stopwords
            ),
        },
    }

    if evaluation:
        params["evaluation"] = evaluation
    if coherence_metrics:
        params["coherence_metrics"] = coherence_metrics
    if extra_info:
        params["extra"] = extra_info

    model_dir.mkdir(parents=True, exist_ok=True)
    params_path = model_dir / "training_parameters.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2, cls=_NumpyJSONEncoder)

    if logger:
        logger.info(f"Parameters saved: {params_path}")
    return params_path


def topic_stability_jaccard(
    models: List[LdaModel],
    topn: int = STABILITY_TOPN_WORDS,
) -> float | None:
    """Topic-stability score across seed models: mean best-match Jaccard.

    For each pair of models, every topic is represented by the set of its
    top-*topn* words and topics are aligned by GREEDY best-match: the
    highest-Jaccard (topic_a, topic_b) pair is matched first, both topics
    are removed, and matching repeats until one model's topics run out
    (a full Hungarian assignment is unnecessary at this granularity).
    The pair's score is the mean Jaccard of the matched topics; the
    returned score is the mean over all model pairs. 1.0 = seeds recover
    identical topics; ~0 = seeds disagree completely. Returns None with
    fewer than two models.
    """
    if len(models) < 2:
        return None

    top_words = [
        [set(w for w, _ in m.show_topic(t, topn=topn)) for t in range(m.num_topics)]
        for m in models
    ]

    pair_scores: List[float] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a_tops, b_tops = top_words[i], top_words[j]
            overlaps = []
            for ai, a in enumerate(a_tops):
                for bi, b in enumerate(b_tops):
                    union = a | b
                    jac = len(a & b) / len(union) if union else 0.0
                    overlaps.append((jac, ai, bi))
            # Deterministic greedy alignment: best Jaccard first, ties by index
            overlaps.sort(key=lambda o: (-o[0], o[1], o[2]))
            used_a: set[int] = set()
            used_b: set[int] = set()
            matched: List[float] = []
            n_match = min(len(a_tops), len(b_tops))
            for jac, ai, bi in overlaps:
                if ai in used_a or bi in used_b:
                    continue
                used_a.add(ai)
                used_b.add(bi)
                matched.append(jac)
                if len(matched) >= n_match:
                    break
            pair_scores.append(float(np.mean(matched)) if matched else 0.0)

    return float(np.mean(pair_scores))


def find_optimal_topics(
    corpus: List[List[Tuple[int, int]]],
    dictionary: Dictionary,
    tokenized_docs: List[List[str]],
    topic_range_start: int = DEFAULT_TOPIC_RANGE_START,
    topic_range_end: int = DEFAULT_TOPIC_RANGE_END,
    topic_range_step: int = DEFAULT_TOPIC_RANGE_STEP,
    sweep_passes: int = DEFAULT_SWEEP_PASSES,
    sweep_iterations: int = DEFAULT_SWEEP_ITERATIONS,
    chunksize: int = DEFAULT_CHUNKSIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_seeds: int = 1,
    holdout_corpus: List[List[Tuple[int, int]]] | None = None,
    logger: logging.Logger | None = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Sweep a range of topic counts and return the k with highest C_v.

    This is standard DH practice (Mimno et al.): train LDA at several k
    values, compute C_v coherence for each, and pick the peak.

    Sweep models train at reduced settings (*sweep_passes* /
    *sweep_iterations*) — enough for a stable *relative* C_v ranking; the
    caller retrains the winning k at full production settings afterwards.

    With *n_seeds* > 1, each candidate k trains n_seeds reduced models
    (random_state, random_state+1, ...), the winning k is picked by MEAN
    C_v instead of a single noisy draw, and each k additionally reports
    the C_v standard deviation (sample sd) plus a topic-stability score
    (see :func:`topic_stability_jaccard`). n_seeds=1 is the exact legacy
    behavior. Secondary metrics (NPMI, U_Mass) are computed on the
    first-seed model only, to keep the sweep cost linear in n_seeds.

    With *holdout_corpus* set (bow docs unseen during training), each k
    also reports held-out ``log_perplexity`` (gensim's per-word likelihood
    bound; higher, i.e. less negative, is better), averaged over seeds.

    Returns:
        best_k: the number of topics with the highest (mean) C_v score.
        results: list of dicts with keys ``k``, ``c_v`` (mean over seeds),
                 ``c_v_sd``, ``c_v_per_seed``, ``stability_jaccard``,
                 ``holdout_log_perplexity``, ``c_npmi``, ``u_mass`` for
                 every tested value, so users can inspect the full curve.
    """
    log = logger or logging.getLogger(__name__)
    n_seeds = max(1, int(n_seeds))

    candidates = list(range(topic_range_start, topic_range_end + 1, topic_range_step))
    seeds = [random_state + s for s in range(n_seeds)]
    log.info(
        f"Optimising num_topics: testing {candidates} "
        f"({len(candidates)} k values x {n_seeds} seed(s) {seeds} at sweep settings: "
        f"passes={sweep_passes}, iterations={sweep_iterations})"
        + (f"; held-out eval on {len(holdout_corpus)} docs" if holdout_corpus else "")
    )

    results: List[Dict[str, Any]] = []
    best_k = candidates[0]
    best_cv = -1.0

    for k in tqdm(candidates, desc="Topic optimisation"):
        seed_models: List[LdaModel] = []
        cv_per_seed: List[float | None] = []
        perplexities: List[float] = []

        for seed in seeds:
            log.info(f"Training LDA with k={k} (seed={seed})...")
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                passes=sweep_passes,
                iterations=sweep_iterations,
                chunksize=chunksize,
                random_state=seed,
                alpha="auto",
                eta="auto",
                per_word_topics=True,
            )
            seed_models.append(model)

            # C_v (primary criterion), per seed
            try:
                cm_cv = CoherenceModel(
                    model=model, texts=tokenized_docs,
                    dictionary=dictionary, coherence="c_v",
                )
                cv = float(cm_cv.get_coherence())
                cv_per_seed.append(cv)
                log.info(f"  k={k} seed={seed}  C_v={cv:.4f}")
            except Exception as e:
                log.warning(f"  k={k} seed={seed}  C_v failed: {e}")
                cv_per_seed.append(None)

            # Held-out log-perplexity (per-word bound; higher = better)
            if holdout_corpus:
                try:
                    perplexities.append(float(model.log_perplexity(holdout_corpus)))
                except Exception as e:
                    log.warning(f"  k={k} seed={seed}  log_perplexity failed: {e}")

        entry: Dict[str, Any] = {"k": k}

        cv_valid = [c for c in cv_per_seed if c is not None]
        if cv_valid:
            mean_cv = float(np.mean(cv_valid))
            entry["c_v"] = mean_cv
            entry["c_v_sd"] = float(np.std(cv_valid, ddof=1)) if len(cv_valid) > 1 else None
            if mean_cv > best_cv:
                best_cv = mean_cv
                best_k = k
        else:
            entry["c_v"] = None
            entry["c_v_sd"] = None
        entry["c_v_per_seed"] = cv_per_seed if n_seeds > 1 else None

        # Topic stability across seeds (None when n_seeds == 1)
        entry["stability_jaccard"] = topic_stability_jaccard(seed_models)

        entry["holdout_log_perplexity"] = (
            float(np.mean(perplexities)) if perplexities else None
        )

        if n_seeds > 1 and entry["c_v"] is not None:
            log.info(
                f"  k={k}  mean C_v={entry['c_v']:.4f}"
                + (f" (sd={entry['c_v_sd']:.4f})" if entry["c_v_sd"] is not None else "")
                + (f"  stability={entry['stability_jaccard']:.3f}"
                   if entry["stability_jaccard"] is not None else "")
            )

        # Secondary metrics on the first-seed model only (cost control)
        first_model = seed_models[0]
        try:
            cm_npmi = CoherenceModel(
                model=first_model, texts=tokenized_docs,
                dictionary=dictionary, coherence="c_npmi",
            )
            entry["c_npmi"] = float(cm_npmi.get_coherence())
        except Exception:
            entry["c_npmi"] = None

        try:
            cm_umass = CoherenceModel(
                model=first_model, corpus=corpus,
                dictionary=dictionary, coherence="u_mass",
            )
            entry["u_mass"] = float(cm_umass.get_coherence())
        except Exception:
            entry["u_mass"] = None

        results.append(entry)

    if all(r.get("c_v") is None for r in results):
        log.error(
            "Every C_v coherence computation failed — model selection did not run; "
            f"returning k={best_k} (the smallest candidate), NOT a data-driven choice."
        )
    else:
        label = "mean C_v" if n_seeds > 1 else "C_v"
        log.info(f"Best num_topics by {label}: {best_k} ({label}={best_cv:.4f})")
    return best_k, results
