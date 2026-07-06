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

from .constants import (
    DOMAIN_STOPWORDS,
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
)

# Combined stopword set for label filtering (module-level to avoid
# rebuilding on every call).  Includes both accented and unaccented
# forms so the accent-stripped w_norm always finds a match.
_ALL_LABEL_STOPWORDS = (
    LABEL_ONLY_STOPWORDS | DOMAIN_STOPWORDS
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
) -> List[List[str]]:
    """Tokenize documents for LDA.

    Since ``lemma_nostop`` is already lemmatized and partially cleaned,
    we only need to split on whitespace and apply minimal filtering.

    When *detect_phrases* is True, gensim ``Phrases`` is used to join
    frequent collocations (e.g. "côte" + "ivoire" → "côte_ivoire",
    "burkina" + "faso" → "burkina_faso").
    """
    if stopwords is None:
        stopwords = set()
    tokenized: List[List[str]] = []
    for doc in docs:
        if not doc or not str(doc).strip():
            tokenized.append([])
            continue
        tokens = [
            t
            for t in str(doc).lower().split()
            if len(t) >= min_token_length and t not in stopwords
        ]
        tokenized.append(tokens)

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

    return tokenized, phraser


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
    """Lowercase and strip accents for robust matching."""
    if not token:
        return ""
    t = token.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return "".join(ch for ch in t if ch.isalnum() or ch in {" ", "-", "_"}).strip()


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


def get_topic_label(model: LdaModel, topic_id: int, top_n: int = 6) -> str:
    """Return a human-readable label for a topic.

    Applies stopword removal and substring-aware deduplication so that
    e.g. "Cote", "Ivoire", "Cote Ivoire" collapse to just "Cote Ivoire".
    """
    # Fetch more candidates than needed so we can filter and still fill top_n slots
    raw_words = model.show_topic(int(topic_id), topn=top_n * 3)

    seen_norms: set[str] = set()
    candidates: List[Tuple[str, str]] = []  # (original, normalized)
    for word, _ in raw_words:
        w_norm = _normalize_token(word)
        if not w_norm:
            continue
        if w_norm in _ALL_LABEL_STOPWORDS:
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
) -> Tuple[int | None, float | None, str | None, str | None]:
    """Predict topics for a single tokenized document.

    Returns (topic_id, probability, label, topk_str) — the dominant topic
    plus a compact top-k distribution string ``"id:prob|id:prob|..."``
    (descending probability, entries below *minimum_probability* dropped).
    Returns (None, None, None, None) for empty docs.

    With *chunk_words* set (models trained on chunks), the document is
    split into chunks, each chunk is inferred separately, and the
    distributions are averaged weighted by chunk length — so a book and
    an article both yield one comparable document-level mixture.
    """
    if not tokens:
        return None, None, None, None

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
            return None, None, None, None
        dist /= total_weight
        ranked = [(int(tid), float(dist[tid])) for tid in np.argsort(dist)[::-1]]
    else:
        bow = dictionary.doc2bow(tokens)
        if not bow:
            return None, None, None, None
        topic_distribution = model.get_document_topics(bow, minimum_probability=0.0)
        if not topic_distribution:
            return None, None, None, None
        ranked = sorted(topic_distribution, key=lambda x: x[1], reverse=True)

    best_topic_id, best_prob = ranked[0]
    label = get_topic_label(model, best_topic_id)
    topk_str = "|".join(
        f"{int(tid)}:{prob:.4f}"
        for tid, prob in ranked[:topk]
        if prob >= minimum_probability
    )
    return int(best_topic_id), float(best_prob), label, topk_str or None


def apply_phraser(tokens: List[str], phraser: Tuple[Phraser, Phraser] | None) -> List[str]:
    """Apply bigram+trigram phrasers to a token list."""
    if phraser is None:
        return tokens
    bigram_phraser, trigram_phraser = phraser
    tokens = bigram_phraser[tokens]
    tokens = trigram_phraser[tokens]
    return list(tokens)


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
) -> Dict[str, List[Any]]:
    """Predict topics for a HuggingFace dataset batch (batched map function).

    Only rows whose ``language`` equals *language* are processed. Skipped
    rows KEEP whatever topic values the batch already carries (None on a
    first run) — so per-language models compose: the French model fills
    French rows, then an English model fills English rows without erasing
    the French ones. When *topic_topk_col* is set, a compact top-k
    distribution string (``"id:prob|id:prob|..."``) is stored alongside
    the dominant topic.
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

    sw = stopwords or set()

    for i, text in enumerate(texts):
        lang = languages[i] if i < len(languages) else None
        if lang is not None and lang != language:
            continue
        if not text or not str(text).strip():
            continue

        tokens = [
            t
            for t in str(text).lower().split()
            if len(t) >= min_token_length and t not in sw
        ]
        tokens = apply_phraser(tokens, phraser)
        tokens = apply_custom_collocations(tokens, CUSTOM_COLLOCATIONS)
        tid, prob, label, topk_str = predict_document(
            model, dictionary, tokens, topk=topk, chunk_words=chunk_words
        )
        topics[i] = tid
        probabilities[i] = prob
        labels[i] = label
        topks[i] = topk_str

    batch[topic_id_col] = topics
    batch[topic_prob_col] = probabilities
    batch[topic_label_col] = labels
    if topic_topk_col is not None:
        batch[topic_topk_col] = topks
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
    coherence_metrics: Dict[str, Any] | None = None,
    extra_info: Dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    alpha: str = "auto",
) -> Path:
    """Save training parameters to JSON for reproducibility."""
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
        },
        "dictionary_filter": {
            "no_below": no_below,
            "no_above": no_above,
        },
        "stopwords": {
            "count": len(stopwords_used),
            "words": sorted(stopwords_used),
        },
    }

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
    logger: logging.Logger | None = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Sweep a range of topic counts and return the k with highest C_v.

    This is standard DH practice (Mimno et al.): train LDA at several k
    values, compute C_v coherence for each, and pick the peak.

    Sweep models train at reduced settings (*sweep_passes* /
    *sweep_iterations*) — enough for a stable *relative* C_v ranking; the
    caller retrains the winning k at full production settings afterwards.

    Returns:
        best_k: the number of topics with the highest C_v score.
        results: list of dicts with keys ``k``, ``c_v``, ``c_npmi``, ``u_mass``
                 for every tested value, so users can inspect the full curve.
    """
    log = logger or logging.getLogger(__name__)

    candidates = list(range(topic_range_start, topic_range_end + 1, topic_range_step))
    log.info(
        f"Optimising num_topics: testing {candidates} "
        f"({len(candidates)} models to train at sweep settings: "
        f"passes={sweep_passes}, iterations={sweep_iterations})"
    )

    results: List[Dict[str, Any]] = []
    best_k = candidates[0]
    best_cv = -1.0

    for k in tqdm(candidates, desc="Topic optimisation"):
        log.info(f"Training LDA with k={k}...")
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=sweep_passes,
            iterations=sweep_iterations,
            chunksize=chunksize,
            random_state=random_state,
            alpha="auto",
            eta="auto",
            per_word_topics=True,
        )

        entry: Dict[str, Any] = {"k": k}

        # C_v (primary criterion)
        try:
            cm_cv = CoherenceModel(
                model=model, texts=tokenized_docs,
                dictionary=dictionary, coherence="c_v",
            )
            cv = cm_cv.get_coherence()
            entry["c_v"] = float(cv)
            log.info(f"  k={k}  C_v={cv:.4f}")
            if cv > best_cv:
                best_cv = cv
                best_k = k
        except Exception as e:
            log.warning(f"  k={k}  C_v failed: {e}")
            entry["c_v"] = None

        # NPMI (secondary)
        try:
            cm_npmi = CoherenceModel(
                model=model, texts=tokenized_docs,
                dictionary=dictionary, coherence="c_npmi",
            )
            entry["c_npmi"] = float(cm_npmi.get_coherence())
        except Exception:
            entry["c_npmi"] = None

        # U_Mass (secondary)
        try:
            cm_umass = CoherenceModel(
                model=model, corpus=corpus,
                dictionary=dictionary, coherence="u_mass",
            )
            entry["u_mass"] = float(cm_umass.get_coherence())
        except Exception:
            entry["u_mass"] = None

        results.append(entry)

    log.info(f"Best num_topics by C_v: {best_k} (C_v={best_cv:.4f})")
    return best_k, results
