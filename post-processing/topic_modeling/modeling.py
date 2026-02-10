"""
modeling.py
-----------
BERTopic model creation, training, loading, and inference utilities.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
import inspect
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from umap import UMAP

from .constants import DOMAIN_STOPWORDS, LABEL_ONLY_STOPWORDS
import unicodedata

try:
    from bertopic.representation import KeyBERTInspired, PartOfSpeech, Merge
    KEYBERT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    KEYBERT_AVAILABLE = False

# Optional: c-TF-IDF with reduction of frequent words to improve labels
try:  # pragma: no cover - optional dependency across versions
    from bertopic._ctfidf import ClassTfidfTransformer  # type: ignore
    CTFIDF_AVAILABLE = True
except Exception:  # older/newer versions may move this symbol
    ClassTfidfTransformer = None  # type: ignore
    CTFIDF_AVAILABLE = False

# Optional: gensim for coherence metrics
try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


def compute_coherence_metrics(
    topic_model: BERTopic,
    docs: List[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Compute topic coherence metrics (C_v, NPMI, U_Mass) for quality assessment.
    
    These metrics are standard in Digital Humanities for evaluating topic interpretability.
    C_v (0-1): Higher is better, measures semantic similarity of top words
    NPMI (-1 to 1): Higher is better, normalized pointwise mutual information
    U_Mass (negative): Less negative is better, based on document co-occurrence
    
    Returns dict with coherence scores and per-topic breakdown.
    """
    if not GENSIM_AVAILABLE:
        logger.warning("gensim non disponible - impossible de calculer les métriques de cohérence. "
                      "Installez avec: pip install gensim")
        return {"error": "gensim not available"}
    
    try:
        # Get topic words from BERTopic
        topic_info = topic_model.get_topic_info()
        topics_words: List[List[str]] = []
        topic_ids: List[int] = []
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:  # Skip outlier topic
                continue
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                words = [word for word, _ in topic_words[:10]]  # Top 10 words
                topics_words.append(words)
                topic_ids.append(topic_id)
        
        if not topics_words:
            logger.warning("Aucun topic valide pour calculer la cohérence")
            return {"error": "no valid topics"}
        
        # Tokenize documents for gensim
        tokenized_docs = [doc.lower().split() for doc in docs if doc and doc.strip()]
        
        if len(tokenized_docs) < 10:
            logger.warning("Pas assez de documents pour calculer la cohérence")
            return {"error": "insufficient documents"}
        
        # Create gensim dictionary
        dictionary = Dictionary(tokenized_docs)
        
        # Compute multiple coherence metrics
        metrics: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "num_topics": len(topics_words),
            "num_documents": len(tokenized_docs),
        }
        
        # C_v coherence (most commonly used in DH)
        try:
            cm_cv = CoherenceModel(
                topics=topics_words,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence='c_v'
            )
            cv_score = cm_cv.get_coherence()
            cv_per_topic = cm_cv.get_coherence_per_topic()
            metrics["c_v"] = {
                "score": float(cv_score),
                "per_topic": {tid: float(score) for tid, score in zip(topic_ids, cv_per_topic)},
                "interpretation": "0-1, higher is better"
            }
            logger.info(f"Cohérence C_v: {cv_score:.4f}")
        except Exception as e:
            logger.warning(f"Impossible de calculer C_v: {e}")
            metrics["c_v"] = {"error": str(e)}
        
        # NPMI coherence
        try:
            cm_npmi = CoherenceModel(
                topics=topics_words,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence='c_npmi'
            )
            npmi_score = cm_npmi.get_coherence()
            npmi_per_topic = cm_npmi.get_coherence_per_topic()
            metrics["npmi"] = {
                "score": float(npmi_score),
                "per_topic": {tid: float(score) for tid, score in zip(topic_ids, npmi_per_topic)},
                "interpretation": "-1 to 1, higher is better"
            }
            logger.info(f"Cohérence NPMI: {npmi_score:.4f}")
        except Exception as e:
            logger.warning(f"Impossible de calculer NPMI: {e}")
            metrics["npmi"] = {"error": str(e)}
        
        # U_Mass coherence (faster, based on document co-occurrence)
        try:
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
            cm_umass = CoherenceModel(
                topics=topics_words,
                corpus=corpus,
                dictionary=dictionary,
                coherence='u_mass'
            )
            umass_score = cm_umass.get_coherence()
            umass_per_topic = cm_umass.get_coherence_per_topic()
            metrics["u_mass"] = {
                "score": float(umass_score),
                "per_topic": {tid: float(score) for tid, score in zip(topic_ids, umass_per_topic)},
                "interpretation": "negative, less negative is better"
            }
            logger.info(f"Cohérence U_Mass: {umass_score:.4f}")
        except Exception as e:
            logger.warning(f"Impossible de calculer U_Mass: {e}")
            metrics["u_mass"] = {"error": str(e)}
        
        # Topic diversity: proportion of unique words across all topics
        all_words = [word for topic in topics_words for word in topic]
        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words) if all_words else 0
        metrics["topic_diversity"] = {
            "score": float(diversity),
            "interpretation": "0-1, higher means more diverse topics"
        }
        logger.info(f"Diversité des topics: {diversity:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques de cohérence: {e}")
        return {"error": str(e)}


def save_model_parameters(
    model_save_path: Path,
    embedding_model_name: str,
    min_topic_size: int,
    umap_params: Dict[str, Any],
    hdbscan_params: Dict[str, Any],
    vectorizer_params: Dict[str, Any],
    stopwords_used: List[str],
    coherence_metrics: Dict[str, Any] | None = None,
    extra_info: Dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Save all model parameters to a JSON file for reproducibility.
    
    This is critical for academic DH work where reproducibility is essential.
    """
    params = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "bertopic_version": None,
            "pipeline_version": "1.0.0",
        },
        "embedding_model": {
            "name": embedding_model_name,
        },
        "clustering": {
            "min_topic_size": min_topic_size,
            "umap": umap_params,
            "hdbscan": hdbscan_params,
        },
        "vectorizer": vectorizer_params,
        "stopwords": {
            "count": len(stopwords_used),
            "words": sorted(stopwords_used),
        },
    }
    
    # Add BERTopic version if available
    try:
        import bertopic
        params["metadata"]["bertopic_version"] = bertopic.__version__
    except Exception:
        pass
    
    if coherence_metrics:
        params["coherence_metrics"] = coherence_metrics
    
    if extra_info:
        params["extra"] = extra_info
    
    params_path = model_save_path / "training_parameters.json"
    try:
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        if logger:
            logger.info(f"Paramètres sauvegardés: {params_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Impossible de sauvegarder les paramètres: {e}")
    
    return params_path


def extract_year_from_date(date_str: str | None) -> int | None:
    """Extract year from a date string (supports YYYY-MM-DD, YYYY, etc.)."""
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    if not date_str:
        return None
    
    # Try YYYY-MM-DD or YYYY/MM/DD format
    try:
        if len(date_str) >= 4:
            year = int(date_str[:4])
            if 1800 <= year <= 2100:  # Reasonable year range
                return year
    except (ValueError, IndexError):
        pass
    
    return None


def compute_topics_over_time(
    topic_model: BERTopic,
    docs: List[str],
    timestamps: List[str | int | None],
    logger: logging.Logger,
    nr_bins: int | None = None,
) -> Tuple[Any | None, Dict[int, int]]:
    """Compute topics over time using BERTopic's built-in method.
    
    Returns:
        - topics_over_time DataFrame (or None if failed)
        - year_mapping: dict mapping doc index to extracted year
    """
    # Extract years from timestamps
    years: List[int | None] = []
    year_mapping: Dict[int, int] = {}
    
    for i, ts in enumerate(timestamps):
        if isinstance(ts, int):
            year = ts if 1800 <= ts <= 2100 else None
        else:
            year = extract_year_from_date(str(ts) if ts else None)
        years.append(year)
        if year is not None:
            year_mapping[i] = year
    
    # Filter to docs with valid years
    valid_indices = [i for i, y in enumerate(years) if y is not None]
    if len(valid_indices) < 10:
        logger.warning(f"Pas assez de documents avec dates valides ({len(valid_indices)}) pour l'analyse temporelle")
        return None, year_mapping
    
    valid_docs = [docs[i] for i in valid_indices]
    valid_years = [years[i] for i in valid_indices]
    
    # Convert years to timestamps for BERTopic
    # BERTopic expects datetime-like objects or strings
    timestamps_str = [f"{y}-01-01" for y in valid_years]
    
    try:
        topics_over_time = topic_model.topics_over_time(
            valid_docs,
            timestamps_str,
            nr_bins=nr_bins,
            datetime_format="%Y-%m-%d",
        )
        
        # Log summary
        if topics_over_time is not None and len(topics_over_time) > 0:
            year_range = f"{min(valid_years)}-{max(valid_years)}"
            logger.info(f"Analyse temporelle: {len(topics_over_time)} entrées, période {year_range}")
        
        return topics_over_time, year_mapping
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul topics_over_time: {e}")
        return None, year_mapping


def create_custom_topic_representation():
    """Create a custom topic representation that prioritizes NOUN/ADJ and adds diversity.

    Uses PartOfSpeech (French) + KeyBERTInspired via Merge when available.
    Falls back to KeyBERTInspired alone, then to None.
    """
    if not KEYBERT_AVAILABLE:
        return None
    try:  # Try POS + KeyBERT with Merge
        # Use 'lg' model for better accuracy as requested in instructions
        pos = PartOfSpeech(model="fr_core_news_lg", allowed_pos={"NOUN", "ADJ"}, top_n=10)
        keybert = KeyBERTInspired(mm_r=True, diversity=0.3)
        return Merge([pos, keybert])
    except Exception:  # spaCy model missing or representation unavailable
        try:
            return KeyBERTInspired()
        except Exception:
            return None


def _normalize_token(token: str) -> str:
    """Lowercase and strip accents/punct for robust matching.

    Keeps alphanumerics; removes combining marks and punctuation.
    """
    if not token:
        return ""
    t = token.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    # keep letters/digits and space/hyphen separators
    cleaned = []
    for ch in t:
        if ch.isalnum() or ch in {" ", "-", "_"}:
            cleaned.append(ch)
    return "".join(cleaned).strip()


def _is_subsumed_by_ngram(token_norm: str, selected_norms: List[str]) -> bool:
    """Check if a unigram is already contained within a selected multi-word ngram.

    For example, "cote" is subsumed by "cote ivoire", and "ivoire" is subsumed
    by "cote ivoire".  This prevents labels like "Ivoire - Cote Ivoire - Cote"
    where three slots say the same thing.
    """
    if " " in token_norm:
        # token is itself a multi-word ngram; don't drop it as a substring
        return False
    for selected in selected_norms:
        if " " not in selected:
            continue  # only check against multi-word ngrams
        parts = selected.split()
        if token_norm in parts:
            return True
    return False


def clean_topic_labels(topic_model: BERTopic, max_words: int = 8) -> BERTopic:
    """Clean topic labels by removing duplicates, subsumed unigrams, and limiting word count."""
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            continue

        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            continue

        # First pass: collect candidates (deduplicated, stopwords removed)
        seen_norms: set[str] = set()
        candidates: List[Tuple[str, str]] = []  # (original_word, normalized)
        for word, _ in topic_words:
            w_norm = _normalize_token(word)
            if not w_norm:
                continue
            if (w_norm in LABEL_ONLY_STOPWORDS) or (w_norm in DOMAIN_STOPWORDS):
                continue
            if w_norm in seen_norms:
                continue
            seen_norms.add(w_norm)
            candidates.append((word, w_norm))

        # Second pass: prefer multi-word ngrams over their component unigrams.
        # Process longer ngrams first so they claim their component words.
        candidates.sort(key=lambda c: c[1].count(" "), reverse=True)

        selected_words: List[str] = []
        selected_norms: List[str] = []
        for word, w_norm in candidates:
            if _is_subsumed_by_ngram(w_norm, selected_norms):
                continue
            selected_words.append(word)
            selected_norms.append(w_norm)
            if len(selected_words) >= max_words:
                break

        if selected_words:
            new_label = " - ".join(selected_words)
            topic_info.loc[topic_info['Topic'] == topic_id, 'Name'] = new_label

    # Apply labels through the public API when available (BERTopic >= 0.16)
    labels_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
    if hasattr(topic_model, "set_topic_labels"):
        topic_model.set_topic_labels(labels_map)
    else:  # fallback for older versions
        try:
            topic_model.topic_labels_ = labels_map  # type: ignore[attr-defined]
        except Exception:
            pass
    return topic_model


def _embed_texts(embedding_model: Any, texts: List[str], batch_size: int, show_progress: bool) -> np.ndarray:
    """Embed texts across BERTopic/SBERT versions.

    Supports:
    - SentenceTransformer.encode (older direct model)
    - EmbeddingModel.embed / embed_documents (BERTopic backends)
    """
    def _call_with_supported(method, texts_arg):
        """Call method(texts, **kwargs) using only supported kwargs.

        Handles variants like show_progress_bar|verbose and optional batch_size.
        Falls back to calling without kwargs if needed.
        """
        try:
            sig = inspect.signature(method)
            params = set(sig.parameters.keys())
        except (TypeError, ValueError):  # builtins or C-accelerated functions
            params = set()

        kwargs = {}
        if 'batch_size' in params:
            kwargs['batch_size'] = batch_size
        # Some apis use show_progress_bar, some use verbose, some none
        if 'show_progress_bar' in params:
            kwargs['show_progress_bar'] = show_progress
        elif 'show_progress' in params:
            kwargs['show_progress'] = show_progress
        elif 'verbose' in params:
            kwargs['verbose'] = show_progress

        try:
            return np.asarray(method(texts_arg, **kwargs))
        except TypeError:
            # Retry with no kwargs at all
            return np.asarray(method(texts_arg))

    # SentenceTransformer from sentence-transformers
    if hasattr(embedding_model, "encode"):
        method = getattr(embedding_model, "encode")
        try:
            sig = inspect.signature(method)
            params = set(sig.parameters.keys())
        except (TypeError, ValueError):
            params = set()

        kwargs = {"convert_to_numpy": True}
        if 'batch_size' in params:
            kwargs['batch_size'] = batch_size
        if 'show_progress_bar' in params:
            kwargs['show_progress_bar'] = show_progress
        try:
            return method(texts, **kwargs)
        except TypeError:
            # Minimal fallback
            return np.asarray(method(texts))

    # BERTopic EmbeddingModel interface
    if hasattr(embedding_model, "embed"):
        return _call_with_supported(embedding_model.embed, texts)
    if hasattr(embedding_model, "embed_documents"):
        return _call_with_supported(embedding_model.embed_documents, texts)
    raise AttributeError("Unsupported embedding model: no encode/embed/embed_documents method found")


def create_bertopic_model(
    embedding_model_name: str,
    min_topic_size: int = 10,
    cpu_only: bool = False,
    embedding_batch_size: int = 32,  # kept for API parity
    # UMAP params (BERTopic best practice defaults)
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    umap_n_components: int = 5,
    umap_metric: str = 'cosine',
    # HDBSCAN params (BERTopic best practice defaults)
    hdbscan_min_samples: int = 10,
    hdbscan_selection_method: str = 'leaf',
    hdbscan_epsilon: float = 0.0,
    # Vectorizer params (min_df=2, max_df=1.0 are hardcoded for BERTopic c-TF-IDF compatibility)
    vectorizer_max_features: int = 25000,
    vectorizer_ngram_min: int = 1,
    vectorizer_ngram_max: int = 3,
    domain_stopwords: List[str] | None = None,
    desired_topics: int | None = None,
) -> BERTopic:
    device = "cpu" if cpu_only else "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model = SentenceTransformer(embedding_model_name, device=device)

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=42,
        # Force single-thread for strict reproducibility across machines (DH requirement)
        n_jobs=1,
    )

    # Optionally adjust cluster size based on desired number of topics
    if desired_topics and desired_topics > 0:
        try:
            # heuristic: cluster size around N/desired_topics will be set by caller using len(docs)
            # Here we ensure min_topic_size is at least 20 to avoid tiny clusters
            min_topic_size = max(20, int(min_topic_size))
        except Exception:
            pass

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=hdbscan_min_samples,
        metric='euclidean',
        cluster_selection_method=hdbscan_selection_method,
        cluster_selection_epsilon=hdbscan_epsilon,
        prediction_data=True,
        approx_min_span_tree=False,
    )

    # Build combined stopwords
    stopwords = set(DOMAIN_STOPWORDS)
    if domain_stopwords:
        stopwords.update({s.lower() for s in domain_stopwords if s})

    # IMPORTANT: BERTopic's c-TF-IDF works on "documents per topic" where each topic's
    # documents are concatenated into a single mega-document. This means the vectorizer
    # operates on N_topics documents (e.g., ~50-100), not N_docs (~12000).
    # 
    # With few topics, min_df/max_df constraints can conflict:
    # - min_df=10 requires term to appear in 10+ topic-documents
    # - max_df=0.9 with 50 topics = term can appear in max 45 topic-documents
    # - This leaves a very narrow valid range, causing sklearn errors
    #
    # Solution: Use min_df=2 (BERTopic best practice) and max_df=1.0 (no upper limit)
    # Frequency filtering is handled by max_features and stopwords instead.
    
    vectorizer_model = CountVectorizer(
        ngram_range=(vectorizer_ngram_min, vectorizer_ngram_max),
        stop_words=sorted(stopwords) if stopwords else None,
        max_features=vectorizer_max_features,
        min_df=2,  # BERTopic best practice - term must appear in at least 2 topics
        max_df=1.0,  # No upper limit - avoid conflict with min_df on few topics
        encoding='utf-8',
        decode_error='replace',
        # Normalize accents to merge e.g., "août" and "aout"
        strip_accents='unicode',
        lowercase=True,
        token_pattern=r'(?u)\b\w\w+\b',
    )

    representation_model = create_custom_topic_representation()

    # Use BERTopic defaults with tuned components; optionally enable c-TF-IDF reduction of frequent words
    ctfidf_model = None
    if CTFIDF_AVAILABLE:
        try:
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        except Exception:
            ctfidf_model = None
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        # Reduce frequent words helps mitigate boilerplate domination in labels
        ctfidf_model=ctfidf_model,  # keep default if unavailable; rely on stopwords/df thresholds
        verbose=True,
        calculate_probabilities=True,
    )

    return topic_model


def fit_topic_model(
    topic_model: BERTopic,
    docs_clean: List[str],
    embed_texts: List[str],
    model_save_path: Path,
    logger: logging.Logger,
    embedding_model_name: str,
    embedding_batch_size: int,
    reduce_outliers_threshold: float | None = None,
    topic_label_max_words: int = 8,
    nr_topics: int | None = None,
) -> BERTopic:
    logger.info("Entraînement du modèle BERTopic (embeddings=OCR, c-TF-IDF=lemma_nostop)...")
    logger.info(f"Nombre de paires docs (clean+OCR): {len(docs_clean)}")

    # Filtrer les paires valides
    logger.info("Filtrage des paires valides...")
    valid_docs_clean: List[str] = []
    valid_embed_texts: List[str] = []
    for dc, et in tqdm(list(zip(docs_clean, embed_texts)), desc="Filtrage des textes"):
        if dc and str(dc).strip() and et and str(et).strip():
            valid_docs_clean.append(str(dc))
            valid_embed_texts.append(str(et))
    logger.info(f"Nombre de paires valides: {len(valid_docs_clean)}")

    if len(valid_docs_clean) < 50:
        logger.warning("Nombre de documents très faible pour un entraînement robuste.")

    logger.info("Entraînement en cours (cela peut prendre plusieurs minutes)...")
    with tqdm(total=100, desc="Entraînement BERTopic") as pbar:
        pbar.update(10)
        logger.info("Calcul des embeddings (OCR)...")
        embeddings = _embed_texts(
            topic_model.embedding_model,
            valid_embed_texts,
            embedding_batch_size,
            show_progress=True,
        )
        topics, probabilities = topic_model.fit_transform(valid_docs_clean, embeddings=embeddings)
        pbar.update(90)

    # Optionally reduce outliers on the training set
    try:
        if reduce_outliers_threshold is not None and reduce_outliers_threshold > 0:
            logger.info(
                f"Réduction des outliers d'entraînement (seuil={reduce_outliers_threshold})…"
            )
            new_topics = topic_model.reduce_outliers(
                valid_docs_clean, topics, probabilities, strategy="c-tf-idf", threshold=reduce_outliers_threshold
            )
            topic_model.update_topics(valid_docs_clean, topics=new_topics)
            logger.info("Outliers d'entraînement réaffectés et représentations mises à jour.")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Réduction des outliers impossible/ignorée: {e}")

    # Optionally reduce number of topics by merging similar ones
    try:
        if nr_topics is not None and nr_topics > 0:
            current_topics = len(topic_model.get_topic_info()) - 1  # -1 for outlier topic
            if current_topics > nr_topics:
                logger.info(f"Réduction du nombre de topics: {current_topics} → {nr_topics} (fusion des topics similaires)...")
                topic_model.reduce_topics(valid_docs_clean, nr_topics=nr_topics)
                new_count = len(topic_model.get_topic_info()) - 1
                logger.info(f"Topics réduits: {current_topics} → {new_count}")
            else:
                logger.info(f"Nombre de topics ({current_topics}) déjà ≤ nr_topics ({nr_topics}), pas de réduction.")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Réduction des topics impossible/ignorée: {e}")

    # Clean topic labels
    try:
        logger.info("Nettoyage des labels de topics (suppression des doublons)...")
        topic_model = clean_topic_labels(topic_model, max_words=topic_label_max_words)
        logger.info(f"Labels de topics nettoyés (max {topic_label_max_words} mots uniques).")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Nettoyage des labels impossible/ignoré: {e}")

    topic_info = topic_model.get_topic_info()
    logger.info(f"Nombre de sujets découverts: {len(topic_info) - 1}")
    logger.info("Aperçu des sujets principaux:")
    for _, row in topic_info.head(10).iterrows():
        if row['Topic'] != -1:
            logger.info(f"  Sujet {row['Topic']}: {row['Name']} ({row['Count']} docs)")

    logger.info(f"Sauvegarde du modèle entraîné vers: {model_save_path}")
    with tqdm(total=100, desc="Sauvegarde du modèle") as pbar:
        pbar.update(20)
        from shutil import rmtree
        if model_save_path.exists():
            rmtree(model_save_path)

        topic_model.save(
            str(model_save_path),
            serialization="pytorch",
            save_ctfidf=True,
            save_embedding_model=embedding_model_name,
        )
        pbar.update(80)

    return topic_model


def load_topic_model(model_path: Path, logger: logging.Logger) -> BERTopic:
    logger.info(f"Chargement du modèle BERTopic depuis: {model_path}")
    with tqdm(total=100, desc="Chargement du modèle") as pbar:
        pbar.update(30)
        try:
            topic_model = BERTopic.load(str(model_path))
        except UnicodeDecodeError as e:
            logger.warning(
                f"Erreur d'encodage UTF-8 lors du chargement ({e}). Tentative de réparation des fichiers JSON..."
            )
            _repair_model_dir_encoding(model_path, logger)
            topic_model = BERTopic.load(str(model_path))
        pbar.update(70)

    topic_info = topic_model.get_topic_info()
    logger.info(f"Modèle chargé avec {len(topic_info) - 1} sujets")
    return topic_model


def _repair_model_dir_encoding(model_dir: Path, logger: logging.Logger) -> None:
    """Repair JSON files in a saved BERTopic directory by re-encoding to UTF-8.

    Some environments may save JSON using a local ANSI codepage (e.g., cp1252) on Windows.
    This function scans .json files and if they are not valid UTF-8, decodes with a few
    common fallbacks, then writes back as UTF-8.
    """
    if not model_dir.exists():
        return
    for json_path in model_dir.glob("**/*.json"):
        try:
            data = json_path.read_bytes()
            try:
                # Fast-path: already UTF-8
                data.decode("utf-8")
                continue
            except UnicodeDecodeError:
                pass

            text = None
            for enc in ("cp1252", "latin-1"):
                try:
                    text = data.decode(enc)
                    logger.info(f"Ré-encodage de '{json_path.name}' depuis {enc} vers UTF-8")
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                # As a last resort, replace errors with placeholders
                text = data.decode("utf-8", errors="replace")
                logger.info(f"Ré-encodage de '{json_path.name}' avec remplacement d'erreurs vers UTF-8")

            json_path.write_text(text, encoding="utf-8", errors="strict")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Impossible de réparer '{json_path}': {e}")


def predict_topics_batch(
    topic_model: BERTopic,
    batch: Dict[str, List[Any]],
    embed_text_col: str,
    topic_id_col: str,
    topic_prob_col: str,
    topic_label_col: str,
    outlier_reassign_threshold: float | None = None,
    embedding_batch_size: int = 16,
) -> Dict[str, List[Any]]:
    texts = batch[embed_text_col]
    languages = batch.get('language', [None] * len(texts))

    french_indices: List[int] = []
    french_texts: List[str] = []
    for i, text in enumerate(texts):
        lang = languages[i] if i < len(languages) else None
        if lang == 'Français':
            french_indices.append(i)
            french_texts.append(" " if text is None or str(text).strip() == "" else str(text))

    topics: List[Any] = [None] * len(texts)
    probabilities: List[Any] = [None] * len(texts)
    topic_labels: List[Any] = [None] * len(texts)

    if french_texts:
        try:
            embeddings = _embed_texts(
                topic_model.embedding_model,
                french_texts,
                embedding_batch_size,
                show_progress=False,
            )
            french_topics, french_probabilities = topic_model.transform(french_texts, embeddings=embeddings)

            topic_info = topic_model.get_topic_info()
            topic_name_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
            # Keep the order as provided by topic_info to better match probabilities ordering across versions
            non_outlier_topic_ids = [int(t) for t in topic_info['Topic'].tolist() if int(t) != -1]

            for idx, french_idx in enumerate(french_indices):
                topic_id = french_topics[idx]
                if topic_id == -1:
                    reassigned = False
                    if outlier_reassign_threshold is not None and french_probabilities is not None:
                        probs_vec = french_probabilities[idx]
                        if isinstance(probs_vec, (list, np.ndarray)) and len(probs_vec) == len(non_outlier_topic_ids):
                            j = int(np.argmax(probs_vec))
                            best_prob = float(probs_vec[j])
                            if best_prob >= outlier_reassign_threshold:
                                topic_id = int(non_outlier_topic_ids[j])
                                topics[french_idx] = topic_id
                                probabilities[french_idx] = best_prob
                                topic_labels[french_idx] = topic_name_map.get(topic_id, f"Topic_{topic_id}")
                                reassigned = True
                    if not reassigned:
                        topics[french_idx] = -1
                        probabilities[french_idx] = 0.0
                        topic_labels[french_idx] = "Outlier"
                else:
                    topics[french_idx] = topic_id
                    if isinstance(french_probabilities[idx], (list, np.ndarray)):
                        probabilities[french_idx] = float(np.max(french_probabilities[idx]))
                    else:
                        probabilities[french_idx] = float(french_probabilities[idx])
                    topic_labels[french_idx] = topic_name_map.get(topic_id, f"Topic_{topic_id}")

        except Exception as e:  # pragma: no cover - defensive
            logging.error(f"Erreur lors de la prédiction des sujets: {e}")
            for french_idx in french_indices:
                topics[french_idx] = -1
                probabilities[french_idx] = 0.0
                topic_labels[french_idx] = "Error"

    batch[topic_id_col] = topics
    batch[topic_prob_col] = probabilities
    batch[topic_label_col] = topic_labels
    return batch
