"""
modeling.py
-----------
BERTopic model creation, training, loading, and inference utilities.
"""
from __future__ import annotations

import logging
from pathlib import Path
import inspect
from typing import Any, Dict, List

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


def create_custom_topic_representation():
    """Create a custom topic representation that prioritizes NOUN/ADJ and adds diversity.

    Uses PartOfSpeech (French) + KeyBERTInspired via Merge when available.
    Falls back to KeyBERTInspired alone, then to None.
    """
    if not KEYBERT_AVAILABLE:
        return None
    try:  # Try POS + KeyBERT with Merge
        pos = PartOfSpeech(model="fr_core_news_md", allowed_pos={"NOUN", "ADJ"}, top_n=10)
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


def clean_topic_labels(topic_model: BERTopic, max_words: int = 8) -> BERTopic:
    """Clean topic labels by removing duplicates and limiting word count."""
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            continue

        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            continue

        seen_words = set()
        unique_words: List[str] = []
        for word, _ in topic_words:
            w_norm = _normalize_token(word)
            if not w_norm:
                continue
            # skip if token or its normalized form is in label-only or vectorizer stopwords
            if (w_norm in LABEL_ONLY_STOPWORDS) or (w_norm in DOMAIN_STOPWORDS):
                continue
            if w_norm in seen_words:
                continue
            seen_words.add(w_norm)
            unique_words.append(word)
            if len(unique_words) >= max_words:
                break

        if unique_words:
            # shorter, cleaner label: join with " - " and no id prefix (BERTopic UI shows id)
            new_label = " - ".join(unique_words)
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
    # UMAP params
    umap_n_neighbors: int = 50,
    umap_min_dist: float = 0.1,
    umap_n_components: int = 10,
    umap_metric: str = 'cosine',
    # HDBSCAN params
    hdbscan_min_samples: int = 5,
    hdbscan_selection_method: str = 'leaf',
    hdbscan_epsilon: float = 0.0,
    # Vectorizer params
    vectorizer_min_df: int = 10,
    vectorizer_max_df: float = 0.9,
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
        n_jobs=-1 if device == "cpu" else 1,
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

    vectorizer_model = CountVectorizer(
        ngram_range=(vectorizer_ngram_min, vectorizer_ngram_max),
        stop_words=sorted(stopwords) if stopwords else None,
        max_features=vectorizer_max_features,
        min_df=vectorizer_min_df,
        max_df=vectorizer_max_df,
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
