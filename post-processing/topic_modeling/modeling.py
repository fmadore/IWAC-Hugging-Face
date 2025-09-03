"""
modeling.py
-----------
BERTopic model creation, training, loading, and inference utilities.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from umap import UMAP

from .constants import DOMAIN_STOPWORDS

try:
    from bertopic.representation import KeyBERTInspired

    KEYBERT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    KEYBERT_AVAILABLE = False


def create_custom_topic_representation():
    """Create a custom topic representation that removes duplicates and creates cleaner labels."""
    if KEYBERT_AVAILABLE:
        return KeyBERTInspired()
    return None


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
            w = word.lower()
            if w not in seen_words:
                seen_words.add(w)
                unique_words.append(word)
                if len(unique_words) >= max_words:
                    break

        if unique_words:
            new_label = f"{topic_id}_" + "_".join(unique_words)
            topic_info.loc[topic_info['Topic'] == topic_id, 'Name'] = new_label

    topic_model.topic_labels_ = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
    return topic_model


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

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=hdbscan_min_samples,
        metric='euclidean',
        cluster_selection_method=hdbscan_selection_method,
        cluster_selection_epsilon=hdbscan_epsilon,
        prediction_data=True,
        approx_min_span_tree=False,
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(vectorizer_ngram_min, vectorizer_ngram_max),
        stop_words=sorted(DOMAIN_STOPWORDS) if DOMAIN_STOPWORDS else None,
        max_features=vectorizer_max_features,
        min_df=vectorizer_min_df,
        max_df=vectorizer_max_df,
        encoding='utf-8',
        decode_error='replace',
        strip_accents=None,
        lowercase=True,
        token_pattern=r'(?u)\b\w\w+\b',
    )

    representation_model = create_custom_topic_representation()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
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
        embeddings = topic_model.embedding_model.encode(
            valid_embed_texts,
            batch_size=embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
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
        topic_model = BERTopic.load(str(model_path))
        pbar.update(70)

    topic_info = topic_model.get_topic_info()
    logger.info(f"Modèle chargé avec {len(topic_info) - 1} sujets")
    return topic_model


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
            embeddings = topic_model.embedding_model.encode(
                french_texts,
                batch_size=embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            french_topics, french_probabilities = topic_model.transform(french_texts, embeddings=embeddings)

            topic_info = topic_model.get_topic_info()
            topic_name_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
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
