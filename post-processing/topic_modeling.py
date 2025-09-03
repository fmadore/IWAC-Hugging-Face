#!/usr/bin/env python3
"""
topic_modeling.py
=================

Ajoute des colonnes avec la modélisation de sujets à un dataset Hugging Face 
existant, basées sur la colonne 'lemma_nostop' (texte lemmatisé sans mots vides).
Le script utilise BERTopic avec CamemBERT pour identifier les sujets principaux
et ajoute les résultats dans de nouvelles colonnes.

IMPORTANT: Seuls les documents avec language='Français' sont traités par le modèle.
Les documents d'autres langues ou sans indication de langue conservent des valeurs
vides (None) dans les colonnes de sujets (topic_id, topic_prob, topic_label).

Usage
-----
    python post-processing/topic_modeling.py [--repo MON_USER/MON_DATASET]

Variables d'environnement
---------------------
HF_TOKEN   Jeton d'accès personnel pour le Hugging Face Hub

Dépendances
-----------
    pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn tqdm torch

Modèles d'embedding français recommandés
---------------------------------------
    --embedding-model dangvantuan/sentence-camembert-base     # Modèle base (110M params)
    --embedding-model dangvantuan/sentence-camembert-large    # Modèle large (336M params, meilleur)
    --embedding-model Lajavaness/sentence-camembert-large     # Alternative optimisée

Optimisations CPU
-----------------
    Pour machines sans GPU, utilisez les options :
    --cpu-only                    # Force l'utilisation du CPU
    --max-documents 10000         # Limite le nombre de documents pour tests
    --embedding-batch-size 16     # Réduit la taille des batches si mémoire limitée
    --min-topic-size 20           # Augmente la taille min des sujets pour réduire le bruit
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfFolder, login
from bertopic import BERTopic
try:
    from bertopic.representation import KeyBERTInspired
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from collections import Counter
from tqdm import tqdm
import torch
import shutil
import json
import unicodedata
import re
import functools
import builtins

# Force UTF-8 encoding for all file operations to prevent issues on Windows
# This is a global patch that affects all `open` calls in the script
original_open = builtins.open
@functools.wraps(original_open)
def patched_open(file, mode='r', *args, **kwargs):
    # Only patch text modes
    if 'b' not in mode:
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
            kwargs.setdefault('errors', 'replace')
    return original_open(file, mode, *args, **kwargs)
builtins.open = patched_open

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

topic_model = None

def create_custom_topic_representation():
    """Create a custom topic representation that removes duplicates and creates cleaner labels."""
    if KEYBERT_AVAILABLE:
        # Use KeyBERTInspired to get diverse keywords
        keybert_model = KeyBERTInspired()
        return keybert_model
    else:
        # Fallback to default representation
        return None

def clean_topic_labels(topic_model, max_words=8):
    """Clean topic labels by removing duplicates and limiting word count."""
    topic_info = topic_model.get_topic_info()
    
    for idx, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:  # Skip outlier topic
            continue
            
        # Get the original topic words
        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            continue
            
        # Extract unique words (case-insensitive deduplication)
        seen_words = set()
        unique_words = []
        
        for word, score in topic_words:
            word_lower = word.lower()
            if word_lower not in seen_words:
                seen_words.add(word_lower)
                unique_words.append(word)
                if len(unique_words) >= max_words:
                    break
        
        # Create new label
        if unique_words:
            new_label = f"{topic_id}_" + "_".join(unique_words)
            # Update the topic info
            topic_info.loc[topic_info['Topic'] == topic_id, 'Name'] = new_label
    
    # Update the topic model's topic info
    topic_model.topic_labels_ = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
    
    return topic_model

def get_available_configs(repo_id: str, token: str) -> List[str]:
    try:
        from huggingface_hub import dataset_info
        info = dataset_info(repo_id, token=token)
        if hasattr(info, 'config_names') and info.config_names:
            return info.config_names
        else:
            return ['articles', 'publications']
    except Exception:
        return ['articles', 'publications']

def choose_config(available_configs: List[str]) -> str:
    if len(available_configs) == 1:
        print(f"Une seule configuration disponible: '{available_configs[0]}'")
        return available_configs[0]
    
    print("Configurations disponibles:")
    for i, config in enumerate(available_configs, 1):
        print(f"  {i}. {config}")
    
    while True:
        try:
            choice = input(f"Choisissez une configuration (1-{len(available_configs)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_configs):
                return available_configs[choice_idx]
            else:
                print(f"Veuillez entrer un nombre entre 1 et {len(available_configs)}.")
        except ValueError:
            print("Veuillez entrer un nombre valide.")
        except KeyboardInterrupt:
            print("\nOpération annulée.")
            exit(0)

def choose_modeling_mode() -> str:
    print("\nMode de modélisation des sujets:")
    print("  1. Entraîner un nouveau modèle BERTopic (recommandé)")
    print("  2. Utiliser un modèle BERTopic existant")
    
    while True:
        try:
            choice = input("Choisissez un mode (1-2): ").strip()
            if choice == "1":
                return "fit"
            elif choice == "2":
                return "predict"
            else:
                print("Veuillez entrer 1 ou 2.")
        except KeyboardInterrupt:
            print("\nOpération annulée.")
            exit(0)

def create_bertopic_model(
    embedding_model_name: str,
    min_topic_size: int = 10,
    cpu_only: bool = False,
    embedding_batch_size: int = 32,
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
    vectorizer_min_df: int = 5,
    vectorizer_max_features: int = 10000,
    vectorizer_ngram_min: int = 1,
    vectorizer_ngram_max: int = 2,
) -> BERTopic:
    # Configuration pour CPU si demandé
    device = "cpu" if cpu_only else "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=42,
        # Optimisation CPU : utiliser tous les cœurs disponibles
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
        stop_words=None,
        max_features=vectorizer_max_features,
        min_df=vectorizer_min_df,
        max_df=0.95,
        encoding='utf-8',  # Explicitly set UTF-8 encoding
        decode_error='replace',  # Replace invalid characters instead of failing
        strip_accents=None,  # Don't strip accents to preserve French characters
        lowercase=True,
        token_pattern=r'(?u)\b\w\w+\b'  # Unicode-aware token pattern
    )
    
    representation_model = create_custom_topic_representation()
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True,
        calculate_probabilities=True
    )
    
    return topic_model

def fit_topic_model(
    texts: List[str],
    model_save_path: Path,
    logger: logging.Logger,
    embedding_model_name: str,
    reduce_outliers_threshold: float | None = None,
    topic_label_max_words: int = 8,
) -> BERTopic:
    global topic_model
    
    logger.info("Entraînement du modèle BERTopic...")
    logger.info(f"Nombre de documents: {len(texts)}")
    
    # Filtrage des textes valides avec barre de progression
    logger.info("Filtrage des textes valides...")
    valid_texts = [text for text in tqdm(texts, desc="Filtrage des textes") if text and text.strip()]
    logger.info(f"Nombre de documents valides: {len(valid_texts)}")
    
    if len(valid_texts) < 50:
        logger.warning("Nombre de documents très faible pour un entraînement robuste.")
    
    logger.info("Entraînement en cours (cela peut prendre plusieurs minutes)...")
    with tqdm(total=100, desc="Entraînement BERTopic") as pbar:
        # L'entraînement BERTopic n'a pas de callback de progression intégré
        # donc nous simulons une progression
        pbar.update(10)  # Début de l'entraînement
        topics, probabilities = topic_model.fit_transform(valid_texts)
        pbar.update(90)  # Fin de l'entraînement

    # Optionally reduce outliers on the training set and update topic representations
    try:
        if reduce_outliers_threshold is not None and reduce_outliers_threshold > 0:
            logger.info(f"Réduction des outliers d'entraînement (seuil={reduce_outliers_threshold})…")
            new_topics = topic_model.reduce_outliers(
                valid_texts, topics, probabilities, strategy="c-tf-idf", threshold=reduce_outliers_threshold
            )
            # Mettre à jour les représentations avec les nouveaux topics
            topic_model.update_topics(valid_texts, topics=new_topics)
            logger.info("Outliers d'entraînement réaffectés et représentations mises à jour.")
    except Exception as e:
        logger.warning(f"Réduction des outliers impossible/ignorée: {e}")
    
    # Clean topic labels to remove duplicates
    try:
        logger.info("Nettoyage des labels de topics (suppression des doublons)...")
        topic_model = clean_topic_labels(topic_model, max_words=topic_label_max_words)
        logger.info(f"Labels de topics nettoyés (max {topic_label_max_words} mots uniques).")
    except Exception as e:
        logger.warning(f"Nettoyage des labels impossible/ignoré: {e}")
    
    topic_info = topic_model.get_topic_info()
    logger.info(f"Nombre de sujets découverts: {len(topic_info) - 1}")
    logger.info(f"Aperçu des sujets principaux:")
    for i, row in topic_info.head(10).iterrows():
        if row['Topic'] != -1:
            logger.info(f"  Sujet {row['Topic']}: {row['Name']} ({row['Count']} docs)")
    
    logger.info(f"Sauvegarde du modèle entraîné vers: {model_save_path}")
    with tqdm(total=100, desc="Sauvegarde du modèle") as pbar:
        pbar.update(20)
        # Remove existing directory if it exists
        if model_save_path.exists():
            shutil.rmtree(model_save_path)
        
        topic_model.save(
            str(model_save_path),
            serialization="pytorch",  # avoid pickle/joblib issues with Python 3.12
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
    batch: Dict[str, List[Any]],
    text_col: str,
    topic_id_col: str,
    topic_prob_col: str,
    topic_label_col: str,
    outlier_reassign_threshold: float | None = None,
) -> Dict[str, List[Any]]:
    global topic_model
    
    texts = batch[text_col]
    languages = batch.get('language', [None] * len(texts))  # Get language info if available
    
    # Identify French entries and prepare texts for processing
    french_indices = []
    french_texts = []
    
    for i, text in enumerate(texts):
        lang = languages[i] if i < len(languages) else None
        # Only process entries with French language
        if lang == 'Français':
            french_indices.append(i)
            if text is None or text.strip() == "":
                french_texts.append(" ")  # Empty placeholder for empty French text
            else:
                french_texts.append(str(text))
    
    # Initialize result arrays with None values
    topics = [None] * len(texts)
    probabilities = [None] * len(texts)
    topic_labels = [None] * len(texts)
    
    # Only process French texts if there are any
    if french_texts:
        try:
            # Transform only French texts
            french_topics, french_probabilities = topic_model.transform(french_texts)
            
            # Get topic information for label mapping
            topic_info = topic_model.get_topic_info()
            topic_name_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
            # Mapping for probability columns (non-outlier topics only)
            non_outlier_topic_ids = [int(t) for t in topic_info['Topic'].tolist() if int(t) != -1]
            
            # Assign results back to the corresponding positions
            for idx, french_idx in enumerate(french_indices):
                topic_id = french_topics[idx]
                
                if topic_id == -1:
                    # Try to reassign outliers to the most probable non-outlier topic when confident
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
                    # Extract max probability for this document
                    if isinstance(french_probabilities[idx], (list, np.ndarray)):
                        probabilities[french_idx] = float(np.max(french_probabilities[idx]))
                    else:
                        probabilities[french_idx] = float(french_probabilities[idx])
                    topic_labels[french_idx] = topic_name_map.get(topic_id, f"Topic_{topic_id}")
            
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction des sujets: {e}")
            # In case of error, set French entries to error values, keep non-French as None
            for french_idx in french_indices:
                topics[french_idx] = -1
                probabilities[french_idx] = 0.0
                topic_labels[french_idx] = "Error"
    # If no French texts in this batch, all values remain None (which is what we want)
    
    batch[topic_id_col] = topics
    batch[topic_prob_col] = probabilities
    batch[topic_label_col] = topic_labels
    
    return batch

# Patch the default JSON encoder so that NumPy types (e.g. np.int64, np.float32, np.ndarray)
# are automatically converted to their native Python equivalents when dumping to JSON. This
# prevents errors such as "TypeError: Object of type int64 is not JSON serializable" that
# can occur when BERTopic attempts to save its configuration files.
class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):  # noqa: D401, N802  (keep signature to respect json API)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Globally patch both the default encoder instance and the class used when
# `json.dump` needs to instantiate a new encoder (e.g. when a non-default
# `indent` argument is supplied).
json.JSONEncoder = _NumpyJSONEncoder
json._default_encoder = _NumpyJSONEncoder()

# Ensure that non-ASCII characters (e.g. é, à, ñ) are written to JSON files plainly
# instead of being escaped as \uXXXX sequences. If callers explicitly set the
# "ensure_ascii" parameter we respect their choice; otherwise we default it to False.
_original_json_dump = json.dump
_original_json_dumps = json.dumps

def _patched_dump(obj, fp, *args, **kwargs):
    kwargs.setdefault("ensure_ascii", False)
    return _original_json_dump(obj, fp, *args, **kwargs)

def _patched_dumps(obj, *args, **kwargs):
    kwargs.setdefault("ensure_ascii", False)
    return _original_json_dumps(obj, *args, **kwargs)

json.dump = _patched_dump
json.dumps = _patched_dumps

def main():
    global topic_model
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Ajoute des colonnes de modélisation de sujets à un dataset Hugging Face."
    )
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection")
    parser.add_argument("--embedding-model", default="dangvantuan/sentence-camembert-base", 
                        help="Modèle d'embedding à utiliser (recommandé: sentence-camembert-base pour le français)")
    parser.add_argument("--min-topic-size", type=int, default=5)
    parser.add_argument("--model-path", default="bertopic_model")
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cpu-only", action="store_true", 
                        help="Force l'utilisation du CPU uniquement (optimisations pour machines sans GPU)")
    parser.add_argument("--max-documents", type=int, default=None,
                        help="Limite le nombre de documents pour les tests (utile pour CPU)")
    parser.add_argument("--embedding-batch-size", type=int, default=16,
                        help="Taille des batches pour les embeddings (réduire si mémoire limitée)")
    parser.add_argument("--min-train-tokens", type=int, default=5,
                        help="Longueur minimale (en tokens) pour inclure un texte dans l'entraînement")
    # UMAP/HDBSCAN/Vectorizer advanced tuning
    parser.add_argument("--umap-n-neighbors", type=int, default=60, help="UMAP n_neighbors (plus grand = moins d'outliers)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--umap-n-components", type=int, default=10, help="UMAP n_components")
    parser.add_argument("--umap-metric", type=str, default="cosine", help="UMAP metric")
    parser.add_argument("--hdbscan-min-samples", type=int, default=3, help="HDBSCAN min_samples (plus petit = moins d'outliers)")
    parser.add_argument("--hdbscan-selection-method", type=str, choices=["eom", "leaf"], default="leaf",
                        help="HDBSCAN cluster_selection_method")
    parser.add_argument("--hdbscan-epsilon", type=float, default=0.0, help="HDBSCAN cluster_selection_epsilon")
    parser.add_argument("--vectorizer-min-df", type=int, default=5, help="CountVectorizer min_df")
    parser.add_argument("--vectorizer-max-features", type=int, default=8000, help="CountVectorizer max_features")
    parser.add_argument("--vectorizer-ngrams", type=str, default="1,2", help="CountVectorizer ngram range 'a,b'")
    # Outlier reduction options
    parser.add_argument("--reduce-outliers-train", type=float, default=0.35,
                        help="Seuil (0-1) pour réduire/réassigner les outliers à l'entraînement via c-TF-IDF. 0=désactivé")
    parser.add_argument("--outlier-reassign-threshold", type=float, default=0.35,
                        help="Réassigne un outlier à la prédiction si la meilleure proba >= seuil (0-1). 0=jamais")
    parser.add_argument("--topic-label-max-words", type=int, default=8,
                        help="Nombre maximum de mots uniques dans les labels de topics (défaut: 8)")
    
    args = parser.parse_args()

    repo_id = args.repo
    embedding_model_name = args.embedding_model
    text_column_name = "lemma_nostop"
    topic_id_column_name = "topic_id"
    topic_prob_column_name = "topic_prob"
    topic_label_column_name = "topic_label"
    min_topic_size = args.min_topic_size
    model_path = Path(args.model_path)
    max_shard_size = args.max_shard_size
    batch_size = args.batch_size
    cpu_only = args.cpu_only
    max_documents = args.max_documents
    embedding_batch_size = args.embedding_batch_size

    # Authentification
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        logger.info("Token Hugging Face non trouvé. Tentative de connexion interactive.")
        try:
            login()
            token = HfFolder.get_token()
            if not token:
                logger.error("Connexion interactive échouée.")
                return
        except Exception as e:
            logger.error(f"Erreur lors de la connexion: {e}")
            return

    # Choix de la configuration
    available_configs = get_available_configs(repo_id, token)
    config_name_choice = choose_config(available_configs)
    logger.info(f"Configuration choisie: '{config_name_choice}'")
    
    # Choix du mode
    modeling_mode = choose_modeling_mode()
    logger.info(f"Mode choisi: '{modeling_mode}'")

    # Chargement du dataset
    logger.info(f"Chargement du dataset '{repo_id}', configuration '{config_name_choice}'...")
    try:
        ds = load_dataset(repo_id, name=config_name_choice, split="train", token=token)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        return

    logger.info(f"Dataset chargé. Nombre de lignes: {len(ds)}")

    # Vérifier la distribution des langues
    if 'language' in ds.column_names:
        languages = ds['language']
        french_count = sum(1 for lang in languages if lang == 'Français')
        other_count = sum(1 for lang in languages if lang and lang != 'Français')
        empty_count = sum(1 for lang in languages if not lang or lang.strip() == '')
        
        logger.info(f"Statistiques des langues:")
        logger.info(f"  - Français: {french_count} (seront traités pour la modélisation)")
        logger.info(f"  - Autres langues: {other_count} (conservés avec colonnes vides)")
        logger.info(f"  - Vides/manquants: {empty_count} (conservés avec colonnes vides)")
        logger.info(f"  - Total: {len(ds)}")
        
        if french_count == 0:
            logger.error("Aucun document français trouvé. La modélisation ne peut pas continuer.")
            return
    else:
        logger.warning("Colonne 'language' non trouvée. Tous les textes seront traités.")

    if text_column_name not in ds.column_names:
        logger.error(f"Colonne '{text_column_name}' non trouvée. Colonnes disponibles: {ds.column_names}")
        return

    # Vérifier si les colonnes de sujets existent déjà
    new_columns = [topic_id_column_name, topic_prob_column_name, topic_label_column_name]
    existing_topic_columns = [col for col in new_columns if col in ds.column_names]
    
    if existing_topic_columns:
        logger.warning(f"Les colonnes suivantes existent déjà et seront écrasées: {existing_topic_columns}")
        try:
            confirm = input("Voulez-vous continuer et écraser ces colonnes? (o/N): ").strip().lower()
            if confirm not in ['o', 'oui', 'y', 'yes']:
                logger.info("Opération annulée par l'utilisateur.")
                return
        except KeyboardInterrupt:
            logger.info("\nOpération annulée.")
            return

    # Préparation du modèle
    if modeling_mode == "fit":
        if cpu_only:
            logger.info("Mode CPU activé - optimisations pour machines sans GPU")
            logger.info(f"Taille des batches d'embeddings: {embedding_batch_size}")
        
        # Parse ngram range
        try:
            ngram_min, ngram_max = [int(x.strip()) for x in args.vectorizer_ngrams.split(",")]
        except Exception:
            ngram_min, ngram_max = 1, 2
        topic_model = create_bertopic_model(
            embedding_model_name,
            min_topic_size,
            cpu_only,
            embedding_batch_size,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_n_components=args.umap_n_components,
            umap_metric=args.umap_metric,
            hdbscan_min_samples=args.hdbscan_min_samples,
            hdbscan_selection_method=args.hdbscan_selection_method,
            hdbscan_epsilon=args.hdbscan_epsilon,
            vectorizer_min_df=args.vectorizer_min_df,
            vectorizer_max_features=args.vectorizer_max_features,
            vectorizer_ngram_min=ngram_min,
            vectorizer_ngram_max=ngram_max,
        )
        
        logger.info("Extraction et validation des textes français pour l'entraînement...")
        
        # Extraire seulement les textes français pour l'entraînement du modèle
        if 'language' in ds.column_names:
            french_texts = []
            for i, (text, lang) in enumerate(zip(ds[text_column_name], ds['language'])):
                if lang == 'Français' and text and text.strip():
                    if len(str(text).split()) >= args.min_train_tokens:
                        french_texts.append(text)
            texts = french_texts
            logger.info(f"Textes français valides extraits: {len(texts)}")
        else:
            texts = ds[text_column_name]
            logger.info("Colonne langue non disponible, utilisation de tous les textes")
        
        # Limitation optionnelle du nombre de documents (utile pour tests CPU)
        if max_documents and len(texts) > max_documents:
            logger.info(f"Limitation à {max_documents} documents pour optimiser les performances CPU")
            texts = texts[:max_documents]
        
        valid_texts = [text for text in tqdm(texts, desc="Validation des textes") if text and text.strip()]
        
        if len(valid_texts) < min_topic_size:
            logger.error(f"Nombre de textes valides ({len(valid_texts)}) < min_topic_size ({min_topic_size})")
            return
        
        topic_model = fit_topic_model(
            valid_texts,
            model_path,
            logger,
            embedding_model_name,
            reduce_outliers_threshold=(args.reduce_outliers_train if args.reduce_outliers_train > 0 else None),
            topic_label_max_words=args.topic_label_max_words,
        )
        
    else:
        if not model_path.exists():
            logger.error(f"Modèle non trouvé: {model_path}")
            return
        
        topic_model = load_topic_model(model_path, logger)

    # Application de la modélisation
    logger.info("Application de la modélisation de sujets...")
    
    ds_processed = ds.map(
        predict_topics_batch,
        fn_kwargs={
            "text_col": text_column_name,
            "topic_id_col": topic_id_column_name,
            "topic_prob_col": topic_prob_column_name,
            "topic_label_col": topic_label_column_name,
            "outlier_reassign_threshold": (args.outlier_reassign_threshold if args.outlier_reassign_threshold and args.outlier_reassign_threshold > 0 else None),
        },
        batched=True,
        batch_size=batch_size,
        desc="Prédiction des sujets",
    )

    logger.info("Modélisation terminée.")
    
    # Statistiques
    logger.info("Calcul des statistiques...")
    topic_ids = ds_processed[topic_id_column_name]
    topic_probs = ds_processed[topic_prob_column_name]
    topic_labels = ds_processed[topic_label_column_name]
    
    # Count processed vs skipped rows
    processed_count = sum(1 for tid in topic_ids if tid is not None)
    skipped_count = sum(1 for tid in topic_ids if tid is None)
    
    logger.info(f"Lignes traitées (français): {processed_count}")
    logger.info(f"Lignes ignorées (non-français/vides): {skipped_count}")
    logger.info(f"Total des lignes: {len(topic_ids)}")
    
    # Filter out None values for statistics
    valid_topic_ids = [tid for tid in topic_ids if tid is not None]
    valid_topic_probs = [prob for prob in topic_probs if prob is not None]
    valid_topic_labels = [label for label in topic_labels if label is not None]
    
    if valid_topic_ids:
        unique_topics = set(valid_topic_ids)
        logger.info(f"Nombre de sujets uniques: {len(unique_topics)}")
        
        valid_probs = [p for p in valid_topic_probs if p > 0]
        if valid_probs:
            logger.info(f"Probabilité moyenne: {np.mean(valid_probs):.3f}")
        else:
            logger.info("Aucune probabilité valide trouvée")
        
        topic_counts = Counter(valid_topic_ids)
        logger.info("Top 10 des sujets les plus fréquents:")
        for topic_id, count in topic_counts.most_common(10):
            # Trouver le label correspondant
            label = next((label for tid, label in zip(valid_topic_ids, valid_topic_labels) if tid == topic_id), f"Topic_{topic_id}")
            logger.info(f"  Sujet {topic_id}: {label} ({count} documents)")
    else:
        logger.warning("Aucun document français n'a été traité pour la modélisation.")
        unique_topics = set()

    # Réorganisation des colonnes
    insert_after_col = "lemma_nostop"
    
    existing_columns = list(ds_processed.column_names)
    
    if insert_after_col in existing_columns:
        insert_index = existing_columns.index(insert_after_col) + 1
        new_column_order = existing_columns[:insert_index]
        
        for col in new_columns:
            if col in existing_columns and col not in new_column_order:
                new_column_order.append(col)
        
        for col in existing_columns[insert_index:]:
            if col not in new_column_order:
                new_column_order.append(col)
        
        ds_processed = ds_processed.select_columns(new_column_order)
        logger.info("Colonnes réorganisées.")

    # Sauvegarde
    logger.info("Sauvegarde du dataset...")
    try:
        commit_message = f"Ajout modélisation sujets ({', '.join(new_columns)}) avec BERTopic"
        ds_processed.push_to_hub(
            repo_id=repo_id,
            config_name=config_name_choice,
            commit_message=commit_message,
            token=token,
            max_shard_size=max_shard_size,
        )
        logger.info("Dataset sauvegardé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        return

    logger.info(f"Processus terminé. Colonnes {new_columns} ajoutées avec succès.")
    logger.info(f"Modèle d'embedding utilisé: {embedding_model_name}")
    if valid_topic_ids:
        logger.info(f"Nombre de sujets découverts: {len(unique_topics)}")
    logger.info(f"Modèle BERTopic sauvegardé à: {model_path}")
    if modeling_mode == "fit":
        logger.info(f"Taille minimale des sujets: {min_topic_size}")
        logger.info(f"Documents français traités: {processed_count}")
        logger.info(f"Documents non-français ignorés: {skipped_count}")
        logger.info(f"Total des documents: {len(ds)}")

if __name__ == "__main__":
    main()