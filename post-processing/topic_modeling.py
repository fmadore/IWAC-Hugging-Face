#!/usr/bin/env python3
"""
topic_modeling.py
=================

Ajoute des colonnes avec la modélisation de sujets à un dataset Hugging Face 
existant, basées sur la colonne 'lemma_nostop' (texte lemmatisé sans mots vides).
Le script utilise BERTopic avec CamemBERT pour identifier les sujets principaux
et ajoute les résultats dans de nouvelles colonnes.

Usage
-----
    python post-processing/topic_modeling.py [--repo MON_USER/MON_DATASET]

Variables d'environnement
---------------------
HF_TOKEN   Jeton d'accès personnel pour le Hugging Face Hub

Dépendances
-----------
    pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn
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
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from collections import Counter

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

topic_model = None

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

def create_bertopic_model(embedding_model_name: str, min_topic_size: int = 10) -> BERTopic:
    embedding_model = SentenceTransformer(embedding_model_name)
    
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine',
        random_state=42
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=None,
        max_features=5000,
        min_df=2,
        max_df=0.95
    )
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_k_words=10,
        verbose=True,
        calculate_probabilities=True
    )
    
    return topic_model

def fit_topic_model(texts: List[str], model_save_path: Path, logger: logging.Logger) -> BERTopic:
    global topic_model
    
    logger.info("Entraînement du modèle BERTopic...")
    logger.info(f"Nombre de documents: {len(texts)}")
    
    valid_texts = [text for text in texts if text and text.strip()]
    logger.info(f"Nombre de documents valides: {len(valid_texts)}")
    
    if len(valid_texts) < 50:
        logger.warning("Nombre de documents très faible pour un entraînement robuste.")
    
    topics, probabilities = topic_model.fit_transform(valid_texts)
    
    topic_info = topic_model.get_topic_info()
    logger.info(f"Nombre de sujets découverts: {len(topic_info) - 1}")
    logger.info(f"Aperçu des sujets principaux:")
    for i, row in topic_info.head(10).iterrows():
        if row['Topic'] != -1:
            logger.info(f"  Sujet {row['Topic']}: {row['Name']} ({row['Count']} docs)")
    
    logger.info(f"Sauvegarde du modèle entraîné vers: {model_save_path}")
    topic_model.save(str(model_save_path))
    
    return topic_model

def load_topic_model(model_path: Path, logger: logging.Logger) -> BERTopic:
    logger.info(f"Chargement du modèle BERTopic depuis: {model_path}")
    topic_model = BERTopic.load(str(model_path))
    
    topic_info = topic_model.get_topic_info()
    logger.info(f"Modèle chargé avec {len(topic_info) - 1} sujets")
    
    return topic_model

def predict_topics_batch(batch: Dict[str, List[Any]], text_col: str, 
                        topic_id_col: str, topic_prob_col: str, topic_label_col: str) -> Dict[str, List[Any]]:
    global topic_model
    
    texts = batch[text_col]
    
    processed_texts = []
    for text in texts:
        if text is None or text.strip() == "":
            processed_texts.append(" ")
        else:
            processed_texts.append(str(text))
    
    try:
        topics, probabilities = topic_model.transform(processed_texts)
        
        # Obtenir les informations des sujets une seule fois pour optimiser
        topic_info = topic_model.get_topic_info()
        topic_name_map = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
        
        topic_labels = []
        for topic_id in topics:
            if topic_id == -1:
                topic_labels.append("Outlier")
            else:
                topic_labels.append(topic_name_map.get(topic_id, f"Topic_{topic_id}"))
        
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction des sujets: {e}")
        topics = [-1] * len(texts)
        probabilities = [0.0] * len(texts)
        topic_labels = ["Error"] * len(texts)
    
    batch[topic_id_col] = topics
    batch[topic_prob_col] = probabilities
    batch[topic_label_col] = topic_labels
    
    return batch

def main():
    global topic_model
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Ajoute des colonnes de modélisation de sujets à un dataset Hugging Face."
    )
    parser.add_argument("--repo", default="fmadore/iwac-newspaper-articles")
    parser.add_argument("--embedding-model", default="sentence-transformers/camembert-base", 
                        help="Modèle d'embedding à utiliser (recommandé: camembert-base pour le français)")
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--model-path", default="bertopic_model")
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--batch-size", type=int, default=100)
    
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
        ds = load_dataset(repo_id, name=config_name_choice, split="train", token=token, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        return

    logger.info(f"Dataset chargé. Nombre de lignes: {len(ds)}")

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
        topic_model = create_bertopic_model(embedding_model_name, min_topic_size)
        
        texts = ds[text_column_name]
        valid_texts = [text for text in texts if text and text.strip()]
        
        if len(valid_texts) < min_topic_size:
            logger.error(f"Nombre de textes valides ({len(valid_texts)}) < min_topic_size ({min_topic_size})")
            return
        
        topic_model = fit_topic_model(valid_texts, model_path, logger)
        
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
            "topic_label_col": topic_label_column_name
        },
        batched=True,
        batch_size=batch_size,
        desc="Prédiction des sujets",
    )

    logger.info("Modélisation terminée.")
    
    # Statistiques
    topic_ids = ds_processed[topic_id_column_name]
    topic_probs = ds_processed[topic_prob_column_name]
    topic_labels = ds_processed[topic_label_column_name]
    
    unique_topics = set(topic_ids)
    logger.info(f"Nombre de sujets uniques: {len(unique_topics)}")
    
    valid_probs = [p for p in topic_probs if p > 0]
    if valid_probs:
        logger.info(f"Probabilité moyenne: {np.mean(valid_probs):.3f}")
    else:
        logger.info("Aucune probabilité valide trouvée")
    
    topic_counts = Counter(topic_ids)
    logger.info("Top 10 des sujets les plus fréquents:")
    for topic_id, count in topic_counts.most_common(10):
        # Trouver le label correspondant
        label = next((label for tid, label in zip(topic_ids, topic_labels) if tid == topic_id), f"Topic_{topic_id}")
        logger.info(f"  Sujet {topic_id}: {label} ({count} documents)")

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
    logger.info(f"Nombre de sujets découverts: {len(unique_topics)}")
    logger.info(f"Modèle BERTopic sauvegardé à: {model_path}")
    if modeling_mode == "fit":
        logger.info(f"Taille minimale des sujets: {min_topic_size}")
        logger.info(f"Nombre de documents traités: {len(ds)}")

if __name__ == "__main__":
    main()