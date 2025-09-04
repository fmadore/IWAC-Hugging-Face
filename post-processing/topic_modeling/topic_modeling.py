#!/usr/bin/env python3
"""
topic_modeling.py
=================

Ajoute des colonnes avec la modélisation de sujets à un dataset Hugging Face.

Ce fichier est désormais un orchestrateur fin exploitant des modules:
- patches.py      -> correctifs globaux (UTF-8, JSON/NumPy)
- utils.py        -> logging, CLI, sélection de configuration
- modeling.py     -> création/chargement/entraînement BERTopic, prédiction par lot
- constants.py    -> constantes partagées (stopwords, etc.)
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfFolder, login
from tqdm import tqdm

import sys

# Ensure package imports work when running this file directly
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from topic_modeling.patches import apply_all_patches  # type: ignore
from topic_modeling.utils import (
    configure_logging,
    get_available_configs,
    choose_config,
    choose_modeling_mode,
    build_arg_parser,
)  # type: ignore
from topic_modeling.modeling import (  # type: ignore
    create_bertopic_model,
    fit_topic_model,
    load_topic_model,
    predict_topics_batch as predict_topics_batch_impl,
)


def main() -> None:
    # Apply global patches (UTF-8 I/O, JSON/NumPy handling)
    apply_all_patches()

    configure_logging()
    logger = logging.getLogger(__name__)

    parser = build_arg_parser()
    args = parser.parse_args()

    repo_id = args.repo
    embedding_model_name = args.embedding_model
    text_column_name = "lemma_nostop"  # used for vectorization/labels (docs_clean)
    embed_text_column_name = "OCR"      # used for embeddings (full French sentences)
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
    if embed_text_column_name not in ds.column_names:
        logger.error(f"Colonne d'embeddings '{embed_text_column_name}' non trouvée. Colonnes disponibles: {ds.column_names}")
        logger.error("Le script attend 'OCR' pour calculer les embeddings.")
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
    topic_model = None
    if modeling_mode == "fit":
        if cpu_only:
            logger.info("Mode CPU activé - optimisations pour machines sans GPU")
            logger.info(f"Taille des batches d'embeddings: {embedding_batch_size}")

        logger.info("Extraction des textes français pour l'entraînement (OCR pour embeddings, lemma_nostop pour labels)...")
        
        # Extraire seulement les textes français; créer deux listes alignées
        if 'language' in ds.column_names:
            docs_clean = []
            embed_texts = []
            for lemma_text, ocr_text, lang in zip(ds[text_column_name], ds[embed_text_column_name], ds['language']):
                if lang == 'Français' and lemma_text and str(lemma_text).strip() and ocr_text and str(ocr_text).strip():
                    if len(str(lemma_text).split()) >= args.min_train_tokens:
                        docs_clean.append(str(lemma_text))
                        embed_texts.append(str(ocr_text))
            logger.info(f"Paires (lemma_nostop + OCR) extraites: {len(docs_clean)}")
        else:
            docs_clean = [str(t) for t in ds[text_column_name]]
            embed_texts = [str(t) for t in ds[embed_text_column_name]]
            logger.info("Colonne langue non disponible, utilisation de tous les textes (OCR/lemma)")
        
    # Limitation optionnelle du nombre de documents (utile pour tests CPU)
        if max_documents and len(docs_clean) > max_documents:
            logger.info(f"Limitation à {max_documents} documents pour optimiser les performances CPU")
            docs_clean = docs_clean[:max_documents]
            embed_texts = embed_texts[:max_documents]
        
        # Petite validation locale
        valid_pairs = [
            (dc, et) for dc, et in tqdm(list(zip(docs_clean, embed_texts)), desc="Validation des paires")
            if dc and str(dc).strip() and et and str(et).strip()
        ]
        if len(valid_pairs) < min_topic_size:
            logger.error(f"Nombre de textes valides ({len(valid_pairs)}) < min_topic_size ({min_topic_size})")
            return
        docs_clean_valid = [p[0] for p in valid_pairs]
        embed_texts_valid = [p[1] for p in valid_pairs]

        # Load extra domain stopwords if provided
        extra_stopwords: list[str] | None = None
        if args.domain_stopwords_file:
            try:
                from pathlib import Path as _P
                path_sw = _P(args.domain_stopwords_file)
                if path_sw.exists():
                    with path_sw.open("r", encoding="utf-8", errors="replace") as f:
                        extra_stopwords = [line.strip().lower() for line in f if line.strip()]
                    logger.info(f"Stopwords additionnels chargés: {len(extra_stopwords)} mots")
                else:
                    logger.warning(f"Fichier de stopwords introuvable: {path_sw}")
            except Exception as e:
                logger.warning(f"Impossible de charger les stopwords additionnels: {e}")

        # Determine dynamic min_cluster_size if desired_topics is provided
        dynamic_min_cluster_size = min_topic_size
        if args.desired_topics and args.desired_topics > 0:
            try:
                dynamic_min_cluster_size = max(20, int(len(docs_clean_valid) / int(args.desired_topics)))
                logger.info(
                    f"min_cluster_size dynamique: {dynamic_min_cluster_size} (docs={len(docs_clean_valid)}, sujets visés={args.desired_topics})"
                )
            except Exception:
                pass

        # Parse ngram range
        try:
            ngram_min, ngram_max = [int(x.strip()) for x in args.vectorizer_ngrams.split(",")]
        except Exception:
            ngram_min, ngram_max = 1, 3

        topic_model = create_bertopic_model(
            embedding_model_name,
            dynamic_min_cluster_size,
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
            vectorizer_max_df=args.vectorizer_max_df,
            vectorizer_max_features=args.vectorizer_max_features,
            vectorizer_ngram_min=ngram_min,
            vectorizer_ngram_max=ngram_max,
            domain_stopwords=extra_stopwords,
            desired_topics=args.desired_topics,
        )

        topic_model = fit_topic_model(
            topic_model,
            docs_clean_valid,
            embed_texts_valid,
            model_path,
            logger,
            embedding_model_name,
            embedding_batch_size,
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
        lambda batch: predict_topics_batch_impl(
            topic_model,
            batch,
            embed_text_col=embed_text_column_name,
            topic_id_col=topic_id_column_name,
            topic_prob_col=topic_prob_column_name,
            topic_label_col=topic_label_column_name,
            outlier_reassign_threshold=(
                args.outlier_reassign_threshold if args.outlier_reassign_threshold and args.outlier_reassign_threshold > 0 else None
            ),
            embedding_batch_size=embedding_batch_size,
        ),
        fn_kwargs={
            # kept for API compatibility, but handled via lambda closure above
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
