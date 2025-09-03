"""
utils.py
--------
Misc utilities: logging, config discovery, and interactive selection.
"""
from __future__ import annotations

import argparse
import logging
from typing import List


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_available_configs(repo_id: str, token: str) -> List[str]:
    try:
        from huggingface_hub import dataset_info

        info = dataset_info(repo_id, token=token)
        if hasattr(info, 'config_names') and info.config_names:
            return list(info.config_names)
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
            raise SystemExit(0)


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
            raise SystemExit(0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ajoute des colonnes de modélisation de sujets à un dataset Hugging Face."
    )
    parser.add_argument("--repo", default="fmadore/islam-west-africa-collection")
    parser.add_argument(
        "--embedding-model",
        default="dangvantuan/sentence-camembert-base",
        help="Modèle d'embedding à utiliser (recommandé: sentence-camembert-base pour le français)",
    )
    parser.add_argument("--min-topic-size", type=int, default=5)
    parser.add_argument("--model-path", default="bertopic_model")
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--cpu-only", action="store_true", help="Force l'utilisation du CPU uniquement (optimisations pour machines sans GPU)"
    )
    parser.add_argument(
        "--max-documents", type=int, default=None, help="Limite le nombre de documents pour les tests (utile pour CPU)"
    )
    parser.add_argument(
        "--embedding-batch-size", type=int, default=16, help="Taille des batches pour les embeddings (réduire si mémoire limitée)"
    )
    parser.add_argument(
        "--min-train-tokens", type=int, default=5, help="Longueur minimale (en tokens) pour inclure un texte dans l'entraînement"
    )
    # UMAP/HDBSCAN/Vectorizer advanced tuning
    parser.add_argument("--umap-n-neighbors", type=int, default=60, help="UMAP n_neighbors (plus grand = moins d'outliers)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--umap-n-components", type=int, default=10, help="UMAP n_components")
    parser.add_argument("--umap-metric", type=str, default="cosine", help="UMAP metric")
    parser.add_argument(
        "--hdbscan-min-samples", type=int, default=3, help="HDBSCAN min_samples (plus petit = moins d'outliers)"
    )
    parser.add_argument(
        "--hdbscan-selection-method", type=str, choices=["eom", "leaf"], default="leaf", help="HDBSCAN cluster_selection_method"
    )
    parser.add_argument("--hdbscan-epsilon", type=float, default=0.0, help="HDBSCAN cluster_selection_epsilon")
    parser.add_argument("--vectorizer-min-df", type=int, default=10, help="CountVectorizer min_df")
    parser.add_argument("--vectorizer-max-df", type=float, default=0.9, help="CountVectorizer max_df")
    parser.add_argument("--vectorizer-max-features", type=int, default=25000, help="CountVectorizer max_features")
    parser.add_argument("--vectorizer-ngrams", type=str, default="1,3", help="CountVectorizer ngram range 'a,b'")
    # Outlier reduction options
    parser.add_argument(
        "--reduce-outliers-train",
        type=float,
        default=0.35,
        help="Seuil (0-1) pour réduire/réassigner les outliers à l'entraînement via c-TF-IDF. 0=désactivé",
    )
    parser.add_argument(
        "--outlier-reassign-threshold",
        type=float,
        default=0.35,
        help="Réassigne un outlier à la prédiction si la meilleure proba >= seuil (0-1). 0=jamais",
    )
    parser.add_argument(
        "--topic-label-max-words", type=int, default=8, help="Nombre maximum de mots uniques dans les labels de topics (défaut: 8)"
    )
    return parser
