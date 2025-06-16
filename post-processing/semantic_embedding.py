#!/usr/bin/env python3
"""
semantic_embedding.py
=====================

Ajoute des colonnes avec les embeddings sémantiques à un dataset Hugging Face 
existant, basées sur la colonne 'descriptionAI' (résumés français générés par Gemini).
Le script charge un dataset, calcule les embeddings à l'aide d'un modèle de 
sentence-transformers, et ajoute les résultats dans une nouvelle colonne.

L'utilisateur est invité à choisir la configuration ('articles' ou 'publications').
Le nom de la nouvelle colonne est : "embedding_descriptionAI".

Usage
-----
    python post-processing/semantic_embedding.py [--repo MON_USER/MON_DATASET]

Exemple:
    python post-processing/semantic_embedding.py --repo fmadore/iwac-newspaper-articles

Variables d'environnement
---------------------
HF_TOKEN   Jeton d'accès personnel pour le Hugging Face Hub (sinon, une
           connexion interactive sera demandée).

Dépendances supplémentaires
-------------------------
    pip install sentence-transformers torch datasets huggingface_hub tqdm
"""
import argparse
import logging
import os
import numpy as np
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from huggingface_hub import HfFolder, login
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# Configuration du logging
def configure_logging() -> None:
    """Configure le logging de base."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Modèle d'embedding global (sera initialisé dans main())
embedding_model = None

def get_available_configs(repo_id: str, token: str) -> List[str]:
    """
    Récupère la liste des configurations disponibles pour un dataset.
    
    Args:
        repo_id (str): ID du repository Hugging Face.
        token (str): Token d'authentification.
    
    Returns:
        List[str]: Liste des noms de configurations disponibles.
    """
    try:
        # Essayer de charger les métadonnées du dataset pour obtenir les configs
        from huggingface_hub import dataset_info
        info = dataset_info(repo_id, token=token)
        if hasattr(info, 'config_names') and info.config_names:
            return info.config_names
        else:
            # Fallback vers les configs connues
            return ['articles', 'publications']
    except Exception:
        # En cas d'erreur, retourner les configs par défaut
        return ['articles', 'publications']

def choose_config(available_configs: List[str]) -> str:
    """
    Demande à l'utilisateur de choisir une configuration parmi celles disponibles.
    
    Args:
        available_configs (List[str]): Liste des configurations disponibles.
    
    Returns:
        str: Nom de la configuration choisie.
    """
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

def choose_update_mode() -> str:
    """
    Demande à l'utilisateur de choisir le mode de mise à jour des embeddings.
    
    Returns:
        str: Mode choisi ('all' pour tout recalculer, 'missing' pour seulement les valeurs manquantes)
    """
    print("\nMode de mise à jour des embeddings:")
    print("  1. Mettre à jour seulement les lignes sans embeddings (recommandé)")
    print("  2. Recalculer tous les embeddings (peut être long)")
    
    while True:
        try:
            choice = input("Choisissez un mode (1-2): ").strip()
            if choice == "1":
                return "missing"
            elif choice == "2":
                return "all"
            else:
                print("Veuillez entrer 1 ou 2.")
        except KeyboardInterrupt:
            print("\nOpération annulée.")
            exit(0)

def compute_embeddings_batch(batch: Dict[str, List[Any]], text_col: str, embedding_col: str, update_mode: str = "all") -> Dict[str, List[Any]]:
    """
    Calcule les embeddings pour un batch de textes.
    
    Args:
        batch (Dict[str, List[Any]]): Batch de données du dataset.
        text_col (str): Nom de la colonne contenant le texte source.
        embedding_col (str): Nom de la colonne où stocker les embeddings.
        update_mode (str): Mode de mise à jour ('all' ou 'missing').
    
    Returns:
        Dict[str, List[Any]]: Batch avec les embeddings ajoutés.
    """
    global embedding_model
    
    texts = batch[text_col]
    existing_embeddings = batch.get(embedding_col, [None] * len(texts))
    
    # Déterminer quels textes ont besoin d'embeddings
    texts_to_process = []
    indices_to_update = []
    processed_texts = []
    
    for i, (text, existing_emb) in enumerate(zip(texts, existing_embeddings)):
        # Vérifier si on doit traiter ce texte
        should_process = False
        
        if update_mode == "all":
            should_process = True
        elif update_mode == "missing":
            # Traiter seulement si l'embedding n'existe pas ou est vide/invalide
            if (existing_emb is None or 
                existing_emb == [] or 
                (isinstance(existing_emb, list) and all(x == 0.0 for x in existing_emb))):
                should_process = True
        
        if should_process:
            if text is None or text == "":
                processed_texts.append("")  # Texte vide pour les valeurs manquantes
            else:
                processed_texts.append(str(text))
            texts_to_process.append(text)
            indices_to_update.append(i)
    
    # Si aucun texte à traiter, retourner le batch tel quel
    if not processed_texts:
        if embedding_col not in batch:
            # Créer la colonne avec les embeddings existants ou vides
            embedding_dim = embedding_model.get_sentence_embedding_dimension()
            batch[embedding_col] = [existing_emb if existing_emb is not None else [0.0] * embedding_dim 
                                  for existing_emb in existing_embeddings]
        return batch
    
    # Calculer les embeddings pour les textes sélectionnés
    try:
        # Utiliser show_progress_bar=False pour éviter les conflits avec tqdm externe
        embeddings = embedding_model.encode(
            processed_texts, 
            batch_size=32,  # Batch size interne pour le modèle
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Convertir en listes pour la compatibilité avec datasets
        new_embeddings_list = [emb.tolist() for emb in embeddings]
        
    except Exception as e:
        logging.error(f"Erreur lors du calcul des embeddings: {e}")
        # En cas d'erreur, créer des embeddings vides
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        new_embeddings_list = [[0.0] * embedding_dim for _ in processed_texts]
    
    # Construire la liste finale des embeddings
    if embedding_col not in batch:
        # Initialiser avec les embeddings existants ou des embeddings vides
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        final_embeddings = [existing_emb if existing_emb is not None else [0.0] * embedding_dim 
                          for existing_emb in existing_embeddings]
    else:
        final_embeddings = batch[embedding_col].copy()
    
    # Mettre à jour seulement les indices sélectionnés
    for idx, new_emb in zip(indices_to_update, new_embeddings_list):
        final_embeddings[idx] = new_emb
    
    # Ajouter les embeddings au batch
    batch[embedding_col] = final_embeddings
    
    return batch

def main():
    global embedding_model
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Ajoute une colonne d'embeddings sémantiques ('embedding_descriptionAI') "
                   "à un dataset Hugging Face, basée sur la colonne 'descriptionAI'."
    )
    parser.add_argument(
        "--repo", 
        default="fmadore/iwac-newspaper-articles", 
        help="ID du repository sur le Hugging Face Hub (ex: utilisateur/nom_dataset)."
    )
    parser.add_argument(
        "--model", 
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Modèle sentence-transformers à utiliser pour les embeddings."
    )
    parser.add_argument(
        "--max-shard-size", 
        default="1GB", 
        help="Taille maximale des shards Parquet lors du push vers le Hub."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=50, 
        help="Taille des batchs pour le traitement .map()."
    )
    
    args = parser.parse_args()

    repo_id = args.repo
    model_name = args.model
    text_column_name = "descriptionAI"  # Colonne source (résumés Gemini)
    embedding_column_name = "embedding_descriptionAI"  # Nouvelle colonne
    max_shard_size = args.max_shard_size
    batch_size = args.batch_size

    # --- Authentification avec le Hub ---
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        logger.info("Token Hugging Face non trouvé. Tentative de connexion interactive.")
        try:
            login()
            token = HfFolder.get_token()
            if not token:
                logger.error("Connexion interactive échouée ou token non obtenu. Veuillez définir HF_TOKEN ou vous connecter manuellement.")
                return
        except Exception as e:
            logger.error(f"Erreur lors de la connexion interactive: {e}")
            return

    # --- Choix de la configuration ---
    available_configs = get_available_configs(repo_id, token)
    config_name_choice = choose_config(available_configs)
    logger.info(f"Configuration choisie: '{config_name_choice}'")
    
    # --- Choix du mode de mise à jour ---
    update_mode = choose_update_mode()
    logger.info(f"Mode de mise à jour choisi: '{update_mode}'")

    # --- Initialisation du modèle d'embedding ---
    logger.info(f"Chargement du modèle d'embedding: {model_name}")
    try:
        # Vérifier si CUDA est disponible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Utilisation du device: {device}")
        
        embedding_model = SentenceTransformer(model_name, device=device)
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Modèle chargé avec succès. Dimension des embeddings: {embedding_dim}")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle d'embedding: {e}")
        return

    # --- Chargement du dataset ---
    logger.info(f"Chargement du dataset '{repo_id}', configuration '{config_name_choice}'...")
    try:
        ds = load_dataset(repo_id, name=config_name_choice, split="train", token=token, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        return

    logger.info(f"Dataset chargé. Nombre de lignes: {len(ds)}")
    logger.info(f"Colonnes disponibles: {ds.column_names}")

    # --- Vérifications des colonnes ---
    if text_column_name not in ds.column_names:
        logger.error(f"La colonne de texte source '{text_column_name}' n'existe pas dans le dataset. Colonnes disponibles: {ds.column_names}")
        return

    if embedding_column_name in ds.column_names:
        if update_mode == "all":
            logger.warning(f"La colonne d'embedding '{embedding_column_name}' existe déjà. Elle sera écrasée.")
        else:
            logger.info(f"La colonne d'embedding '{embedding_column_name}' existe déjà. Seules les valeurs manquantes seront calculées.")
    else:
        logger.info(f"La colonne d'embedding '{embedding_column_name}' sera créée.")

    # --- Statistiques sur la colonne source ---
    texts = ds[text_column_name]
    non_empty_texts = [t for t in texts if t is not None and t.strip() != ""]
    logger.info(f"Statistiques de la colonne '{text_column_name}':")
    logger.info(f"  - Total d'entrées: {len(texts)}")
    logger.info(f"  - Entrées non vides: {len(non_empty_texts)}")
    logger.info(f"  - Entrées vides/None: {len(texts) - len(non_empty_texts)}")
    
    if non_empty_texts:
        avg_length = sum(len(t) for t in non_empty_texts) / len(non_empty_texts)
        logger.info(f"  - Longueur moyenne des textes non vides: {avg_length:.1f} caractères")
    
    # --- Statistiques sur les embeddings existants (si mode 'missing') ---
    if update_mode == "missing" and embedding_column_name in ds.column_names:
        existing_embeddings = ds[embedding_column_name]
        valid_embeddings = 0
        empty_embeddings = 0
        
        for emb in existing_embeddings:
            if emb is None or emb == [] or (isinstance(emb, list) and all(x == 0.0 for x in emb)):
                empty_embeddings += 1
            else:
                valid_embeddings += 1
        
        logger.info(f"Statistiques des embeddings existants:")
        logger.info(f"  - Embeddings valides: {valid_embeddings}")
        logger.info(f"  - Embeddings manquants/vides: {empty_embeddings}")
        logger.info(f"  - Pourcentage à traiter: {(empty_embeddings/len(existing_embeddings)*100):.1f}%")

    # --- Application du calcul des embeddings ---
    logger.info(f"Calcul des embeddings (colonne: '{embedding_column_name}') pour la colonne '{text_column_name}'...")
    
    ds_processed = ds.map(
        compute_embeddings_batch,
        fn_kwargs={
            "text_col": text_column_name, 
            "embedding_col": embedding_column_name,
            "update_mode": update_mode
        },
        batched=True,
        batch_size=batch_size,
        desc=f"Calcul des embeddings ({'tous' if update_mode == 'all' else 'manquants seulement'})",
    )

    logger.info("Calcul des embeddings terminé.")
    
    # --- Vérification des résultats ---
    embeddings_sample = ds_processed[embedding_column_name][:3]
    logger.info(f"Aperçu des embeddings (3 premiers):")
    for i, emb in enumerate(embeddings_sample):
        if emb:
            logger.info(f"  Embedding {i+1}: dimension {len(emb)}, premiers éléments: {emb[:5]}")
        else:
            logger.info(f"  Embedding {i+1}: vide")

    # --- Réorganisation des colonnes ---
    # Placer la nouvelle colonne après 'descriptionAI'
    insert_after_col = "descriptionAI"
    new_embedding_cols = [embedding_column_name]
    logger.info(f"Réorganisation des colonnes pour placer {new_embedding_cols} après '{insert_after_col}'.")
    
    existing_columns = list(ds_processed.column_names)
    
    if insert_after_col in existing_columns:
        # Trouver l'index de la colonne après laquelle insérer
        insert_index = existing_columns.index(insert_after_col) + 1
        
        # Créer la nouvelle liste de colonnes
        new_columns = existing_columns[:insert_index]
        
        # Ajouter les nouvelles colonnes d'embedding
        for col in new_embedding_cols:
            if col in existing_columns and col not in new_columns:
                new_columns.append(col)
        
        # Ajouter le reste des colonnes
        for col in existing_columns[insert_index:]:
            if col not in new_columns:
                new_columns.append(col)
        
        # Réorganiser le dataset
        ds_processed = ds_processed.select_columns(new_columns)
        logger.info(f"Colonnes réorganisées. Nouvel ordre: {ds_processed.column_names}")
    else:
        logger.warning(f"Colonne de référence '{insert_after_col}' non trouvée. Les nouvelles colonnes seront ajoutées à la fin.")

    # --- Sauvegarde du dataset traité ---
    logger.info(f"Sauvegarde du dataset traité vers le Hub Hugging Face (repo: '{repo_id}', config: '{config_name_choice}')...")
    try:
        commit_message = f"Ajout colonne '{embedding_column_name}' (embeddings sémantiques) basée sur '{text_column_name}' avec {model_name} (config: {config_name_choice})"
        ds_processed.push_to_hub(
            repo_id=repo_id,
            config_name=config_name_choice,
            commit_message=commit_message,
            token=token,
            max_shard_size=max_shard_size,
        )
        logger.info("Dataset traité et sauvegardé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du dataset sur le Hub: {e}")
        return

    logger.info(f"Processus terminé. Colonne '{embedding_column_name}' {'mise à jour' if embedding_column_name in ds.column_names else 'ajoutée'} avec succès.")
    logger.info(f"Mode de traitement: {update_mode}")
    logger.info(f"Modèle utilisé: {model_name}")
    logger.info(f"Dimension des embeddings: {embedding_dim}")

if __name__ == "__main__":
    main()
