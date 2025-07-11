#!/usr/bin/env python3
"""
calculate_word_count.py
=======================

Ajoute une colonne avec le nombre de mots à un dataset Hugging Face existant.
Le script charge un dataset depuis le repository Hugging Face 'fmadore/iwac-newspaper-articles',
compte les mots dans la colonne 'OCR', et ajoute ces comptes dans une nouvelle
colonne nommée 'nb_mots'. Le dataset mis à jour est ensuite poussé vers le Hub.

L'utilisateur est invité à choisir la configuration ('articles', 'publications' ou 'documents')
à traiter.

Usage
-----
    python post-processing/calculate_word_count.py

Exemple:
    python post-processing/calculate_word_count.py
    (Le script demandera ensuite la configuration)

Variables d'environnement
---------------------
HF_TOKEN   Jeton d'accès personnel pour le Hugging Face Hub (sinon, une
           connexion interactive sera demandée).
"""
import logging
import os
import re
from datasets import load_dataset, Dataset
from huggingface_hub import HfFolder, login

def configure_logging() -> None:
    """Configure le logging de base."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def count_words(text: str) -> int:
    """
    Compte le nombre de mots dans une chaîne de caractères.
    Les mots sont simplement séparés par des espaces.
    Retourne 0 si le texte est None ou vide.
    """
    if not text:
        return 0
    # Utilise une expression régulière pour mieux gérer les séparateurs multiples
    # et la ponctuation simple attachée aux mots.
    words = re.findall(r"\b\w+\b", str(text).lower())
    return len(words)

def add_word_count_batch(batch: dict, text_col: str, count_col: str) -> dict:
    """
    Applique le comptage de mots à un batch d'exemples.
    """
    if text_col not in batch:
        # Si la colonne de texte n'est pas dans ce batch (peut arriver avec des datasets hétérogènes)
        # ou si le batch est vide, retourner le batch tel quel ou avec une colonne de comptes vide.
        if count_col not in batch:
             batch[count_col] = [0] * len(batch.get(next(iter(batch)), [])) # Crée une colonne de zéros
        return batch

    texts_in_batch: list = batch[text_col]
    word_counts = [count_words(str(text)) if text is not None else 0 for text in texts_in_batch]
    batch[count_col] = word_counts
    return batch

def main():
    configure_logging()
    logger = logging.getLogger(__name__)

    # Hardcoded values
    repo_id = "fmadore/iwac-newspaper-articles"
    text_column_fixed = "OCR"
    count_column_name = "nb_mots"  # Changed here
    max_shard_size = "1GB"
    batch_size = 1000

    # --- Choix de la configuration par l'utilisateur ---
    config_name_choice = ""
    while config_name_choice not in ["articles", "publications", "documents"]:
        try:
            config_name_choice = input("Quelle configuration traiter? ('articles', 'publications' ou 'documents'): ").strip().lower()
            if config_name_choice not in ["articles", "publications", "documents"]:
                logger.warning("Entrée invalide. Veuillez choisir 'articles', 'publications' ou 'documents'.")
        except KeyboardInterrupt:
            logger.info("\nOpération annulée par l'utilisateur.")
            return
        except EOFError:
            logger.error("\nEntrée non détectée (EOF). Veuillez exécuter le script dans un terminal interactif.")
            return
    
    logger.info(f"Configuration sélectionnée: {config_name_choice}")
    # text_column_fixed = "OCR" # Déjà défini plus haut

    # --- Authentification avec le Hub ---
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        logger.info("Token Hugging Face non trouvé. Tentative de connexion interactive.")
        login()
        token = HfFolder.get_token()
        if not token:
            logger.error("Échec de la connexion au Hugging Face Hub. Veuillez fournir un token via HF_TOKEN ou vous connecter.")
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

    if text_column_fixed not in ds.column_names:
        logger.error(f"La colonne de texte '{text_column_fixed}' n'existe pas dans le dataset. Colonnes disponibles: {ds.column_names}")
        return

    if count_column_name in ds.column_names:
        logger.warning(f"La colonne de comptage '{count_column_name}' existe déjà. Elle sera écrasée.")

    # --- Application du comptage de mots ---
    logger.info(f"Calcul du nombre de mots pour la colonne '{text_column_fixed}' et stockage dans '{count_column_name}'...")
    
    ds_processed = ds.map(
        add_word_count_batch,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={
            "text_col": text_column_fixed,
            "count_col": count_column_name,
        },
        desc=f"Comptage des mots dans '{text_column_fixed}'"
    )
    logger.info(f"Comptage des mots terminé. Aperçu de la nouvelle colonne (premiers 5) pour '{count_column_name}': {ds_processed[count_column_name][:5]}")

    # --- Réorganisation des colonnes ---
    logger.info(f"Réorganisation des colonnes pour placer '{count_column_name}' après '{text_column_fixed}'.")
    current_columns = ds_processed.column_names
    
    # Enlever la colonne de comptage de sa position actuelle (généralement à la fin)
    # pour la réinsérer au bon endroit. Si elle n'y est pas pour une raison quelconque, pas de souci.
    if count_column_name in current_columns:
        current_columns.remove(count_column_name)
    
    try:
        ocr_index = current_columns.index(text_column_fixed)
        new_column_order = current_columns[:ocr_index+1] + [count_column_name] + current_columns[ocr_index+1:]
        ds_processed = ds_processed.select_columns(new_column_order)
        logger.info(f"Nouvel ordre des colonnes: {ds_processed.column_names}")
    except ValueError:
        logger.error(f"La colonne de référence '{text_column_fixed}' n'a pas été trouvée pour la réorganisation. Le dataset sera poussé sans réorganisation des colonnes.")

    # --- Push du dataset mis à jour vers le Hub ---
    logger.info(f"Push du dataset mis à jour vers '{repo_id}' (configuration '{config_name_choice}')...")
    try:
        ds_processed.push_to_hub(
            repo_id,
            config_name=config_name_choice,
            token=token,
            max_shard_size=max_shard_size,
            commit_message=f"Ajout de la colonne de comptage de mots '{count_column_name}' basée sur '{text_column_fixed}' (config: {config_name_choice})",
        )
        logger.info("Dataset poussé avec succès vers le Hub.")
    except Exception as e:
        logger.error(f"Erreur lors du push du dataset vers le Hub: {e}")

    logger.info("Script terminé.")

if __name__ == "__main__":
    main()
