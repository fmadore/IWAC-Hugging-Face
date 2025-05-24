#!/usr/bin/env python3
"""
calculate_lexical_richness.py
=============================

Ajoute des colonnes avec la richesse lexicale (Type-Token Ratio, TTR) et la
lisibilité (score de Flesch) à un dataset Hugging Face existant, basées sur
la colonne 'OCR'. Le script charge un dataset, calcule ces métriques, et ajoute
les scores dans de nouvelles colonnes.

L'utilisateur est invité à choisir la configuration ('articles' ou 'publications').
Les noms des nouvelles colonnes sont codés en dur : "Richesse_Lexicale_OCR" et "Lisibilite_OCR".

Usage
-----
    python post-processing/calculate_lexical_richness.py [--repo MON_USER/MON_DATASET]

Exemple:
    python post-processing/calculate_lexical_richness.py
    (Le script demandera ensuite la configuration et le nom du repo si non fourni)

Variables d'environnement
---------------------
HF_TOKEN   Jeton d'accès personnel pour le Hugging Face Hub (sinon, une
           connexion interactive sera demandée).

Dépendances supplémentaires
-------------------------
    pip install textstat
"""
import argparse
import logging
import os
import re
from datasets import load_dataset, Dataset
from huggingface_hub import HfFolder, login
import textstat # Ajout de l'import

def configure_logging() -> None:
    """Configure le logging de base."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def calculate_ttr(text: str) -> float:
    """
    Calcule le Type-Token Ratio (TTR) pour une chaîne de caractères.
    Retourne 0.0 si le texte est None, vide, ou ne contient pas de mots.
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Utilise re.findall pour obtenir les mots, similaire à calculate_word_count.py
    # Les mots sont des séquences de caractères alphanumériques.
    tokens = re.findall(r"\b\w+\b", text.lower())
    
    if not tokens:
        return 0.0
    
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    
    # Éviter la division par zéro si total_tokens est 0 (bien que 'if not tokens' devrait le couvrir)
    return unique_tokens / total_tokens if total_tokens > 0 else 0.0

def calculate_readability(text: str) -> float:
    """
    Calcule le score de lisibilité de Flesch pour une chaîne de caractères.
    Retourne un score (plus il est élevé, plus c'est facile à lire),
    ou 0.0 si le texte est None, vide, ou si le calcul échoue.
    textstat.set_lang('fr') doit avoir été appelé au préalable.
    """
    if not text or not isinstance(text, str):
        return 0.0  # Ou une autre valeur pour indiquer une donnée non traitable
    try:
        # textstat.set_lang('fr') est appelé une fois dans main()
        score = textstat.flesch_reading_ease(text)
        return score
    except Exception: # Peut échouer si le texte est trop court ou malformé
        return 0.0 # Ou une valeur spécifique pour indiquer un échec de calcul

def add_text_metrics_batch(batch: dict, text_col: str, richness_col: str, readability_col: str) -> dict:
    """
    Applique le calcul du TTR et de la lisibilité à un batch d'exemples.
    """
    if text_col not in batch:
        # Si la colonne de texte n'est pas dans ce batch
        if richness_col not in batch:
             batch[richness_col] = [0.0] * len(batch.get(next(iter(batch)), []))
        if readability_col not in batch:
             batch[readability_col] = [0.0] * len(batch.get(next(iter(batch)), []))
        return batch

    texts_in_batch: list = batch[text_col]
    
    richness_scores = [calculate_ttr(str(text)) if text is not None else 0.0 for text in texts_in_batch]
    readability_scores = [calculate_readability(str(text)) if text is not None else 0.0 for text in texts_in_batch]
    
    batch[richness_col] = richness_scores
    batch[readability_col] = readability_scores
    return batch

def main():
    configure_logging()
    logger = logging.getLogger(__name__)
    
    textstat.set_lang('fr') # Configurer la langue pour textstat globalement

    parser = argparse.ArgumentParser(description="Ajoute des colonnes de richesse lexicale ('Richesse_Lexicale_OCR') et de lisibilité ('Lisibilite_OCR') à un dataset Hugging Face, basées sur la colonne 'OCR'.")
    parser.add_argument("--repo", default="fmadore/iwac-newspaper-articles", help="ID du repository sur le Hugging Face Hub (ex: utilisateur/nom_dataset).")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille maximale des shards Parquet lors du push vers le Hub.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Taille des batchs pour le traitement .map().")
    
    args = parser.parse_args()

    repo_id = args.repo
    text_column_name = "OCR"  # Hardcoded
    richness_column_name = "Richesse_Lexicale_OCR"  # Hardcoded
    readability_column_name = "Lisibilite_OCR"      # Hardcoded
    max_shard_size = args.max_shard_size
    batch_size = args.batch_size

    # --- Choix de la configuration par l'utilisateur ---
    config_name_choice = ""
    while config_name_choice not in ["articles", "publications"]:
        try:
            config_name_choice = input("Quelle configuration traiter? ('articles' ou 'publications'): ").strip().lower()
            if config_name_choice not in ["articles", "publications"]:
                logger.warning("Choix invalide. Veuillez entrer 'articles' ou 'publications'.")
        except KeyboardInterrupt:
            logger.info("\nOpération annulée par l'utilisateur.")
            return
        except EOFError: # Gère le cas où l'entrée est redirigée et se termine
            logger.error("Fin de fichier atteinte lors de la demande de configuration. Le script ne peut pas continuer.")
            return
    
    logger.info(f"Configuration sélectionnée: {config_name_choice}")

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

    # --- Chargement du dataset ---
    logger.info(f"Chargement du dataset '{repo_id}', configuration '{config_name_choice}'...")
    try:
        ds = load_dataset(repo_id, name=config_name_choice, split="train", token=token, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        return

    logger.info(f"Dataset chargé. Nombre de lignes: {len(ds)}")
    logger.info(f"Colonnes disponibles: {ds.column_names}")

    if text_column_name not in ds.column_names:
        logger.error(f"La colonne de texte source (hardcodée) '{text_column_name}' n'existe pas dans le dataset. Colonnes disponibles: {ds.column_names}")
        return

    if richness_column_name in ds.column_names:
        logger.warning(f"La colonne de richesse lexicale '{richness_column_name}' existe déjà. Elle sera écrasée.")
    if readability_column_name in ds.column_names:
        logger.warning(f"La colonne de lisibilité '{readability_column_name}' existe déjà. Elle sera écrasée.")

    # --- Application du calcul des métriques ---
    logger.info(f"Calcul du TTR (col: '{richness_column_name}') et de la lisibilité (col: '{readability_column_name}') pour la colonne '{text_column_name}'...")
    
    ds_processed = ds.map(
        add_text_metrics_batch,
        fn_kwargs={"text_col": text_column_name, "richness_col": richness_column_name, "readability_col": readability_column_name},
        batched=True,
        batch_size=batch_size,
        desc="Calcul des métriques textuelles",
    )

    logger.info(f"Calcul du TTR terminé. Aperçu (premiers 5) pour '{richness_column_name}': {ds_processed[richness_column_name][:5]}")
    logger.info(f"Calcul de la lisibilité terminé. Aperçu (premiers 5) pour '{readability_column_name}': {ds_processed[readability_column_name][:5]}")

    # --- Réorganisation des colonnes ---
    logger.info(f"Réorganisation des colonnes pour placer '{richness_column_name}' et '{readability_column_name}' après '{text_column_name}'.")
    column_order = [text_column_name, richness_column_name, readability_column_name] + [col for col in ds_processed.column_names if col not in [text_column_name, richness_column_name, readability_column_name]]
    ds_processed = ds_processed.select(column_order)

    # --- Sauvegarde du dataset traité ---
    logger.info(f"Sauvegarde du dataset traité vers le Hub Hugging Face (repo: '{repo_id}')...")
    try:
        commit_message = f"Ajout colonnes '{richness_column_name}' (TTR) et '{readability_column_name}' (Lisibilité) basées sur '{text_column_name}' (config: {config_name_choice})"
        ds_processed.push_to_hub(
            repo_id,
            commit_message=commit_message,
            token=token,
            max_shard_size=max_shard_size,
        )
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du dataset sur le Hub: {e}")
        return

    logger.info("Dataset traité et sauvegardé avec succès.")

if __name__ == "__main__":
    main()
