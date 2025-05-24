#!/usr/bin/env python3
"""
calculate_sentiment.py
======================

Ajoute des colonnes avec l'analyse de sentiment (label et score) à un dataset
Hugging Face existant, basées sur la colonne 'OCR'. Le script charge un dataset,
calcule ces métriques à l'aide du modèle cmarkea/distilcamembert-base-sentiment,
et ajoute les résultats dans de nouvelles colonnes.

L'utilisateur est invité à choisir la configuration ('articles' ou 'publications').
Les noms des nouvelles colonnes sont codés en dur : "sentiment_label" et "sentiment_score".

Usage
-----
    python post-processing/calculate_sentiment.py [--repo MON_USER/MON_DATASET]

Exemple:
    python post-processing/calculate_sentiment.py
    (Le script demandera ensuite la configuration et le nom du repo si non fourni)

Variables d'environnement
---------------------
HF_TOKEN   Jeton d'accès personnel pour le Hugging Face Hub (sinon, une
           connexion interactive sera demandée).

Dépendances supplémentaires
-------------------------
    pip install transformers torch  # ou tensorflow à la place de torch
"""
import argparse
import logging
import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfFolder, login
from transformers import pipeline # Pour l'analyse de sentiment

# Configuration du logging (identique à l'autre script)
def configure_logging() -> None:
    """Configure le logging de base."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Initialisation du pipeline de sentiment (sera fait dans main pour gérer les ressources)
sentiment_pipeline = None

def calculate_sentiment_batch(texts: list[str]) -> list[dict]:
    """
    Calcule le sentiment pour une liste de textes.
    Retourne une liste de dictionnaires avec 'label' and 'score'.
    """
    global sentiment_pipeline
    if not sentiment_pipeline:
        # Ce cas ne devrait pas arriver si initialisé dans main
        raise RuntimeError("Sentiment pipeline non initialisé.")
    
    if not texts:
        return []
    
    # Gérer les textes None ou vides pour éviter les erreurs dans le pipeline
    processed_texts = [text if isinstance(text, str) and text.strip() else " " for text in texts]
    
    try:
        # Le pipeline retourne une liste de dictionnaires, ex: [{'label': 'POSITIVE', 'score': 0.99}]
        # S truncation=True pour gérer les textes longs, max_length peut être ajusté
        results = sentiment_pipeline(processed_texts, truncation=True, max_length=512) 
        
        # S'assurer que le format est correct même si le pipeline a un comportement inattendu
        # pour les entrées " " (utilisées pour remplacer None/vide)
        # Le modèle cmarkea renvoie typiquement 'POSITIVE', 'NEGATIVE', 'NEUTRAL' ou des labels numériques selon la config.
        # Pour cmarkea/distilcamembert-base-sentiment, les labels sont 'POSITIVE', 'NEGATIVE', 'NEUTRAL'.
        # Si le texte était " ", le pipeline pourrait retourner un résultat par défaut.
        # Nous allons standardiser pour avoir label et score.
        final_results = []
        for i, res in enumerate(results):
            if not isinstance(res, dict) or 'label' not in res or 'score' not in res:
                # Si le résultat n'est pas conforme, mettre des valeurs par défaut
                # Cela peut arriver si le texte original était None ou vide.
                original_text_is_empty = not (isinstance(texts[i], str) and texts[i].strip())
                if original_text_is_empty:
                    final_results.append({'label': 'N/A', 'score': 0.0})
                else: # Cas d'erreur inattendu du pipeline
                    final_results.append({'label': 'ERROR', 'score': 0.0})
            else:
                final_results.append(res)
        return final_results

    except Exception as e:
        logging.getLogger(__name__).error(f"Erreur lors de l'analyse de sentiment par batch: {e}")
        # Retourner des valeurs par défaut pour tout le batch en cas d'erreur majeure
        return [{'label': 'ERROR', 'score': 0.0} for _ in texts]


def add_sentiment_metrics_batch(batch: dict, text_col: str, label_col: str, score_col: str) -> dict:
    """
    Applique le calcul du sentiment à un batch d'exemples.
    """
    if text_col not in batch:
        # Si la colonne de texte n'est pas dans ce batch, ajouter des colonnes vides
        num_rows = len(batch.get(next(iter(batch)), [])) # Nombre de lignes dans le batch
        if label_col not in batch:
             batch[label_col] = ['N/A'] * num_rows
        if score_col not in batch:
             batch[score_col] = [0.0] * num_rows
        return batch

    texts_in_batch: list = batch[text_col]
    
    sentiment_results = calculate_sentiment_batch(texts_in_batch)
    
    batch[label_col] = [res['label'] for res in sentiment_results]
    batch[score_col] = [res['score'] for res in sentiment_results]
    return batch

def main():
    global sentiment_pipeline # Pour pouvoir l'assigner
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Ajoute des colonnes d'analyse de sentiment ('sentiment_label', 'sentiment_score') à un dataset Hugging Face, basées sur la colonne 'OCR'.")
    parser.add_argument("--repo", default="fmadore/iwac-newspaper-articles", help="ID du repository sur le Hugging Face Hub (ex: utilisateur/nom_dataset).")
    parser.add_argument("--max-shard-size", default="1GB", help="Taille maximale des shards Parquet lors du push vers le Hub.")
    parser.add_argument("--batch-size", type=int, default=100, help="Taille des batchs pour le traitement .map(). Attention: un batch size élevé avec des modèles transformers peut consommer beaucoup de RAM/VRAM.")
    
    args = parser.parse_args()

    repo_id = args.repo
    text_column_name = "OCR"  # Hardcoded
    sentiment_label_col_name = "sentiment_label"  # Hardcoded
    sentiment_score_col_name = "sentiment_score"  # Hardcoded
    max_shard_size = args.max_shard_size
    batch_size = args.batch_size # Peut nécessiter d'être plus petit pour les transformers

    # --- Initialisation du pipeline de sentiment ---
    logger.info("Initialisation du pipeline d'analyse de sentiment avec cmarkea/distilcamembert-base-sentiment...")
    try:
        # Utiliser device=0 pour GPU si disponible, sinon -1 pour CPU
        # Le choix du device peut être rendu plus intelligent (vérifier cuda)
        sentiment_pipeline = pipeline("sentiment-analysis", model="cmarkea/distilcamembert-base-sentiment", device=-1) # device=-1 for CPU, device=0 for GPU
        logger.info("Pipeline de sentiment initialisé.")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du pipeline de sentiment: {e}")
        logger.error("Assurez-vous que 'transformers' et 'torch' (ou 'tensorflow') sont installés.")
        return

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
        except EOFError:
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

    if sentiment_label_col_name in ds.column_names:
        logger.warning(f"La colonne de label de sentiment '{sentiment_label_col_name}' existe déjà. Elle sera écrasée.")
    if sentiment_score_col_name in ds.column_names:
        logger.warning(f"La colonne de score de sentiment '{sentiment_score_col_name}' existe déjà. Elle sera écrasée.")

    # --- Application du calcul des métriques de sentiment ---
    logger.info(f"Calcul du sentiment (cols: '{sentiment_label_col_name}', '{sentiment_score_col_name}') pour la colonne '{text_column_name}'...")
    
    ds_processed = ds.map(
        add_sentiment_metrics_batch,
        fn_kwargs={"text_col": text_column_name, "label_col": sentiment_label_col_name, "score_col": sentiment_score_col_name},
        batched=True,
        batch_size=batch_size, # Un batch size plus petit est souvent nécessaire pour les transformers
        desc="Calcul du sentiment",
    )

    logger.info(f"Calcul du sentiment terminé.")
    logger.info(f"Aperçu (premiers 5) pour '{sentiment_label_col_name}': {ds_processed[sentiment_label_col_name][:5]}")
    logger.info(f"Aperçu (premiers 5) pour '{sentiment_score_col_name}': {ds_processed[sentiment_score_col_name][:5]}")
    
    # --- Réorganisation des colonnes ---
    insert_after_col = "lemma_nostop"  # MODIFIÉ: Colonne après laquelle insérer les nouvelles métriques
    logger.info(f"Réorganisation des colonnes pour placer '{sentiment_label_col_name}' et '{sentiment_score_col_name}' après '{insert_after_col}'.")
    
    existing_columns = list(ds_processed.column_names)
    
    if insert_after_col not in existing_columns:
        logger.warning(f"La colonne '{insert_after_col}' n'a pas été trouvée. Les nouvelles colonnes seront ajoutées à la fin.")
        ordered_columns = [col for col in existing_columns if col not in [sentiment_label_col_name, sentiment_score_col_name]]
        if sentiment_label_col_name in existing_columns: # Devrait toujours y être après .map()
            ordered_columns.append(sentiment_label_col_name)
        if sentiment_score_col_name in existing_columns: # Devrait toujours y être
            ordered_columns.append(sentiment_score_col_name)
    else:
        ordered_columns = []
        for col in existing_columns:
            # Exclure les nouvelles colonnes pour les ajouter au bon endroit
            if col == sentiment_label_col_name or col == sentiment_score_col_name:
                continue
            ordered_columns.append(col)
            if col == insert_after_col:
                if sentiment_label_col_name in existing_columns:
                    ordered_columns.append(sentiment_label_col_name)
                if sentiment_score_col_name in existing_columns:
                    ordered_columns.append(sentiment_score_col_name)
    
    if len(ordered_columns) == len(existing_columns) and set(ordered_columns) == set(existing_columns):
        ds_processed = ds_processed.select_columns(ordered_columns)
        logger.info(f"Nouvel ordre des colonnes: {ds_processed.column_names}")
    else:
        logger.warning("La réorganisation des colonnes a été sautée car une incohérence a été détectée. "
                         f"Colonnes attendues: {existing_columns}, Colonnes réorganisées proposées: {ordered_columns}. "
                         "Vérifiez les noms des colonnes et la logique de réorganisation.")

    # --- Sauvegarde du dataset traité --- 
    logger.info(f"Sauvegarde du dataset traité vers le Hub Hugging Face (repo: '{repo_id}', config: '{config_name_choice}')...")
    try:
        commit_message = f"Ajout colonnes '{sentiment_label_col_name}' et '{sentiment_score_col_name}' basées sur '{text_column_name}' (config: {config_name_choice})"
        ds_processed.push_to_hub(
            repo_id=repo_id,
            config_name=config_name_choice,
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
