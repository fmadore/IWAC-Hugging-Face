#!/usr/bin/env python3
"""
calculate_total_stats.py
========================

Calcule les statistiques totales (nombre de mots et de pages) pour tous les
datasets IWAC (articles, publications, documents) et sauvegarde les résultats
dans un fichier JSON.

Le script charge chaque configuration depuis le Hub Hugging Face, calcule les
totaux et fournit des statistiques détaillées par type de contenu.

Usage
-----
    python calculate_total_stats.py [--repo MON_USER/MON_DATASET] [--output stats.json]

Exemple:
    python calculate_total_stats.py --repo fmadore/islam-west-africa-collection --output iwac_stats.json

Variables d'environnement
---------------------
HF_TOKEN   Jeton d'accès personnel pour le Hugging Face Hub (sinon, une
           connexion interactive sera demandée).
"""
import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from datasets import load_dataset
from huggingface_hub import HfFolder, login
import pandas as pd

def configure_logging() -> None:
    """Configure le logging de base."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def safe_sum_column(df: pd.DataFrame, column_name: str) -> int:
    """
    Calcule la somme d'une colonne en gérant les valeurs manquantes et les types.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
        column_name (str): Nom de la colonne à sommer
    
    Returns:
        int: Somme des valeurs valides de la colonne
    """
    if column_name not in df.columns:
        return 0
    
    # Remplacer les valeurs manquantes par 0, convertir en numérique
    values = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
    return int(values.sum())

def count_non_empty_rows(df: pd.DataFrame, column_name: str) -> int:
    """
    Compte le nombre de lignes non vides pour une colonne donnée.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
        column_name (str): Nom de la colonne à analyser
    
    Returns:
        int: Nombre de lignes avec des valeurs non vides
    """
    if column_name not in df.columns:
        return 0
    
    # Compter les lignes où la valeur n'est pas nulle, vide ou NaN
    non_empty = df[column_name].notna() & (df[column_name] != "") & (df[column_name] != 0)
    return int(non_empty.sum())

def count_unique_newspapers(df: pd.DataFrame) -> int:
    """
    Compte le nombre de journaux uniques dans la colonne 'newspaper'.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
    
    Returns:
        int: Nombre de journaux uniques
    """
    if "newspaper" not in df.columns:
        return 0
    
    # Filtrer les valeurs non nulles et non vides, puis compter les uniques
    newspapers = df["newspaper"].dropna()
    newspapers = newspapers[newspapers != ""]
    return len(newspapers.unique())

def load_dataset_safe(repo_id: str, config_name: str, token: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Charge un dataset de manière sécurisée avec gestion d'erreurs.
    
    Args:
        repo_id (str): ID du repository Hugging Face
        config_name (str): Nom de la configuration
        token (Optional[str]): Token d'authentification
    
    Returns:
        Optional[pd.DataFrame]: DataFrame du dataset ou None en cas d'erreur
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Chargement du dataset '{repo_id}', configuration '{config_name}'...")
        ds = load_dataset(
            repo_id, 
            name=config_name, 
            split="train", 
            token=token, 
            trust_remote_code=True
        )
        df = ds.to_pandas()
        logger.info(f"Dataset '{config_name}' chargé avec succès: {len(df)} lignes, {len(df.columns)} colonnes")
        return df
    except Exception as e:
        logger.warning(f"Impossible de charger le dataset '{config_name}': {e}")
        return None

def calculate_dataset_stats(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Calcule les statistiques pour un dataset donné.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
        dataset_name (str): Nom du dataset pour les logs
    
    Returns:
        Dict[str, Any]: Dictionnaire avec les statistiques
    """
    logger = logging.getLogger(__name__)
    
    if df is None or df.empty:
        logger.warning(f"Dataset '{dataset_name}' vide ou non disponible")
        return {
            "total_records": 0,
            "total_words": 0,
            "total_pages": 0,
            "records_with_word_count": 0,
            "records_with_page_count": 0,
            "records_with_ocr": 0,
            "unique_newspapers": 0,
            "columns_available": []
        }
    
    # Déterminer les noms de colonnes à utiliser selon le dataset
    word_count_col = "nb_mots"
    page_count_col = "nb_pages"
    
    # Pour les publications, vérifier si d'autres colonnes existent
    if dataset_name == "publications":
        # Colonnes alternatives possibles pour les pages
        possible_page_cols = ["nb_pages", "pages", "num_pages", "page_count", "bibo:numPages"]
        for col in possible_page_cols:
            if col in df.columns:
                page_count_col = col
                break
    
    # Afficher les colonnes disponibles pour debug
    logger.info(f"Colonnes disponibles dans '{dataset_name}': {list(df.columns)}")
    logger.info(f"Utilisation de '{word_count_col}' pour les mots et '{page_count_col}' pour les pages")
    
    # Calculs de base
    total_records = len(df)
    total_words = safe_sum_column(df, word_count_col)
    total_pages = safe_sum_column(df, page_count_col)
    
    # Comptes de disponibilité
    records_with_word_count = count_non_empty_rows(df, word_count_col)
    records_with_page_count = count_non_empty_rows(df, page_count_col) 
    records_with_ocr = count_non_empty_rows(df, "OCR")
    
    # Compter les journaux uniques (seulement pour articles et publications)
    unique_newspapers = 0
    if dataset_name in ["articles", "publications"]:
        unique_newspapers = count_unique_newspapers(df)
    
    stats = {
        "total_records": total_records,
        "total_words": total_words,
        "total_pages": total_pages,
        "records_with_word_count": records_with_word_count,
        "records_with_page_count": records_with_page_count,
        "records_with_ocr": records_with_ocr,
        "unique_newspapers": unique_newspapers,
        "columns_available": list(df.columns),
        "word_count_column_used": word_count_col,
        "page_count_column_used": page_count_col
    }
    
    logger.info(f"Statistiques pour '{dataset_name}':")
    logger.info(f"  - Enregistrements totaux: {total_records:,}")
    logger.info(f"  - Mots totaux: {total_words:,} (colonne: {word_count_col})")
    logger.info(f"  - Pages totales: {total_pages:,} (colonne: {page_count_col})")
    logger.info(f"  - Avec comptage de mots: {records_with_word_count:,}")
    logger.info(f"  - Avec comptage de pages: {records_with_page_count:,}")
    logger.info(f"  - Avec contenu OCR: {records_with_ocr:,}")
    if dataset_name in ["articles", "publications"]:
        logger.info(f"  - Journaux uniques: {unique_newspapers:,}")
    
    return stats

def main():
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Calcule les statistiques totales pour tous les datasets IWAC."
    )
    parser.add_argument(
        "--repo", 
        default="fmadore/islam-west-africa-collection", 
        help="ID du repository sur le Hugging Face Hub"
    )
    parser.add_argument(
        "--output", 
        default="iwac_total_stats.json", 
        help="Nom du fichier de sortie JSON"
    )
    
    args = parser.parse_args()

    repo_id = args.repo
    output_file = args.output

    # --- Authentification avec le Hub ---
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        logger.info("Token Hugging Face non trouvé. Tentative de connexion interactive.")
        try:
            login()
            token = HfFolder.get_token()
        except Exception as e:
            logger.error(f"Échec de la connexion au Hugging Face Hub: {e}")
            return

    # Configurations à traiter
    configs = ["articles", "publications", "documents"]
    
    # Structure pour stocker tous les résultats
    all_stats = {
        "metadata": {
            "repository": repo_id,
            "generated_at": datetime.now().isoformat(),
            "script_version": "1.0"
        },
        "datasets": {},
        "totals": {
            "total_records": 0,
            "total_words": 0,
            "total_pages": 0,
            "total_records_with_word_count": 0,
            "total_records_with_page_count": 0,
            "total_records_with_ocr": 0,
            "unique_newspapers_total": 0
        }
    }

    # Traiter chaque configuration
    all_newspapers = set()  # Pour compter les journaux uniques à travers tous les datasets
    
    for config in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Traitement de la configuration: {config}")
        logger.info(f"{'='*50}")
        
        df = load_dataset_safe(repo_id, config, token)
        stats = calculate_dataset_stats(df, config)
        
        # Stocker les statistiques
        all_stats["datasets"][config] = stats
        
        # Ajouter aux totaux globaux
        all_stats["totals"]["total_records"] += stats["total_records"]
        all_stats["totals"]["total_words"] += stats["total_words"]
        all_stats["totals"]["total_pages"] += stats["total_pages"]
        all_stats["totals"]["total_records_with_word_count"] += stats["records_with_word_count"]
        all_stats["totals"]["total_records_with_page_count"] += stats["records_with_page_count"]
        all_stats["totals"]["total_records_with_ocr"] += stats["records_with_ocr"]
        
        # Collecter les journaux uniques pour le total global
        if config in ["articles", "publications"] and df is not None and "newspaper" in df.columns:
            newspapers = df["newspaper"].dropna()
            newspapers = newspapers[newspapers != ""]
            all_newspapers.update(newspapers.unique())
    
    # Calculer le total de journaux uniques
    all_stats["totals"]["unique_newspapers_total"] = len(all_newspapers)

    # Afficher les totaux finaux
    logger.info(f"\n{'='*60}")
    logger.info("STATISTIQUES TOTALES IWAC")
    logger.info(f"{'='*60}")
    totals = all_stats["totals"]
    logger.info(f"Enregistrements totaux: {totals['total_records']:,}")
    logger.info(f"Mots totaux: {totals['total_words']:,}")
    logger.info(f"Pages totales: {totals['total_pages']:,}")
    logger.info(f"Journaux uniques (total): {totals['unique_newspapers_total']:,}")
    logger.info(f"Enregistrements avec comptage de mots: {totals['total_records_with_word_count']:,}")
    logger.info(f"Enregistrements avec comptage de pages: {totals['total_records_with_page_count']:,}")
    logger.info(f"Enregistrements avec contenu OCR: {totals['total_records_with_ocr']:,}")

    # Calculs de pourcentages
    if totals['total_records'] > 0:
        word_coverage = (totals['total_records_with_word_count'] / totals['total_records']) * 100
        page_coverage = (totals['total_records_with_page_count'] / totals['total_records']) * 100
        ocr_coverage = (totals['total_records_with_ocr'] / totals['total_records']) * 100
        
        all_stats["totals"]["word_count_coverage_percent"] = round(word_coverage, 2)
        all_stats["totals"]["page_count_coverage_percent"] = round(page_coverage, 2)
        all_stats["totals"]["ocr_coverage_percent"] = round(ocr_coverage, 2)
        
        logger.info(f"\nCouverture des données:")
        logger.info(f"Comptage de mots: {word_coverage:.1f}%")
        logger.info(f"Comptage de pages: {page_coverage:.1f}%")
        logger.info(f"Contenu OCR: {ocr_coverage:.1f}%")

    # Sauvegarder les résultats
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"\nStatistiques sauvegardées dans: {output_file}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")

    # Créer aussi un résumé en format texte lisible
    summary_file = output_file.replace('.json', '_summary.txt')
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("STATISTIQUES TOTALES IWAC\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Repository: {repo_id}\n")
            f.write(f"Généré le: {all_stats['metadata']['generated_at']}\n\n")
            
            f.write("TOTAUX GÉNÉRAUX\n")
            f.write("-" * 30 + "\n")
            f.write(f"Enregistrements totaux: {totals['total_records']:,}\n")
            f.write(f"Mots totaux: {totals['total_words']:,}\n")
            f.write(f"Pages totales: {totals['total_pages']:,}\n")
            f.write(f"Journaux uniques (total): {totals['unique_newspapers_total']:,}\n\n")
            
            f.write("DÉTAIL PAR DATASET\n")
            f.write("-" * 30 + "\n")
            for config, stats in all_stats["datasets"].items():
                f.write(f"\n{config.upper()}\n")
                f.write(f"  Enregistrements: {stats['total_records']:,}\n")
                f.write(f"  Mots: {stats['total_words']:,}\n")
                f.write(f"  Pages: {stats['total_pages']:,}\n")
                f.write(f"  Avec OCR: {stats['records_with_ocr']:,}\n")
                if config in ["articles", "publications"]:
                    f.write(f"  Journaux uniques: {stats['unique_newspapers']:,}\n")
            
            if totals['total_records'] > 0:
                f.write(f"\nCOUVERTURE DES DONNÉES\n")
                f.write(f"-" * 30 + "\n")
                f.write(f"Comptage de mots: {all_stats['totals']['word_count_coverage_percent']:.1f}%\n")
                f.write(f"Comptage de pages: {all_stats['totals']['page_count_coverage_percent']:.1f}%\n")
                f.write(f"Contenu OCR: {all_stats['totals']['ocr_coverage_percent']:.1f}%\n")
        
        logger.info(f"Résumé textuel sauvegardé dans: {summary_file}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du résumé: {e}")

    logger.info("\nScript terminé avec succès!")

if __name__ == "__main__":
    main()
