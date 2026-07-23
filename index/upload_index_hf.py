#!/usr/bin/env python3
"""
upload_index_hf.py
==================

Extrait les données d'index depuis l'API Omeka S d'IWAC, calcule les statistiques
de fréquence depuis les datasets articles et publications, et pousse le tout
sur le Hugging Face Hub.

Usage
-----
    python upload_index_hf.py \
        --repo fmadore/islam-west-africa-collection \
        --max-shard-size 1GB

Variables d'environnement
------------------------
  OMEKA_BASE_URL        Base URL de l'API, ex. https://islam.zmo.de/api
  OMEKA_KEY_IDENTITY    Identité de la clé Omeka
  OMEKA_KEY_CREDENTIAL  Credential de la clé Omeka
  HF_TOKEN              Jeton d'accès personnel Hugging Face (facultatif si
                        vous appelez login() de manière interactive)
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter

# Add parent directory to path for iwac_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from rich.console import Console
from iwac_common.omeka_client import OmekaApiClient, conn_manager, fetch_iiif_thumbnail_url
from iwac_common.field_mappers import extract_added_date, get_value
from iwac_common.upload_runner import UploadSpec, run_upload

logger = logging.getLogger("upload")
console = Console()

load_dotenv()


# Orchestration + CLI + Rich console/logging live in
# iwac_common.upload_runner. Index keeps a `post_map` hook for its
# cross-subset frequency statistics (computed from the articles /
# publications / references subsets before merging).


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

def _get_value_only(item: Dict[str, Any], field: str) -> str:
    """Extrait seulement les valeurs @value d'un champ Omeka (pour dcterms:identifier)"""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        # Ne prendre que les @value, ignorer les autres types
        parts = [str(v.get("@value", "")) for v in val if isinstance(v, dict) and v.get("@value")]
        return "|".join(filter(None, parts))
    if isinstance(val, dict):
        return val.get("@value", "")
    return str(val)


def _get_value_with_lang(item: Dict[str, Any], field: str, preferred_lang: str = "fr") -> str:
    """Extrait une valeur en privilégiant une langue spécifique"""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        # Chercher d'abord la langue préférée
        for v in val:
            if isinstance(v, dict) and v.get("@language") == preferred_lang:
                return v.get("@value", "")
        # Sinon prendre la première valeur disponible
        for v in val:
            if isinstance(v, dict) and v.get("@value"):
                return v.get("@value", "")
    elif isinstance(val, dict):
        return val.get("@value", "")
    return str(val)


def _get_display_title(item: Dict[str, Any], field: str) -> str:
    """Extrait les display_title d'un champ"""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        parts = [str(v.get("display_title", "")) for v in val if v.get("display_title")]
        return "|".join(filter(None, parts))
    elif isinstance(val, dict):
        return val.get("display_title", "")
    return ""


def _get_resource_class_type(item: Dict[str, Any]) -> str:
    """Mappe le resource_class_id vers le type correspondant"""
    resource_class_mapping = {
        9: "Lieux",
        94: "Personnes", 
        96: "Organisations",
        54: "Événements",
        244: "Sujets"  # Par défaut, sera affiné selon l'item_set
    }
    
    resource_class = item.get("o:resource_class")
    if not resource_class or not isinstance(resource_class, dict):
        return ""
    
    class_id = resource_class.get("o:id")
    if not class_id:
        return ""
    
    # Cas spécial pour la classe 244 (Sujets/Notices d'autorité)
    if class_id == 244:
        # Vérifier l'item_set pour distinguer Sujets vs Notices d'autorité
        item_sets = item.get("o:item_set", [])
        if isinstance(item_sets, list):
            for item_set in item_sets:
                if isinstance(item_set, dict):
                    item_set_id = item_set.get("o:id")
                    if item_set_id == 1:
                        return "Sujets"
                    elif item_set_id == 267:
                        return "Notices d'autorité"
        # Si pas d'item_set spécifique trouvé, retourner "Sujets" par défaut
        return "Sujets"
    
    return resource_class_mapping.get(class_id, "")


def _get_item_set_ids(item: Dict[str, Any]) -> str:
    """Extrait les IDs des item sets (gardé pour compatibilité si nécessaire ailleurs)"""
    if "o:item_set" not in item or item["o:item_set"] is None:
        return ""
    item_sets = item["o:item_set"]
    if isinstance(item_sets, list):
        ids = []
        for item_set in item_sets:
            if isinstance(item_set, dict) and "o:id" in item_set:
                ids.append(str(item_set["o:id"]))
        return "|".join(ids)
    return ""


async def map_index_item(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item d'index Omeka en dict plat pour HF datasets."""
    
    added_date = extract_added_date(item)

    # Fetch thumbnail URL
    session = await conn_manager.get()
    thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)

    return {
        "o:id": item["o:id"],
        "identifier": _get_value_only(item, "dcterms:identifier"),
        "added_date": added_date, # Date when item was added to Omeka
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "thumbnail": thumbnail_url,
        "Titre": item.get("o:title", ""),
        "Titre alternatif": get_value(item, "dcterms:alternative"),
        "Type": _get_resource_class_type(item),
        "Description": _get_value_with_lang(item, "dcterms:description", "fr"),
        "Date création": get_value(item, "dcterms:created"),
        "date": get_value(item, "dcterms:date"),
        "Relation": _get_display_title(item, "dcterms:relation"),
        "Remplacé par": get_value(item, "dcterms:isReplacedBy"),
        "Partie de": _get_display_title(item, "dcterms:isPartOf"),
        "spatial": get_value(item, "dcterms:spatial"),
        "A une partie": _get_display_title(item, "dcterms:hasPart"),
        "Prénom": get_value(item, "foaf:firstName"),
        "Nom": get_value(item, "foaf:lastName"),
        "Genre": _get_display_title(item, "foaf:gender"),
        "Naissance": get_value(item, "foaf:birthday"),
        "Coordonnées": get_value(item, "curation:coordinates"),
    }


# ---------------------------------------------------------------------------
# Calcul des statistiques de fréquence
# ---------------------------------------------------------------------------

def extract_terms_from_field(field_value: str) -> List[str]:
    """Extrait les termes d'un champ qui peut contenir des valeurs multiples séparées par |"""
    if not field_value or pd.isna(field_value):
        return []
    return [term.strip() for term in str(field_value).split("|") if term.strip()]


def _accumulate_term_stats(
    term_stats: Dict[str, Dict[str, Any]],
    row: pd.Series,
    fields: List[str],
) -> None:
    """Met à jour fréquence / première-dernière occurrence / pays pour tous les
    termes (séparés par |) des colonnes ``fields`` d'une ligne."""
    date_str = row.get('pub_date', '') or row.get('date', '')
    country = row.get('country', '')
    for field in fields:
        for term in extract_terms_from_field(row.get(field, '')):
            stats = term_stats[term]
            stats['frequency'] += 1
            if country:
                stats['countries'].add(country)
            if date_str:
                if not stats['first_occurrence'] or date_str < stats['first_occurrence']:
                    stats['first_occurrence'] = date_str
                if not stats['last_occurrence'] or date_str > stats['last_occurrence']:
                    stats['last_occurrence'] = date_str


def calculate_frequency_stats(articles_df: pd.DataFrame, publications_df: pd.DataFrame, references_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calcule les statistiques de fréquence pour tous les termes des colonnes subject, spatial et author des datasets articles et publications,
    ainsi que les colonnes author, editor et publisher du dataset references"""
    term_stats = defaultdict(lambda: {
        'frequency': 0,
        'first_occurrence': None,
        'last_occurrence': None,
        'countries': set()
    })

    sources = [
        (articles_df, ['subject', 'spatial', 'author'], 'articles'),
        (publications_df, ['subject', 'spatial', 'author'], 'publications'),
        (references_df, ['author', 'editor', 'publisher'], 'references'),
    ]
    for df, fields, name in sources:
        logger.info(f"Calculating frequency stats from {name} dataset...")
        if df.empty:
            continue
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {name}"):
            _accumulate_term_stats(term_stats, row, fields)

    # Convertir les sets en chaînes séparées par |
    result = {}
    for term, stats in term_stats.items():
        result[term] = {
            'frequency': stats['frequency'],
            'first_occurrence': stats['first_occurrence'] or '',
            'last_occurrence': stats['last_occurrence'] or '',
            'countries': "|".join(sorted(stats['countries'])) if stats['countries'] else ''
        }

    logger.info(f"Calculated frequency stats for {len(result)} unique terms")
    return result


async def load_reference_datasets(token: Optional[str], repo: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les datasets articles, publications et references depuis Hugging Face Hub"""
    articles_df = pd.DataFrame()
    publications_df = pd.DataFrame()
    references_df = pd.DataFrame()

    try:
        logger.info(f"Loading articles dataset from {repo}...")
        articles_ds = load_dataset(repo, name="articles", split="train", token=token, download_mode="force_redownload", verification_mode="no_checks")
        articles_df = articles_ds.to_pandas()
        logger.info(f"Loaded {len(articles_df)} articles")
    except Exception as e:
        logger.warning(f"Could not load articles dataset: {e}")

    try:
        logger.info(f"Loading publications dataset from {repo}...")
        publications_ds = load_dataset(repo, name="publications", split="train", token=token, download_mode="force_redownload", verification_mode="no_checks")
        publications_df = publications_ds.to_pandas()
        logger.info(f"Loaded {len(publications_df)} publications")
    except Exception as e:
        logger.warning(f"Could not load publications dataset: {e}")

    try:
        logger.info(f"Loading references dataset from {repo}...")
        references_ds = load_dataset(repo, name="references", split="train", token=token, download_mode="force_redownload", verification_mode="no_checks")
        references_df = references_ds.to_pandas()
        logger.info(f"Loaded {len(references_df)} references")
    except Exception as e:
        logger.warning(f"Could not load references dataset: {e}")

    return articles_df, publications_df, references_df




# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

async def _attach_frequency_stats(
    new_df: pd.DataFrame, api: OmekaApiClient, repo: str, token: Optional[str]
) -> pd.DataFrame:
    """post_map hook: enrich index rows with corpus-wide term frequency
    statistics (occurrences + first/last year + countries) computed from the
    articles, publications and references subsets, joined on the index Titre
    (controlled vocabulary). Runs after mapping, before the Hub merge."""
    articles_df, publications_df, references_df = await load_reference_datasets(token, repo)
    frequency_stats = calculate_frequency_stats(articles_df, publications_df, references_df)

    console.print("[blue]→[/blue] Attaching frequency statistics to index records...")
    new_df["frequency"] = 0
    new_df["first_occurrence"] = ""
    new_df["last_occurrence"] = ""
    new_df["countries"] = None
    for idx, row in new_df.iterrows():
        titre = row.get("Titre", "")
        if titre in frequency_stats:
            stats = frequency_stats[titre]
            new_df.at[idx, "frequency"] = stats["frequency"]
            new_df.at[idx, "first_occurrence"] = stats["first_occurrence"]
            new_df.at[idx, "last_occurrence"] = stats["last_occurrence"]
            new_df.at[idx, "countries"] = stats["countries"]
        else:
            new_df.at[idx, "countries"] = ""
    return new_df


SPEC = UploadSpec(
    config_name="index",
    resource_class_ids=(9, 94, 96, 54, 244),  # Lieux, Personnes, Organisations, Événements, Sujets/Notices
    map_item=map_index_item,
    title="🗂️ IWAC Index Upload",
    cache_dir=".cache_omk_index",
    description="Publie l'index IWAC sur le Hub HF",
    post_map=_attach_frequency_stats,
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
