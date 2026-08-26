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

import asyncio
import os
import sys
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Add parent directory to path for iwac_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from rich.console import Console
from iwac_common.omeka_client import OmekaApiClient, conn_manager, fetch_iiif_thumbnail_url
from iwac_common.field_mappers import (
    extract_added_date,
    get_value,
    get_value_by_language,
)
from iwac_common.upload_runner import UploadSpec, run_upload
from iwac_common.schema import SUBSETS
from iwac_common.hub import (
    ConcurrentHubWriteError,
    HubBaselineUnavailableError,
    get_repo_revision,
)

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
        # fallback=True: an authority record's description is of unpredictable
        # language, so any value beats none. untagged_matches stays off to
        # preserve the exact precedence the local helper had before it moved
        # into iwac_common (tagged 'fr' wins, then first value of any kind).
        "Description": get_value_by_language(
            item, "dcterms:description", "fr", fallback=True
        ),
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
    termes (séparés par |) des colonnes ``fields`` d'une ligne.

    ``frequency`` compte des **items**, pas des mentions : les termes sont
    dédupliqués sur la ligne avant comptage, si bien qu'une autorité citée à la
    fois en ``subject`` et en ``spatial`` (ou deux fois dans le même champ)
    n'est comptée qu'une fois. Sans cela « fréquence » mélangeait deux
    grandeurs — le nombre de documents et le nombre de champs où le terme
    apparaît — et seules les autorités présentes dans plusieurs champs étaient
    gonflées, ce qui rendait la comparaison entre entités fausse.
    """
    date_str = row.get('pub_date', '') or row.get('date', '')
    country = row.get('country', '')
    terms = {
        term
        for field in fields
        for term in extract_terms_from_field(row.get(field, ''))
    }
    for term in terms:
        stats = term_stats[term]
        stats['frequency'] += 1
        if country:
            stats['countries'].add(country)
        if date_str:
            if not stats['first_occurrence'] or date_str < stats['first_occurrence']:
                stats['first_occurrence'] = date_str
            if not stats['last_occurrence'] or date_str > stats['last_occurrence']:
                stats['last_occurrence'] = date_str


#: Colonnes scannées par subset. Chaque colonne est une liste d'autorités
#: séparées par ``|`` dont les valeurs correspondent exactement au ``Titre``
#: d'une ligne d'index (vocabulaire contrôlé) — d'où la comparaison par
#: appartenance exacte après découpage, jamais par sous-chaîne.
FREQUENCY_SOURCE_FIELDS: Dict[str, List[str]] = {
    'articles': ['subject', 'spatial', 'author'],
    'publications': ['subject', 'spatial', 'author'],
    # ``subject``/``spatial`` ont longtemps manqué ici, alors que les colonnes
    # existent et sont peuplées (spatial sur 854 des 867 lignes, subject sur
    # 313) : les lieux et thèmes d'une référence ne comptaient nulle part.
    # 228 entrées d'index y gagnent 1 708 occurrences, et 227 d'entre elles
    # n'en tiraient aucune — « Extrémisme violent » et « Contre-terrorisme »
    # affichaient une fréquence de 0 alors que des références les portent en
    # sujet. Même classe de bug que les signatures côté IwacSearch.
    'references': ['subject', 'spatial', 'author', 'editor', 'publisher'],
    # ``creator``/``publisher`` plutôt que ``author``/``newspaper`` : c'est le
    # nom des colonnes du subset audiovisuel. ``publisher`` fait entrer les
    # chaînes YouTube (des foaf:Organization, donc des lignes d'index) dans
    # les agrégats, ce qui leur donne enfin une fréquence réelle.
    'audiovisual': ['subject', 'spatial', 'creator', 'publisher'],
}


def calculate_frequency_stats(
    articles_df: pd.DataFrame,
    publications_df: pd.DataFrame,
    references_df: pd.DataFrame,
    audiovisual_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, Any]]:
    """Calcule fréquence / première-dernière occurrence / pays pour chaque
    autorité, en balayant les colonnes de ``FREQUENCY_SOURCE_FIELDS``.

    ``frequency`` est un nombre d'items : voir :func:`_accumulate_term_stats`
    pour la déduplication par ligne."""
    term_stats = defaultdict(lambda: {
        'frequency': 0,
        'first_occurrence': None,
        'last_occurrence': None,
        'countries': set()
    })

    frames = {
        'articles': articles_df,
        'publications': publications_df,
        'references': references_df,
        'audiovisual': audiovisual_df,
    }
    sources = [
        (frames[name], fields, name)
        for name, fields in FREQUENCY_SOURCE_FIELDS.items()
        if frames[name] is not None
    ]
    for df, fields, name in sources:
        logger.info(f"Calculating frequency stats from {name} dataset...")
        if df.empty:
            continue
        for _, row in df.iterrows():
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


async def load_reference_datasets(
    token: Optional[str], repo: str
) -> Dict[str, pd.DataFrame]:
    """Load every mandatory input from one stable Hub revision.

    Frequency statistics are a corpus-wide derived layer.  A missing input is
    not equivalent to an empty corpus, so any load error aborts rather than
    overwriting good statistics with zeros.

    Returns one frame per key of :data:`FREQUENCY_SOURCE_FIELDS`.
    """
    before = get_repo_revision(repo, token=token)

    def load_one(config_name: str) -> pd.DataFrame:
        logger.info("Loading %s dataset from %s at %s...", config_name, repo, before)
        try:
            dataset = load_dataset(
                repo,
                name=config_name,
                split="train",
                token=token,
                revision=before,
                download_mode="force_redownload",
                verification_mode="no_checks",
            )
        except Exception as exc:  # noqa: BLE001
            raise HubBaselineUnavailableError(
                f"Index enrichment requires '{config_name}', but it could not be "
                f"loaded from {repo} at {before}: {exc}"
            ) from exc
        frame = dataset.to_pandas()
        logger.info("Loaded %d %s rows", len(frame), config_name)
        return frame

    names = list(FREQUENCY_SOURCE_FIELDS)
    frames = await asyncio.gather(
        *(asyncio.to_thread(load_one, name) for name in names)
    )
    after = get_repo_revision(repo, token=token)
    if after != before:
        raise ConcurrentHubWriteError(
            f"{repo} changed from {before} to {after} while index inputs were loading; "
            "refusing mixed-revision frequency statistics."
        )
    return dict(zip(names, frames))




# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

async def _attach_frequency_stats(
    new_df: pd.DataFrame, api: OmekaApiClient, repo: str, token: Optional[str]
) -> pd.DataFrame:
    """post_map hook: enrich index rows with corpus-wide term frequency
    statistics (occurrences + first/last year + countries) computed from every
    content subset in FREQUENCY_SOURCE_FIELDS, joined on the index Titre
    (controlled vocabulary). Runs after mapping, before the Hub merge."""
    frames = await load_reference_datasets(token, repo)
    frequency_stats = calculate_frequency_stats(
        frames["articles"],
        frames["publications"],
        frames["references"],
        frames["audiovisual"],
    )

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
    resource_class_ids=SUBSETS["index"].resource_class_ids,
    map_item=map_index_item,
    title="🗂️ IWAC Index Upload",
    cache_dir=".cache_omk_index",
    description="Publie l'index IWAC sur le Hub HF",
    post_map=_attach_frequency_stats,
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
