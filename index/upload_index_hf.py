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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from collections import defaultdict

# Add parent directory to path for iwac_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from iwac_common.omeka_client import (
    OmekaApiClient,
    conn_manager,
    fetch_iiif_thumbnail_url,
    item_page_url,
)
from iwac_common.field_mappers import (
    extract_added_date,
    get_display_titles,
    get_literal_values,
    get_value,
    get_value_by_language,
    parse_coordinates,
    parse_pub_date,
)
from iwac_common.upload_runner import UploadSpec, run_upload
from iwac_common.schema import SUBSETS
from iwac_common.hub import (
    ConcurrentHubWriteError,
    HubBaselineUnavailableError,
    get_repo_revision,
    load_hub_columns,
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


async def map_index_item(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item d'index Omeka en dict plat pour HF datasets."""
    
    session = await conn_manager.get()
    thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
    coordinates = get_value(item, "curation:coordinates")
    latitude, longitude = parse_coordinates(coordinates)

    return {
        "o:id": item["o:id"],
        "identifier": get_literal_values(item, "dcterms:identifier"),
        "added_date": extract_added_date(item),  # Date when item was added to Omeka
        "iwac_url": item_page_url(item["o:id"]),
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
        "Relation": get_display_titles(item, "dcterms:relation"),
        "Remplacé par": get_value(item, "dcterms:isReplacedBy"),
        "Partie de": get_display_titles(item, "dcterms:isPartOf"),
        "spatial": get_value(item, "dcterms:spatial"),
        "A une partie": get_display_titles(item, "dcterms:hasPart"),
        "Prénom": get_value(item, "foaf:firstName"),
        "Nom": get_value(item, "foaf:lastName"),
        "Genre": get_display_titles(item, "foaf:gender"),
        "Naissance": get_value(item, "foaf:birthday"),
        "Coordonnées": coordinates,
        # Analysable companions of the "lat, lng" string; null when it is
        # absent, multiple or malformed.
        "latitude": latitude,
        "longitude": longitude,
    }


# ---------------------------------------------------------------------------
# Calcul des statistiques de fréquence
# ---------------------------------------------------------------------------

def extract_terms_from_field(field_value: Any) -> List[str]:
    """Extrait les termes d'un champ qui peut contenir des valeurs multiples séparées par |"""
    if field_value is None or (isinstance(field_value, float) and pd.isna(field_value)):
        return []
    if not field_value:
        return []
    return [term.strip() for term in str(field_value).split("|") if term.strip()]


#: ``row, field → set of authority keys``. The default keys by title.
TermResolver = Callable[[Mapping[str, Any], str], Set[str]]


def title_terms(row: Mapping[str, Any], field: str) -> Set[str]:
    """Authority keys of one field, by exact title (the historical join)."""
    return set(extract_terms_from_field(row.get(field)))


def make_authority_resolver(title_to_ids: Mapping[str, Sequence[str]]) -> TermResolver:
    """Resolve a field to index ``o:id`` values.

    The ``<field>_ids`` column (the linked authorities' Omeka ids, written by
    the upload mappers since the ids columns were added) is authoritative when
    the row has one: two authorities sharing a title no longer pool their
    counts, and a renamed authority keeps counting before every content subset
    has been re-uploaded. Rows without ids — content subsets not re-uploaded
    since, or fields holding only free-text literals — fall back to the exact
    title match, mapped to every index row carrying that title (the previous
    behaviour, homonyms included).
    """

    def resolve(row: Mapping[str, Any], field: str) -> Set[str]:
        ids = extract_terms_from_field(row.get(f"{field}_ids"))
        if ids:
            return set(ids)
        return {
            index_id
            for title in extract_terms_from_field(row.get(field))
            for index_id in title_to_ids.get(title, ())
        }

    return resolve


def _occurrence_bounds(date_value: Any) -> Tuple[str, str]:
    """``(first, last)`` comparable date strings, or ``("", "")``.

    Only dates ``parse_pub_date`` recognises take part: a free-text value such
    as ``s.d.`` used to win ``last_occurrence`` by sorting after every digit. A
    ``YYYY-MM/YYYY-MM`` range contributes its start to the first occurrence and
    its end to the last.
    """
    text = str(date_value or "").strip()
    year, _precision = parse_pub_date(text)
    if year is None:
        return "", ""
    parts = [p.strip() for p in text.split("/")]
    return parts[0], parts[-1]


def _accumulate_term_stats(
    term_stats: Dict[str, Dict[str, Any]],
    row: Mapping[str, Any],
    fields: List[str],
    resolve: TermResolver = title_terms,
) -> None:
    """Met à jour fréquence / première-dernière occurrence / pays pour toutes
    les autorités des colonnes ``fields`` d'une ligne.

    ``frequency`` compte des **items**, pas des mentions : les termes sont
    dédupliqués sur la ligne avant comptage, si bien qu'une autorité citée à la
    fois en ``subject`` et en ``spatial`` (ou deux fois dans le même champ)
    n'est comptée qu'une fois. Sans cela « fréquence » mélangeait deux
    grandeurs — le nombre de documents et le nombre de champs où le terme
    apparaît — et seules les autorités présentes dans plusieurs champs étaient
    gonflées, ce qui rendait la comparaison entre entités fausse.

    ``country`` peut être multiple (``Benin|Nigeria`` sur une référence) : chaque
    pays est compté séparément, au lieu d'entrer comme une seule valeur composée.
    """
    first, last = _occurrence_bounds(row.get('pub_date') or row.get('date'))
    countries = extract_terms_from_field(row.get('country'))
    terms = set().union(*(resolve(row, field) for field in fields)) if fields else set()
    for term in terms:
        stats = term_stats[term]
        stats['frequency'] += 1
        stats['countries'].update(countries)
        if first and (not stats['first_occurrence'] or first < stats['first_occurrence']):
            stats['first_occurrence'] = first
        if last and (not stats['last_occurrence'] or last > stats['last_occurrence']):
            stats['last_occurrence'] = last


#: Colonnes scannées par subset. Chaque colonne est une liste d'autorités
#: séparées par ``|``; la colonne ``<champ>_ids`` correspondante porte leurs
#: ``o:id`` et sert de clé de jointure quand elle est présente (voir
#: :func:`make_authority_resolver`). À défaut, la comparaison se fait par
#: appartenance exacte au ``Titre`` après découpage, jamais par sous-chaîne.
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


def frequency_input_columns(config_name: str) -> List[str]:
    """Columns the frequency pass reads from one content subset."""
    fields = FREQUENCY_SOURCE_FIELDS[config_name]
    return ["o:id", "pub_date", "country", *fields, *(f"{f}_ids" for f in fields)]


def calculate_frequency_stats(
    articles_df: pd.DataFrame,
    publications_df: pd.DataFrame,
    references_df: pd.DataFrame,
    audiovisual_df: Optional[pd.DataFrame] = None,
    *,
    resolve: TermResolver = title_terms,
) -> Dict[str, Dict[str, Any]]:
    """Calcule fréquence / première-dernière occurrence / pays pour chaque
    autorité, en balayant les colonnes de ``FREQUENCY_SOURCE_FIELDS``.

    Les clés sont ce que renvoie ``resolve`` : des titres par défaut, des
    ``o:id`` d'index avec :func:`make_authority_resolver`.

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
        for row in df.to_dict("records"):
            _accumulate_term_stats(term_stats, row, fields, resolve)

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

    Only the columns the pass reads are fetched (column-pruned parquet reads):
    a few MB instead of every subset's full text and embeddings, which the old
    force-redownload of four complete configs cost on every index upload.

    Returns one frame per key of :data:`FREQUENCY_SOURCE_FIELDS`.
    """
    before = get_repo_revision(repo, token=token)

    def load_one(config_name: str) -> pd.DataFrame:
        logger.info("Loading %s dataset from %s at %s...", config_name, repo, before)
        try:
            frame = load_hub_columns(
                repo,
                config_name,
                revision=before,
                columns=frequency_input_columns(config_name),
                token=token,
                console=console,
            )
        except Exception as exc:  # noqa: BLE001
            raise HubBaselineUnavailableError(
                f"Index enrichment requires '{config_name}', but it could not be "
                f"loaded from {repo} at {before}: {exc}"
            ) from exc
        missing = [c for c in ("o:id", *FREQUENCY_SOURCE_FIELDS[config_name])
                   if c not in frame.columns]
        if missing:
            raise HubBaselineUnavailableError(
                f"'{config_name}' at {before} lacks {missing}; refusing partial "
                "frequency statistics."
            )
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


def attach_frequency_stats(new_df: pd.DataFrame, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Add ``frequency``/``first_occurrence``/``last_occurrence``/``countries``
    to the index rows, keyed by each row's ``o:id``."""
    title_to_ids: Dict[str, List[str]] = defaultdict(list)
    for index_id, titre in zip(new_df["o:id"].astype(str), new_df["Titre"].fillna("")):
        if titre:
            title_to_ids[titre].append(index_id)
    stats = calculate_frequency_stats(
        frames["articles"],
        frames["publications"],
        frames["references"],
        frames.get("audiovisual"),
        resolve=make_authority_resolver(title_to_ids),
    )
    ids = new_df["o:id"].astype(str)
    empty = {'frequency': 0, 'first_occurrence': '', 'last_occurrence': '', 'countries': ''}
    for column in ("frequency", "first_occurrence", "last_occurrence", "countries"):
        new_df[column] = [stats.get(i, empty)[column] for i in ids]
    return new_df


# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

async def _attach_frequency_stats(
    new_df: pd.DataFrame, api: OmekaApiClient, repo: str, token: Optional[str]
) -> pd.DataFrame:
    """post_map hook: enrich index rows with corpus-wide term frequency
    statistics (occurrences + first/last date + countries) computed from every
    content subset in FREQUENCY_SOURCE_FIELDS, joined on authority ``o:id``
    (title fallback, see :func:`make_authority_resolver`). Runs after mapping,
    before the Hub merge."""
    frames = await load_reference_datasets(token, repo)
    console.print("[blue]→[/blue] Attaching frequency statistics to index records...")
    return attach_frequency_stats(new_df, frames)


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
