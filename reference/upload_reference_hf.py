#!/usr/bin/env python3
"""
upload_reference_hf.py
=====================

Extrait les références bibliographiques (resource_class_id = [35, 43, 88, 40, 82, 178, 52, 77, 305])
depuis l'API Omeka S d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le Hugging Face
Hub comme subset 'references' du miroir privé (fmadore/islam-west-africa-collection-full).

La colonne OCR (bibo:content, texte intégral privé côté Omeka) n'est poussée
que vers le repo privé; publish_public.py produit la projection publique.

Usage
-----
    python upload_reference_hf.py \
        --repo fmadore/islam-west-africa-collection-full \
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
import re
import logging
import pandas as pd
from typing import Dict, Any, List

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from iwac_common.omeka_client import OmekaApiClient
from iwac_common.field_mappers import extract_added_date, get_value, is_content_public
from iwac_common.upload_runner import UploadSpec, run_upload
from iwac_common.schema import SUBSETS

load_dotenv()

# Rich logging itself is configured by iwac_common.upload_runner; this is the
# named logger the field-mapping warnings below write to.
logger = logging.getLogger("upload")


# Orchestration (fetch of all 9 reference classes → map loop → outer merge →
# validate → push), the CLI (--repo/--max-shard-size/--no-cache/--dry-run/
# --force-shrink/--stale-rows) and the Rich console/logging setup live in
# iwac_common.upload_runner.


# Reference resource classes. The shared runner fetches each in turn and
# aborts the whole run on any class fetch failure or truncation — safer than
# the old continue-on-error, which could silently drop a class (and, via the
# outer merge, hide it as blank-Omeka rows).
RESOURCE_CLASSES = SUBSETS["references"].resource_class_ids

# Resource class mapping
RESOURCE_CLASS_MAPPING = {
    35: 'Article de revue',
    43: 'Chapitre',
    88: 'Thèse',
    40: 'Livre',
    82: 'Rapport',
    178: 'Compte rendu',
    52: 'Ouvrage collectif',
    77: 'Communication',
    305: 'Article de blog'
}

# Country mapping based on item sets
COUNTRY_ITEM_SETS = {
    2193: 'Benin',
    2212: 'Burkina Faso',
    2217: 'Côte d\'Ivoire',
    2222: 'Niger',
    2225: 'Nigeria',
    2228: 'Togo'
}


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

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


# ``bibo:doi`` holds a URI value that is *usually* a DOI (as an https://doi.org/
# link) but is sometimes a plain article/repository URL (ethnographiques.org,
# hdl.handle.net, …). We normalise real DOIs to their bare form
# (``10.xxxx/yyyy``) and route anything that is not a DOI to the URL column.
_DOI_PREFIX_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)
_DOI_CORE_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def _extract_doi_raw_values(item: Dict[str, Any]) -> List[str]:
    """Return the raw ``bibo:doi`` values (URI ``@id`` or literal ``@value``)."""
    val = item.get("bibo:doi")
    if not val:
        return []
    if isinstance(val, dict):
        val = [val]
    if not isinstance(val, list):
        return [str(val).strip()] if str(val).strip() else []
    out: List[str] = []
    for v in val:
        if isinstance(v, dict):
            s = str(v.get("@id") or v.get("@value") or "").strip()
        else:
            s = str(v).strip()
        if s:
            out.append(s)
    return out


def split_doi_and_urls(item: Dict[str, Any]) -> tuple[str, List[str]]:
    """Split ``bibo:doi`` values into (bare DOIs, non-DOI URLs).

    - ``https://doi.org/10.1163/x`` / bare ``10.1163/x`` → DOI ``10.1163/x``
    - ``https://www.ethnographiques.org/...``, ``https://hdl.handle.net/...``
      → returned as URLs (belong in the ``URL`` column, not ``doi``)
    """
    dois: List[str] = []
    urls: List[str] = []
    for raw in _extract_doi_raw_values(item):
        core = _DOI_PREFIX_RE.sub("", raw).strip()
        if _DOI_CORE_RE.match(core):
            dois.append(core)
        else:
            urls.append(raw)
    return "|".join(dois), urls


def _get_iwac_identifier(item: Dict[str, Any], field: str) -> str:
    """Extract identifier values that start with 'iwac-reference'"""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        for v in val:
            identifier = str(v.get("display_title") or v.get("@value") or v.get("@id", ""))
            if identifier.startswith("iwac-reference"):
                return identifier
    elif isinstance(val, dict):
        identifier = val.get("display_title", "") or val.get("@value", "")
        if identifier.startswith("iwac-reference"):
            return identifier
    else:
        identifier = str(val)
        if identifier.startswith("iwac-reference"):
            return identifier
    return ""


def _get_resource_class(item: Dict[str, Any]) -> str:
    """Extract resource class information and map to human-readable name"""
    if "o:resource_class" in item and isinstance(item["o:resource_class"], dict):
        class_id = item["o:resource_class"].get("o:id")
        if class_id and class_id in RESOURCE_CLASS_MAPPING:
            return RESOURCE_CLASS_MAPPING[class_id]
        elif class_id:
            return str(class_id)  # Return ID as string if not in mapping
    return ""


def _get_countries_from_item_sets(item: Dict[str, Any]) -> str:
    """Map country based on item set IDs"""
    countries = []
    if "o:item_set" in item and isinstance(item["o:item_set"], list):
        for item_set in item["o:item_set"]:
            if isinstance(item_set, dict) and "o:id" in item_set:
                item_set_id = item_set["o:id"]
                if item_set_id in COUNTRY_ITEM_SETS:
                    countries.append(COUNTRY_ITEM_SETS[item_set_id])
    return "|".join(countries) if countries else ""


async def map_reference(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka de référence en dict plat pour HF datasets."""

    # Map country based on item set IDs
    country = _get_countries_from_item_sets(item)

    # Custom logic to extract URL from fabio:hasURL, prioritizing @id
    fabio_has_url_data = item.get("fabio:hasURL")
    extracted_fabio_url = ""
    if isinstance(fabio_has_url_data, list):
        urls = []
        for v_item in fabio_has_url_data:
            if isinstance(v_item, dict):
                id_val = v_item.get("@id")
                if id_val and isinstance(id_val, str):
                    urls.append(id_val)
        if urls:
            extracted_fabio_url = "|".join(urls)
    elif isinstance(fabio_has_url_data, dict):
        id_val = fabio_has_url_data.get("@id")
        if id_val and isinstance(id_val, str): # Ensure it's a non-empty string
            extracted_fabio_url = id_val
    elif isinstance(fabio_has_url_data, str) and fabio_has_url_data: # If it's already a non-empty string
        extracted_fabio_url = fabio_has_url_data
    # If none of the above, extracted_fabio_url remains ""

    # Normalise bibo:doi: keep real DOIs (bare form) in ``doi``, and fold any
    # non-DOI URL mistakenly stored there into the ``URL`` column instead.
    doi_clean, doi_urls = split_doi_and_urls(item)
    if doi_urls:
        url_parts = [u for u in extracted_fabio_url.split("|") if u]
        for u in doi_urls:
            if u not in url_parts:
                url_parts.append(u)
        extracted_fabio_url = "|".join(url_parts)

    # Keep volume as string (can contain multiple values like "1|2")
    volume_str = get_value(item, "bibo:volume")

    # Keep issue as string (can contain multiple values like "3|4")
    issue_str = get_value(item, "bibo:issue")

    # Convert edition to int
    edition_str = get_value(item, "bibo:edition")
    edition_int = ""
    if edition_str:
        try:
            edition_int = int(edition_str)
        except ValueError:
            logger.warning(
                f"Could not convert edition '{edition_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Convert chapter to int
    chapter_str = get_value(item, "bibo:chapter")
    chapter_int = ""
    if chapter_str:
        try:
            chapter_int = int(chapter_str)
        except ValueError:
            logger.warning(
                f"Could not convert chapter '{chapter_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Convert nb_pages to int
    nb_pages_str = get_value(item, "bibo:numPages")
    nb_pages_int = ""
    if nb_pages_str:
        try:
            nb_pages_int = int(nb_pages_str)
        except ValueError:
            logger.warning(
                f"Could not convert nb_pages '{nb_pages_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    # Convert page start/end to int
    page_start_str = get_value(item, "bibo:pageStart")
    page_start_int = ""
    if page_start_str:
        try:
            page_start_int = int(page_start_str)
        except ValueError:
            logger.warning(
                f"Could not convert pageStart '{page_start_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    page_end_str = get_value(item, "bibo:pageEnd")
    page_end_int = ""
    if page_end_str:
        try:
            page_end_int = int(page_end_str)
        except ValueError:
            logger.warning(
                f"Could not convert pageEnd '{page_end_str}' to int for item {item['o:id']}. Defaulting to empty."
            )

    added_date = extract_added_date(item)

    # Full text (bibo:content) — kept as the OCR column. This is private on
    # the Omeka side, so the references subset must only be pushed to the
    # PRIVATE repo; publish_public.py strips OCR before the public push.
    content_text = get_value(item, "bibo:content")
    nb_mots = count_words(content_text)

    return {
        "o:id": item["o:id"],
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "identifier": _get_iwac_identifier(item, "dcterms:identifier"),
        "added_date": added_date,
        "o:resource_class": _get_resource_class(item),
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "bibo:authorList"),
        "editor": get_value(item, "bibo:editorList"),
        "review_of": get_value(item, "bibo:reviewOf"),
        "publisher": get_value(item, "dcterms:publisher"),
        "pub_date": get_value(item, "dcterms:date"),
        "type": get_value(item, "dcterms:type"),
        "book_title": get_value(item, "dcterms:alternative"),
        "chapter": chapter_int,
        "volume": volume_str,
        "issue": issue_str,
        "abstract": get_value(item, "dcterms:abstract"),
        "edition": edition_int,
        "nb_pages": nb_pages_int,
        "page_start": page_start_int,
        "page_end": page_end_int,
        "extent": get_value(item, "dcterms:extent"),
        "is_part_of": get_value(item, "dcterms:isPartOf"),
        "provenance": get_value(item, "dcterms:provenance"),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "doi": doi_clean,
        "URL": extracted_fabio_url,
        "OCR": content_text,
        "OCR_is_public": is_content_public(item),
        "nb_mots": nb_mots,
        "country": country,
    }




# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

def _cast_mixed_columns_to_str(final_df: pd.DataFrame) -> pd.DataFrame:
    """References mix ints and strings in these bibliographic columns
    (e.g. "12-15" vs 12); cast to str so Arrow gets a stable schema. Empty
    values stay empty strings, not the literal 'nan'."""
    mixed_type_columns = ["chapter", "edition", "nb_pages", "page_start", "page_end"]
    for col in mixed_type_columns:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype(str).replace("nan", "")
    return final_df


SPEC = UploadSpec(
    config_name="references",
    resource_class_ids=RESOURCE_CLASSES,  # 9 bibliographic classes
    map_item=map_reference,
    title="📚 IWAC References Upload",
    cache_dir=".cache_omk_references",
    description="Publie les références bibliographiques IWAC sur le Hub HF",
    # References keep Hub-only rows (deleted Omeka items) via an outer merge,
    # dropping a few legacy columns; --stale-rows drop removes them.
    merge_how="outer",
    merge_suffixes=("", "_old"),
    columns_to_exclude=("o:item_set", "o:media/file", "iiif_manifest", "thumbnail"),
    supports_stale_rows=True,
    post_merge=_cast_mixed_columns_to_str,
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
