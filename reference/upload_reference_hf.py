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
import pandas as pd
from typing import Dict, Any, List

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from iwac_common.omeka_client import OmekaApiClient, item_page_url
from iwac_common.field_mappers import (
    countries_from_item_sets,
    extract_added_date,
    get_resource_ids,
    get_uri_value,
    get_value,
    is_content_public,
    parse_pub_date,
    split_by_language,
)
from iwac_common.text_utils import count_words
from iwac_common.upload_runner import UploadSpec, run_upload
from iwac_common.schema import COUNTRY_ITEM_SETS, SUBSETS

load_dotenv()

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

# Country-specific reference collections live in iwac_common.schema.


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

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


async def map_reference(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka de référence en dict plat pour HF datasets."""

    # Normalise bibo:doi: keep real DOIs (bare form) in ``doi``, and fold any
    # non-DOI URL mistakenly stored there into the ``URL`` column instead.
    url_parts = [u for u in get_uri_value(item, "fabio:hasURL").split("|") if u]
    doi_clean, doi_urls = split_doi_and_urls(item)
    for u in doi_urls:
        if u not in url_parts:
            url_parts.append(u)

    pub_date = get_value(item, "dcterms:date")
    pub_year, pub_date_precision = parse_pub_date(pub_date)

    # A reference often carries its abstract twice, in French and in English,
    # as two literals of one property; get_value() pipe-joined them. The
    # English-tagged one now has its own column; untagged values stay in
    # ``abstract`` (their language is unknown), and an English-only abstract
    # appears in both, so ``abstract`` never loses one.
    abstract, abstract_en = split_by_language(item, "dcterms:abstract", "en")

    # Full text (bibo:content) — kept as the OCR column. This is private on
    # the Omeka side for most references: publish_public.py masks it per row
    # (OCR_is_public), so only the private repo carries it in full.
    content_text = get_value(item, "bibo:content")

    return {
        "o:id": item["o:id"],
        "iwac_url": item_page_url(item["o:id"]),
        "identifier": _get_iwac_identifier(item, "dcterms:identifier"),
        "added_date": extract_added_date(item),
        "o:resource_class": _get_resource_class(item),
        "title": get_value(item, "dcterms:title"),
        "author": get_value(item, "bibo:authorList"),
        "author_ids": get_resource_ids(item, "bibo:authorList"),
        "editor": get_value(item, "bibo:editorList"),
        "editor_ids": get_resource_ids(item, "bibo:editorList"),
        "review_of": get_value(item, "bibo:reviewOf"),
        "publisher": get_value(item, "dcterms:publisher"),
        "publisher_ids": get_resource_ids(item, "dcterms:publisher"),
        "pub_date": pub_date,
        "pub_year": pub_year,
        "pub_date_precision": pub_date_precision,
        "type": get_value(item, "dcterms:type"),
        "book_title": get_value(item, "dcterms:alternative"),
        # Bibliographic numbers stay verbatim strings: the source holds ranges
        # and roman numerals ("12-15", "iv"), which an int conversion used to
        # blank before the column was cast back to str anyway.
        "chapter": get_value(item, "bibo:chapter"),
        "volume": get_value(item, "bibo:volume"),  # may hold several: "1|2"
        "issue": get_value(item, "bibo:issue"),  # may hold several: "3|4"
        "abstract": abstract,
        "abstract_en": abstract_en,
        "edition": get_value(item, "bibo:edition"),
        "nb_pages": get_value(item, "bibo:numPages"),
        "page_start": get_value(item, "bibo:pageStart"),
        "page_end": get_value(item, "bibo:pageEnd"),
        "extent": get_value(item, "dcterms:extent"),
        "is_part_of": get_value(item, "dcterms:isPartOf"),
        "provenance": get_value(item, "dcterms:provenance"),
        "subject": get_value(item, "dcterms:subject"),
        "subject_ids": get_resource_ids(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "spatial_ids": get_resource_ids(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "doi": doi_clean,
        "URL": "|".join(url_parts),
        "OCR": content_text,
        "OCR_is_public": is_content_public(item),
        "nb_mots": count_words(content_text),
        "country": countries_from_item_sets(
            item, COUNTRY_ITEM_SETS["references"], first_only=False
        ),
    }


# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

BIBLIOGRAPHIC_STRING_COLUMNS = ("chapter", "edition", "nb_pages", "page_start", "page_end")


def _blank_missing_bibliographic_numbers(final_df: pd.DataFrame) -> pd.DataFrame:
    """Hub-only rows kept by the outer merge have no Omeka values; give these
    string columns ``""`` rather than null, as the other rows have."""
    for col in BIBLIOGRAPHIC_STRING_COLUMNS:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna("").astype(str)
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
    post_merge=_blank_missing_bibliographic_numbers,
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
