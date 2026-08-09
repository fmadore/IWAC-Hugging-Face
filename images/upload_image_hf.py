#!/usr/bin/env python3
"""
upload_image_hf.py
==================

Extrait les photographies (resource_class_id = 58, ``bibo:Image``) depuis l'API
Omeka S d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le
Hugging Face Hub comme subset ``images`` du miroir privé complet
``fmadore/islam-west-africa-collection-full``.

Les photographies sont des clichés de terrain (mosquées, radios islamiques, …)
pris par le curateur dans les cinq pays du corpus. Elles ne portent quasiment
pas de texte libre (2 descriptions sur 30) ; leur contenu visuel est capté en
aval par ``embedding_image`` (voir ``post-processing/semantic_embedding_images.py``).

L'image elle-même n'est PAS stockée dans le dataset : on garde seulement un
pointeur d'URL (``image_url``), conformément à la convention « pas de média
binaire dans HF ».

Usage
-----
    python images/upload_image_hf.py \
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
from typing import Dict, Any

# Add parent directory to path for iwac_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from iwac_common.omeka_client import (
    OmekaApiClient,
    conn_manager,
    fetch_iiif_thumbnail_url,
    fetch_primary_media_url,
)
from iwac_common.field_mappers import extract_added_date, get_value
from iwac_common.upload_runner import UploadSpec, run_upload
from iwac_common.schema import SUBSETS

load_dotenv()


# Orchestration + CLI + Rich console/logging live in
# iwac_common.upload_runner. The `images` subset has no OCR/full-text
# columns; the merge preserves the computed `embedding_image` from
# post-processing/semantic_embedding_images.py.

# Photographs (bibo:Image). The "Photograph" resource template's default class
# (33) is unused; every photograph item is filed under class 58.
IMAGE_RESOURCE_CLASS_ID = SUBSETS["images"].resource_class_ids[0]

# Each photograph belongs to exactly one country-specific item set. Map the
# item-set id to the canonical country label used across the dataset (matches
# country_mapper.py, incl. the un-accented raw ``Benin``).
ITEM_SET_COUNTRY = {
    2192: "Benin",           # Photographies (Bénin)
    2211: "Burkina Faso",    # Photographies (Burkina Faso)
    2216: "Côte d'Ivoire",   # Photographies (Côte d'Ivoire)
    2220: "Niger",           # Photographies (Niger)
    2227: "Togo",            # Photographies (Togo)
}


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

def _get_display_title(item: Dict[str, Any], field: str) -> str:
    """Extract display_title from a field (pipe-joined for lists)."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        titles = [str(v["display_title"]) for v in val
                  if isinstance(v, dict) and "display_title" in v]
        return "|".join(filter(None, titles))
    if isinstance(val, dict) and "display_title" in val:
        return str(val["display_title"])
    return ""


def _get_at_value(item: Dict[str, Any], field: str) -> str:
    """Extract @value from a field (pipe-joined for lists)."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        values = [str(v["@value"]) for v in val
                  if isinstance(v, dict) and "@value" in v]
        return "|".join(filter(None, values))
    if isinstance(val, dict) and "@value" in val:
        return str(val["@value"])
    return ""


def _get_rights_label(item: Dict[str, Any], field: str = "dcterms:rights") -> str:
    """Rights statements carry a human ``o:label`` alongside the ``@id`` URI;
    prefer the label, fall back to the URI, then to ``@value``."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, dict):
        val = [val]
    if not isinstance(val, list):
        return ""
    parts = [str(v.get("o:label") or v.get("@id") or v.get("@value") or "")
             for v in val if isinstance(v, dict)]
    return "|".join(filter(None, parts))


def _get_country_from_item_sets(item: Dict[str, Any]) -> str:
    """Map the item's country-specific photo collection to a country label."""
    for s in item.get("o:item_set", []) or []:
        country = ITEM_SET_COUNTRY.get(s.get("o:id"))
        if country:
            return country
    return ""


async def map_image_item(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka photographie (bibo:Image) en dict plat pour HF."""

    # Original image URL from the primary media (the actual JPEG).
    image_url = await fetch_primary_media_url(
        item,
        api,
        affected_fields=("image_url",),
    )

    added_date = extract_added_date(item)

    # IIIF thumbnail + manifest (only meaningful when media exists). Fall back
    # to the item's baked-in ``large`` derivative if the IIIF manifest is
    # unavailable.
    session = await conn_manager.get()
    thumbnail_url = ""
    iiif_manifest_url = ""
    if image_url:
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        if not thumbnail_url:
            thumbnail_url = (item.get("thumbnail_display_urls") or {}).get("large", "")
        iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"

    return {
        "o:id": item["o:id"],
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": added_date,
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "iiif_manifest": iiif_manifest_url,
        "image_url": image_url,
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "type": _get_display_title(item, "dcterms:type"),
        "creator": get_value(item, "dcterms:creator"),
        "pub_date": get_value(item, "dcterms:date"),
        "description": get_value(item, "dcterms:description"),
        "rights": _get_rights_label(item),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "coordinates": _get_at_value(item, "curation:coordinates"),
        "country": _get_country_from_item_sets(item),
    }




# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

SPEC = UploadSpec(
    config_name="images",
    resource_class_ids=SUBSETS["images"].resource_class_ids,
    map_item=map_image_item,
    title="🖼️ IWAC Images Upload",
    cache_dir=".cache_omk_images",
    description="Publie les photographies IWAC sur le Hub HF",
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
