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

from iwac_common.omeka_client import (
    OmekaApiClient,
    conn_manager,
    fetch_iiif_thumbnail_url,
    fetch_primary_media_url,
    iiif_manifest_url,
    item_page_url,
)
from iwac_common.field_mappers import (
    visibility_metadata,
    countries_from_item_sets,
    extract_added_date,
    get_display_titles,
    get_literal_values,
    get_resource_ids,
    get_rights_label,
    get_value,
    parse_coordinates,
    parse_pub_date,
)
from iwac_common.upload_runner import UploadSpec, run_upload
from iwac_common.schema import COUNTRY_ITEM_SETS, SUBSETS



# Orchestration + CLI + Rich console/logging live in
# iwac_common.upload_runner. The `images` subset has no OCR/full-text
# columns; the merge preserves the computed `embedding_image` from
# post-processing/semantic_embedding_images.py.

# Photographs (bibo:Image). The "Photograph" resource template's default class
# (33) is unused; every photograph item is filed under class 58.
IMAGE_RESOURCE_CLASS_ID = SUBSETS["images"].resource_class_ids[0]

# Each photograph belongs to exactly one country-specific item set; the
# mapping lives in iwac_common.schema.COUNTRY_ITEM_SETS["images"].
ITEM_SET_COUNTRY = COUNTRY_ITEM_SETS["images"]


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

async def map_image_item(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka photographie (bibo:Image) en dict plat pour HF."""

    # Original image URL from the primary media (the actual JPEG).
    image_url = await fetch_primary_media_url(
        item,
        api,
        affected_fields=("image_url", "thumbnail", "iiif_manifest"),
    )

    pub_date = get_value(item, "dcterms:date")
    pub_year, pub_date_precision = parse_pub_date(pub_date)
    coordinates = get_literal_values(item, "curation:coordinates")
    latitude, longitude = parse_coordinates(coordinates)

    # IIIF thumbnail + manifest (only meaningful when media exists). Fall back
    # to the item's baked-in ``large`` derivative if the IIIF manifest is
    # unavailable.
    thumbnail_url = ""
    manifest_url = ""
    if image_url:
        session = await conn_manager.get()
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        if not thumbnail_url:
            thumbnail_url = (item.get("thumbnail_display_urls") or {}).get("large", "")
        manifest_url = iiif_manifest_url(item["o:id"])

    return {
        "o:id": item["o:id"],
        **visibility_metadata(item, "images"),
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": extract_added_date(item),
        "iwac_url": item_page_url(item["o:id"]),
        "iiif_manifest": manifest_url,
        "image_url": image_url,
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "type": get_display_titles(item, "dcterms:type"),
        "creator": get_value(item, "dcterms:creator"),
        "creator_ids": get_resource_ids(item, "dcterms:creator"),
        "pub_date": pub_date,
        "pub_year": pub_year,
        "pub_date_precision": pub_date_precision,
        "description": get_value(item, "dcterms:description"),
        "rights": get_rights_label(item),
        "subject": get_value(item, "dcterms:subject"),
        "subject_ids": get_resource_ids(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "spatial_ids": get_resource_ids(item, "dcterms:spatial"),
        "coordinates": coordinates,
        "latitude": latitude,
        "longitude": longitude,
        "country": countries_from_item_sets(item, ITEM_SET_COUNTRY),
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
