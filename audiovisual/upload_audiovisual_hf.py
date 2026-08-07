#!/usr/bin/env python3
"""
upload_audiovisual_hf.py
========================

Extrait les documents audiovisuels (resource_class_id = 38) depuis l'API Omeka S
d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le Hugging Face
Hub comme subset 'audiovisual' du repository fmadore/islam-west-africa-collection-full.

Usage
-----
    python audiovisual/upload_audiovisual_hf.py \
        --max-shard-size 1GB

Options CLI (partagées, voir iwac_common.upload_runner)
------------------------------------------------------
  --repo            Repository Hugging Face cible (défaut : miroir privé complet)
  --max-shard-size  Taille max d'un shard Parquet (ex. 500MB, 1GB)
  --no-cache        Ignore le cache local des réponses Omeka (TTL 24h)
  --dry-run         Fetch + map + merge, mais ne pousse rien
  --force-shrink    Autorise un dataset nettement plus petit que celui du Hub

Variables d'environnement
------------------------
  OMEKA_BASE_URL        Base URL de l'API, ex. https://islam.zmo.de/api
  OMEKA_KEY_IDENTITY    Identité de la clé Omeka
  OMEKA_KEY_CREDENTIAL  Credential de la clé Omeka
  HF_TOKEN              Jeton d'accès personnel Hugging Face (facultatif si
                        vous appelez login() de manière interactive)
"""

import os
import re
import sys
from typing import Dict, Any, Optional

# Add parent directory to path for iwac_common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from iwac_common.omeka_client import (
    OmekaApiClient,
    conn_manager,
    fetch_iiif_thumbnail_url,
)
from iwac_common.field_mappers import (
    extract_added_date,
    get_value,
    get_value_by_language,
    is_content_public,
)
from iwac_common.upload_runner import UploadSpec, run_upload

load_dotenv()


# Orchestration (fetch → map loop → merge → validate → push), the CLI
# (--repo, --max-shard-size, --no-cache, --dry-run, --force-shrink) and the
# Rich console/logging setup live in iwac_common.upload_runner.


# ---------------------------------------------------------------------------
# Fonctions d'aide pour mapper les champs Omeka → plat
# ---------------------------------------------------------------------------

def count_words(text: Optional[str]) -> int:
    """Compte le nombre de mots dans une chaîne. Retourne 0 si vide/None.

    Même logique que ``reference/upload_reference_hf.py`` : les mots sont les
    séquences alphanumériques (``\\b\\w+\\b``), ce qui gère la ponctuation et
    les séparateurs multiples.
    """
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", str(text).lower()))


def _get_display_title(item: Dict[str, Any], field: str) -> str:
    """Extract display_title from a field."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        titles = []
        for v in val:
            if isinstance(v, dict) and "display_title" in v:
                titles.append(str(v["display_title"]))
        return "|".join(filter(None, titles))
    elif isinstance(val, dict) and "display_title" in val:
        return str(val["display_title"])
    return ""


def _get_at_value(item: Dict[str, Any], field: str) -> str:
    """Extract @value from a field."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        values = []
        for v in val:
            if isinstance(v, dict) and "@value" in v:
                values.append(str(v["@value"]))
        return "|".join(filter(None, values))
    elif isinstance(val, dict) and "@value" in val:
        return str(val["@value"])
    return ""


async def map_audiovisual_document(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka audiovisuel en dict plat pour HF datasets."""

    primary_url = ""
    if item.get("o:primary_media"):
        mid = item["o:primary_media"]["@id"].split("/")[-1]
        mdata = await api.fetch_media_data(mid)
        primary_url = mdata.get("o:original_url", "")

    # Extract publisher and determine country
    publisher = _get_display_title(item, "dcterms:publisher")
    country = "Nigeria"  # Fixed country for all audiovisual documents

    added_date = extract_added_date(item)

    # Full text / transcription (bibo:content), kept as the OCR column to
    # match the other content subsets. Only ~4/47 audiovisual items carry a
    # transcription. Private on the Omeka side, so ``OCR_is_public`` drives
    # per-row masking in publish_public.py; the audiovisual subset must only
    # be pushed to the PRIVATE repo.
    content_text = get_value(item, "bibo:content")
    nb_mots = count_words(content_text)

    # Fetch thumbnail URL and set IIIF manifest URL only if media exists
    session = await conn_manager.get()
    thumbnail_url = ""
    iiif_manifest_url = ""

    if primary_url:
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"

    return {
        "o:id": item["o:id"],
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": added_date,
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        "iiif_manifest": iiif_manifest_url,
        "PDF": primary_url,  # Keeping as PDF for consistency, though it might be video/audio
        "thumbnail": thumbnail_url,
        "title": get_value(item, "dcterms:title"),
        "creator": get_value(item, "dcterms:creator"),
        "publisher": publisher,
        "country": country,
        "pub_date": get_value(item, "dcterms:date"),
        # Empty for all 47 rows today, but kept the same shape as articles and
        # documents so populating it later (iwac-mcp-server TODO) is a data
        # change rather than a schema change. See upload_newspaper_hf.py.
        "descriptionAI": get_value_by_language(
            item, "bibo:shortDescription", "fr", untagged_matches=True
        ),
        "descriptionAI_en": get_value_by_language(
            item, "bibo:shortDescription", "en"
        ),
        "volume": _get_at_value(item, "bibo:volume"),
        "issue": _get_at_value(item, "bibo:issue"),
        "is_part_of": _get_at_value(item, "dcterms:isPartOf"),
        "extent": _get_at_value(item, "dcterms:extent"),
        "medium": _get_display_title(item, "dcterms:medium"),
        "subject": get_value(item, "dcterms:subject"),
        "spatial": get_value(item, "dcterms:spatial"),
        "language": get_value(item, "dcterms:language"),
        "source": get_value(item, "dcterms:source"),
        "OCR": content_text,
        "nb_mots": nb_mots,
        "OCR_is_public": is_content_public(item),
    }


# ---------------------------------------------------------------------------
# Spec + entry point (shared pipeline in iwac_common.upload_runner)
# ---------------------------------------------------------------------------

SPEC = UploadSpec(
    config_name="audiovisual",
    resource_class_ids=(38,),  # Audiovisual documents
    map_item=map_audiovisual_document,
    title="🎬 IWAC Audiovisual Upload",
    cache_dir=".cache_omk_audiovisual",
    description="Publie les documents audiovisuels IWAC sur le Hub HF",
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
