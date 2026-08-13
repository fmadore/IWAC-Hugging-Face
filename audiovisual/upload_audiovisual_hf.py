#!/usr/bin/env python3
"""
upload_audiovisual_hf.py
========================

Extrait les documents audiovisuels (resource_class_id = 38) depuis l'API Omeka S
d'IWAC, les convertit en dataset Arrow/Parquet et les pousse sur le Hugging Face
Hub comme subset 'audiovisual' du repository fmadore/islam-west-africa-collection-full.

Deux populations, une seule classe
----------------------------------
Depuis 2026-08-12 la classe 38 contient deux types d'items qu'il faut
distinguer, d'où la colonne ``source_type`` :

- ``deposited`` (template 19) — DVD/CD déposés au projet, avec un vrai fichier
  vidéo/audio : ``PDF`` pointe le fichier et ``iiif_manifest`` est utilisable.
- ``youtube`` (template 23) — vidéos YouTube intégrées, ingérées par
  ``Audiovisual/youtube_sync.py`` (dépôt IWAC-automation). Le média n'a **aucun
  fichier** (l'ingester ne garde que des vignettes) : ``o:original_url`` est
  null, donc ``PDF`` reste vide et le manifeste IIIF n'a aucun canvas — c'est
  ``URL`` (``fabio:hasURL``) qui porte le lien canonique de visionnage.

Aucun binaire YouTube n'est téléchargé : le dataset ne stocke que des métadonnées,
du texte dérivé (la transcription) et des pointeurs d'URL.

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
    fetch_primary_media_url,
)
from iwac_common.field_mappers import (
    extract_added_date,
    get_rights_label,
    get_uri_value,
    get_value,
    get_value_by_language,
    is_content_public,
)
from iwac_common.upload_runner import UploadSpec, run_upload
from iwac_common.schema import SUBSETS
from country_mapper import get_country_from_newspaper

load_dotenv()


# Resource template that marks an embedded YouTube video. Class 38 alone no
# longer identifies a kind of thing (see the module docstring).
YOUTUBE_TEMPLATE_ID = 23

# Country-specific YouTube collections. Unlike the legacy sets — 2183
# "Enregistrements audio" and 2184 "Collection de sermons islamiques sur
# vidéo", which group by format rather than place — every item in these is
# from one country's channels, so membership settles the country outright.
# Bénin is pre-declared and still empty; the ingester creates one set per
# country as channels are approved.
ITEM_SET_COUNTRY = {
    2194: "Benin",           # Vidéos YouTube (Bénin)
    108260: "Burkina Faso",  # Vidéos YouTube (Burkina Faso)
}

# ``dcterms:spatial`` links to the French-labelled place authority, while the
# dataset's canonical country labels are the ones country_mapper emits (note
# the un-accented ``Benin``/``Nigeria``). Places that are not one of the six
# collection countries — cities like Zaria, or neighbours like Mali — are
# absent on purpose: they must not resolve a country.
SPATIAL_COUNTRY = {
    "Bénin": "Benin",
    "Benin": "Benin",
    "Burkina Faso": "Burkina Faso",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "Niger": "Niger",
    "Nigéria": "Nigeria",
    "Nigeria": "Nigeria",
    "Togo": "Togo",
}

# ISO 8601 duration as Omeka stores it in dcterms:extent: "PT6M51S" from the
# YouTube API (second precision), "PT45M" on the deposited recordings.
_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


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


def _source_type(item: Dict[str, Any]) -> str:
    """``youtube`` for an embedded video, ``deposited`` for a DVD/CD recording.

    The template is authoritative — it is what the ingester sets and what the
    admin form reflects — so it is read first; the URL host is a fallback for
    an item filed by hand without the template.
    """
    template = item.get("o:resource_template") or {}
    if template.get("o:id") == YOUTUBE_TEMPLATE_ID:
        return "youtube"
    if "youtube.com/" in get_uri_value(item, "fabio:hasURL") or "youtu.be/" in get_uri_value(
        item, "fabio:hasURL"
    ):
        return "youtube"
    return "deposited"


def _parse_iso_duration_seconds(value: Optional[str]) -> Optional[int]:
    """ISO 8601 duration → whole seconds, or ``None`` when unparseable.

    ``extent`` stays as the verbatim Omeka string; this is the analysable
    companion, because the two populations write different precisions
    ("PT6M51S" from the YouTube API against "PT45M" on the deposited
    recordings) and neither sorts or sums as text.
    """
    if not value:
        return None
    match = _ISO_DURATION_RE.match(str(value).strip())
    if not match or not any(match.groupdict().values()):
        return None
    parts = {k: float(v) for k, v in match.groupdict().items() if v is not None}
    total = (
        parts.get("days", 0.0) * 86400
        + parts.get("hours", 0.0) * 3600
        + parts.get("minutes", 0.0) * 60
        + parts.get("seconds", 0.0)
    )
    return int(round(total))


def _resolve_country(item: Dict[str, Any]) -> str:
    """Country for one audiovisual item, or ``""`` when nothing settles it.

    Three layers, first match wins — no subset-wide default, which used to
    stamp "Nigeria" on every row and would now mislabel every Burkinabè video:

    1. the country-specific item set (settles all YouTube items);
    2. the linked ``dcterms:spatial`` place, when it names one of the six
       collection countries (settles the deposited Nigerian recordings, which
       carry both "Nigéria" and a city such as "Zaria");
    3. the publisher, via the shared outlet index.

    An unresolved item is left blank on purpose: an empty country is a visible
    gap, a guessed one is a silent error.
    """
    for item_set in item.get("o:item_set") or []:
        country = ITEM_SET_COUNTRY.get((item_set or {}).get("o:id"))
        if country:
            return country

    for value in item.get("dcterms:spatial") or []:
        country = SPATIAL_COUNTRY.get(str((value or {}).get("display_title") or "").strip())
        if country:
            return country

    for publisher in _get_display_title(item, "dcterms:publisher").split("|"):
        country = get_country_from_newspaper(publisher.strip())
        if country:
            return country
    return ""


async def map_audiovisual_document(item: Dict[str, Any], api: OmekaApiClient) -> Dict[str, Any]:
    """Transforme un item Omeka audiovisuel en dict plat pour HF datasets."""

    primary_url = await fetch_primary_media_url(
        item, api, affected_fields=("PDF",)
    )

    publisher = _get_display_title(item, "dcterms:publisher")
    country = _resolve_country(item)
    source_type = _source_type(item)

    added_date = extract_added_date(item)

    # Full text / transcription (bibo:content), kept as the OCR column to
    # match the other content subsets. Sparse and growing: transcription is a
    # separate opt-in pass, not part of ingest, so most rows have none and
    # ``nb_mots`` is 0 for them. ``OCR_is_public`` drives the per-row masking
    # in publish_public.py; every upload here targets the PRIVATE repo.
    content_text = get_value(item, "bibo:content")
    nb_mots = count_words(content_text)

    # A YouTube media has thumbnail derivatives but no file, so the two
    # pointers part company: the thumbnail is real and worth keeping, while
    # the IIIF manifest resolves 200 with **zero canvases** — a link that
    # opens an empty viewer. Publish the manifest only when there is an actual
    # file behind it; take the thumbnail whenever the item has any media.
    session = await conn_manager.get()
    thumbnail_url = ""
    iiif_manifest_url = ""

    if item.get("o:primary_media"):
        thumbnail_url = await fetch_iiif_thumbnail_url(item["o:id"], session)
        if not thumbnail_url:
            thumbnail_url = (item.get("thumbnail_display_urls") or {}).get("large", "")
    if primary_url:
        iiif_manifest_url = f"https://islam.zmo.de/iiif/3/{item['o:id']}/manifest"

    extent = _get_at_value(item, "dcterms:extent")

    return {
        "o:id": item["o:id"],
        "identifier": get_value(item, "dcterms:identifier"),
        "added_date": added_date,
        "iwac_url": f"https://islam.zmo.de/s/afrique_ouest/item/{item['o:id']}",
        # Empty on a YouTube row: that media stores no file, so there is no
        # canvas to show. Kept for the deposited recordings.
        "iiif_manifest": iiif_manifest_url,
        # Legacy name, kept for compatibility: the deposited recordings' file
        # is video/audio, never a PDF. Empty for YouTube — see ``URL``.
        "PDF": primary_url,
        "thumbnail": thumbnail_url,
        # Canonical external address: the YouTube watch URL, or whatever
        # source link a deposited recording carries. Same meaning as the
        # ``URL`` column on articles/publications/references.
        "URL": get_uri_value(item, "fabio:hasURL"),
        "source_type": source_type,
        "title": get_value(item, "dcterms:title"),
        "creator": get_value(item, "dcterms:creator"),
        # Linked authority (the channel's foaf:Organization, or the producer
        # of a deposited recording), pipe-joined if an item ever carries more
        # than one.
        "publisher": publisher,
        "country": country,
        "pub_date": get_value(item, "dcterms:date"),
        # The human-authored blurb — the public YouTube description on an
        # embedded video. Distinct from ``descriptionAI``, which is reserved
        # for the model-written bibo:shortDescription.
        "description": get_value(item, "dcterms:description"),
        # Empty for every row today, but kept the same shape as articles and
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
        "extent": extent,
        "duration_seconds": _parse_iso_duration_seconds(extent),
        # The carrier as catalogued: "DVD"/"CD" for a deposited recording,
        # "Vidéo sur le web" for an embedded one. Read ``source_type`` to
        # branch on the record's shape; this field states what the object is.
        "medium": _get_display_title(item, "dcterms:medium"),
        "type": _get_display_title(item, "dcterms:type"),
        "rights": get_rights_label(item),
        # Provenance — who deposited or ingested the item, as on `documents`.
        # Distinct from ``creator``.
        "contributor": _get_display_title(item, "dcterms:contributor"),
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
    resource_class_ids=SUBSETS["audiovisual"].resource_class_ids,
    map_item=map_audiovisual_document,
    title="🎬 IWAC Audiovisual Upload",
    cache_dir=".cache_omk_audiovisual",
    description="Publie les documents audiovisuels IWAC sur le Hub HF",
    # Nullable: an item whose dcterms:extent is missing or unparseable has no
    # duration, which is not the same as a duration of zero.
    int_columns=("duration_seconds",),
)


if __name__ == "__main__":
    sys.exit(run_upload(SPEC))
