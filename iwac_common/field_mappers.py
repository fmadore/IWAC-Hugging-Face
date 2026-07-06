"""Shared Omeka → flat-record field helpers.

Extracted from helper functions that were duplicated verbatim across the 6
upload scripts. Subset-specific helpers (``_get_iwac_identifier``,
``_get_value_with_lang``, ``_get_resource_class``, etc.) intentionally stay in
their original scripts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def get_value(item: Dict[str, Any], field: str) -> str:
    """Extract a flat string value from an Omeka property.

    Falls back through ``display_title`` → ``@value`` → ``@id`` and
    pipe-joins lists. Returns ``""`` when the field is absent or null.
    """
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, list):
        parts = [
            str(v.get("display_title") or v.get("@value") or v.get("@id", ""))
            for v in val
        ]
        return "|".join(filter(None, parts))
    if isinstance(val, dict):
        return val.get("display_title", "") or val.get("@value", "")
    return str(val)


def is_content_public(item: Dict[str, Any], field: str = "bibo:content") -> bool:
    """True iff ``field``'s full text is publicly visible on Omeka.

    Omeka value dicts carry a per-value ``is_public`` flag; the
    authenticated API returns private values too (with the flag False). We
    treat the text as public only when there is non-empty text **and every**
    non-empty value is public — a mixed item (some values private) is
    treated as private, the safe direction, so the private text never leaks
    into the public HF projection. Cross-checked against the anonymous API:
    public-flagged items return their text anonymously, private-flagged
    ones return nothing.

    Consumed by ``post-processing/publish_public.py`` (via the
    ``OCR_is_public`` column) to keep public full text public while
    stripping private full text.
    """
    val = item.get(field)
    if not val:
        return False
    if isinstance(val, dict):
        val = [val]
    if not isinstance(val, list):
        return False
    text_vals = [
        v for v in val
        if isinstance(v, dict) and str(v.get("@value") or v.get("display_title") or "").strip()
    ]
    return bool(text_vals) and all(v.get("is_public") is True for v in text_vals)


def get_media_ids(item: Dict[str, Any]) -> str:
    """Pipe-joined ``o:media`` IDs, or ``""`` if none."""
    if "o:media" in item and isinstance(item["o:media"], list):
        return "|".join(str(m["o:id"]) for m in item["o:media"])
    return ""


def to_int_or_none(value: Any) -> Optional[int]:
    """Best-effort int conversion. Returns ``None`` for empty / unparseable input."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except (ValueError, TypeError):
            return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def extract_added_date(item: Dict[str, Any]) -> str:
    """Extract the ISO date (``YYYY-MM-DD``) from an item's ``o:created``.

    Returns ``""`` when the field is missing or unparseable. A warning is
    logged for malformed values.
    """
    if "o:created" not in item or not isinstance(item["o:created"], dict):
        return ""
    created_value = item["o:created"].get("@value", "")
    if not created_value:
        return ""
    try:
        return created_value.split("T")[0]
    except Exception:
        logger.warning(
            "Could not parse added date '%s' for item %s",
            created_value,
            item.get("o:id"),
        )
        return ""


__all__ = [
    "get_value",
    "is_content_public",
    "get_media_ids",
    "to_int_or_none",
    "extract_added_date",
]
