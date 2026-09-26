"""Shared Omeka → flat-record field helpers.

Extracted from helper functions that were duplicated verbatim across the 6
upload scripts. Subset-specific helpers (``_get_iwac_identifier``,
``_get_value_with_lang``, ``_get_resource_class``, etc.) intentionally stay in
their original scripts.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple


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
        return (
            val.get("display_title", "")
            or val.get("@value", "")
            or val.get("@id", "")
        )
    return str(val)


def get_value_by_language(
    item: Dict[str, Any],
    field: str,
    language: str,
    *,
    untagged_matches: bool = False,
    fallback: bool = False,
) -> str:
    """Extract ONE ``@language``-tagged literal from an Omeka property.

    The language-aware counterpart to :func:`get_value`, which pipe-joins every
    value it finds. That join is wrong for a property carrying one literal per
    language — ``bibo:shortDescription`` since the summariser went bilingual —
    because it produces ``"résumé|summary"``, a single string whose two halves
    can only be recovered by splitting on a delimiter the prose is allowed to
    contain, in an order Omeka never promised to keep stable.

    Only the *first* match is returned; a second literal in the same language
    is a data error upstream, so it is logged rather than silently joined back
    into the shape this function exists to avoid.

    Args:
        language: the ``@language`` tag to select, e.g. ``"fr"``.
        untagged_matches: also accept values carrying **no** ``@language``.
            Set on the French slot: every summary written before the pipeline
            went bilingual is an untagged French literal, and without this they
            would all read as missing.
        fallback: if nothing matched, return the first value that has any
            ``@value`` at all. Right for a descriptive field of unpredictable
            language (``index``'s ``Description``); wrong for a per-language
            column, where it would file French prose under English.
    """
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, dict):
        val = [val]
    if not isinstance(val, list):
        return str(val)

    matches = [
        v for v in val
        if isinstance(v, dict) and v.get("@value")
        and (v.get("@language") == language
             or (untagged_matches and not v.get("@language")))
    ]
    if matches:
        if len(matches) > 1:
            logger.warning(
                "Item %s has %d '%s' values for %s; keeping the first",
                item.get("o:id"), len(matches), language, field,
            )
        return str(matches[0].get("@value", ""))

    if fallback:
        for v in val:
            if isinstance(v, dict) and v.get("@value"):
                return str(v.get("@value", ""))
    return ""


def get_uri_value(item: Dict[str, Any], field: str) -> str:
    """Extract a URI property's ``@id``, pipe-joined for lists.

    The counterpart to :func:`get_value` for ``uri``-typed properties such as
    ``fabio:hasURL``. ``get_value`` would work by accident today — a URI value
    carries neither ``display_title`` nor ``@value``, so the fallback chain
    reaches ``@id`` last — but it would silently prefer a stray ``@value`` if
    one were ever entered, which for a link column means publishing a label
    where consumers expect a resolvable address.
    """
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        val = [val]
    if not isinstance(val, list):
        return ""
    uris = [
        str(v["@id"]) for v in val
        if isinstance(v, dict) and isinstance(v.get("@id"), str) and v["@id"]
    ]
    return "|".join(uris)


def get_rights_label(item: Dict[str, Any], field: str = "dcterms:rights") -> str:
    """Rights statements carry a human ``o:label`` beside the ``@id`` URI;
    prefer the label, fall back to the URI, then to ``@value``."""
    if field not in item or item[field] is None:
        return ""
    val = item[field]
    if isinstance(val, dict):
        val = [val]
    if not isinstance(val, list):
        return ""
    parts = [
        str(v.get("o:label") or v.get("@id") or v.get("@value") or "")
        for v in val if isinstance(v, dict)
    ]
    return "|".join(filter(None, parts))


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


def _values(item: Dict[str, Any], field: str) -> List[Dict[str, Any]]:
    """The property's value dicts, whatever shape Omeka returned."""
    val = item.get(field)
    if isinstance(val, dict):
        return [val]
    if isinstance(val, list):
        return [v for v in val if isinstance(v, dict)]
    return []


def get_display_titles(item: Dict[str, Any], field: str) -> str:
    """Pipe-joined ``display_title`` of a property's linked resources.

    Unlike :func:`get_value`, literal values are ignored. This replaces three
    diverging local copies (index, audiovisual, images); one of them wrote the
    string ``"None"`` for a null title, another crashed on a non-dict value.
    """
    return "|".join(
        str(v["display_title"]) for v in _values(item, field) if v.get("display_title")
    )


def get_literal_values(item: Dict[str, Any], field: str) -> str:
    """Pipe-joined ``@value`` literals of a property (linked values ignored)."""
    return "|".join(
        str(v["@value"]) for v in _values(item, field)
        if v.get("@value") not in (None, "")
    )


def get_resource_ids(item: Dict[str, Any], field: str) -> str:
    """Pipe-joined Omeka ids (``value_resource_id``) of a property's linked items.

    The companion to the label columns (``subject``, ``spatial``, ``author``…),
    which carry only display titles. Joining a label column to ``index.Titre``
    merges homonyms, misses a renamed authority until every content subset is
    re-uploaded, and cannot tell a linked authority from a free-text literal;
    joining these ids to ``index.o:id`` does none of that. Literals have no id
    and are skipped, so the list is not positionally aligned with the labels.
    """
    ids = []
    for v in _values(item, field):
        rid = v.get("value_resource_id")
        if rid is None or rid == "":
            continue
        ids.append(str(rid))
    return "|".join(dict.fromkeys(ids))


def split_by_language(
    item: Dict[str, Any], field: str, language: str = "en"
) -> Tuple[str, str]:
    """``(primary, <language>)`` literals of a property that may be bilingual.

    For free-text fields whose values arrive as one literal per language
    (``dcterms:abstract`` on references: a French abstract and its English
    translation, which :func:`get_value` pipe-joined into one string):

    - the second element is the first value tagged ``language``;
    - the first element is every *other* value, pipe-joined exactly as before
      (untagged values stay there: their language is unknown, and splitting
      them would be a guess), or — when ``language`` is the only language
      present — that same value, so the primary column never loses an
      abstract that happens to be English-only.
    """
    literals = [
        (str(v["@value"]), str(v.get("@language") or "").lower())
        for v in _values(item, field) if v.get("@value") not in (None, "")
    ]
    wanted = language.lower()
    tagged = [text for text, lang in literals if lang == wanted]
    others = [text for text, lang in literals if lang != wanted]
    secondary = tagged[0] if tagged else ""
    if len(tagged) > 1:
        logger.warning(
            "Item %s has %d '%s' values for %s; keeping the first",
            item.get("o:id"), len(tagged), language, field,
        )
    primary = "|".join(others) if others else secondary
    return primary, secondary


def countries_from_item_sets(
    item: Dict[str, Any], mapping: Mapping[int, str], *, first_only: bool = True
) -> str:
    """Country label(s) from an item's membership in country-specific item sets.

    ``first_only`` returns the first match (subsets where an item belongs to one
    country collection); otherwise every match, pipe-joined in item-set order.
    """
    countries: List[str] = []
    for item_set in item.get("o:item_set") or []:
        if not isinstance(item_set, dict):
            continue
        country = mapping.get(item_set.get("o:id"))
        if country and country not in countries:
            countries.append(country)
            if first_only:
                break
    return "|".join(countries)


_ISO_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def _iso_date(text: str) -> Optional[Tuple[int, Optional[int], Optional[int]]]:
    match = _ISO_DATE_RE.match(text.strip())
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else None
    day = int(match.group(3)) if match.group(3) else None
    if month is not None and not 1 <= month <= 12:
        return None
    if day is not None and not 1 <= day <= 31:
        return None
    return year, month, day


def parse_pub_date(value: Any) -> Tuple[Optional[int], str]:
    """``(pub_year, pub_date_precision)`` for a raw ``dcterms:date`` string.

    ``pub_date`` stays verbatim; these are its analysable companions, because
    the collection mixes ``YYYY-MM-DD``, ``YYYY-MM``, bare years and
    ``YYYY-MM/YYYY-MM`` ranges, which neither sort nor bin reliably as text.

    Precision is one of ``day``, ``month``, ``year``, ``range`` (the year is
    the range's start), ``other`` (a value this parser does not recognise —
    no year is guessed) or ``""`` (no date).
    """
    text = str(value or "").strip()
    if not text:
        return None, ""
    parts = text.split("/")
    if len(parts) == 2:
        start, end = _iso_date(parts[0]), _iso_date(parts[1])
        if start and end:
            return start[0], "range"
        return None, "other"
    parsed = _iso_date(text)
    if parsed is None:
        return None, "other"
    year, month, day = parsed
    return year, ("day" if day else "month" if month else "year")


def parse_coordinates(value: Any) -> Tuple[Optional[float], Optional[float]]:
    """``(latitude, longitude)`` from a ``"lat, lng"`` string, else ``(None, None)``.

    ``curation:coordinates`` is stored as one string; several values, a
    malformed pair or an out-of-range coordinate yield nulls rather than a
    guess.
    """
    text = str(value or "").strip()
    if not text or "|" in text:
        return None, None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        return None, None
    try:
        lat, lng = float(parts[0]), float(parts[1])
    except ValueError:
        return None, None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None, None
    return lat, lng


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
    "get_display_titles",
    "get_literal_values",
    "get_resource_ids",
    "split_by_language",
    "countries_from_item_sets",
    "parse_pub_date",
    "parse_coordinates",
    "get_value_by_language",
    "get_uri_value",
    "get_rights_label",
    "is_content_public",
    "get_media_ids",
    "to_int_or_none",
    "extract_added_date",
]
