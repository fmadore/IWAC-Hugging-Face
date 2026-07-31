"""Canonical definition of the AI sentiment annotator panel.

One place to rotate models. Before this module the panel was restated in the
uploader, the publisher, the agreement script and the tests, which is how a
vendor slot silently changed model without anything recording it.

Two rules encoded here:

1. **Column prefixes name the model, not the vendor.** The Omeka properties are
   vendor-keyed (``iwac:gemini*``, ``iwac:chatgpt*``, ``iwac:mistral*``) and
   carry no ``iwac:*Model`` value annotation, so the property alone cannot say
   what produced a value. The 2026-01/02 corpus ran on gemini-3-flash-preview,
   gpt-5-mini and ministral-14b-2512 (campaign window verified from
   ``o:modified``), which is what the prefixes below record.

2. **Re-running sentiment with a newer model adds a new entry — never reuses
   one.** A model whose Omeka properties have been retired keeps its columns on
   the Hub with ``omeka_prefix=None``: the uploader then stops producing them,
   and ``hub_merge`` preserves them as frozen historical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: The six per-model fields, as ``(column suffix, Omeka property suffix)``.
#: ``subjectivite_score`` is stored as a ``resource:item`` link on Omeka rather
#: than a numeric literal, so readers resolve it via ``_get_subjectivity_score``.
DIMENSION_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("centralite_islam_musulmans", "Centralite"),
    ("centralite_justification", "CentraliteJustification"),
    ("polarite", "Polarite"),
    ("polarite_justification", "PolariteJustification"),
    ("subjectivite_score", "SubjectiviteScore"),
    ("subjectivite_justification", "SubjectiviteJustification"),
)

#: Suffixes whose values are free prose (needed by the publisher's prose guard).
JUSTIFICATION_SUFFIXES: Tuple[str, ...] = (
    "centralite_justification",
    "polarite_justification",
    "subjectivite_justification",
)


@dataclass(frozen=True)
class SentimentModel:
    """One annotator in the panel."""

    prefix: str
    """HF column prefix, e.g. ``gemini3flash`` -> ``gemini3flash_polarite``."""

    label: str
    """Human-readable model name for reports and the dataset card."""

    model_id: str
    """Exact provider model id that produced the values."""

    omeka_prefix: Optional[str]
    """Omeka property prefix (``iwac:gemini``), or ``None`` once the properties
    have been retired from Omeka and the Hub columns are frozen history."""

    campaign: str
    """When this model annotated the corpus, for the dataset card."""

    run_config: str = ""
    """Sampling configuration actually used for this model's campaign.

    Reproducibility, not decoration: the generation-1 panel did not share one
    config (Ministral capped output at 512 tokens, GPT-5 mini set no
    temperature at all), so a column cannot be re-derived without it.
    """

    @property
    def frozen(self) -> bool:
        """True once Omeka no longer holds this model's properties."""
        return self.omeka_prefix is None

    def column(self, suffix: str) -> str:
        return f"{self.prefix}_{suffix}"

    def columns(self) -> Tuple[str, ...]:
        return tuple(self.column(suffix) for suffix, _ in DIMENSION_FIELDS)

    def justification_columns(self) -> Tuple[str, ...]:
        return tuple(self.column(suffix) for suffix in JUSTIFICATION_SUFFIXES)

    def omeka_property(self, omeka_suffix: str) -> str:
        if self.omeka_prefix is None:
            raise ValueError(
                f"{self.prefix} is frozen: its Omeka properties were retired, so "
                "its Hub columns are historical and must not be re-derived."
            )
        return f"{self.omeka_prefix}{omeka_suffix}"


#: The live panel, in report order.
#:
#: ``prefix`` is the exact provider model id with ``-`` and ``.`` folded to
#: ``_``. Verbose, deliberately: a column named for the precise id is the only
#: thing that makes an annotation reproducible years later, and the vendor-keyed
#: names it replaced could not even say which Gemini answered.
#:
#: ``run_config`` is the generation's sampling configuration, recovered from
#: commit 07fb007 (2026-01-27), which was the live code for the whole campaign.
#: It is NOT uniform across the panel, and reproducing a value needs the exact
#: settings below — note in particular Ministral's ``max_tokens=512`` cap, and
#: that no model ran with any reasoning/thinking parameter at all.
PANEL: Tuple[SentimentModel, ...] = (
    SentimentModel(
        prefix="gemini_3_flash_preview",
        label="Gemini 3 Flash",
        model_id="gemini-3-flash-preview",
        omeka_prefix="iwac:gemini",
        campaign="2026-01/2026-02",
        run_config="temperature=0.2; response_schema; no thinking_level",
    ),
    SentimentModel(
        prefix="gpt_5_mini",
        label="GPT-5 mini",
        model_id="gpt-5-mini",
        omeka_prefix="iwac:chatgpt",
        campaign="2026-01/2026-02",
        run_config="response_format schema only; no temperature; no reasoning_effort",
    ),
    SentimentModel(
        prefix="ministral_14b_2512",
        label="Ministral 3 14B",
        model_id="ministral-14b-2512",
        omeka_prefix="iwac:mistral",
        campaign="2026-01/2026-02",
        run_config="temperature=0.2; max_tokens=512; response_format schema",
    ),
)

#: Vendor-keyed column names used on the Hub before the 2026-07 rename. Passed
#: as ``columns_to_exclude`` so the merge drops them rather than preserving them
#: next to the renamed columns.
LEGACY_VENDOR_COLUMNS: Tuple[str, ...] = tuple(
    f"{vendor}_{suffix}"
    for vendor in ("gemini", "chatgpt", "mistral")
    for suffix, _ in DIMENSION_FIELDS
)


def active_models() -> Tuple[SentimentModel, ...]:
    """Panel members still backed by Omeka properties (the uploader reads these)."""
    return tuple(m for m in PANEL if not m.frozen)


def prefixes() -> Tuple[str, ...]:
    return tuple(m.prefix for m in PANEL)


def all_columns() -> Tuple[str, ...]:
    return tuple(col for m in PANEL for col in m.columns())


def all_justification_columns() -> Tuple[str, ...]:
    return tuple(col for m in PANEL for col in m.justification_columns())


__all__ = [
    "DIMENSION_FIELDS",
    "JUSTIFICATION_SUFFIXES",
    "SentimentModel",
    "PANEL",
    "LEGACY_VENDOR_COLUMNS",
    "active_models",
    "prefixes",
    "all_columns",
    "all_justification_columns",
]
