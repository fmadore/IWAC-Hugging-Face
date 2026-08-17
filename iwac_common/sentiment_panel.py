"""Canonical definition of the AI sentiment annotator panel.

One place to rotate models. Before this module the panel was restated in the
uploader, the publisher, the agreement script and the tests, which is how a
vendor slot silently changed model without anything recording it.

Three rules encoded here:

1. **Column prefixes name the model, not the vendor.** Generation-1 Omeka
   properties are vendor-keyed (``iwac:gemini*``, ``iwac:chatgpt*``,
   ``iwac:mistral*``) and carry no ``iwac:*Model`` value annotation, so the
   property alone cannot say what produced a value. The 2026-01/02 corpus ran
   on gemini-3-flash-preview, gpt-5-mini and ministral-14b-2512 (campaign
   window verified from ``o:modified``), which is what the prefixes below
   record. Generation 2 fixes this at the source: its properties are keyed by
   model (``iwac:gpt56Luna*``), which is the *only* place the model is
   recorded — Omeka does not index value annotations, so provenance cannot
   live there.

2. **Re-running sentiment with a newer model adds a new entry — never reuses
   one.** A model whose Omeka properties have been retired keeps its columns on
   the Hub with ``omeka_prefix=None``: the uploader then stops producing them,
   and ``hub_merge`` preserves them as frozen historical data.

3. **A generation boundary is a change of instrument, not a version bump.**
   Generation 2 runs a rewritten prompt (fingerprint ``d14ace9ac192``) and asks
   for subjectivité as a *label* where generation 1 asked for the integer 1-5,
   so its ``*_subjectivite_score`` column is a string where generation 1's is a
   float (see :attr:`SentimentModel.subjectivite_is_label`). Any comparison
   across the boundary confounds model change with prompt change; convert one
   side with :data:`SUBJECTIVITE_ORDER` and say so.

Not in the panel, deliberately: ``iwac:deepseekV4Flash*`` (the retired April
preview — 11,482 real annotations on Omeka, superseded by 0731, never to be
repointed), ``iwac:gemini35FlashLite*`` (held the generation-2 Google slot from
2026-07-31 until Gemma took it on 2026-08-14 and wrote nothing in between —
verified at 0 items on the day of the swap, so nothing was mixed), and
``iwac:gemini36Flash*`` / ``iwac:qwen35A3b*`` / ``iwac:qwen35A10b*`` (reserved
or dropped, zero values). Twelve property prefixes in Omeka vocabulary 10
describe seven columns' worth of annotators here — and the five empty ones stay
installed, since on this instance the vocabulary update adds what the TTL
declares without deleting what it omits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

#: The six per-model fields, as ``(column suffix, Omeka property suffix)``.
#: Every one of centralité, polarité and subjectivité is stored as a
#: ``resource:item`` link on Omeka (never a literal), so readers resolve them
#: through ``display_title`` rather than ``@value``.
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

#: The subjectivité suffix, named once: it is ``_score`` on both generations for
#: continuity with the live Omeka property (``SubjectiviteScore``) and the
#: already-published generation-1 column, even though generation 2 stores a
#: label in it. Renaming a published column is worse than a slightly wrong name.
SUBJECTIVITE_SUFFIX = "subjectivite_score"

#: Subjectivité labels, weakest→strongest, and their ordinal rank.
#:
#: Doubles as the bridge across the generation boundary: a generation-1 column
#: already holds this rank as a float (the integer the model returned), and a
#: generation-2 column holds the label, so mapping the label through this dict
#: puts both on one scale. ``Non applicable`` is deliberately absent — it is not
#: a point on the scale. Mirrors ``SUBJECTIVITE_ORDER`` in the annotation
#: pipeline's ``sentiment_core.py``; keep the two identical.
SUBJECTIVITE_ORDER: Dict[str, int] = {
    "Très objectif": 1,
    "Plutôt objectif": 2,
    "Mixte": 3,
    "Plutôt subjectif": 4,
    "Très subjectif": 5,
}


@dataclass(frozen=True)
class SentimentModel:
    """One annotator in the panel."""

    prefix: str
    """HF column prefix, e.g. ``gpt_5_6_luna`` -> ``gpt_5_6_luna_polarite``."""

    label: str
    """Human-readable model name for reports and the dataset card."""

    model_id: str
    """Exact provider model id that produced the values."""

    omeka_prefix: Optional[str]
    """Omeka property prefix (``iwac:gemini``), or ``None`` once the properties
    have been retired from Omeka and the Hub columns are frozen history."""

    campaign: str
    """When this model annotated the corpus, for the dataset card."""

    generation: int = 1
    """Annotation generation.

    1 = vendor-keyed Omeka properties, one prompt (``84bf993``), subjectivité
    requested as an integer 1-5. 2 = model-keyed properties, the rewritten
    prompt, subjectivité requested as a label. The generation is what decides
    the ``*_subjectivite_score`` column's *type*, so it is data here rather
    than prose in a docstring.
    """

    run_config: str = ""
    """Sampling configuration actually used for this model's campaign.

    Reproducibility, not decoration: neither generation shares one config
    (Ministral capped output at 512 tokens and GPT-5 mini set no temperature at
    all; in generation 2 only Luna gets the middle reasoning level it asked
    for), so a column cannot be re-derived without it.
    """

    prompt_fingerprint: str = ""
    """``sentiment_core.prompt_fingerprint`` of the prompt that produced the
    values. Empty for generation 1, which predates fingerprinting: its prompt is
    the one at commit ``84bf993``. Two columns with different fingerprints are
    not two readings of the same question."""

    @property
    def frozen(self) -> bool:
        """True once Omeka no longer holds this model's properties."""
        return self.omeka_prefix is None

    @property
    def subjectivite_is_label(self) -> bool:
        """True when this model was asked for a subjectivité *label*, so its
        ``*_subjectivite_score`` column is a string and must not be averaged,
        cast to Int64, or compared to a generation-1 float without conversion
        through :data:`SUBJECTIVITE_ORDER`."""
        return self.generation >= 2

    @property
    def subjectivite_column(self) -> str:
        return self.column(SUBJECTIVITE_SUFFIX)

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


#: The panel, **newest generation first**.
#:
#: Order is not cosmetic: it is the order the sentiment columns take on the Hub
#: (``_sentiment_columns_last`` sorts by this tuple) and the order every report
#: lists models in. The live panel leads because it is what a reader should reach
#: for; the frozen generation trails it as history.
#:
#: ``prefix`` is the exact provider model id with ``-`` and ``.`` folded to
#: ``_``. Verbose, deliberately: a column named for the precise id is the only
#: thing that makes an annotation reproducible years later, and the vendor-keyed
#: names it replaced could not even say which Gemini answered.
#:
#: **Generation 2** is model-keyed, ran the rewritten prompt ``d14ace9ac192``,
#: and is the panel any new analysis should use. Provenance below is measured
#: from the annotation pipeline's own cache (``cache/sentiment_v2.jsonl``), not
#: inferred. A member is added once it has annotated something and not before:
#: pre-declaring one makes the uploader emit six empty columns that the publisher
#: will happily ship. That is why ``gemini-3.5-flash-lite`` never appeared here
#: in the two weeks it held the Google slot, and why ``gemma-4-31b-it``, which
#: replaced it, did on the day its first corpus pass began.
#:
#: **Generation 1 is frozen** (``omeka_prefix=None``). Its Omeka properties are
#: being deleted from the archive; the six columns per model stay on the Hub as
#: historical data because the uploader no longer emits them and ``hub_merge``
#: preserves what it does not overwrite. Verified before freezing: all three
#: models are populated on 12,286 of 12,356 Hub rows, matching Omeka exactly, so
#: nothing was still owed. Its ``run_config`` was recovered from commit 07fb007
#: (2026-01-27), the live code for that whole campaign.
PANEL: Tuple[SentimentModel, ...] = (
    SentimentModel(
        prefix="gpt_5_6_luna",
        label="GPT-5.6 Luna",
        model_id="gpt-5.6-luna",
        omeka_prefix="iwac:gpt56Luna",
        campaign="2026-08-03",
        generation=2,
        # The only member that got the middle reasoning level the panel asks
        # for; the other two have no medium and round up. 2.7 h for the corpus.
        run_config="reasoning_effort=medium; response schema; no temperature",
        prompt_fingerprint="d14ace9ac192",
    ),
    SentimentModel(
        prefix="mistral_small_2603",
        label="Mistral Small 4",
        model_id="mistral-small-2603",
        omeka_prefix="iwac:mistralSmall2603",
        campaign="2026-08-05",
        generation=2,
        run_config=(
            "temperature=0.3; reasoning_effort=high (API accepts only none|high; "
            "medium rounded up); response schema"
        ),
        prompt_fingerprint="d14ace9ac192",
    ),
    SentimentModel(
        prefix="deepseek_v4_flash_0731",
        label="DeepSeek V4 Flash 0731",
        model_id="deepseek/deepseek-v4-flash-0731",
        omeka_prefix="iwac:deepseekV4Flash0731",
        campaign="2026-08-03/2026-08-05",
        generation=2,
        # Temperature is vendor-owned and V4 runs at 1.0, so re-annotating the
        # same article is a fresh reading rather than a correction: a 1,485-item
        # repair pass returned a different centralité on 19 of them.
        run_config=(
            "temperature=1.0; reasoning_effort=high (API accepts only low|high|max; "
            "medium rounded up); response schema"
        ),
        prompt_fingerprint="d14ace9ac192",
    ),
    SentimentModel(
        prefix="gemma_4_31b_it",
        label="Gemma 4 31B",
        model_id="google/gemma-4-31b-it",
        omeka_prefix="iwac:gemma431bIt",
        # Corpus complete on 2026-08-17: 12,298 on centralité and polarité,
        # matching the other three exactly, and 12,055 on subjectivité.
        campaign="2026-08-14/2026-08-15",
        generation=2,
        # Routed through OpenRouter under data_collection=deny, never the Gemini
        # API: Gemma is free of charge there with no paid tier, and Google states
        # free-tier content is used to improve its products — unacceptable for
        # whole archival articles. (That route also caps this model at 16k input
        # tokens/minute, so it buys no speed either.)
        #
        # Read the depth as requested, not as measured: OpenRouter fans the call
        # across third-party backends serving the same weights, and they disagree
        # on what an effort level means — `medium` and `high` are
        # indistinguishable through this route.
        run_config=(
            "reasoning_effort=high (model has only MINIMAL|HIGH; medium rounded "
            "up); response schema; no temperature; OpenRouter, data_collection=deny"
        ),
        prompt_fingerprint="d14ace9ac192",
    ),
    SentimentModel(
        prefix="gemini_3_flash_preview",
        label="Gemini 3 Flash",
        model_id="gemini-3-flash-preview",
        omeka_prefix=None,  # was iwac:gemini — retired 2026-08
        campaign="2026-01/2026-02",
        generation=1,
        run_config="temperature=0.2; response_schema; no thinking_level",
    ),
    SentimentModel(
        prefix="gpt_5_mini",
        label="GPT-5 mini",
        model_id="gpt-5-mini",
        omeka_prefix=None,  # was iwac:chatgpt — retired 2026-08
        campaign="2026-01/2026-02",
        generation=1,
        run_config="response_format schema only; no temperature; no reasoning_effort",
    ),
    SentimentModel(
        prefix="ministral_14b_2512",
        label="Ministral 3 14B",
        model_id="ministral-14b-2512",
        omeka_prefix=None,  # was iwac:mistral — retired 2026-08
        campaign="2026-01/2026-02",
        generation=1,
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


def frozen_models() -> Tuple[SentimentModel, ...]:
    """Members whose Omeka properties are gone; their Hub columns are history."""
    return tuple(m for m in PANEL if m.frozen)


def generation(gen: int) -> Tuple[SentimentModel, ...]:
    """Members of one annotation generation.

    The unit any agreement or consensus figure should be computed over: models
    from different generations answered a different prompt on a different scale,
    so a κ across the boundary measures the rewrite as much as the models.
    """
    return tuple(m for m in PANEL if m.generation == gen)


def latest_generation() -> Tuple[SentimentModel, ...]:
    """The newest generation present in the panel — the head of ``PANEL``."""
    return generation(max(m.generation for m in PANEL))


def prefixes() -> Tuple[str, ...]:
    return tuple(m.prefix for m in PANEL)


def all_columns() -> Tuple[str, ...]:
    return tuple(col for m in PANEL for col in m.columns())


def all_justification_columns() -> Tuple[str, ...]:
    return tuple(col for m in PANEL for col in m.justification_columns())


def numeric_subjectivite_columns() -> Tuple[str, ...]:
    """Subjectivité columns holding the 1-5 integer, i.e. the ones the uploader
    may cast to ``Int64``. A generation-2 column holds a label and casting it
    would blank the whole column."""
    return tuple(m.subjectivite_column for m in PANEL if not m.subjectivite_is_label)


def label_subjectivite_columns() -> Tuple[str, ...]:
    """Subjectivité columns holding a label string."""
    return tuple(m.subjectivite_column for m in PANEL if m.subjectivite_is_label)


__all__ = [
    "DIMENSION_FIELDS",
    "JUSTIFICATION_SUFFIXES",
    "SUBJECTIVITE_SUFFIX",
    "SUBJECTIVITE_ORDER",
    "SentimentModel",
    "PANEL",
    "LEGACY_VENDOR_COLUMNS",
    "active_models",
    "frozen_models",
    "generation",
    "latest_generation",
    "prefixes",
    "all_columns",
    "all_justification_columns",
    "numeric_subjectivite_columns",
    "label_subjectivite_columns",
]
