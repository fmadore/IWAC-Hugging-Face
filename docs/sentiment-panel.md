# Sentiment panel: generations and campaigns

The sentiment annotations on `articles` are produced upstream, in Omeka S, by
[iwac-ai-pipelines](https://github.com/fmadore/iwac-ai-pipelines); this
pipeline imports them and computes panel agreement and consensus
(`post-processing/sentiment_agreement.py`). The panel is defined once, in
`PANEL` in `iwac_common/sentiment_panel.py`. This page records the campaign
history behind the columns. Counts and dates are as recorded at the time of
each campaign; recount them from the dataset revision you analyse before
citing them.

## Column naming

Each model writes six columns under a prefix that names the exact model, never
the vendor slot: `{prefix}_centralite_islam_musulmans`, `{prefix}_polarite`,
`{prefix}_subjectivite_score` and the three matching `*_justification`
columns (for example `gpt_5_6_luna_polarite`). A newer model always gets a new
prefix, so no two generations of a model can collide in the same column.

Column order follows the panel: `PANEL` is ordered newest generation first,
and the uploader's `post_merge` hook sorts the sentiment block by it, so the
current panel precedes its history rather than trailing it.

## Generations

| Generation | Models (registry ids) | Campaign | Subjectivité | Status |
|---|---|---|---|---|
| 2 | `gpt-5.6-luna`, `mistral-small-2603`, `deepseek/deepseek-v4-flash-0731`, `google/gemma-4-31b-it`, `Qwen/Qwen3.8-27B`, `claude-haiku-5-5` | 2026-08; Haiku 2026-10 | label (`string`) | Live |
| 1 | `gemini-3-flash-preview`, `gpt-5-mini`, `ministral-14b-2512` | 2026-01/02 | integer 1–5 (nullable `int64`; `float64` in revisions published before the canonical types) | Frozen |

Generation 2, as recorded for its campaigns:

- The first four models were complete at 12,298 articles on centralité and
  polarité; subjectivité is where they diverge.
- Qwen3.8 joined on 2026-08-25 at 12,098 articles and stays there: 153
  articles were retired after four attempts each, so its shortfall is a finding
  about the model, not a run to repair. It was served from a self-hosted vLLM
  instance (see the README's acknowledgements); `qwen3_8_27b` names that route,
  and the same weights reached through another provider must never join the
  panel beside it.
- Claude Haiku 5.5 joined on 2026-10-22 **without a pilot**, at the curator's
  decision, so no agreement or self-consistency figure exists for it yet. Its
  2026-10-23 pass reached 16,617 articles, well beyond the rest of the panel;
  rows only it answered carry a single vote and therefore no consensus.

Generation 1 is frozen. Its Omeka properties were deleted in 2026-08; the 18
columns remain on the Hub as historical data. They survive through the merge,
not through a copy: the uploader stops emitting a frozen model's columns and
`hub_merge` preserves every Hub column the fresh frame does not carry.
Emitting those columns empty would overwrite the values without changing the
row count, so no guard would fire — which is why `tests/test_sentiment_panel.py`
covers that pair directly.

## The generation boundary

A generation boundary is a change of instrument, not a version bump.
Generation 2 ran a rewritten prompt (fingerprint `d14ace9ac192`) and asked for
subjectivité as a label rather than a number, so `{prefix}_subjectivite_score`
is a string in generation 2 and an integer in generation 1. Comparisons across
the boundary confound the models with the prompt rewrite.
`sentiment_agreement.py` therefore takes `--generation` and defaults to the
newest; `--generation all` is exploratory reporting only and cannot publish a
pooled consensus. `SUBJECTIVITE_ORDER` maps a label to its 1–5 rank when one
scale is needed for both.

## Consensus columns

`iwac-process sentiment-agreement --generation <n> --push` writes
generation-specific columns: `consensus_g<n>_polarite`,
`consensus_g<n>_centralite`, `consensus_g<n>_subjectivite_score`,
`sentiment_g<n>_disagreement` and `consensus_g<n>_instrument_id`. The
instrument id fingerprints each member's `INSTRUMENT_FIELDS` (model id,
generation, sampling configuration, prompt fingerprint) and
`CONSENSUS_RULE_VERSION`; column prefixes, labels, campaigns and Omeka
properties are left out, so freezing a generation keeps its id.

- A label consensus is the strict majority (more than half of the models that
  voted). It is `""` when at least two models rated the row and none has a
  strict majority, and null when fewer than two valid ratings exist.
- The subjectivité consensus is the median of at least two scores; with an
  even number of raters it averages the two middle ranks, so `.5` values
  appear by design.
- The disagreement column pipe-joins the disputed dimensions (no majority for
  polarité or centralité; a subjectivité range of 2 or more). It is `""` when
  no assessable dimension is disputed and null when no dimension has two valid
  ratings.

The report gives Krippendorff's alpha with the **ordinal** coefficient as the
primary reading (interval and nominal secondary), a nominal alpha for polarité
that includes "Non applicable", pairwise Cohen's kappa and exact-agreement
rates. The dispute share is computed over rows with at least two raters. The
subjectivité dispute rule fires more often the more raters a row has, so
dispute rates are not comparable across generations of different panel size;
the report records the panel size beside them. Agreement among related models
is not a human-validated accuracy estimate.
