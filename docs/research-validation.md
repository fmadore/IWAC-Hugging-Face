# Research validation and corpus diagnostics

The analysis tools describe the digitized IWAC corpus. They cannot establish
representativeness of the historical press, public opinion, or Muslim life.
Outlet availability, acquisition, preservation, OCR, date uncertainty and model
coverage are part of the research design and should accompany substantive claims.
These commands create local reports only: they do not upload, deduplicate, label,
retrain or alter the source dataset.

Install the project and analysis dependencies as described in the root README.
The commands below use a verified local mirror by default. An explicit
`--input snapshot.parquet` or `--input snapshot.csv` works without network access.
It records the file's SHA-256; an absent revision is recorded as `unknown`, never
invented. `--revision` supplied with `--input` is a user declaration, not a verified
Hub association. Without `--input`, `--revision` pins the Hub or verified mirror
revision. `--source hub` opts into a read-only remote load using existing credentials.

Every output CSV carries the source repository and revision. A run manifest records
arguments, input and output hashes, code state and package versions; complete output
files are archived by the shared run-manifest helper. Keep the matching input snapshot
or immutable source revision as well. Publication-quality runs should use a clean
checkout and a locked environment. Files derived from private sources require the
same access controls as the original working data.

## 1. Audit what the analysis includes

```bash
iwac-analyze coverage --input snapshot.parquet \
  --year-min 1990 --year-max 2025 --languages Français \
  --require lemma_nostop lda_topic_id --min-outlet-year 5
```

`corpus_coverage_cells.csv` reports document and enrichment/annotation counts by
country, outlet, year, language, text-access status, date precision, and
country–outlet–year. Fractions always state a document denominator; annotation
columns remain generation-specific. Multi-valued metadata strings are counted as
joint categories, not exploded into duplicate documents. Missing language is its
own category and is never silently assumed to be French.

`corpus_coverage_inclusion.csv` links each source ID to its inclusion decision and
all applicable exclusion reasons. Ambiguous dates and date ranges remain visible
in the audit but are excluded from the point-year cohort; this choice differs from
assigning a range's start year. Text availability and public text access are separate
variables. Empty or absent enrichment values are unavailable, including embeddings;
a numeric topic ID of zero is a valid populated value.

The stable-outlet flag identifies outlets with at least `--min-outlet-year`
eligible documents **in every year of the requested window**, including years with
zero records. Outlet IDs are preferred to names and scoped by country. Select a
meaningful year window: the broad default 1900–2100 intentionally yields no stable
outlets in most research corpora. Use the exported source IDs to rerun an analysis on
that cohort, compare results with the complete cohort, and discuss the difference.
The flag supports a sensitivity analysis; it does not reweight or repair the archive.

## 2. Build an independent annotation exercise

```bash
iwac-analyze annotation-review --input snapshot.parquet \
  --per-stratum 3 --challenge-size 30 --seed 42
```

The default strata are country, decade, newspaper and language. A stable hash of
seed and source ID selects up to three records per stratum without replacement.
Shuffling input rows does not alter selection. The sheet records each stratum's
population, sample size, inclusion probability and design weight. Very many small
strata can produce a large sample; adjust `--strata country decade language` or
`--per-stratum` to fit a predeclared sampling plan. This tool's sampling population
is the entire supplied snapshot, including unannotated rows; filter the input in a
reproducible way if the intended target population is narrower.

A disjoint challenge set oversamples within-generation model disagreements. Supply
`--difficult-ids expert_flags.txt` to prioritize sources identified as difficult by a
historian; unknown IDs fail rather than silently disappearing. Selected challenge
cases have **no probability weight**. A difficult source already selected in the
probability sample stays there and is not duplicated. `--generation` selects one
canonical annotation generation; generations and prompts are never pooled.

Reviewers receive blank fields for two independent annotators and adjudication,
plus evaluation target, quoted stance, negative event, uncertainty and notes.
Machine predictions and selection reasons are omitted from the annotation sheet;
a separate selection audit is for study management, not initial annotation.
Source links allow close reading in the archive. Excerpts are omitted by default.
`--excerpt-chars 1500` explicitly includes text from `--text-column OCR`; it may
export restricted text and should only be used for an authorized review workspace.
Short excerpts can omit attribution or a reversal of stance, so read the whole source.

Before annotation, freeze the codebook, target definitions and adjudication procedure.
Distinguish a negative event from negative evaluation of Muslims, and quoted voices
from an article's own stance. Pilot on a separate set, revise the codebook, then
freeze it before evaluating the held-out sample. Preserve each independent label,
adjudication, reviewer identity and uncertainty. Compute per-class errors and
agreement with uncertainty appropriate to the design; for corpus-wide estimates,
apply the recorded probability weights. Report the challenge set separately.

**Generating a sheet supplies no gold labels and demonstrates no model accuracy.**
Expert reading and adjudication remain empirical research work. Model agreement can
identify review priorities but cannot substitute for this validation.

## 3. Review topics against their sources

```bash
iwac-analyze topic-review --input snapshot.parquet \
  --representatives 5 --borderline 5
```

For each `(model_id, topic_id)`, the sheet includes documents with the largest
stored dominant-topic probabilities, then disjoint borderline documents with the
smallest dominant-minus-runner-up margin. Borderline selection requires at least
two valid, consistent saved top-k probabilities. It never renormalizes a truncated
distribution or substitutes low dominant probability for a missing runner-up.
This is purposive evidence for interpretation, not a random validation sample.

New immutable `lda-sha256:` identities and explicitly fingerprinted historical
`lda-legacy-sha256:` bundles are accepted. Mutable old directory names are excluded
and listed in an exclusion ledger. `--allow-legacy-model-names` opts into reviewing
those historical estimates, clearly marked identity unverified. This flag does not
make two similarly named models equivalent or reconstruct the original bundle.

Record an approved label, description, fit judgment, supporting source IDs, reviewer
and notes. Review both typical and borderline texts and compare alternative models
through their words and linked texts. Keep topic identity scoped to its exact bundle:
the same topic count and random seed do not preserve the meaning of topic 17 after
retraining. Do not treat coherence alone as a historically validated quality score.

## 4. Screen and validate reprints

```bash
iwac-analyze reprints --input snapshot.parquet \
  --min-cosine 0.93 --min-jaccard 0.45 --shingle-size 3 --top-k 20
```

A pair must pass both normalized embedding cosine similarity and lexical Jaccard
similarity of case-folded word trigrams. The supplied thresholds are starting points
for calibration, not validated IWAC operating points. Semantic similarity without
lexical overlap suggests shared subject matter, not necessarily copying. OCR errors,
abridgments, translations and boilerplate can respectively hide or create matches.
The default text field is OCR; `--text-column` is explicit because lemmatized text
changes the meaning of lexical evidence.

The command requires a single stored embedding configuration fingerprint. Mixed
known configurations or dimensions fail. Legacy rows lacking fingerprints require
`--allow-unverified-embeddings` and are counted as unverified in the summary. Equal
vector dimensions alone do not prove model compatibility. The command does not
claim that the legacy model identity or input provenance has been recovered.

Only the highest `--top-k` semantic neighbors per item receive lexical checks.
The union of directed neighbor lists forms unique candidate pairs. Results are
deterministic across input order, including ties. Blockwise matrix multiplication
limits memory to approximately `block-size × eligible-documents` similarities,
but exact search remains quadratic in corpus size. The summary counts truncated
neighbor lists and lexical comparisons; increasing top-k can improve recall while
increasing lexical work. `--max-days 30` optionally requires exact day dates and
restricts pairs to a 30-day gap; partial dates are explicitly excluded instead of
being interpreted as January 1. No date-window restriction applies by default.

All output pairs are marked `candidate_unverified`, with blank `human_reprint`,
reviewer and notes fields. No item is removed or automatically assigned to a reprint
cluster. Review the original sources, including attribution and editorial changes,
before deciding whether they are reprints, adaptations, common dispatches or merely
related texts. A connected component of candidate pairs is not a verified reprint
family because similarity is not transitive.

For calibration, create an independently adjudicated CSV:

```csv
id_a,id_b,human_reprint,reviewer,notes
101,102,true,reviewer-id,Same dispatch with a changed headline
101,103,false,reviewer-id,Shared subject but independently written
```

```bash
iwac-analyze reprints --input snapshot.parquet --evaluated-pairs adjudicated.csv
```

Labels may be `true/false`, `yes/no`, or `1/0`; blank labels remain unknown. Duplicate
pairs, invalid labels and IDs outside the snapshot are rejected. The summary reports
confusion counts and precision/recall **only on the supplied adjudicated pairs**.
Include independently sampled non-candidate pairs to examine missed reprints; a
review of candidates alone cannot establish corpus recall. Use separate calibration
and evaluation sets so thresholds are not assessed on the same cases used to choose
them. Report any sensitivity of historical trends to retaining one version versus
all members of expert-validated reprint families.
