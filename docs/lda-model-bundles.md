# LDA model identity and evaluation

New fits write immutable model bundles under `--model-path/bundles/<sha256>/`.
`current.json` selects the latest bundle; previous fits remain available. Each
bundle's manifest covers the model, dictionary, phrase models, frozen
preprocessing, labels and training parameters. Readers verify all artifact
hashes before loading. `lda_model_name` now carries `lda-sha256:<digest>`.
Always join on **both `lda_model_name` and `lda_topic_id`**: fixing the number
of topics does not preserve topic meanings after a refit.

Prediction restores the saved stopwords, post-phrase exclusions, custom
collocations, language, chunking and labels. Incompatible explicit language,
chunking, stopword or label overrides are rejected. Changing these settings
requires a new fit. Missing language is excluded unless
`--include-unknown-language` is supplied deliberately.

Existing model directories remain usable through an explicit
`--allow-legacy-preprocessing` option. Their saved stopwords and available
settings are restored, but missing collocation settings must be reconstructed
from current code. Such artifacts receive `lda-legacy-sha256:<digest>`; this
identifies the available bytes plus effective reconstructed preprocessing,
not a fully recovered historical instrument. Changed reconstructed settings
produce a new legacy identity.
Refitting is the route to a complete frozen bundle.

## Held-out evaluation

`--optimize-topics --holdout 0.1` splits source documents **before** fitting
phrases, vocabulary or model parameters. Holdout texts are transformed by the
training-only phrase models and dictionary. Chunking follows this split, so
one source cannot leak through different chunks. Exact duplicate texts remain
together; `--holdout-group-column <column>` additionally keeps known reprint or
source groups together. Its default is `o:id`. The fraction applies to grouped
units; actual held-out document counts and IDs are recorded in the bundle.

The sweep's held-out scores describe these evaluation models. The chosen
specification is subsequently refitted on the full eligible corpus for use in
discovery. That production fit is not a held-out test. A pinned-topic fit with
no sweep does not perform held-out evaluation, and records that fact.

Coherence is a diagnostic, not a universal quality grade. Inspect seed
stability, alternative preprocessing, representative documents and borderline
assignments before interpreting a topic historically. These checks do not
replace expert validation or an account of archive selection.

## Prediction exports and prevalence

Each prediction pass writes a separate directory under `predictions/`, with
`topics.csv`, full distributions in `doc_topics.parquet`, input/model metadata
and an archived run manifest. Exported distributions carry per-document text
hashes. They are never merged with a prior model's distributions.

To reuse a particular export in prevalence analysis, pass:

```bash
python analyses/topic_prevalence.py --model-path lda_model \
  --theta-path lda_model/predictions/<run>/doc_topics.parquet
```

Reuse checks the model identity, dataset repository/configuration, file hash,
document IDs, text hashes and distribution values. A changed dataset revision
is acceptable only insofar as those item inputs still match; newly eligible
items are inferred with the frozen model. A stale existing item is an error.
The metadata also verifies the effective preprocessing fingerprint. Inference
uses a document-specific initialization derived from its token sequence, so
skipping cached rows does not change estimates for uncached rows.

Prevalence defaults to descriptive slopes. `--trend-test mann-kendall` enables
an explicitly conditional test requiring independent annual observations;
neither its p-values nor BH adjustment correct temporal dependence or changing
archive coverage. Within-year bands resample documents by default. Use
`--bootstrap-unit newspaper` to resample whole outlet clusters where outlet
metadata is complete; fewer than two outlets yield undefined bands. Both
choices condition on the fitted model and exclude archive-selection, OCR and
model-estimation uncertainty. Frequency tables describe eligible archived
documents, not all press coverage or public opinion.

Only valid single-year, month or day dates enter annual prevalence. Date ranges,
invalid dates and missing dates are counted as exclusions instead of being
assigned to a guessed start year. Cluster bootstrap prefers authority outlet
IDs; when only labels exist, it scopes them by country to separate homonyms.
