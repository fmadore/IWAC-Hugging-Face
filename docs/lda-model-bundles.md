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

## Source-document units

Long subsets train on chunks (`--chunk-words`), but evidence is counted per
**source document**, the `--holdout-group-column` unit (default `o:id`):

- the dictionary's `--no-below` and `--no-above` count source documents, not
  chunks, so one long book cannot push a term over the floor by repetition;
- a compound learned by phrase detection is applied only if it occurs in at
  least `--phrase-min-docs` source documents (`DEFAULT_PHRASE_MIN_DOCS` in
  `constants.py`, overridable per preset).

Fits that record `phrase_min_docs` carry frozen preprocessing version 2 with
`document_frequency_unit: source_document`. Bundles fitted before keep their
recorded behaviour at prediction; the rule applies from the next `--mode fit`.

## Choosing k: the sweep is a diagnostic

k is pinned per subset in `CONFIG_PRESETS` (`num_topics`, with
`language_overrides` for the English references model). The sweep reports and
exits; it does not fit, write a bundle or push:

```bash
iwac-process lda --config references --mode fit --sweep-only --stability-seeds 3 --holdout 0.1
```

`--optimize-topics` behaves the same as `--sweep-only` unless
`--select-k-from-sweep` is also given. That flag is the explicit opt-in to fit
the production model at the sweep's highest mean C_v, and it is refused when k
is pinned by a preset or `--num-topics`. The report, written under
`<model root>/sweeps/<timestamp>/sweep_report.json`, gives per k the mean C_v
with its seed standard deviation, the multi-seed stability (mean best-match
Jaccard of top words) and, with `--holdout`, the held-out bound.

## Held-out evaluation

`--holdout 0.1` on a sweep splits source documents **before** fitting
phrases, vocabulary or model parameters. Holdout texts are transformed by the
training-only phrase models and dictionary. Chunking follows this split, so
one source cannot leak through different chunks. Exact duplicate texts remain
together; `--holdout-group-column <column>` additionally keeps known reprint or
source groups together. Its default is `o:id`. The fraction applies to grouped
units; actual held-out document counts and IDs are recorded in the sweep
report (and in the bundle when `--select-k-from-sweep` fits a model).

The held-out figure is gensim's `LdaModel.log_perplexity` per-word bound on the
held-out documents. Theta is inferred from the same held-out words it scores,
so it is an optimistic, report-only number, not a document-completion
estimate. A production fit trained on the full eligible corpus is not a
held-out test. A pinned-topic fit with no sweep does not perform held-out
evaluation, and records that fact.

## Labels

Topic labels keep relevance order (LDAvis-style, λ = 0.6, unless
`--no-relevance-labels`); compound length confers no priority. Subsumption only
removes redundancy: a unigram contained in an already selected compound is
skipped, and a compound arriving after one of its unigrams takes that
unigram's slot, at the rank the unigram earned.

## Minimum evidence at prediction

A document with fewer in-vocabulary tokens than `--min-predict-tokens` (default:
the model's training floor, `--min-train-tokens`) receives null topic columns
but keeps `lda_model_name`, so a reader can tell "assessed, too little
evidence" from "not processed". The theta export records `n_vocab_tokens` and
`below_min_evidence` for every document.

## Reviewed exclusions

`TOPIC_MODEL_EXCLUSIONS` in `constants.py` lists, per subset and `o:id`, rows
that stay in the dataset (public where they are public) but never enter topic
modelling, each with its reviewed reason. A fit drops them before phrase
detection and dictionary building and records the applied list as
`topic_model_exclusions` in `training_parameters.json`. Prediction applies the
registry from current code whatever bundle it loads: an excluded row gets null
topic columns but keeps `lda_model_name`, like a row below the evidence floor,
and carries no theta (the export metadata lists `excluded_document_ids`).
`iwac-analyze topics` leaves the same rows out and counts them
(`docs_excluded_by_registry`).

The list is deliberately not part of the frozen preprocessing: it selects rows
rather than transforming text, so existing bundles keep their identity and
their theta exports stay reusable, and a bundle fitted before an entry was added
still nulls that row at its next prediction.

Eight `publications` rows are excluded: the Islam Info issues 23757, 24190,
24233, 24234, 24635, 25286, 25287 and 25289. Their text is mostly tabular lists
of pilgrims' names, which formed a spurious name-list topic (topic 18 of the
periodical bundle fitted before the exclusion). They must stay excluded unless
the lists are removed from their text.

Coherence is a diagnostic, not a universal quality grade. Inspect seed
stability, alternative preprocessing, representative documents and borderline
assignments before interpreting a topic historically. These checks do not
replace expert validation or an account of archive selection.

## Prediction exports and prevalence

Each prediction pass writes a separate directory under
`<model root>/predictions/<timestamp>/`, with `topics.csv` (the label, top words
and document counts per `lda_topic_id`), full distributions in
`doc_topics.parquet`, input/model metadata
and an archived run manifest. Exported distributions carry per-document text
hashes. They are never merged with a prior model's distributions.

To reuse a particular export in prevalence analysis, pass:

```bash
iwac-analyze topics --model-path lda_model \
  --theta-path lda_model/predictions/<run>/doc_topics.parquet
```

Without `--model-path`, prevalence loads the `--config` subset's preset model
(`--language Anglais` for the English references model), as the LDA stage
resolves it. A bundle whose recorded training subset or language differs from
the request is refused; `--allow-cross-corpus-model` applies it deliberately
and the comparison is recorded in the summary and run manifest.

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

Only valid single-year, month or day dates enter annual prevalence, under the
rule every analysis shares (`analyses/_research.py:point_years`). Date ranges,
invalid dates and missing dates are counted as exclusions instead of being
assigned to a guessed start year. Cluster bootstrap prefers authority outlet
IDs; when only labels exist, it scopes them by country to separate homonyms.
