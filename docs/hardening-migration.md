# Pipeline and methodology hardening: migration guide

This implementation changes code and tests. It does not retroactively correct
existing Hub values, regenerate topic models, run paid inference or validate
historical claims. Apply the following sequence first to scratch repositories,
then retain the exact validated code/data/model/environment revisions used in
research. Do not mix outputs from the previous and new methods without labeling
the difference.

## 1. Install and set destinations

Use Python 3.12 or 3.13 and a fresh environment:

```bash
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip check
```

Install the required spaCy model packages for the languages you process. They
are separate trained artifacts, not part of the Python dependency lock; preserve
their distributions and record the versions captured in enrichment provenance.
For a wheel install, use the package's `nlp` and `analysis` extras (or `all`) as
needed, and set `IWAC_WORK_DIR` to a writable directory.

Set both repository overrides in the process environment or the selected
`IWAC_ENV_FILE`. Settings load before defaults are constructed. Full writes
require a verified private target even if `--repo` explicitly names another
destination. The public destination must differ from the full source. Test
against scratch repos without minting a DOI or squashing history.

## 2. Refresh source permissions and invalidate stale outputs

Run an authenticated dry run followed by ingestion for **each of the seven
subsets**. Both Omeka credentials are required to avoid a partial anonymous
mirror. Use an account with complete-source access.

```bash
iwac-upload articles --dry-run
iwac-upload articles
```

Repeat for `publications`, `documents`, `references`, `index`, `images` and
`audiovisual`. Do not add `--initialize` to existing subsets merely to bypass
an unreadable baseline. Review intentional shrinkage before using its override.

Every row now carries explicit parent visibility and restricted flat-column
names. Legacy mirrors without this evidence cannot publish. Private parents
are omitted; restricted properties and reconstructive text are masked. Existing
approved OCR derivatives retain their explicit publication policy. Annotation
privacy also propagates to dependent consensus. Private provenance columns are
excluded from public output.

Source changes clear dependent values by default. Pending stale queues from
previous runs are replayed even when Omeka and Hub text already match. Resolve
queues before expensive enrichment. `--preserve-derived` defers this cleanup
explicitly and retains the queue until a verified invalidating write. See
[the detailed recovery contract](ingest-contract.md).

References absent from the complete source listing now drop by default.
`--stale-rows keep` retains their full historical private record. Failed mapper
or media refresh overrides preserve values privately until successful refetch.

## 3. Regenerate affected enrichments

Text and image embeddings and lemmas persist input/configuration fingerprints.
Existing nonempty values without compatible provenance are recomputed once,
even in incremental mode. Plan inference cost accordingly. Source and method
changes then permit selective recomputation. Empty inputs clear obsolete values.

Embedding 2 requests now use distinct content objects, instruction-prefixed
text and bounded batches; its unsupported task-type parameter is omitted.
Conservative byte-bounded chunking changes long-document aggregation. Old and
new vectors should not be treated as a single calibrated similarity space.
Image identity covers downloaded and re-encoded bytes, so a URL whose content
changed is detectable. Incremental image validation still downloads the image.
Readable configuration JSON accompanies each fingerprint privately. Related-item
and reprint tools reject mixed known configurations; legacy/public vectors that
lack provenance require an explicit `--allow-unverified-embeddings` exploratory
option, which does not establish compatibility.

Lemmatization skips missing/ambiguous language and multilingual records by
default; `--multilingual primary` explicitly applies the first metadata language.
French readability requires monolingual French, not simply a first-listed French
label. Invalid/inapplicable inputs remain null. MATTR preserves Unicode combining
marks and applies French clitic handling only to French inputs. These policy
changes require rerunning lexical/lemma stages before comparing their outputs.

```bash
iwac-process embeddings --help
iwac-process image-embeddings --help
iwac-process lemmas --help
iwac-process lexical --help
iwac-process related --help
```

Run the relevant stages using their documented subset/update options. New
lemma outputs can invalidate topic assignments; changed candidate embeddings
or corpus membership invalidate related-item rankings across the subset.
Rebuild those dependents after their prerequisites. Historical enrichments
whose source changed before provenance/worklists existed require a deliberate
full recomputation from a known source snapshot.

## 4. Refit and identify topic models

See [the model bundle contract](lda-model-bundles.md) for artifact layout and
verification. New fits write immutable bundles under `<model-root>/bundles/<digest>` and an
atomic `current.json` pointer. `lda_model_name` contains `lda-sha256:<digest>`;
topic IDs must always be interpreted jointly with that value. Archive previous
models instead of treating a new fit with the same k as the same topic system.

The evaluation split is formed before phrase detection and vocabulary fitting.
Exact duplicates and an explicit `--holdout-group-column` stay together.
After model selection, a production fit may use all eligible documents; its
training performance is not the held-out estimate. A sweep holdout used to
inspect/select settings is a validation set, not a final untouched test set.
Keep a separate external evaluation set for final predictive claims.

```bash
iwac-process lda --config articles --mode fit --model-path lda_model_articles
iwac-process lda --config articles --mode predict --model-path lda_model_articles
```

Fits freeze preprocessing; prediction reloads it and verifies bundle hashes.
Exports live in separate run directories and include document-input hashes.
`iwac-analyze topics --theta-path <run>/doc_topics.parquet` reuses only compatible
rows, rejecting conflicting model identities or stale inputs. Legacy models
need explicit `--allow-legacy-preprocessing` for exploratory reconstruction or,
preferably, a new fit. Unknown language is excluded by default; including it
requires an explicit flag and a sensitivity check.

## 5. Recompute generation-specific sentiment and analyses

```bash
iwac-process sentiment-agreement --generation 2
iwac-process sentiment-agreement --generation 2 --push
iwac-analyze topic-sentiment --generation 2
```

Consensus columns now include the generation and an instrument fingerprint;
legacy generic consensus is not reused for topic sentiment. `--generation all`
is exploratory reporting only and cannot publish a pooled consensus. Constant
labels yield undefined chance-corrected agreement. Undefined values are not
zero agreement or perfect reliability.

Topic sentiment groups by immutable model identity plus topic ID. Medians,
category shares, sample sizes and missingness are primary. `--ordinal-means`
explicitly assumes equal scale spacing. Agreement among related LLMs is not a
human-validated accuracy estimate.

Topic trends default to descriptive slopes. Optional Mann–Kendall testing
states its annual-independence assumption. `--bootstrap-unit newspaper`
resamples whole outlets within years; intervals remain conditional on the
observed archive and fitted model, not uncertainty in archive selection or the
historical population. Check stable-outlet cohorts and reprint sensitivity.
Keyness uses the same year/language eligibility for each comparison; authority
bursts use IDs and export denominator/tagging exposure.

## 6. Review evidence and publish a complete projection

Use [the research validation tools](research-validation.md) to audit corpus
coverage, create blinded weighted annotation samples, interpret topics against
sources and adjudicate reprint candidates. Human labels, threshold calibration,
expert topic interpretation and historically justified inclusion rules remain
research work. This implementation supplies workflows, not empirical findings.

```bash
iwac-publish-public --dry-run
iwac-publish-public
iwac-mirror --dataset public
```

The publisher stages all selected subsets plus exact README schemas and commits
them atomically under a parent revision precondition. A remote conflict requires
reloading the baseline and rerunning; never force a stale write. Record the final
verified public revision before making figures, citations or release metadata.
No DOI or release is created by this migration.

Analysis runs retain full output bytes and environment manifests under unique
run directories. Keep the input snapshot or immutable Hub revision, exact model
bundle, Python lock and spaCy artifacts together. Use a clean Git commit for a
publication run: untracked code is deliberately not copied into run archives.
