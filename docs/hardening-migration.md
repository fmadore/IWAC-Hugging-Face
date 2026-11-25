# Pipeline and methodology hardening: migration guide

This implementation changes code and tests. It does not retroactively correct
existing Hub values, regenerate topic models, run paid inference or validate
historical claims. Apply the following sequence first to scratch repositories,
then retain the exact validated code/data/model/environment revisions used in
research. Do not mix outputs from the previous and new methods without labeling
the difference.

Sections 1–6 below describe the October 2026 migration. The next section is
the re-run that the November 2026 hardening commit adds on top of it.

## Re-run after the November 2026 hardening (2026-11-16)

The hardening commit of 2026-11-16 changes how several columns are computed
but, like every code change, moves no data. This is the ordered re-run that
applies it. Run every step against the **private** repository, one push at a
time: wait for each command to finish (including its verification) before
starting the next. Try the whole sequence on scratch repositories first
(`IWAC_HF_PRIVATE_REPO` / `IWAC_HF_PUBLIC_REPO`).

The order matters for three reasons. Uploads come before everything because
they refresh visibility metadata and clear stale derived cells that later
stages refill. Lemmas come before LDA because topic models train on
`lemma_nostop`. Publication comes last because it projects whatever the
private repository holds at that moment.

### 1. Uploads, `index` last

```bash
iwac-upload articles --dry-run
iwac-upload articles
iwac-upload publications
iwac-upload references
iwac-upload audiovisual
iwac-upload documents
iwac-upload images
iwac-upload index
```

What changes: `country` is filled for articles and publications with several
publishers (`country_mapper.get_country_from_outlets`); the references and
audiovisual mappers compute `nb_mots` with the row language; every subset casts
all registry `int_columns` after the merge; IIIF manifest lookups retry
(`IIIF_MAX_TRIES`); `index` thumbnails are set only for records with primary
media. `documents` and `images` need a run only to pick up the retries and the
integer casts.

`index` goes last. Its `frequency`, `first_occurrence`, `last_occurrence` and
`countries` now count **public parent items only** (`item_is_public` exactly
true) from articles, publications, references and audiovisual on the Hub, and
a column listed in a row's `private_fields` contributes nothing. A public item
whose OCR is private still counts. The upload refuses to run if any of those
content subsets lacks `item_is_public` or `private_fields`, so refresh them
first.

An upload clears `related_articles` across the whole subset whenever an item's
`item_is_public` changed, membership changed or an embedding source changed.
If an upload also cleared embedding cells (the dry run's "Source text changed"
table lists them), refill them before step 7 with
`iwac-process embeddings --config <subset> -y` (or
`iwac-process image-embeddings --config images -y`). Embedding values and
configuration hashes are otherwise unchanged by this commit.

### 2. Word count, then lexical metrics

Word count on each of `articles`, `publications`, `documents`, `references`
and `audiovisual`; lexical metrics on `articles` only:

```bash
iwac-process word-count --config <subset> -y   # all five text subsets
iwac-process lexical --config articles -y
```

`Richesse_Lexicale_OCR` and `Lisibilite_OCR` are allowlisted for `articles`
only (`iwac_common/public_columns.json`), and on 2026-11-24 no other subset
carried them, even privately. Running `lexical` elsewhere would add columns
that `iwac-publish-public` refuses as unreviewed; publishing them would need a
rights review per subset, not a silenced guard.

What changes: tokenizer version 4 (`TOKENIZER_VERSION` in
`iwac_common/text_utils.py`). For a row whose languages include French an
elided clitic is a word of its own (`l'église` counts 2, like `la mosquée`);
other languages keep an apostrophe inside the word. `n'`/`m'` before a
consonant is a name and stays one word whatever its case (`N'diaye`,
`M'BOW`), since French elides only before a vowel or mute h; before a vowel it
is the clitic, so an all-caps `N'EST` splits like `n'est`. `nb_mots` and the MATTR
token stream come from the same function with the row language. `nb_mots` now
carries private provenance columns, and both stages compare stored provenance
with the current text, language and configuration, so the default `missing`
mode recomputes every count and metric written under an older tokenizer or
without provenance; `--update-mode all` is not needed. Readability
(`Lisibilite_OCR`) stays monolingual-French only and is null below
`MIN_READABILITY_WORDS` words or `MIN_READABILITY_SENTENCES` sentences, above
`MAX_WORDS_PER_SENTENCE` words per sentence, or without letters (100, 3 and 100
in `post-processing/calculate_lexical_richness.py`); the thresholds are part of
its configuration hash.

### 3. Lemmas

```bash
iwac-process lemmas --config articles -y
iwac-process lemmas --config publications -y
iwac-process lemmas --config documents -y
iwac-process lemmas --config references -y
iwac-process lemmas --config references --language Anglais -y
```

The default language is `Français`; `references` needs the second, English
pass. What changes: `LEMMA_LOGIC_VERSION` 4. Hyphenated and apostrophe words
stay whole (`côte-d'ivoire`, `sous-préfet`, `aujourd'hui`, `quelqu'un`);
transliterations the French tokenizer splits are rejoined (`jama'at`, `da'wa`,
`qur'an`); N'/M' before a consonant is a name whatever its case (`n'guessan`,
`n'diaye`, `m'bow`), while an all-caps `N'EST` is the clitic; a word broken by
a line-end hyphen is rejoined when the model's vectors know the joined form
(`organi-`/`sation` → `organisation`, while `sous-`/`préfet` keeps its hyphen);
a token negated by an attached `n't` takes its lemma's stop status (`won't` →
`will`); bare elided clitics are dropped and typographic apostrophes
normalised; a lemma never contains whitespace. Word counts do not rejoin
line-end hyphens, since `nb_mots` has no lexicon. The version is part of the
provenance hash, so `missing` mode recomputes every row: on a CPU this is a
long run. The resume cache is SQLite
(`.cache_lemmas/*.sqlite3`) and is checkpointed incrementally, so an
interrupted run resumes by re-running the same command. `--dry-run` prints the
plan without running spaCy.

### 4. Refit the topic models

```bash
iwac-process lda --config articles --mode fit -y
iwac-process lda --config publications --mode fit -y
iwac-process lda --config references --mode fit -y
iwac-process lda --config references --language Anglais --mode fit -y
```

Each fit writes a new immutable bundle under the preset's model root
(`lda_model`, `lda_model_publications`, `lda_model_references`,
`lda_model_references_en`), then predicts and pushes the topic columns. What
changes: dictionary `no_below`/`no_above` count source documents rather than
chunks; a learned compound must occur in at least `phrase_min_docs` source
documents (`DEFAULT_PHRASE_MIN_DOCS`); the stopword tiers were revised (generic
words that anchor domain compounds moved to `FRAGMENT_STOPWORDS`, underscore
entries to `JUNK_COMPOUND_STOPWORDS`, collocations added in lemma form, and the
pre-phrase set is language-aware through `LANGUAGE_PRE_PHRASE_EXEMPTIONS`);
labels keep relevance order; rows below the minimum-evidence floor
(`--min-predict-tokens`) get null topic columns but keep `lda_model_name`. See
[the model bundle contract](lda-model-bundles.md). A new fit renumbers topics:
`lda_topic_id` is meaningful only with its `lda_model_name`.

### 5. Hijri dates

```bash
iwac-process hijri --config articles --update-mode all -y
```

Repeat for `publications`, `documents`, `audiovisual` and `images`. What
changes: only a complete date in exactly `YYYY-MM-DD` with ASCII digits is
converted (partial dates and ranges stay null, as before); an ISO datetime is
converted from its date part; dates outside the Umm al-Qura table and
impossible calendar dates stay null and are reported on separate rows. Stored
values for valid dates do not change. The Hijri columns carry no provenance,
so only `all` re-examines rows that already hold a value.

### 6. Sentiment agreement

```bash
iwac-process sentiment-agreement --generation 2 --push
iwac-process sentiment-agreement --generation 1 --push
```

What changes: `CONSENSUS_RULE_VERSION` (in `iwac_common/sentiment_panel.py`,
now v3) changes each generation's instrument fingerprint, which also hashes
only the instrument-defining fields of each model, so every stored
`consensus_g<n>_instrument_id` is stale until this step runs. Consensus and disagreement
columns are null for rows with fewer than two valid ratings; `""` means "rated
by at least two models, no strict majority" for a label and "no assessable
dimension disputed" for the disagreement column. Ordinal Krippendorff alpha is
the primary coefficient in the report (interval and nominal secondary), and
polarité also gets a nominal alpha that includes "Non applicable". The dispute
share is over rows with at least two raters and is not comparable across
generations, whose panels differ in size.

### Before the next embedding run

The lock regenerated on 2026-11-16 moves `onnxruntime` to 1.32.0 and Pillow to
13.0.0. Pillow 13.0.0 gave decoded pixels and model patches bit-identical to
12.3.0 on 23 synthetic images (JPEG, PNG, WebP, TIFF; Linux, Python 3.12);
Windows and real photographs were not checked, and no vector was compared
under the new `onnxruntime`. The lock's `tokenizers` 0.23.3 gave token ids
identical to the verified 0.22.1 on 404 texts (400 `articles` rows in document
format, 403,878 tokens, plus Arabic, NFD and emoji cases and the image token
sequence; 2026-11-24). Before embedding new rows, re-embed a few stored rows
and confirm the vectors match what is on the Hub. No step in this plan
re-embeds unless an upload cleared embedding cells.

### 7. Related items

```bash
iwac-process related --config articles --push
iwac-process related --config documents --push
```

`publications` is not in this step: it carries no `related_articles` column
today, and the column is not allowlisted for it, so adding one would stop the
publication step until reviewed. What changes: the neighbour pool is the public items only
(`item_is_public` exactly true); private rows get null; a public item whose OCR
is private remains a neighbour. Run this after steps 1 and any embedding refill,
because both visibility and vectors define the pool.

### 8. Publish the projection

```bash
iwac-publish-public --dry-run
iwac-publish-public
```

The projection strips any `related_articles` entry that names an unpublished
id, and validation refuses one that remains (`NEIGHBOUR_ID_COLUMNS`,
`strip_unpublished_neighbours` in `iwac_common/write_policy.py`). Columns
approved with `--approve-columns` are written to `public_columns.json` only
after the verified push succeeds.

### 9. Calibration and cited analyses

```bash
iwac-mirror --dataset private
iwac-analyze similarity-calibration
```

The calibration now splits by lexical reprint family into item-disjoint halves
(`SPLIT_VERSION` 3) and reports bootstrap intervals; the figures previously
quoted for 0.91 and 0.97 came from a per-pair split. The re-run on 2026-11-25
confirmed 0.91 and chose 0.965 on the calibration half; 0.97 was kept. Its
held-out figures are in
[the research validation protocol](research-validation.md). Re-run any
analysis whose output you cite, since word counts, lemmas, topics, consensus,
neighbours and the shared date rule (`analyses/_research.py:point_years`) all
changed underneath them.

**DOI.** Before minting a dataset DOI, verify the published projection itself,
not just the publisher's guards: load each content subset from the public repo
and confirm that no row flagged `OCR_is_public = false` carries `OCR`,
`lemma_text` or `lemma_nostop`, and that no `related_articles` entry names an id
absent from the published subset. Mint only on
`fmadore/islam-west-africa-collection`, at a deliberate release point
(CLAUDE.md, "DOIs").

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

Text and image embeddings, lemmas, word counts and lexical metrics persist
input/configuration fingerprints.
Existing nonempty values without compatible provenance are recomputed once,
even in incremental mode. Plan inference cost accordingly. Source and method
changes then permit selective recomputation. Empty inputs clear obsolete values.

Embeddings now come from `google/embeddinggemma-2`, run locally from its
pinned fp32 ONNX export (`post-processing/_gemma_encoder.py`), with the model
card's `title: … | text: …` document prompt and byte-bounded chunking. The
vectors published before October 2026 are `gemini-embedding-2` vectors of raw
text and cannot be reproduced by current code; every embedding column
(`embedding_OCR` on articles and documents, `embedding_abstract` on
references, `embedding_tableOfContents`, `embedding_image`) must be regenerated
in one release, because text and image vectors share one space. Old and new
vectors are not a single calibrated similarity space: recalibrate cosine
thresholds afterwards.

References embed their abstract, not their full text. Half the reference texts
exceed 58k characters and the longest run to 1.8M: averaging hundreds of
chunks yields a centroid of genre and bibliography rather than argument, and
notes and bibliographies outweigh the thesis. References without an abstract
keep an empty vector. The run that writes `embedding_abstract` removes the
retired `embedding_OCR` from references; consumers must read the new column.

On a CPU the articles re-embed alone is a long job (about 16 h; a re-embed of
every text column is about 30 h); each finished batch is checkpointed, so an interrupted run resumes by re-running the same command.
Image identity covers the downloaded bytes and URL, so a URL whose content
changed is detectable. Incremental image validation still downloads the image.
Readable configuration JSON accompanies each fingerprint privately. Related-item
and reprint tools reject mixed known configurations; legacy/public vectors that
lack provenance require an explicit `--allow-unverified-embeddings` exploratory
option, which does not establish compatibility.

Lemmatization skips missing/ambiguous language and multilingual records by
default; `--multilingual primary` explicitly applies the first metadata language.
French readability requires monolingual French, not simply a first-listed French
label, and enough evidence (the minimum word, sentence and words-per-sentence
thresholds described above). Invalid/inapplicable inputs remain null. MATTR preserves Unicode combining
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
iwac-process lda --config articles --mode fit
iwac-process lda --config articles --mode predict
```

Both use the preset's model root (`lda_model` for articles, the default that
`iwac-analyze topics --model-path` also reads). Pass `--model-path` only to
work on a separate root, and pass the same value to every later command.

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
category shares, sample sizes and missingness are primary. Polarity,
centrality and subjectivity each export plain shares (over every assigned row,
`n`) and `*_of_scored` shares (over that dimension's consensus count:
`n_polarity_consensus`, `n_centrality_scored`, `n_subjectivity_scored`). The
ranking, country and year tables filter on `n_consensus_max`, the larger of the
polarity (Non applicable included) and centrality consensus counts, so a topic
whose finding is low centrality is not dropped. `--ordinal-means`
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
