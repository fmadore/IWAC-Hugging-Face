# IWAC Code & Methods Audit — July 2026

A full-repo examination following the completed `REFACTORING_ROADMAP.md`. That roadmap
(Tiers 1–3 + post-roadmap pass) is done and verified; everything below is **new** —
found by re-reading all 30 Python files after the refactor. Organized by priority:
bugs first, then pipeline safety, then refactoring, then DH-methodology improvements
and new analysis ideas.

---

## 1. Bugs to fix (correctness, ordered by severity)

### B1 — `islamic-publications` empty-Omeka branch: `NameError` + dangerous push
`islamic-publications/upload_Islamic_publications_hf.py:215` uses `token_to_use`
inside `load_dataset(...)`, but the variable is only assigned at line 246 — the
branch crashes with `NameError` before doing anything. Even if fixed, this branch
is the only place in the repo that **pushes an empty dataset over live Hub data**
when Omeka returns zero items (lines 240–260) — every other script safely skips the
push. Its fallback schemas (lines 221–235) also omit `OCR_is_public` (and one omits
`tableOfContents`), so firing it would break `publish_public.py`'s masking contract.
**Recommendation:** delete the push-empty behavior entirely; warn and exit like the
other six scripts.

### B2 — Embedding cache has no parameter fingerprint
`post-processing/semantic_embedding.py` keys its resume cache by `o:id` only
(lines 498, 514–518); neither the key nor the filename encodes `MODEL_NAME`,
`dimensionality`, or `task_type`. Abort a 768-dim run, re-run at 1536 (or switch
task type), and stale vectors are silently restored into the new column. Same gap in
`semantic_embedding_images.py:289-298`. **Fix:** put a `{model}-{dim}-{task}` slug in
the cache filename; invalidate on mismatch.

### B3 — Partial-chunk documents get permanent truncated embeddings
`semantic_embedding.py:307-316`: if some chunks of a long document fail in a batch,
the surviving chunks are averaged and cached; in `missing` mode the row is now
non-empty and never retried. A partially-embedded document is worse than a missing
one. Related: `--update-mode all` silently reuses cache entries from a previous run
(line 531 skips cached ids; `_save_completed_to_cache` never overwrites), which
contradicts "recompute all".

### B4 — Chunk-average is unweighted and ignores overlap
`_embedding_utils.py:97-107` takes a plain mean over chunk vectors: a 2,000-char
tail chunk weighs the same as a 28,000-char body chunk, and the 2,000-char overlap
region is double-counted. Use a length-weighted mean (and numpy instead of the
Python double loop). Also worth documenting: stored embeddings are **not**
L2-normalized (`related_articles.py` normalizes at read time, correctly — but other
consumers may assume unit vectors).

### B5 — Tokenization drift: keyness doesn't lowercase
The "lowercase → split → len≥2 → stopwords" tokenizer is re-implemented four times
(`modeling.py:96-100` training, `modeling.py:494-500` predict,
`topic_prevalence.py:65-69`, `keyness_bursts.py:237`) and has already drifted:
**keyness skips `.lower()`**, so it case-splits tokens the LDA side merges. Extract
one shared `tokenize(text, stopwords)` helper (natural home: `iwac_common` or
`lda_topic_modeling/constants.py`) and use it in all four places.

### B6 — `publish_public.py` prose-length guard has holes
`find_suspect_columns` (lines 105–124) only flags a new column when
`dtype == object`, values are scalar `str`, and mean > 3000 **or** max > 30000
chars. A medium-length text column (~2k chars), a `list[str]` column, or a pandas
`StringDtype` column all pass silently. The guard is a backstop, but the safe design
is an **explicit allowlist**: maintain the set of known-public columns per subset in
`iwac_common/repos.py` and abort on *any* unknown column, regardless of length or
dtype. New columns then require a one-line, reviewed classification.

### B7 — Merge fan-out and dtype drift in `hub_merge`
`iwac_common/hub_merge.py:124` merges on `o:id` with no uniqueness assertion — a
duplicate `o:id` already on the Hub multiplies rows silently. Add
`assert existing_df["o:id"].is_unique` (and the same for `new_df`). Separately,
left/outer merges inject `NaN` into preserved computed columns; only 4 of 7 scripts
re-cast dtypes afterwards (`index`, `audiovisual`, `images` do not), so preserved
numeric columns on those configs can flip int→float between runs.

### Smaller fixes
- `lda_topic_modeling.py`: a failed Hub push logs and returns **exit code 0**
  (lines 543–545) — scripted callers see success. Return non-zero on failure paths.
- `find_optimal_topics` (`modeling.py:657-691`): if every C_v computation throws, it
  silently returns the smallest k. Warn loudly when selection degrades.
- `modeling.py:79`: `tokenize_documents` annotated `-> List[List[str]]` but returns
  a `(tokens, phraser)` tuple.
- `calculate_word_count.py:327`: `ds.map` without `load_from_cache_file=False` can
  serve a stale HF map cache (lexical_richness handles this; word_count doesn't).
- Per-item media fetch is try/except-wrapped only in publications; in articles /
  document / audiovisual / images a single bad media record drops the whole item
  (e.g. `articles/upload_newspaper_hf.py:128-132`).
- Dead imports left by the refactor: `get_media_ids` (4 scripts), `load_dataset`
  (4 scripts), `huggingface_hub` (3), `IntPrompt`/`tqdm`/`os` in
  `lda_topic_modeling.py`, `json`/`aiohttp`/`Union` in `reference`, plus a no-op
  `fetch_media_data` re-definition at `reference/upload_reference_hf.py:154-155`.

---

## 2. Pipeline safety rails (highest-value structural work)

The most serious systemic risk in the repo: **a partial Omeka fetch silently shrinks
the Hub dataset.**

1. `OmekaApiClient.fetch_items` (`omeka_client.py:246-256`) stops paginating as soon
   as a page comes back short. A transient short-but-200 response mid-run truncates
   the fetch and reports success.
2. The truncated frame flows into a `how="left"` merge, so every missing item is
   **dropped from the Hub config**.
3. No script compares `len(new_df)` to the existing row count before `push_to_hub` —
   the only pre-push validation anywhere is `o:id` non-null.

Three cheap, high-value guards:

- **Reconcile against the API's total count.** Omeka S returns an
  `Omeka-S-Total-Results` header; assert fetched ≈ reported before proceeding.
- **Row-count tripwire in `hub_merge` / push helper.** Abort (or require
  `--force-shrink`) if the new dataset is more than, say, 5% smaller than the
  existing config. Deletions are rare and deliberate in this corpus; a guard costs
  nothing.
- **Schema check before push.** Compare `final_df.columns` against the existing
  config's columns; require an explicit flag to add/drop columns.

Also in this bucket:

- **`reference`'s outer-merge stale rows.** Items deleted from Omeka persist as
  blank-Omeka rows; `hub_merge.py:136-139` prints "review before pushing" but
  nothing gates the push. Either prompt/flag-gate it or drop them explicitly.
- **`--no-cache` exists only in `reference`.** The other six always trust the 24h
  cache — an Omeka edit followed by a re-run within 24h pushes stale content. Add
  the flag to all upload scripts (it's one argparse line once main() is shared).
- **Fetch-level try/except per resource class** exists in `reference` and `index`
  but not the other five — one exhausted retry kills the entire run.

---

## 3. Remaining refactoring opportunities

The roadmap removed the infrastructure duplication; what's left is **orchestration
duplication** — roughly 650–700 lines across the seven upload scripts:

| Block | Where | ~Lines |
|---|---|---|
| "Step 4" convert → validate → cast → push | all 7 (e.g. `articles:298-350`, `reference:500-562`) | ~320 |
| Mapping loop with progress bar | all 7 (`articles:253-269`, …) | ~100 |
| `display_config_panel` | 4 scripts | ~55 |
| `fabio:hasURL` extraction (verbatim ×3) | articles, publications, reference | ~48 |
| argparse `__main__` block | all 7 | ~55 |
| module header (sys.path, dotenv, logging) | all 7 | ~105 |
| local `_get_display_title` / `_get_at_value` / `count_words` / rights-label variants | audiovisual, index, images, document, reference | ~60 |

**Suggested shape:** one `iwac_common/upload_runner.py` with
`run_upload(subset_config)` where `subset_config` bundles config name, cache dir,
resource-class ids, a `map_item` coroutine, and optional dtype casts. Each upload
script shrinks to its Omeka→column mapper plus a config object — which is the part
that genuinely differs per subset. This also kills the two-tier world where
articles/document/publications/reference use Rich and audiovisual/index/images still
use tqdm + plain logging (the roadmap intended to unify this; the three tqdm scripts
never migrated — CLAUDE.md mandates Rich).

Post-processing side:

- **The deferred lexical_richness/word_count scaffold merge** is still worth doing:
  auth → pick config → per-row metric on `OCR` → `ds.map` → reorder column → push is
  copy-pasted, and they even share the same `\b\w+\b` tokenizer regex
  (`calculate_word_count.py:161` = `calculate_lexical_richness.py:79`). A tiny
  "column-metric runner" in `_common.py` serves both (and future metrics).
- **Embedding retry ladder duplicated**: `embed_texts_with_retry`
  (`semantic_embedding.py:182-212`) ≡ `embed_images_with_retry`
  (`semantic_embedding_images.py:133-169`), plus duplicated constants, cache-restore
  loop, pyarrow column build, and dry-run/push panels → extend `_embedding_utils.py`
  or add `_gemini_client.py` (the roadmap's Tier 2b idea, never fully realized).
- **Four different config-pickers** (`_common.choose_config` is used by exactly one
  script; lemmatize, word_count, semantic_embedding each roll their own with
  hardcoded, mutually inconsistent subset lists). Converge on the `_common` one.
- **Generic cache helpers live in `_embedding_utils`** with embedding-specific type
  hints, but lemmatize stores strings through them (`lemmatize_update_hf.py:60,259`).
  Move to `_common.py` (or `_cache.py`) with honest signatures.

### CLI consistency (one afternoon, big usability win)
- `--config` everywhere except `publish_public.py --configs`; standardize.
- Recompute semantics use four vocabularies: `--mode all|empty` (lemmatize),
  `--update-mode missing|all` (embeddings, richness), `-y` confirm (word_count),
  `--push` opt-in (sentiment, related). Pick one (`--update-mode missing|all` is the
  best fit) and alias the old flags.
- `--dry-run` missing from word_count, lemmatize, sentiment_agreement,
  related_articles.
- `calculate_lexical_richness.py` is the only metric script that **cannot run
  non-interactively** (no `--config`, no mode flag).
- The two embedding siblings behave differently with no flags (text prompts for
  update-mode; images defaults silently).

### Testing & CI (currently zero of both)
The repo has no tests and no CI. The highest-leverage subset is small and pure:
- `publish_public.py` masking + guard logic (**this is the privacy boundary** — it
  deserves tests more than anything else in the repo: null `OCR_is_public` → masked,
  unknown column → abort, list[str] column → abort once B6 is fixed).
- `dunning_g2`, `kleinberg_bursts`, `chunk_tokens`, `average_embeddings`,
  `calculate_mattr`, `merge_with_hub_dataset` (fan-out, `_old` suffix, dtype casts),
  the shared tokenizer from B5.
A GitHub Actions workflow doing `pip install -e . && pytest` + import-smoke of all
scripts (~2 min, CPU) would have caught B1 (`NameError`) and the dead imports.
`requirements.txt` has upper caps but no lower bounds — CI is what makes floating
lower bounds safe.

---

## 4. DH methodology — improvements to existing methods

### 4.1 Topic modeling (LDA)
What's already good: phrasers saved/reloaded so train≠predict can't drift; phrase
detection before chunking; length-weighted chunk→document aggregation
(`modeling.py:404-417`); domain-aware stopword curation that protects Islamic
terminology; per-language passes that compose without erasing each other.

Improvements, in rough value order:

1. **Multi-seed stability.** Everything rests on one `random_state=42`; the k-sweep
   scores each k with a single noisy C_v draw. Run 3–5 seeds per k, average C_v, and
   report topic stability (e.g. mean Jaccard overlap of top-10 words across seeds,
   aligned by Hungarian matching). This is the difference between "we found 30
   topics" and "these 30 topics are robust" — reviewers ask.
2. **Held-out evaluation.** Coherence is computed in-sample; the reduced-passes
   sweep assumes rank preservation (the docstring admits this is unverified). Hold
   out 10% of documents and report held-out log-perplexity alongside C_v.
3. **Disambiguate per-language topic IDs.** FR and EN models share the
   `lda_topic_id` column with colliding ID spaces — topic 5 (FR) ≠ topic 5 (EN),
   distinguishable only via `language`. Either offset EN ids (e.g. +100) or add an
   `lda_model_name` column so downstream users can't accidentally pool them.
4. **Export full document–topic matrices.** `lda_topic_topk` truncates to top-3;
   `topic_prevalence.py` has to re-run inference to get full distributions. Persist
   theta (docs × topics parquet in the model dir, or raise topk) so analyses don't
   depend on having the model + exact preprocessing locally. Note
   `topic_prevalence.py:129-132` infers on whole-document BoW and ignores
   `chunk_words` — correct for the articles model, silently inconsistent with
   training if pointed at the chunked publications/references models. Exported theta
   removes that trap entirely.
5. **FREX/relevance-weighted labels.** Labels are top-probability words, so
   corpus-common words can dominate several topics. LDAvis-style relevance
   (λ≈0.6) or FREX scoring makes labels more distinctive — cheap to add to
   `get_topic_label`.
6. **Embedding-based topic comparison.** The corpus already has Gemini document
   embeddings computed and stored — clustering them (HDBSCAN/k-means + c-TF-IDF
   labeling, i.e. the BERTopic recipe without any GPU work, since the expensive part
   is already paid for) gives an independent topic structure to triangulate against
   LDA. Agreement strengthens claims; disagreement is itself interesting (LDA sees
   vocabulary, embeddings see semantics — OCR noise affects them differently).

### 4.2 Statistical rigor in `analyses/` (the biggest methodological gap)
Neither analysis script attaches uncertainty or significance to anything it reports:

- **`topic_prevalence.py` trends** are raw OLS slopes ranked by size
  (lines 159–177, 224–230): no p-values/CIs, no multiple-comparison control across
  ~30 topics, unweighted years (a 20-doc year = a 2,000-doc year), slope computed on
  `solid_years` but `mean_prevalence` over all years, `peak_year` a raw argmax.
  *Fix:* bootstrap documents within years for prevalence CIs; n-weighted regression
  or Mann–Kendall for trend; Benjamini–Hochberg across topics; enforce the
  `min_docs` guard on year×country cells (currently advisory only).
- **`keyness_bursts.py` G²** emits top-25 per slice with no critical value and no
  FDR (line 108) — the classic keyness pitfall; with thousands of token×slice tests
  some "distinctive vocabulary" is noise. *Fix:* report log-ratio as effect size,
  filter by G² > threshold with BH correction; dedup repeated subjects per document
  (`|`-split currently double-counts, line 257–262).
- **Kleinberg bursts**: 2-state variant is fine, but the year index collapses
  calendar gaps (`T` = observed years only, lines 252–253), so a burst can silently
  span a missing year; and the base rate is over subject-tagged docs only. Document
  or fix both; consider multi-level bursts (s = 2, 4, 8) for graded intensity.

### 4.3 Lexical metrics
- **`Richesse_Lexicale_OCR` mixes two metrics**: MATTR for texts > 50 tokens, plain
  TTR below (`calculate_lexical_richness.py:85-86`) — short vs long texts aren't
  comparable. Emit `None` below the window (or a separate column) instead.
- **French elision**: the `\b\w+\b` tokenizer splits `l'islam` → `l` + `islam`,
  `aujourd'hui` → `aujourd` + `hui`, inflating token/type counts in MATTR *and*
  word_count. A tokenizer that strips elided clitics (`[ldjmnstc]'`) is ~3 lines.
- **Flesch Reading Ease on French OCR** (`textstat.flesch_reading_ease` after
  `set_lang('fr')`) is English-calibrated; on OCR with unreliable sentence
  boundaries it's mostly noise. Either switch to a French-calibrated formula
  (Kandel–Moles) or drop/rename the column so it doesn't over-promise.
- **OCR quality column (new, high value):** a per-document OCR-quality score
  (dictionary hit-rate of tokens against a French lexicon + short-token ratio) lets
  every downstream analysis weight or filter noisy scans, and would flag which
  richness/keyness outliers are artifacts. Cheap to compute, useful everywhere.

### 4.4 Sentiment agreement
Largely sound (quadratic-weighted κ on true ordinals, missing-tolerant Krippendorff
α). Two refinements: pin the κ weight matrix to the full 1–5 scale rather than
observed categories (`sentiment_agreement.py:103` — otherwise κ isn't comparable
across dimensions/pairs), and document that two-rater medians produce `.5` consensus
values.

### 4.5 Lemmatization
- Stopword test compares the **lemma** against spaCy's surface-form stopword list
  (`lemmatize_update_hf.py:256`) — mostly works for French but is conceptually
  mismatched; test the token's surface form (or `token.is_stop`).
- Language filter is exact/whitespace-fragile (`:202-204`): `"Français | Anglais"`
  with spaces fails membership. Strip components before comparing.
- Bilingual rows are processed by **both** language passes; in `--mode all` the
  second pass overwrites the first (order-dependent output), in `empty` it's
  skipped. Decide a rule (e.g. first language listed wins) and enforce it in both
  modes.

---

## 5. New analyses worth building (grounded in what the dataset already has)

Ordered by (research value × implementation ease):

1. **Topic × sentiment × time.** The two richest computed layers — LDA topics and
   the 3-model sentiment consensus — are never joined. "Which topics attract the
   most negative polarité, and how does that shift after 2015 (or differ
   Benin vs Burkina Faso)?" is a flagship DH finding sitting in existing columns.
   One script, no new computation.
2. **Entity co-occurrence networks from the authority file.** `articles.subject`
   joins exactly to `index.Titre` (controlled vocabulary), and index entries are
   typed (persons/orgs/places) with coordinates for `Lieux`. Subject co-occurrence
   within articles → actor networks per country/period; Louvain communities;
   centrality trajectories of key figures/organizations over decades. This exploits
   the corpus's most distinctive asset (a curated authority file) and no other tool
   in the repo touches it.
3. **Press-ecology comparisons.** `newspaper` + `country` are on every article:
   per-newspaper topic profiles and keyness (state vs. private vs. religious press
   treat Islam differently — measurable with the existing G² machinery, one
   `groupby` away).
4. **Reprint/wire-copy detection.** `related_articles` already computes near-dup
   cosine pairs but only reports each row's top-1 neighbor
   (`related_articles.py:172-180`). Union-find over all pairs ≥ 0.95 → reprint
   clusters: which stories circulated across borders, and with what lag. Also
   flags duplicates that distort topic prevalence counts.
5. **Diachronic semantic drift.** Word2vec per decade on `lemma_nostop` (gensim,
   CPU-fine, already a dependency) with Procrustes alignment: how the words around
   *islam*, *voile*, *terrorisme*, *confrérie* shift across decades — a
   well-established DH method (HistWords) the corpus is ideally sized for.
6. **Geography of attention.** `spatial` + `index.Coordonnées`: map place-mention
   frequency over time; distinguish local vs. foreign framing (e.g. how much
   coverage of Islam in Benin is actually about Nigeria/Mali/Middle East). Pairs
   naturally with the burst detector for event geography.
7. **Agenda-setting: press vs. academy.** The `references` subset (academic
   literature) vs. `articles` (press): per-subject or per-topic time series of
   both, with lead–lag correlation — does scholarship follow press attention or
   vice versa? Few corpora can even ask this; IWAC has both sides in one dataset.

---

## 6. Suggested execution order

| Phase | Content | Effort |
|---|---|---|
| 1. Correctness | B1 (delete push-empty branch), B2/B3 (cache fingerprint + partial-chunk), B5 (shared tokenizer), B6 (allowlist guard), B7 (uniqueness assert), exit codes | 1–2 days |
| 2. Safety rails | Total-count reconciliation, row-count tripwire, schema check, `--no-cache` everywhere, gate reference stale rows | 1–2 days |
| 3. Tests + CI | pytest for publish_public masking, G²/bursts/MATTR/merge/chunkers; GH Actions import-smoke + tests | 1–2 days |
| 4. Refactor | `upload_runner` extraction (~650 lines), metric-script scaffold, embedding-client dedup, CLI unification | 3–5 days |
| 5. Methods | Multi-seed/held-out LDA eval, theta export, FDR + bootstrap in analyses, French tokenizer/elision, OCR-quality column | 1 week, incremental |
| 6. New analyses | Topic×sentiment, entity networks, press ecology, reprints, drift, geography, press-vs-academy | pick per research agenda |

Phases 1–3 protect the dataset (the live Hub data is the irreplaceable artifact
here); 4 makes the codebase cheap to extend; 5 makes existing claims defensible;
6 is new research surface.
