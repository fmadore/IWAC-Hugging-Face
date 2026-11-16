# IWAC Hugging Face Pipeline

Python pipeline that mirrors the [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) (IWAC) from its Omeka S archive into versioned [Hugging Face](https://huggingface.co/datasets/fmadore/islam-west-africa-collection) datasets.

[![Collection: IWAC](https://img.shields.io/badge/Collection-IWAC-1f6feb?style=flat-square)](https://islam.zmo.de/s/westafrica/)
[![Hugging Face dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffd21e?style=flat-square&labelColor=555)](https://huggingface.co/datasets/fmadore/islam-west-africa-collection)
[![CI](https://github.com/fmadore/IWAC-Hugging-Face/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/fmadore/IWAC-Hugging-Face/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-3fb950?style=flat-square)](LICENSE)

[![Dataset DOI](https://img.shields.io/badge/Dataset%20DOI-10.57967%2Fhf%2F9857-0a7bbb?style=flat-square)](https://doi.org/10.57967/hf/9857)
[![Software DOI](https://img.shields.io/badge/Software%20DOI-10.5281%2Fzenodo.21805704-0a7bbb?style=flat-square)](https://doi.org/10.5281/zenodo.21805704)

## Context

The [Islam West Africa Collection](https://islam.zmo.de/s/westafrica/) is an open-access digital database documenting Islam and Muslim communities in Benin, Burkina Faso, Côte d'Ivoire, Niger, Nigeria, and Togo since the 1960s. Created by [Frédérick Madore](https://www.frederickmadore.com/) and hosted at the Leibniz-Zentrum Moderner Orient (ZMO) in Berlin, it is a growing archive curated in [Omeka S](https://omeka.org/s/). Counts depend on the source date and dataset revision; report them from the snapshot used in your research.

This pipeline reads the Omeka S REST API, flattens selected resource classes into tabular subsets, and computes semantic embeddings, lemmatised text, topic assignments, lexical metrics, and Islamic-calendar dates. It also imports sentiment annotations generated upstream in Omeka and computes panel aggregates. The resulting Hugging Face dataset can be loaded in one line.

It is the data layer behind the collection's [visualisations](https://github.com/fmadore/IwacVisualizations) and its MCP server, and a companion to [iwac-ai-pipelines](https://github.com/fmadore/iwac-ai-pipelines), which handles the LLM-assisted curation happening upstream inside Omeka S.

## The two-repo split

Much of the collection's full text is **private on the Omeka S source** — rights-restricted newspaper scans, for instance — while a large share is public. The dataset is therefore split across two Hub repos:

| Repo | Visibility | Role |
|------|-----------|------|
| [`fmadore/islam-west-africa-collection-full`](https://huggingface.co/datasets/fmadore/islam-west-africa-collection-full) | Private | Complete superset, full text for all rows. The canonical target of **every** upload and post-processing script. |
| [`fmadore/islam-west-africa-collection`](https://huggingface.co/datasets/fmadore/islam-west-africa-collection) | Public | The citable projection. Written **only** by `post-processing/publish_public.py` (`iwac-publish-public`). |

The public projection omits private parent items (`item_is_public`), masks restricted Omeka properties (`private_fields`), and keeps `OCR`, `lemma_text` and `lemma_nostop` only where `OCR_is_public` is true. Missing visibility metadata fails closed. Reviewed computed derivatives — embeddings, topics, metrics — are retained for public items even when their OCR is restricted; this is a publication decision, not a guarantee that derived outputs cannot disclose source information. Two published columns are built from public items only: the `index` aggregates (`frequency`, `first_occurrence`, `last_occurrence`, `countries`) and the `related_articles` neighbour lists. New columns require review in [`public_columns.json`](iwac_common/public_columns.json).

Every write goes through one verified gateway that requires a private destination for the full data and revalidates the projection for the public repo. Data and dataset-card schema are committed together under a server-side revision precondition, then verified at that exact commit. Uploads fail closed on incomplete Omeka enumeration, mapper or media failures and unreadable Hub baselines, and changed source values invalidate dependent enrichments by default. The [ingestion and recovery contract](docs/ingest-contract.md) gives the details, including the override flags.

## Dataset subsets

Seven subsets, each mapped from one or more Omeka S resource classes:

| Subset | Contents |
|--------|----------|
| `articles` | Newspaper articles — the analytical core of the collection |
| `publications` | Islamic periodicals and their issues |
| `documents` | Archival and institutional documents |
| `references` | Scholarly references (books, chapters, journal articles) |
| `index` | Authority records: persons, places, organisations, events |
| `audiovisual` | Deposited recordings and embedded YouTube videos, with transcriptions where available (`source_type` tells the two apart) |
| `images` | Fieldwork photographs |

Content subsets join to `index` authority records, which is what makes entity-level analysis possible across the corpus. Join on the `*_ids` columns (`subject_ids`, `spatial_ids`, `author_ids`, `newspaper_ids`, `creator_ids`, `publisher_ids`, `editor_ids`) against `index.o:id`: they carry the linked authorities' Omeka ids, so homonyms stay apart and a renamed authority still matches. The label columns beside them hold display titles, pipe-joined.

A few columns exist to save every consumer from re-parsing the same strings: `pub_year` and `pub_date_precision` (`day`, `month`, `year`, `range`, `other`) beside the verbatim `pub_date`; `latitude`/`longitude` beside the `"lat, lng"` coordinate string on `index` and `images`; and `abstract_en` on `references`, the English literal that used to be pipe-joined into `abstract`.

```python
from datasets import load_dataset

articles = load_dataset("fmadore/islam-west-africa-collection", name="articles", split="train")
```

## What the pipeline computes

| Stage | Script | Output |
|-------|--------|--------|
| Semantic embeddings | `post-processing/semantic_embedding.py` | EmbeddingGemma 2 vectors (run locally at a pinned revision) over full text (abstracts for `references`), chunked and averaged for long documents |
| Image embeddings | `post-processing/semantic_embedding_images.py` | EmbeddingGemma 2 vectors of the photographs, in the same space as the text embeddings |
| Lemmatisation | `lemmatize_update_hf.py` | spaCy lemmas, with and without stopwords, per language |
| Topic modeling | `post-processing/lda_topic_modeling/` | LDA topic id, probability, label, top-k terms and model identity; a `topics.csv` lookup (label, top words, document counts per `lda_topic_id`) is written with each prediction run under `<model root>/predictions/<timestamp>/` |
| Lexical metrics | `post-processing/calculate_word_count.py`, `calculate_lexical_richness.py` | Word count (one language-aware definition, shared with the upload mappers), MATTR lexical richness, French Flesch readability (monolingual French rows only) |
| Islamic calendar | `post-processing/calculate_hijri_dates.py` | Hijri year, month, and day (Umm al-Qura) |
| Model agreement | `post-processing/sentiment_agreement.py` | Agreement across the sentiment panel and generation-specific consensus columns |
| Related items | `post-processing/related_articles.py` | Top-k nearest public neighbours by embedding |

### Sentiment panel

Sentiment annotations (centralité, polarité, subjectivité, each with a justification) are generated upstream in Omeka by an LLM panel and imported here; `iwac_common/sentiment_panel.py` is the registry. Each model's columns carry a prefix naming the exact model (for example `gpt_5_6_luna_polarite`), so two generations never share a column.

- **Generation 2 is the live panel**: `gpt-5.6-luna`, `mistral-small-2603`, `deepseek/deepseek-v4-flash-0731`, `google/gemma-4-31b-it`, `Qwen/Qwen3.8-27B` and `claude-haiku-5-5`. Its subjectivité is a label (`string`).
- **Generation 1 is frozen history**: `gemini-3-flash-preview`, `gpt-5-mini` and `ministral-14b-2512`, whose subjectivité is an integer 1–5 (nullable `int64`; `float64` in revisions published before the canonical types).
- **Do not pool across the boundary.** Generation 2 ran a rewritten prompt, so cross-generation comparisons confound the models with the instrument. Consensus columns are generation-specific (`consensus_g2_*`, `sentiment_g2_disagreement`) and null where fewer than two models rated a row.

Coverage differs by model; [the sentiment panel page](docs/sentiment-panel.md) records each campaign, the consensus rules and the agreement report.

## Repository layout

```
articles/  audiovisual/  document/  images/       Upload scripts, one per subset:
index/     islamic-publications/  reference/      Omeka S -> Hugging Face

iwac_common/        Shared infrastructure: Omeka client, fail-closed Hub
                    gateway, schema registry, merge, mappers, upload runner
iwac_pipeline/      Installed `iwac-*` command entry points
post-processing/    Computed columns + publish_public.py
analyses/           Report-only analyses; write to analyses/output/, never
                    add Hub columns
tests/              Unit tests and import smoke tests (run in CI)
data/               fetch_datasets.py — revision-pinned local Parquet (or CSV)
                    mirror for offline work
```

## Installation

Requires **Python >= 3.12**. CI tests Python 3.12 on Linux and Windows and Python 3.13 on Linux. Development is CPU-only throughout; the pipeline deliberately prefers CPU-viable models such as spaCy's `*_lg` pipelines over transformer equivalents.

Windows PowerShell:

```powershell
git clone https://github.com/fmadore/IWAC-Hugging-Face.git
Set-Location IWAC-Hugging-Face
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install -e . --no-deps
Copy-Item .env.example .env
```

Linux or macOS:

```bash
git clone https://github.com/fmadore/IWAC-Hugging-Face.git
cd IWAC-Hugging-Face
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install -e . --no-deps
cp .env.example .env
```

The lock includes runtime and development dependencies and hashes, with platform markers for the CI environments. To develop against the supported dependency ranges instead:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest
```

Both editable and wheel installs expose `iwac-upload`, `iwac-mirror`, `iwac-publish-public`, `iwac-process`, and `iwac-analyze`. Core dependencies are declared by the package; `pip install ".[all]"` adds the NLP and analysis dependencies, or select `.[nlp]` / `.[analysis]` (the tables under [Usage](#usage) say which commands need which). CI builds the wheel and runs every command's `--help` from an empty working directory. With a wheel, set `IWAC_WORK_DIR` to a writable working directory for data, caches, models and reports; the default is the current directory. Source checkouts keep their existing output locations.

The lemmatisation step additionally needs spaCy models:

```bash
python -m spacy download fr_core_news_lg
python -m spacy download en_core_web_lg
```

## Configuration

Copy `.env.example` to `.env`. What is actually required:

- **Uploads** need `OMEKA_KEY_IDENTITY` and `OMEKA_KEY_CREDENTIAL`, dry runs included: an anonymous listing omits private items, so an upload refuses to start without both. `OMEKA_BASE_URL` is optional and defaults to the production API, `https://islam.zmo.de/api`.
- **Anything that reads or writes the private repo** needs a Hugging Face token with access to it: `HF_TOKEN`, or else the login cached by `hf auth login`. Write-capable commands without either start an interactive login.

The embedding stages need no API key: they download the pinned EmbeddingGemma 2 ONNX export (~1.1 GB, plus ~0.7 GB for the vision encoder) on first use.

Set `IWAC_HF_PRIVATE_REPO` and `IWAC_HF_PUBLIC_REPO` to redirect the pipeline at scratch datasets before running write-capable commands; the full destination must be private. `IWAC_ENV_FILE` selects an explicit dotenv file; process environment variables always win over dotenv. The remaining optional settings (`IWAC_WORK_DIR`, `IWAC_PUBLIC_BASE_URL`, `IWAC_LOCK_DIR`, `IWAC_STATE_DIR`) are described in `.env.example`.

## Usage

The installed commands are the supported interface; the script paths they wrap (`python post-processing/calculate_word_count.py`, …) remain runnable. Every command takes `--help`.

The flow runs in three stages, in order:

```bash
# 1. Upload — fetch from Omeka S, merge into the private repo
iwac-upload articles --dry-run
iwac-upload articles

# 2. Enrich — compute derived columns on the private repo
iwac-process word-count --config articles -y

# 3. Publish — project the private repo into the public one
iwac-publish-public --dry-run
iwac-publish-public
```

`iwac-upload` takes one of `articles`, `publications`, `documents`, `references`, `index`, `audiovisual` or `images`, and forwards its remaining flags to that subset's parser. Upload `index` after the content subsets: its aggregates read them from the Hub. `iwac-mirror --dataset private` (or `public`) writes the revision-pinned local mirror; without `--dataset` it asks interactively.

### Enrichment stages (`iwac-process <stage>`)

| Stage | What it does | Needs |
|-------|--------------|-------|
| `word-count` | `nb_mots` from `OCR`, language-aware | core |
| `lexical` | MATTR richness and French Flesch readability | `nlp` (textstat) |
| `lemmas` | spaCy `lemma_text` / `lemma_nostop`, one language per pass | `nlp` (spaCy) and the spaCy models |
| `lda` | Fit or predict LDA topic columns | `analysis` (gensim) |
| `hijri` | Umm al-Qura `hijri_year` / `hijri_month` / `hijri_day` from `pub_date` | `nlp` (hijridate) |
| `embeddings` | EmbeddingGemma 2 text embeddings | `nlp` (onnxruntime, tokenizers) |
| `image-embeddings` | EmbeddingGemma 2 embeddings of `images` | `nlp` (onnxruntime, tokenizers, Pillow) |
| `sentiment-agreement` | Panel agreement report; `--push` adds consensus columns | core |
| `related` | Top-k related public items from embeddings; `--push` adds `related_articles` | core |

`word-count`, `lexical`, `lemmas`, `hijri`, `embeddings` and `image-embeddings` share two flags. `--update-mode missing` computes only rows without a current value and `--update-mode all` recomputes every row. Without the flag a stage uses `missing` when given `-y` or run without a terminal, and asks otherwise; `-y` never selects a full recompute, so a scripted one needs `--update-mode all -y`. Before an `all` run, `word-count`, `lexical`, `lemmas` and `hijri` ask for confirmation unless given `-y` or `--dry-run` or run without a terminal; the embedding stages do not ask. Word count, lexical metrics, lemmas and embeddings store input and configuration hashes, so `missing` also recomputes values whose text, language or method changed; Hijri dates do not, so a parser change needs `all`. A stage that changes nothing pushes nothing. `lda` takes `--mode fit|predict` and `-y`; `sentiment-agreement` and `related` are report-only unless given `--push`.

**Changing code does nothing to published data until its stage is rerun.** The [migration guide](docs/hardening-migration.md) gives the current re-run order.

### Analyses (`iwac-analyze <analysis>`)

Report-only: they write to `analyses/output/` with a run manifest and never add Hub columns.

| Analysis | What it does | Needs |
|----------|--------------|-------|
| `topics` | Probability-weighted topic prevalence over time | `analysis` (gensim) |
| `topic-sentiment` | Topic × sentiment consensus × time × country | core |
| `keyness` | Keyness (Dunning G²) and subject burst detection | `analysis` (scipy) |
| `entities` | Authority co-occurrence networks from `index` | core |
| `coverage` | Corpus composition, missingness and an explicit inclusion rule | core |
| `annotation-review` | Blinded human annotation sheets and a challenge set | core |
| `reprints` | Reprint candidates from semantic and lexical evidence | core |
| `similarity-calibration` | Re-derives cosine cutoffs for the current vector space | core |
| `topic-review` | Representative and borderline texts for expert topic review | core |

All share `--input`, `--source local|hub` (`csv` is accepted for `local`), `--revision` and `--output-dir`; `topics`, `topic-sentiment`, `keyness` and `entities` also take an `--ids-file` cohort. See the [research validation protocol](docs/research-validation.md).

Two properties of this flow are easy to get wrong:

**Pushes to one repo must be sequential.** The writer enforces this locally with a repo-scoped lock and rejects a Hub revision that changed after computation began. A lock left behind by a crashed process on this host is reclaimed on the next run; one whose owner is still alive, or which was written by another machine, fails closed — wait for that writer rather than deleting the lock file.

**Uploads merge rather than overwrite.** Each upload fetches from Omeka, loads the existing Hub rows, identifies columns that exist only on the Hub, and merges them back on `o:id`. That is what keeps embeddings and topics alive across a re-upload rather than blanking them.

## Reproducibility

For a publication, record the code commit, the dataset repository and full revision SHA, the exact command arguments, and the installed dependency and spaCy model versions. Archive the model artifacts and analysis outputs used for the paper. Use the hashed `requirements-lock.txt` for the tested research environment; dependency ranges support maintenance and have a separate CI leg. The spaCy model packages must also be retained at the versions recorded in run provenance. See the [migration guide](docs/hardening-migration.md) and [research validation protocol](docs/research-validation.md).

Pin dataset reads to the revision used in the paper:

```python
articles = load_dataset(
    "fmadore/islam-west-africa-collection",
    name="articles", split="train", revision="<full dataset commit SHA>",
)
```

LDA fits produce hash-verified immutable bundles with frozen preprocessing. `lda_model_name` contains `lda-sha256:<digest>`; topic IDs have meaning only within that identity. See the [model bundle contract](docs/lda-model-bundles.md).

Sentiment consensus is generation-specific and records its instrument fingerprint. Topic sentiment computes the selected panel from its raw annotations; ordinal medians and category shares are primary, with equal-spacing means explicit. Constant-label agreement is undefined rather than reported as perfect reliability. Topic trends are descriptive by default; optional inferential assumptions are recorded, and outlet-cluster bootstrap intervals condition on the observed archive and model.

Report runs archive **complete output bytes**, hashes, arguments, dataset revision, environment versions and code state under `analyses/output/runs/<run-id>/`. Latest filenames remain convenient pointers; earlier runs are retained. A dirty checkout records its tracked diff and names of untracked files, not their contents, so use a clean commit for a fully reconstructable publication run.

The corpus-coverage, blinded-annotation, topic-review and reprint-candidate commands support empirical validation. They do not establish model accuracy or historical representativeness: those require expert reading, adjudication and sensitivity analysis. See [the protocol and command examples](docs/research-validation.md).

`iwac-mirror --dataset private` creates typed local `data/iwac_*.parquet` files (`--format csv` for the legacy export) from one pinned Hub revision, with `data/mirror_manifest.json` recording the SHA, row counts and file hashes. Offline readers verify that manifest and refuse an interrupted or mixed-revision mirror.

CI compiles every module, runs `ruff check` for undefined names and dead imports, tests the complete production tree with a 70% `iwac_common` coverage floor, checks dependency consistency, and runs the supported Linux/Windows/Python matrix. It separately runs every installed command's `--help` from a clean wheel install and tests the latest supported dependency ranges. Dependabot tracks Python and GitHub Action updates, while pull requests receive GitHub's dependency review.

## Further documentation

- [Ingestion, privacy and recovery contract](docs/ingest-contract.md) — visibility columns, failure overrides, invalidation and stale-row handling
- [Migration guide](docs/hardening-migration.md) — the current, ordered re-run after a method change
- [LDA model bundles](docs/lda-model-bundles.md) — model identity, the k sweep, held-out evaluation and prediction exports
- [Sentiment panel](docs/sentiment-panel.md) — generations, campaigns and consensus rules
- [Research validation protocol](docs/research-validation.md) — coverage audits, blinded annotation, topic review and reprint calibration
- [Publication readiness review](docs/publication-review.md) — the 2026-09 review and its follow-up status
- [Research backlog](RESEARCH_BACKLOG.md) and [contributing guide](CONTRIBUTING.md)
- Historical: [September 2026 refactoring runbook](docs/refactoring-2026-09-runbook.md), [v1.0.1 release notes](docs/release-1.0.1.md)

## Limitations and caveats

**Public full text is incomplete.** Private parent items are omitted; restricted full text and source properties are masked in the remaining rows. Treat access-related selection bias explicitly. Reviewed derivatives can support some analyses of public items whose OCR is private, but do not recover omitted private parents. Enrichment coverage varies by subset, field, and processing run: new items and failed or inapplicable computations can have missing values. Report the usable row count for each analysis.

**LLM sentiment is non-deterministic and opaque.** The same text sent twice may score differently — measurably so: re-annotating 1,485 articles with `deepseek/deepseek-v4-flash-0731`, which the vendor runs at temperature 1.0, returned a different centrality for 19 of them. A re-run is a fresh reading, not a correction, and the models' reasoning cannot be traced. This is why sentiment runs as a model panel with a published agreement measure and per-model justification columns, rather than as a single score presented as ground truth. Treat disagreement as information about the item, not as noise to be averaged away.

**Metrics keyed to a French or English lexicon mis-score the collection's own material.** `Lisibilite_OCR` applies the French Flesch formula, so it is computed only for rows whose sole language is French and that carry enough text to score, and is null everywhere else, the Ewé, Kabiyè, and Dendi items included. MATTR (`Richesse_Lexicale_OCR`) carries no lexicon and is kept for every language, but its values are not comparable across languages.

**The number of topics is pinned, not swept.** On the smaller subsets C_v coherence cannot choose *k*: seed-to-seed variation exceeds the differences between *k* values. *k* is fixed per subset and language in `CONFIG_PRESETS`; the sweep is a report-only diagnostic. Topic IDs are meaningful only beside `lda_model_name`.

**The Hijri date is a stored convention.** `hijri_*` uses `hijridate` (Umm al-Qura), as the collection's visualisations do. A browser's or Node's `Intl` calendar disagrees with it on most pre-2000 dates, usually by a day, so use the stored columns rather than converting `pub_date` yourself. Day-level lunar aggregates are sensitive to the converter; month-level ones are robust. Only complete `YYYY-MM-DD` dates carry a lunar date.

**Topic-model stopwords are a scholarly choice, not cleanup.** Islamic organisations, religious events, figures and titles are the object of study and must survive into topic labels. The stopword tiers in `lda_topic_modeling/constants.py` filter a fragment such as `al` or `grand` when it stands alone but keep it inside `al_azhar` or `grand_imam`. Adding a stopword changes what the models mean and only takes effect on a re-fit.

## Related repositories

- [iwac-ai-pipelines](https://github.com/fmadore/iwac-ai-pipelines) — LLM-assisted curation upstream in Omeka S: OCR, HTR, NER, summarisation, transcription
- [IwacVisualizations](https://github.com/fmadore/IwacVisualizations) — visualisations built on this dataset

## Citation

The pipeline and the data it produces are separate objects, and which one you cite depends on what your work relies on:

- **This pipeline** — the code in this repository. Cite [`10.5281/zenodo.21805704`](https://doi.org/10.5281/zenodo.21805704), the concept DOI, which always resolves to the newest release. Metadata comes from [`CITATION.cff`](CITATION.cff).
- **The dataset** — cite [`10.57967/hf/9857`](https://doi.org/10.57967/hf/9857). Hugging Face assigns a new DOI per revision and marks the previous one outdated, so check the [dataset page](https://huggingface.co/datasets/fmadore/islam-west-africa-collection) for the DOI matching the revision you loaded.
- **The collection itself** — the underlying archive:

> Madore, Frédérick. *Islam West Africa Collection*. Leibniz-Zentrum Moderner Orient. https://islam.zmo.de/s/westafrica/

## Acknowledgements

One member of the sentiment panel is not a vendor API. `Qwen/Qwen3.8-27B` was served from a self-hosted vLLM instance on *festus*, the cluster of the Bayreuth Centre for High Performance Computing (BZHPC), across one L40S and two H100 shards; the serving harness lives in [iwac-ai-pipelines](https://github.com/fmadore/iwac-ai-pipelines). Everything else the pipeline computes — embeddings, spaCy lemmas, LDA, lexical metrics — ran off the cluster and is not covered by this.

The centre states that its funding acknowledgement is mandatory rather than a courtesy, so any publication whose results depend on the `qwen3_8_27b_*` columns carries it verbatim:

> Calculations were performed using the festus-cluster of the Bayreuth Centre for High Performance Computing (https://www.bzhpc.uni-bayreuth.de), funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - 523317330.

Wording checked against the [festus system guide](https://www.hpc.uni-bayreuth.de/clusters/festus/) on 15 September 2026; re-check it there rather than copying it from here. `CITATION.cff` has no field for a funding acknowledgement under CFF 1.2.0, so it is not recorded there.

## License

[MIT](LICENSE) © 2025-2026 Frédérick Madore

The MIT license covers the pipeline code in this repository only. The dataset is a separate work, recorded in [`CITATION.cff`](CITATION.cff) under CC BY-NC-SA 4.0 (`CC-BY-NC-SA-4.0`); check the [dataset card](https://huggingface.co/datasets/fmadore/islam-west-africa-collection) for the terms of the revision you use. The collection's underlying materials carry their own rights, which vary by item and are recorded in the Omeka S source.
