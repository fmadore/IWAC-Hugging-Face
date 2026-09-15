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
| [`fmadore/islam-west-africa-collection`](https://huggingface.co/datasets/fmadore/islam-west-africa-collection) | Public | The citable projection. Written **only** by `post-processing/publish_public.py`. |

The projection **masks full text per row rather than stripping it wholesale**. `OCR`, `lemma_text`, and `lemma_nostop` survive wherever `OCR_is_public` is true — a flag derived from the per-value `is_public` attribute on Omeka's `bibo:content` field. Measure text availability in the revision you use; it changes as the archive grows and source permissions change. Reviewed derived fields — embeddings, topics, sentiment and its justifications, `descriptionAI`, lexical metrics — are retained. This allowlist is a publication policy, not a guarantee that derived outputs cannot disclose or quote source information.

Because a leak here would be unrecoverable, `publish_public.py` aborts rather than guessing: if a content subset lacks `OCR_is_public`, or if any column is absent from the per-subset allowlist in [`iwac_common/public_columns.json`](iwac_common/public_columns.json). Adding a legitimately new column means editing that allowlist deliberately.

The uploads carry equivalent rails. Hub baselines fail closed on auth, network, schema, or config errors; a genuinely new config requires `--initialize`. `hub_merge` refuses a frame under 95% of the Hub's current row count; `fetch_items` requires and exactly reconciles the `Omeka-S-Total-Results` header; any mapper or media lookup failure aborts by default. The explicit `--allow-map-failures` and `--allow-media-failures` overrides retain the affected prior Hub row/fields instead of replacing them with blanks.

One rail runs *after* the push instead of before it. `push_to_hub` refreshes a config's byte sizes in the dataset card but not its feature list, so any push that adds or drops a column leaves the card declaring the old schema — and `load_dataset` then raises `CastError: column names don't match`, making the subset unloadable for every consumer, this pipeline's own next run included. It happened twice on 2026-08-06, to the private mirror and then to the public citable dataset. [`iwac_common/hub.py`](iwac_common/hub.py) is now the only write gateway: it validates IDs and embedding dimensions, rejects a changed source revision, acquires a local repo lock, pushes, repairs the card through [`card_sync.py`](iwac_common/card_sync.py), and verifies the exact published revision. That verification is split by cost: `card_sync` compares the card's declared features against the parquet footer on the Hub, and the row-level check then reads only the `o:id` column rather than re-downloading every embedding. If that columnar read is unavailable it falls back to a full reload — never to skipping the check. A test rejects any direct `push_to_hub` call outside this gateway.

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

Content subsets join to `index` authority records, which is what makes entity-level analysis possible across the corpus.

```python
from datasets import load_dataset

articles = load_dataset("fmadore/islam-west-africa-collection", name="articles", split="train")
```

## What the pipeline computes

| Stage | Script | Output |
|-------|--------|--------|
| Semantic embeddings | `post-processing/semantic_embedding.py` | Gemini embeddings over full text, chunked and averaged for long documents |
| Image embeddings | `post-processing/semantic_embedding_images.py` | Embeddings over downscaled images |
| Lemmatisation | `lemmatize_update_hf.py` | spaCy lemmas, with and without stopwords, per language |
| Topic modeling | `post-processing/lda_topic_modeling/` | LDA topic id, probability, label, and top-k terms |
| Lexical metrics | `post-processing/calculate_lexical_richness.py`, `calculate_word_count.py` | Word count, lexical richness, readability |
| Islamic calendar | `post-processing/calculate_hijri_dates.py` | Hijri year, month, and day (Umm al-Qura) |
| Sentiment panel | `iwac_common/sentiment_panel.py` | Registry and consensus helpers for centrality, polarity, and subjectivity annotations imported from Omeka; model inference happens upstream |
| Related items | `post-processing/related_articles.py` | Nearest neighbours by embedding |
| Model agreement | `post-processing/sentiment_agreement.py` | Inter-model agreement across the sentiment panel |

The sentiment panel writes columns keyed by the exact model id, so that no two generations of a model can collide in the same column. Two generations now sit side by side on the Hub, the live one first:

| Generation | Models | Campaign | Subjectivité | Status |
|---|---|---|---|---|
| 2 | `gpt-5.6-luna`, `mistral-small-2603`, `deepseek-v4-flash-0731`, `google/gemma-4-31b-it`, `Qwen/Qwen3.8-27B` | 2026-08 | label (`string`) | Live — use this panel. The first four are complete at 12,298 articles on centralité and polarité; subjectivité is where they diverge. Qwen3.8 joined on 2026-08-25 at 12,098 and stays there: 153 articles were retired after four attempts each, so its shortfall is a finding about the model, not a run to repair |
| 1 | `gemini-3-flash-preview`, `gpt-5-mini`, `ministral-14b-2512` | 2026-01/02 | integer 1–5 (`float64`) | Frozen. The Omeka properties were deleted in 2026-08; the 18 columns remain on the Hub as historical data |

Column order follows that table: `PANEL` in `iwac_common/sentiment_panel.py` is ordered newest-generation-first, and the uploader's `post_merge` hook sorts the sentiment block by it, so the current panel precedes its history rather than trailing it.

A generation boundary is a change of instrument, not a version bump: generation 2 ran a rewritten prompt (fingerprint `d14ace9ac192`) and asked for subjectivité as a label rather than a number, so `{model}_subjectivite_score` is a string in generation 2 and a float in generation 1. Comparisons across the boundary confound the models with the prompt rewrite; `sentiment_agreement.py` therefore takes `--generation` and defaults to the newest. `SUBJECTIVITE_ORDER` maps a label to its 1–5 rank when one scale is needed for both.

Generation 1 survives on the Hub through the merge, not through a copy: the uploader stops emitting a frozen model's columns and `hub_merge` preserves every Hub column the fresh frame does not carry. Emitting those columns empty would overwrite the values without changing the row count, so no guard would fire — which is why `tests/test_sentiment_panel.py` covers that pair directly.

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
data/               fetch_datasets.py — local CSV mirrors for offline work
```

## Installation

Requires **Python >= 3.12**. CI tests Python 3.12 on Linux and Windows and Python 3.13 on Linux. Development is CPU-only throughout; the pipeline deliberately prefers CPU-viable models such as spaCy's `*_lg` pipelines over transformer equivalents.

Windows PowerShell:

```powershell
git clone https://github.com/fmadore/IWAC-Hugging-Face.git
Set-Location IWAC-Hugging-Face
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
Copy-Item .env.example .env
```

Linux or macOS:

```bash
git clone https://github.com/fmadore/IWAC-Hugging-Face.git
cd IWAC-Hugging-Face
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
cp .env.example .env
```

For development, install `requirements-dev.txt` instead; it includes the runtime dependencies, pytest, coverage, and the undefined-name check:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest
```

The editable install exposes `iwac-upload`, `iwac-mirror`, and `iwac-publish-public` from the checkout. Keep the checkout in place: these commands load scripts at their repository paths. Standalone wheel installation is not currently supported, and runtime dependencies must be installed separately as above.

The lemmatisation step additionally needs spaCy models:

```bash
python -m spacy download fr_core_news_lg
python -m spacy download en_core_web_lg
```

## Configuration

Copy `.env.example` to `.env` and fill in `OMEKA_BASE_URL`, `OMEKA_KEY_IDENTITY`, `OMEKA_KEY_CREDENTIAL`, `HF_TOKEN`, and — for the embedding scripts — `GOOGLE_API_KEY`.

Set `IWAC_HF_PRIVATE_REPO` and `IWAC_HF_PUBLIC_REPO` to redirect the pipeline at a scratch dataset. Do this before running anything that writes to the Hub for the first time.

## Usage

The original script paths remain supported. The installed commands give the common operations one discoverable surface:

```bash
iwac-upload --help
iwac-mirror --help
iwac-publish-public --help
```

`iwac-upload` accepts `articles`, `publications`, `index`, `references`, `audiovisual`, `documents`, or `images` as its subset argument.

The flow runs in three stages, in order:

```bash
# 1. Upload — fetch from Omeka S, merge into the private repo
iwac-upload articles --dry-run
iwac-upload articles

# 2. Post-process — compute derived columns on the private repo
python post-processing/calculate_word_count.py --update-mode empty

# 3. Publish — project the private repo into the public one
iwac-publish-public --dry-run
iwac-publish-public
```

Two properties of this flow are easy to get wrong:

**Pushes to one repo must be sequential.** The writer enforces this locally with a repo-scoped lock and rejects a Hub revision that changed after computation began. That prevents overlapping processes on one machine and detects most remote lost updates; it is still good operational practice to finish one job before starting another. A lock left behind by a crashed process on this host is reclaimed automatically on the next run; one whose owner is still alive, or which was written by another machine, fails closed — wait for that writer rather than deleting the lock file.

**Uploads merge rather than overwrite.** Each upload fetches from Omeka, loads the existing Hub rows, identifies columns that exist only on the Hub, and merges them back on `o:id`. That is what keeps embeddings and topics alive across a re-upload rather than blanking them.

Post-processing scripts share a `--update-mode` flag: `empty` fills only missing values (the cheap default for incremental runs), `all` recomputes every row. **Changing a computation does nothing to published data until you re-run its script with `--update-mode all`** — a method change without a re-run silently leaves the old values in place.

## Reproducibility

For a publication, record the code commit, the dataset repository and full revision SHA, the exact command arguments, and the installed dependency and spaCy model versions. Archive the model artifacts and analysis outputs used for the paper. The dependency ranges in `requirements.txt` support maintenance; they are not a frozen research environment. See [the publication review](docs/publication-review.md) for remaining release work.

Pin dataset reads to the revision used in the paper:

```python
articles = load_dataset(
    "fmadore/islam-west-africa-collection",
    name="articles", split="train", revision="<full dataset commit SHA>",
)
```

Topic models use a fixed seed (42), write their parameters to `training_parameters.json`, and record coherence metrics alongside the model. Omeka responses are cached atomically in `.cache_omk*` for 24 hours; cache keys include the API host and credential identity so staging/public responses cannot be confused with production/private ones. Lemma and embedding resume caches are fingerprinted by the configuration that produced them — spaCy model plus `LEMMA_LOGIC_VERSION`, embedding model plus dimension and task — so a cache written under a different configuration is ignored rather than silently mixed in. These caches are deleted on a successful push, which means a leftover cache file is a reliable signal of an interrupted run.

`iwac-mirror --dataset private` creates the local `data/iwac_*.csv` files from one pinned Hub revision. Files are staged first and `data/mirror_manifest.json` records the repository SHA, row counts, and SHA-256 hashes. Offline consumers verify that manifest and refuse an interrupted or mixed-revision mirror.

CI compiles every module, rejects undefined names, runs the unit/contract/import-smoke suite with a 70% `iwac_common` coverage floor, executes `pip check`, and tests the supported Linux/Windows/Python matrix. Dependabot tracks both Python and GitHub Action updates, while pull requests receive GitHub's dependency review.

## Limitations and caveats

**Public full text is incomplete.** Full text is masked per row by the access status of the source item, so analyses requiring that text cover a subset of the material. Treat access-related selection bias explicitly. The public projection retains the private mirror's rows and reviewed derived columns, allowing some analyses to include source-private items. Enrichment coverage still varies by subset, field, and processing run: new items and failed or inapplicable computations can have missing values. Report the usable row count for each analysis.

**LLM sentiment is non-deterministic and opaque.** The same text sent twice may score differently — measurably so: re-annotating 1,485 articles with `deepseek-v4-flash-0731`, which the vendor runs at temperature 1.0, returned a different centrality for 19 of them. A re-run is a fresh reading, not a correction, and the models' reasoning cannot be traced. This is why sentiment runs as a model panel with a published agreement measure and per-model justification columns, rather than as a single score presented as ground truth. Treat disagreement as information about the item, not as noise to be averaged away.

**Metrics keyed to a French or English lexicon mis-score the collection's own material.** Readability and lexical-richness measures have no valid reading for the Ewé, Kabiyè, and Dendi items. Those are scored null rather than low: a metric that ranks correctly transcribed African-language sources as garbage is worse than no metric.

**The number of topics is pinned, not swept.** On the smaller subsets, C_v coherence cannot choose *k* — a three-seed sweep on `references` placed every *k* from 12 to 32 within 0.014 mean C_v while a single *k* varied by up to 0.035 across seeds, so successive re-fits each produced a confident-looking but different "best k". Because *k* defines what `lda_topic_id` means, an auto-sweep would renumber every topic on each re-fit. *k* is therefore fixed per language in `CONFIG_PRESETS` and judged by multi-seed stability and documents-per-topic instead.

**The Hijri converter is a compatibility contract.** `calculate_hijri_dates.py` uses `hijridate` (Umm al-Qura) because the collection's visualisation pipeline does. Measured on the live `articles` subset, the ICU tables behind a browser's or Node's `Intl` disagree with it on 75% of pre-2000 dates and none from 2000 onward. Storing the lunar date as a column rather than deriving it per consumer is what keeps the website, the MCP server, and any notebook in agreement. Day-level lunar aggregates are sensitive to this choice; month-level ones are robust, as only 0.86% of articles shift lunar month.

**Topic-model stopwords are a scholarly choice, not cleanup.** Islamic organisations, religious events, figures, and titles are the object of study and must survive into topic labels. The stopword tiers in `lda_topic_modeling/constants.py` are ordered so that a fragment like `al` is filtered when it stands alone but preserved inside `al_azhar` or `dar_al_hadith`. Adding a stopword changes what the models mean and only takes effect on a re-fit.

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

One member of the sentiment panel is not a vendor API. `Qwen/Qwen3.8-27B` was served from a self-hosted vLLM instance on *festus*, the cluster of the Bayreuth Centre for High Performance Computing (BZHPC), across one L40S and two H100 shards; the serving harness lives in [iwac-ai-pipelines](https://github.com/fmadore/iwac-ai-pipelines). Everything else the pipeline computes — Gemini embeddings, spaCy lemmas, LDA, lexical metrics — ran off the cluster and is not covered by this.

The centre states that its funding acknowledgement is mandatory rather than a courtesy, so any publication whose results depend on the `qwen3_8_27b_*` columns carries it verbatim:

> Calculations were performed using the festus-cluster of the Bayreuth Centre for High Performance Computing (https://www.bzhpc.uni-bayreuth.de), funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - 523317330.

Wording checked against the [festus system guide](https://www.hpc.uni-bayreuth.de/clusters/festus/) on 15 September 2026; re-check it there rather than copying it from here. `CITATION.cff` has no field for a funding acknowledgement under CFF 1.2.0, so it is not recorded there.

## License

[MIT](LICENSE) © 2025-2026 Frédérick Madore

The license covers the pipeline code in this repository. The collection's underlying materials carry their own rights, which vary by item and are recorded in the Omeka S source.
