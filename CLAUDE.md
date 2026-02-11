# CLAUDE.md

## Project Overview

Python scripts to manage the **Islam West Africa Collection (IWAC)** dataset on Hugging Face Hub.

- **Dataset**: https://huggingface.co/datasets/fmadore/islam-west-africa-collection
- **Source**: Omeka S API at https://islam.zmo.de/api

Dataset subsets:
- `articles` — Newspaper articles (resource_class_id = 36)
- `audiovisual` — Audiovisual documents (resource_class_id = 38)
- `documents` — General documents
- `publications` — Islamic publications
- `references` — Academic references
- `index` — Index entries

## Architecture

### Upload Scripts

Each subset has an upload script that fetches from Omeka S API, maps fields to flat columns, merges with the existing HF dataset (preserving computed columns), and pushes to Hub.

- `articles/upload_newspaper_hf.py` — Articles (main template)
- `audiovisual/upload_audiovisual_hf.py`
- `document/upload_documents_hf.py`
- `index/upload_index_hf.py`
- `islamic-publications/upload_Islamic_publications_hf.py`
- `reference/upload_reference_hf.py`

AI sentiment analysis (Gemini, ChatGPT, Mistral) is fetched directly from the Omeka API in upload scripts — there is no separate sentiment computation step.

### Post-Processing Scripts

Scripts that enrich the dataset with computed columns:
- `lemmatize_update_hf.py` — French lemmatization with spaCy
- `post-processing/calculate_sentiment.py` — Transformer-based sentiment (distilcamembert)
- `post-processing/calculate_lexical_richness.py` — Text statistics
- `post-processing/calculate_word_count.py` — Word counts
- `post-processing/semantic_embedding.py` — Sentence embeddings
- `post-processing/topic_modeling/` — Shared topic modeling utilities (stopwords, patches)
- `post-processing/lda_topic_modeling/` — LDA topic modeling (gensim)

### Utilities

- `country_mapper.py` — Maps newspaper names to countries (Benin, Burkina Faso, Côte d'Ivoire, Niger, Togo)
- `data/fetch_datasets.py` — Download datasets locally

## Running Scripts

Always use the project's virtual environment:
```
.venv\Scripts\python script_name.py
```

## Environment Variables

Required in `.env`:
```
OMEKA_BASE_URL=https://islam.zmo.de/api
OMEKA_KEY_IDENTITY=your_key
OMEKA_KEY_CREDENTIAL=your_credential
HF_TOKEN=your_huggingface_token
```

## Hardware Constraints

- Development is **CPU only** (no GPU)
- Use CPU-optimized models: spaCy `fr_core_news_lg` (not transformer models), lightweight embedding models
- Consider batch sizes and model complexity for CPU performance

## Code Style

### Always use Rich for console output

Use `rich` instead of plain `print` or basic logging. Follow the pattern in `articles/upload_newspaper_hf.py`:
- `RichHandler` for logging
- `Progress` bars with `SpinnerColumn`, `BarColumn`, `TaskProgressColumn`, `TimeElapsedColumn`
- `console.status()` for indeterminate operations
- `Panel` for important info, `Table` for structured data
- Status icons: `[green]✓[/green]`, `[yellow]⚠[/yellow]`, `[red]✗[/red]`, `[blue]→[/blue]`, `[yellow]ℹ[/yellow]`

### Async patterns

Upload scripts use `async`/`await` with `aiohttp`. Shared `ConnectionManager` singleton manages the HTTP session. `async_retry` decorator provides exponential backoff.

### Caching

API responses cached in `.cache_omk*` (gitignored) as gzipped JSON, 24h TTL.

### Dataset merge logic

1. Fetch fresh data from Omeka API
2. Load existing dataset from HF Hub
3. Identify columns only in existing dataset (computed columns)
4. Merge on `o:id`, keeping computed columns for existing items

## Omeka Field Mappings

Core fields:
- `o:id` → `o:id` (primary key)
- `dcterms:title` → `title`
- `dcterms:creator` → `author`
- `dcterms:date` → `pub_date`
- `dcterms:publisher` → `newspaper`
- `dcterms:subject` → `subject` (pipe-separated)
- `dcterms:language` → `language`
- `dcterms:spatial` → `spatial`
- `bibo:content` → `OCR`
- `fabio:hasURL` → `URL`

AI sentiment fields (for each model: gemini, chatgpt, mistral):
- `iwac:{model}Centralite` → `{model}_centralite_islam_musulmans`
- `iwac:{model}CentraliteJustification` → `{model}_centralite_justification`
- `iwac:{model}Polarite` → `{model}_polarite`
- `iwac:{model}PolariteJustification` → `{model}_polarite_justification`
- `iwac:{model}SubjectiviteScore` → `{model}_subjectivite_score`
- `iwac:{model}SubjectiviteJustification` → `{model}_subjectivite_justification`

## Dependencies

Key libraries: `datasets`, `huggingface_hub`, `aiohttp`, `aiofiles`, `rich`, `pandas`, `pyarrow`, `spacy`, `gensim`

## Digital Humanities Guidelines

This is a research dataset studying Islam in West Africa. Follow these principles:

### Domain relevance
- **Never filter out domain-specific terms** as stopwords: Islamic organizations (COSIM, FAIB, UIB), religious events (Ramadan, Tabaski, Maouloud), religious figures and titles
- These terms are core to the research — they must appear in topic labels, analyses, and visualizations
- Only remove noise: OCR artifacts, English stopwords from non-French docs, generic boilerplate

### Topic modeling
- Target ~30 topics for ~12K documents (use `--optimize-topics` to sweep)
- Prioritize topic coherence (C_v metric, ≥ 0.5 is good)
- Keep Islamic/religious terminology in topic labels
- Custom collocations in `constants.py` force domain-specific multi-word joins

### Reproducibility
- Save parameters in `training_parameters.json`
- Document coherence metrics
- Use fixed random seeds (42)

## Language

- Code comments: English or French (codebase uses both)
- Console messages: English preferred for new code
- Documentation: English
