# CLAUDE.md

Python pipeline that mirrors the **Islam West Africa Collection (IWAC)** from an
Omeka S archive (https://islam.zmo.de/api) into Hugging Face datasets.

## Load the `iwac-data` skill first

Before touching anything that reads, transforms, or pushes the dataset, load the
**`iwac-data` skill**. It is the single source of truth for per-subset schemas,
field mappings, Omeka resource classes, sentiment field shapes, the authority
join, and the end-to-end Omeka → HF flow. Do not re-derive any of that here — and
if the pipeline changes (new computed column, new subset, changed class ID),
update the skill's `references/omeka-to-hf-mapping.md` to match.

## The two-repo split (read this before any push)

`OCR` is private on the Omeka side, so the Hub side is split:

- **`fmadore/islam-west-africa-collection-full`** (private) — the canonical
  target of *every* upload and post-processing script. Complete superset,
  including full text for all rows. `PRIVATE_REPO_ID` in `iwac_common/repos.py`.
- **`fmadore/islam-west-africa-collection`** (public) — the citable projection,
  written **only** by `post-processing/publish_public.py`. Never write to it by
  any other route.

The projection **masks full text per row**, it does not blanket-strip it.
`OCR` / `lemma_text` / `lemma_nostop` survive for rows where `OCR_is_public` is
true (the per-value `is_public` flag on `bibo:content`, emitted by every upload
mapper via `is_content_public()`). Everything else — embeddings, LDA, sentiment
and justifications, `descriptionAI`, metrics — is always projected.

Two guards exist because a leak here is unrecoverable: `publish_public.py` aborts
if a content subset lacks `OCR_is_public`, and aborts on any column missing from
the per-subset allowlist in `iwac_common/public_columns.json`. If a new column is
legitimate, add it to the allowlist deliberately — never silence the guard.

## Non-obvious gotchas

**Code changes never move data.** Editing a computation does nothing to the Hub
until you re-run the script that owns that column, usually with
`--update-mode all` (`--mode all` for `lemmatize_update_hf.py`). A method change
without a re-run leaves the old values in place, silently.

**Pushes to one repo must be sequential.** `push_to_hub` rewrites the whole
config and the shared README metadata. Two scripts pushing concurrently — even to
different subsets — will clobber each other's columns via lost update. Finish one
before starting the next.

**Uploads merge rather than overwrite.** Each upload fetches from Omeka, loads
the existing Hub rows, identifies columns present only on the Hub (the computed
ones), and merges them back on `o:id`. That is what keeps embeddings and topics
alive across an upload. `reference/` merges `how="outer"` — the others use
`"left"`.

**Import-smoke tests cannot catch undefined names** used inside `main()` or in a
rarely-taken branch. CI runs `pyflakes` for exactly that reason; a refactor that
drops an import will otherwise pass tests and crash at runtime.

**Caches:** Omeka responses in `.cache_omk*` (gzipped JSON, 24h TTL); lemma and
embedding resume caches are deleted on a successful push, so a leftover cache
file means a previous run was interrupted — check it predates no relevant code
change before resuming.

## Running things

```
.venv\Scripts\python script_name.py
```

`iwac_common` and `country_mapper` are editable-installed (`pip install -e .
--no-deps`), so they import from any working directory; scripts keep sys.path
fallbacks for uninstalled venvs.

Required in `.env`: `OMEKA_BASE_URL`, `OMEKA_KEY_IDENTITY`,
`OMEKA_KEY_CREDENTIAL`, `HF_TOKEN`, and `GOOGLE_API_KEY` for the embedding
scripts. `IWAC_HF_PRIVATE_REPO` / `IWAC_HF_PUBLIC_REPO` optionally redirect the
repo IDs at `iwac_common/repos.py` — useful for testing against a scratch repo.

Development is **CPU only**. Prefer CPU-viable models (spaCy `*_lg`, not
transformers) and keep batch sizes realistic for it.

## Conventions

Console output uses `rich` — `RichHandler` for logging, `Progress` for long
loops, `Panel`/`Table` for structured output, and the `✓ ⚠ ✗ → ℹ` status icons.
Match the surrounding scripts rather than inventing a new presentation.

Upload scripts are `async`/`aiohttp` over a shared `ConnectionManager` singleton,
with exponential backoff via `async_retry`.

Comments may be English or French (the codebase uses both); new console messages
and documentation are English.

## Digital humanities guidelines

This is a research dataset on Islam in West Africa, and the analytical choices
carry scholarly weight.

**Never treat domain vocabulary as noise.** Islamic organizations (COSIM, FAIB,
UIB), religious events (Ramadan, Tabaski, Maouloud), religious figures and titles
must survive into topic labels, analyses, and visualizations — they *are* the
object of study. Only strip genuine noise: OCR artifacts, generic boilerplate,
English stopwords in French documents.

**Watch for metrics that penalise the collection's own material.** Anything
keyed to a French or English lexicon will mis-score the Ewé, Kabiyè, and Dendi
items. Score them as null rather than as low quality; a metric that ranks
correctly-transcribed African-language sources as garbage is worse than no metric.

**Topic modeling:** ~30 topics for ~12K documents; prioritise C_v coherence
(≥ 0.5 is good); domain collocations are forced in `constants.py`.

**Reproducibility:** fixed seed 42, parameters saved to `training_parameters.json`,
coherence metrics recorded. Report-only analyses write to `analyses/output/`
(gitignored) and never add Hub columns.
