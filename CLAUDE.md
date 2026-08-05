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

The uploads have their own rails, and the same rule applies: an abort is
information, not an obstacle. `hub_merge` refuses a frame under 95% of the Hub's
row count (`--force-shrink`), `fetch_items` raises on a short read against the
`Omeka-S-Total-Results` header, and a mass media-lookup failure aborts
(`--allow-media-failures`). That last one exists because the others guard row
*count*: when the Omeka host is unreachable, every media sub-fetch fails,
`PDF`/`thumbnail` come back empty, and the merge overwrites good Hub URLs with
blanks while the row count stays identical. If it fires, fix connectivity —
overriding it is almost never right.

## DOIs: mint at release points, never on every update

The dataset DOI is minted on the Hub, from the **public** repo's settings, and
**never** on `-full`. Hugging Face states the action plainly: it *cannot be
undone*, and the repo can no longer be deleted, renamed, transferred, or made
private.

That last clause is the reason this has its own section. Flipping the public
dataset private is the emergency lever if a leak is ever found; minting removes
it permanently. So the order is fixed:

1. Check the published projection first: load each content subset from the
   public repo and confirm no row flagged `OCR_is_public = false` still carries
   `OCR` / `lemma_text` / `lemma_nostop`. The `publish_public.py` guards cover
   what gets *written* and say nothing about what is already *there*. Verified
   clean on 2026-08-05: 14,797 rows, 5,914 source-private, none leaking.
2. Only then mint, and only on `fmadore/islam-west-africa-collection`.

**Mint at deliberate release points, not on every pipeline push.** The Hub has
no concept DOI: each "Generate new DOI" supersedes the last and marks it
outdated, so a DOI per update produces a trail of stale identifiers and citations
that resolve to a superseded revision. Pick a state worth citing — a completed
enrichment pass, not an incremental column refresh — and mint that.

The **code** DOI is a separate object: Zenodo, on the GitHub repo, whose concept
DOI always resolves to the newest version. That is what the commented block in
`CITATION.cff` is reserved for. The dataset DOI belongs on the dataset card.

## Non-obvious gotchas

**Code changes never move data.** Editing a computation does nothing to the Hub
until you re-run the script that owns that column, with `--update-mode all` —
uniform across every script since the Tier D CLI unification, `lemmatize`
included (its `--mode all|empty` still works but is the legacy alias). A method
change without a re-run leaves the old values in place, silently.

**The Hijri converter is a compatibility contract, not an implementation
detail.** `calculate_hijri_dates.py` uses `hijridate` (Umm al-Qura) because
IwacVisualizations' `generate_on_this_day.py` does, and measured on the live
`articles` subset the ICU tables behind a browser's or Node's `Intl` disagree
with it on **75 % of pre-2000 dates** (2,365 of 3,152) and on none from 2000 on.
That is the reason the lunar date is a stored column rather than something each
consumer derives: the website's day buckets, the MCP server's lunar tools and
any notebook now agree by construction. Swapping the converter would silently
re-file thousands of 1960s–90s items. Only 0.86 % of articles change lunar
*month*, so month-level aggregates are robust either way — day-level ones are
not. Not computed for `references`: an academic imprint date has no meaningful
lunar reading.

**Pushes to one repo must be sequential.** `push_to_hub` rewrites the whole
config and the shared README metadata. Two scripts pushing concurrently — even to
different subsets — will clobber each other's columns via lost update. Finish one
before starting the next.

**LDA stopwords come in tiers, and the tier decides the outcome.** Which set a
word goes in matters more than whether it is in one at all:

- `DOMAIN_STOPWORDS` — stripped *before* gensim's phrase detection, so nothing
  here can ever appear inside a compound. Right for digitisation noise
  (`camscanner`) and for citation apparatus whose whole family you want gone:
  `paris` alone anchored 32 compounds (`paris_cnrs`, `afrique_noir_paris`).
  Wrong for a fragment like `al`, which is junk alone but carries 82 compounds
  (`al_qadr`, `al_qaïda`, `al_azhar`, `dar_al_hadith`) — the religious events,
  organisations and titles the collection exists to study.
- `FRAGMENT_STOPWORDS` — filtered one pass *after* phrasing, so `al`/`el`/`page`
  vanish alone and survive in `al_azhar`, `el_hadj`, `page_facebook`.
- `JUNK_COMPOUND_STOPWORDS` — the mirror case: whole phrases that are apparatus
  though each part is legitimate (`university_press` goes, `university_medina`
  stays). Never put a bare word here; it would veto nothing but itself.
- `ARTIFACT_LABEL_STOPWORDS` — vetoes a label word-by-word, so compounds from
  models trained before a fix stop surfacing pending a re-fit.

Adding a modeling stopword only takes effect on `--mode fit`; `--mode predict`
refreshes labels alone. Before adding one, check what it costs: a token absent
from the `articles` (press) dictionary but present in `references` is citation
apparatus, which is how `oxford`/`indiana`/`press` were cleared and why
`berlin` and `licence` were not.

**Uploads merge rather than overwrite.** Each upload fetches from Omeka, loads
the existing Hub rows, identifies columns present only on the Hub (the computed
ones), and merges them back on `o:id`. That is what keeps embeddings and topics
alive across an upload. `reference/` merges `how="outer"` — the others use
`"left"`.

**Import-smoke tests cannot catch undefined names** used inside `main()` or in a
rarely-taken branch. CI runs `pyflakes` for exactly that reason; a refactor that
drops an import will otherwise pass tests and crash at runtime.

**Caches:** Omeka responses in `.cache_omk*` (gzipped JSON, 24h TTL). Lemma and
embedding resume caches are deleted on a successful push, so a leftover file
means an interrupted run. Both are fingerprinted by the config that produced
them (spaCy model + `LEMMA_LOGIC_VERSION`; embedding model + dim + task), so a
cache from a different configuration is ignored rather than silently mixed in —
no manual date-checking needed. Bump `LEMMA_LOGIC_VERSION` whenever the
lemmatisation output changes for identical input.

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

**C_v cannot choose k on the small subsets — do not let it.** On `references`
a 3-seed sweep put every k from 12 to 32 within 0.014 mean C_v while a single
k varied by up to 0.035 across seeds, so the ranking is pure noise: four
successive re-fits picked 24, 16, 24 and 32, each time with a confident-looking
"best k". k is therefore pinned in `CONFIG_PRESETS` (`num_topics`, per language
via `language_overrides`) rather than swept, because k defines what
`lda_topic_id` means and an auto-sweep renumbers every topic on each re-fit.
Judge k by the multi-seed *stability* score (mean best-match Jaccard between
seeds, `--stability-seeds`) and by documents-per-topic; it falls systematically
as k rises where C_v does not. Re-check only if a corpus grows substantially.

**Reproducibility:** fixed seed 42, parameters saved to `training_parameters.json`,
coherence metrics recorded. Report-only analyses write to `analyses/output/`
(gitignored) and never add Hub columns.
