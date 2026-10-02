# CLAUDE.md

Python pipeline that mirrors the **Islam West Africa Collection (IWAC)** from an
Omeka S archive (https://islam.zmo.de/api) into Hugging Face datasets.

## Canonical ingestion contract

Before changing ingestion or publication, read `docs/ingest-contract.md`,
`iwac_common/schema.py`, and `iwac_common/public_columns.json`. These checked-in
files are the canonical mapping/privacy contract. If the external `iwac-data`
skill is available, keep its mapping reference synchronized as supplementary
documentation; the repository does not require access to that external skill.

## The two-repo split (read this before any push)

`OCR` is private on the Omeka side, so the Hub side is split:

- **`fmadore/islam-west-africa-collection-full`** (private) — the canonical
  target of *every* upload and post-processing script. Complete superset,
  including full text for all rows. `PRIVATE_REPO_ID` in `iwac_common/repos.py`.
- **`fmadore/islam-west-africa-collection`** (public) — the citable projection,
  written **only** by `post-processing/publish_public.py`. Never write to it by
  any other route.

Publication removes private parent items, masks private source properties and
OCR/reconstructive lemma text, and retains explicitly reviewed derivatives.
Missing `item_is_public`, `private_fields`, or required `OCR_is_public` fails
closed. A new public column needs an intentional rights review; do not silence
an unknown-column guard. Private annotation dependencies also restrict their
consensus. See `iwac_common/write_policy.py`.

Every Hub write uses `iwac_common.hub.push_dataset_verified` or the batch
`push_datasets_verified`. Full-data mode verifies that the destination is
private. Public mode independently validates projection policy. Parquet files
and exact card metadata commit together under a server-side parent-SHA
precondition, followed by revision-pinned schema/ID verification. Never call
`Dataset.push_to_hub` from a script or bypass this gateway.

Uploads require authenticated source access and exact count reconciliation.
Mapper/media failures abort by default; explicit overrides preserve affected
baseline data privately until a successful refresh. Changed inputs invalidate
`schema.DERIVED_FROM` dependencies by default. `--preserve-derived` stages a
durable queue before writing, and the next invalidating upload replays it.
Only a verified successful invalidating write clears that queue. Resolve it
before enrichment. References default to dropping absent source rows;
`--stale-rows keep` retains their complete baseline record privately.

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

`publish_public.py --squash` refuses any repo carrying a `doi:` tag (or whose
tags cannot be read): squashing would make the cited revision unreachable.

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
until its stage runs again. Embedding and lemma stages use persisted provenance
to detect incompatible method/input changes in incremental mode; other stages
may require `--update-mode all`. Follow `docs/hardening-migration.md`.

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

**Pushes to one repo should be sequential.** A machine-local lock prevents
local overlap and the server-side parent commit precondition rejects remote
conflicts. Restart from a fresh pinned baseline after a conflict. Never remove
a live writer's lock to proceed.

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

Adding a modeling stopword only takes effect on `--mode fit`; prediction uses
the bundle's frozen preprocessing. Before adding one, check what it costs: a token absent
from the `articles` (press) dictionary but present in `references` is citation
apparatus, which is how `oxford`/`indiana`/`press` were cleared and why
`berlin` and `licence` were not.

**Uploads merge rather than overwrite.** Each upload fetches from Omeka, loads
the existing Hub rows, identifies columns present only on the Hub (the computed
ones), and merges them back on `o:id`. That is what keeps embeddings and topics
alive across an upload. References also drop source-absent rows by default; explicit retention keeps
the full historical row privately.

**Import-smoke tests cannot catch undefined names** used inside `main()` or in a
rarely-taken branch. CI runs `ruff check` for exactly that reason — F821 for the
undefined name a dropped import leaves behind, F401 so the dead imports don't
accumulate (rules pinned in `pyproject.toml`; a deliberate availability check
carries `# noqa: F401`). Without it a refactor passes the tests and crashes at
runtime.

**Caches:** Omeka responses in `.cache_omk*` (gzipped JSON, 24h TTL). Lemma and
embedding resume caches are deleted on a successful push, so a leftover file
means an interrupted run. Both are fingerprinted by the config that produced
them (spaCy model + `LEMMA_LOGIC_VERSION`; embedding model + dim + task) and by
the repository, and every entry carries a hash of its input text
(`_embedding_utils.make_entry`), so a cache from a different configuration,
repo or text version is ignored rather than silently mixed in — no manual
date-checking needed. Bump `LEMMA_LOGIC_VERSION` whenever the lemmatisation
output changes for identical input.

**Never round-trip a whole subset through pandas to fix one column's type.**
`to_pandas()` turns every nullable int column into `float64`; that is how
`lda_topic_id`/`nb_pages`/`hijri_*` reached the public card as floats. Declare
the output types instead (`map_with_progress(..., output_types=...)`), which
also avoids the `datasets` failure when a first batch is all `None` ("Couldn't
cast array of type int64 to null"). The gateway's conform step is the backstop,
not the method.

## Running things

```
.venv\Scripts\python script_name.py
```

Editable and wheel installs expose five console entry points, which are the
preferred way to drive the pipeline:

```
iwac-upload <subset>        # articles|publications|index|references|audiovisual|documents|images
iwac-mirror --dataset private
iwac-publish-public
iwac-process <stage>
iwac-analyze <analysis>
```

`iwac-upload` forwards its remaining flags to the subset's own parser, so
`iwac-upload articles --dry-run --no-cache` works. They are thin wrappers
(`iwac_pipeline/cli.py`) that import the same implementations as modules. Setuptools maps the
historical directories to package names; keep both source and wheel routes tested.

`iwac_common` and `country_mapper` are editable-installed (`pip install -e .
--no-deps`), so they import from any working directory; scripts keep sys.path
fallbacks for uninstalled venvs.

**The local mirror is revision-pinned.** `iwac-mirror` writes typed Parquet
(`--format csv` for the legacy export) and `data/mirror_manifest.json`,
recording the Hub SHA, row counts, and SHA-256 per file;
`load_subset_dataframe(source="local")` (alias `"csv"`) verifies it and refuses
an interrupted or mixed-revision mirror. Deleting the manifest does not make the
files usable again — re-run `iwac-mirror`.

**Hub reads take only the columns they need.** `hub.load_hub_columns` reads the
requested parquet columns at a pinned revision (full-load fallback);
`load_subset_dataframe` uses it whenever `columns=` is given. Report-only
analyses write `<script>.manifest.json` (code SHA, dataset revision, arguments,
output hashes) beside their outputs.

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
`Lisibilite_OCR` (French Flesch) is therefore computed for monolingual French rows
only and nulled everywhere else, in every update mode.

**Join authorities on ids, not titles.** Content subsets carry `*_ids` beside
each linked-authority label column; the index frequency and `entity_networks`
join them to `index.o:id` and fall back to the exact title only for rows
without ids.

**Topic modeling:** preserve domain collocations in `constants.py`. Coherence
is a diagnostic, not a validated quality threshold. Compare seed stability,
source-linked expert interpretation, and sensitivity to corpus composition.

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

## Hardening and research validation

Read `docs/hardening-migration.md` before refreshing legacy outputs. LDA bundle
identity, frozen preprocessing and theta input hashes define topic provenance.
Holdout groups and exact duplicates must be separated before vocabulary or
phrase fitting. Consensus is generation-specific; constant-label reliability
is undefined. Trends are descriptive unless explicit assumptions are selected.

Use `docs/research-validation.md` for coverage/cohort audits, blinded probability
sampling, source-linked topic review and reprint calibration. Automated tests
and model agreement do not supply human gold labels or establish historical
representativeness. Report generation archives full outputs and environments;
keep a clean code commit and the exact input revision for publication.
