# Refactoring pass, September 2026 — what to run after merging

Code changes never move data. This branch changes how columns are typed,
computed and joined, but the Hub keeps its current values until the owning
script runs again. This page lists what changes, which runs apply it, and
what can only be done by hand.

## What changed, by effect on the dataset

| Change | Takes effect on the Hub when |
|---|---|
| Canonical types: nullable `int64` for `nb_pages`, `hijri_*`, `pub_year`, `nb_mots`, `lda_topic_id`, `frequency`, `duration_seconds` and the generation-1 subjectivité scores; `list<float32>` embeddings (≈ ⅓ smaller parquet) | the next push of each subset, by any script — the write gateway conforms every push |
| New columns: `*_ids` (authority ids), `pub_year`, `pub_date_precision`, `references.abstract_en`, `latitude`/`longitude` (`index`, `images`) | the next upload of each subset |
| Stable column order (the Hub's order is kept; new columns slot in beside their mapper neighbour) | the next upload |
| `index.frequency` joined on authority ids; multi-country rows counted per country; unparseable dates excluded from `first_occurrence`/`last_occurrence` | the next `index` upload, **after** the content subsets carry `*_ids` |
| `nb_mots` single definition (references, audiovisual changed; the other subsets already used it) | the next references/audiovisual upload |
| `Lisibilite_OCR` only for primary-French rows | `calculate_lexical_richness.py --update-mode all`, per subset |
| references bibliographic numbers kept verbatim (`"iv, 250"` was blanked) | the next references upload |

## Recommended order

1. Refresh the editable install: `pip install -e . --no-deps`.
2. Dry-run each upload first and read two things: the "New column(s) not on
   the Hub yet" line (should list only the columns above) and any
   "Source text changed since the Hub copy" table.

   ```
   iwac-upload articles --dry-run
   ```

3. Upload the content subsets, then `index` last (its frequency pass reads the
   other subsets' new `*_ids` from the Hub):

   ```
   iwac-upload articles
   iwac-upload publications
   iwac-upload documents
   iwac-upload references
   iwac-upload audiovisual
   iwac-upload images
   iwac-upload index
   ```

   If a dry run reported changed source text, decide per subset between
   `--invalidate-derived` (clear the stale values, then re-run the affected
   stages in `missing` mode) and keeping them (they are recorded in
   `.iwac_state/stale_derived/`).
4. Re-score readability, one subset at a time (pushes must stay sequential):

   ```
   python post-processing/calculate_lexical_richness.py --config articles --update-mode all
   ```

   Repeat for `publications`, `documents`, `references` and `audiovisual`.
   Word counts and Hijri dates need no re-run.
5. Publish the projection. The new columns are already allowlisted in
   `iwac_common/public_columns.json`; the plan table should show no surprises.

   ```
   iwac-publish-public --dry-run
   iwac-publish-public
   ```

6. Refresh any local mirror: `iwac-mirror --dataset private` (now Parquet).

## By hand — outside this repository

- **Dataset card** (`README.md` on both Hub repos; `card_sync` only repairs the
  feature list):
  - document the new columns listed above;
  - replace `float64` with `int64` for the integer columns, and describe the
    embeddings as float32 sequences;
  - `index.frequency`: counts *items* in articles, publications, references and
    audiovisual, joined on authority id (the card currently omits
    audiovisual);
  - `first_occurrence`/`last_occurrence`: the earliest/latest parseable
    `pub_date`, whose precision varies (not always `YYYY-MM-DD`);
  - `Lisibilite_OCR`: French rows only, null elsewhere.
- **`iwac-data` skill**: update `references/omeka-to-hf-mapping.md` with the new
  columns, the `*_ids` join and the canonical types (CLAUDE.md requires it; the
  skill does not live in this repository).
- **Consumers** (IwacVisualizations, the MCP server): embeddings arrive as
  float32; integer columns as `int64` with nulls instead of `float64`; the
  `entity_networks` node `Id` is now the authority `o:id` (the label is in
  `Label`); the local mirror is Parquet unless `--format csv` is passed.
- **DOI**: this pass is a reasonable release point once step 5 is done and
  the projection has been checked as described in CLAUDE.md.

## Not done in this pass

- Packaging the upload scripts as importable modules (review finding 5).
- A locked, archived publication environment (finding 6).
- Atomic multi-subset publication (finding 8); public commits now name their
  private source revision, which makes a partial publish traceable.
- A long-format sentiment config, and native list columns in place of the
  pipe-joined label columns — both are schema changes consumers must opt into.
