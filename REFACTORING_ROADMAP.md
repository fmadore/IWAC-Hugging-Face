# IWAC Refactoring Roadmap

**Branch:** `claude/refactor-codebase-YP8F3`
**Goal:** Reduce duplication, improve maintainability, preserve every existing function. No behavioral changes.
**Estimated total reduction:** ~930 lines (~12%) across 6 upload scripts + 5 post-processing scripts.

---

## Guiding principles

1. **No functional regressions.** Every script must produce the same dataset rows and columns it produces today.
2. **One concern per PR-sized commit.** Each tier below is independently shippable and reviewable.
3. **Subset-specific quirks stay in subset scripts.** Shared code goes in `iwac_common/` (uploads) and `post-processing/_common.py` (post-processing).
4. **Verify before extracting.** When two scripts disagree (e.g. `reference` uses `how='outer'` while others use `how='left'`), confirm the difference is intentional and preserve it via parameters — don't normalize silently.
5. **CPU-only constraints stay.** No heavier models or GPU-only deps.
6. **Rich console standard preserved.** CLAUDE.md mandates Rich; bring `audiovisual` and `index` (currently plain `logging`) in line as a side benefit, not a forced rewrite.

---

## Verified findings (with file:line references)

### Upload scripts — confirmed duplicates

| Component | articles | audiovisual | document | islamic-publications | index | reference |
|---|---|---|---|---|---|---|
| `Cache` class | L93 | L76 | L88 | L97 | L77 | L120 |
| `ConnectionManager` | L128 | L111 | L123 | L132 | L112 | L155 |
| `async_retry` | L152 | L135 | L147 | L156 | L136 | L179 |
| `OmekaApiClient` | L172 | L155 | L167 | L176 | L156 | L200 |
| `Config` dataclass | L78 | L61 | L73 | L82 | L62 | L78 |
| Date extraction (`o:created` → `added_date`) | L368-376 | L311-318 | L342-350 | L346-355 | — | L446-454 |
| `pd.merge` on `o:id` | L536 | L424 | L493 | L557 | L709 | L613 |

### Differences that must be preserved

- **`reference/upload_reference_hf.py:613`** uses `how='outer'` with `suffixes=('', '_old')`. The other 5 use `how='left'`. The shared merge helper must accept a `how` parameter and a `suffixes` parameter.
- **Cache directories differ**: `.cache_omk` (articles & islamic-publications — note: same dir, possibly accidental shared cache), `.cache_omk_audiovisual`, `.cache_omk_documents`, `.cache_omk_references`, `.cache_omk_index`. Preserved as `Config.CACHE_DIR` per script.
- **Resource class IDs**: articles=36, audiovisual=38, document=49, islamic-publications=60. `reference` loops over 9 IDs (35, 43, 88, 40, 82, 178, 52, 77, 305). `index` loops over 5 IDs (9, 94, 96, 54, 244).
- **Progress bar style**: 4 scripts use Rich `Progress`; `audiovisual` and `index` use `tqdm`. Standardizing on Rich is desirable but optional — first pass should keep behavior identical.
- **Sentiment fields**: only `articles` and `islamic-publications` extract the gemini/chatgpt/mistral × 6-field block. `audiovisual`, `document`, `reference`, `index` do not.
- **Country logic**: 5 different mechanisms (newspaper-name lookup, hardcoded "Nigeria", item-set→country dict, etc.). These stay in the subset scripts.

### Post-processing — confirmed duplicates

| Pattern | lemmatize_update_hf | calculate_lexical_richness | calculate_word_count | semantic_embedding | lda_topic_modeling |
|---|---|---|---|---|---|
| HF auth (`get_token` → `login`) | L182-185 | L356-370 | L342-352 | L180+ | similar |
| `load_dataset` | L199 | L392 | L367 | L517 | L229 |
| `push_to_hub` | L261 | L553 | L475 | L760 | L469 |
| Config picker (interactive) | hardcoded | L372-383 | L327-336 | L180+ | similar |

`calculate_lexical_richness.py` and `calculate_word_count.py` (586 + 580 lines) share ~80–90% scaffold with different per-row functions.

---

## Tiered plan

Each tier is one or more focused commits. Tiers are ordered by impact-to-risk ratio. Tier N+1 does not depend on N being merged, but they're easiest to review in order.

### Tier 1a — Extract Omeka infrastructure (~400 lines saved) ⭐ start here

**Why first:** Pure infrastructure move. No field-shape changes. The lowest-risk extraction; validates the `iwac_common/` package approach.

**Create `iwac_common/__init__.py`** — empty/docstring.

**Create `iwac_common/omeka_client.py`** containing:
- `Config` dataclass (with `CACHE_DIR` parameter so each subset can keep its dir)
- `Cache` class (gzip JSON, 24h TTL — verbatim)
- `ConnectionManager` singleton + module-level `conn_manager`
- `async_retry` decorator
- `OmekaApiClient` base class with `_get`, `request`, `fetch_items_page`, `fetch_items` (Rich progress), `fetch_media_data`

**Migrate the 6 upload scripts** to import from `iwac_common.omeka_client` instead of defining their own. Keep each script's `Config` instance with the right `CACHE_DIR`. Keep the script-local `logger` so log messages still attribute correctly.

**Acceptance:**
- All 6 scripts import successfully.
- `python -c "import articles.upload_newspaper_hf"` etc. (or the import block via a smoke test) runs without error.
- Diff stat: each script loses ~110 lines of identical code.

**Decisions taken:**
- `audiovisual` and `index` get migrated to Rich `Progress` as part of this work (CLAUDE.md mandates Rich; the shared `OmekaApiClient` uses Rich progress). This is a *visual* change to those two scripts, not a behavioral one.
- The shared `OmekaApiClient` keeps `fetch_items(rcid)` returning items for a single class. Scripts that loop over multiple classes (`reference`, `index`) handle the loop in their own code, exactly as they do today.

---

### Tier 1b — Extract field-mapping helpers (~150 lines saved)

**Create `iwac_common/field_mappers.py`** containing:
- `get_value(item, field)` — the canonical `display_title → @value → @id` extractor with pipe-join for lists. Currently duplicated 6×.
- `extract_date(item, field='o:created')` — the `T`-split → ISO date helper. Currently duplicated 5×.
- `to_int_or_none(value)` — the `try: int(x) except: None` pattern used for `nb_pages`, `chapter`, `edition`, `page_start`, `page_end`.
- `extract_sentiment_block(item, model)` for `model in {"gemini", "chatgpt", "mistral"}` returning the 6-field dict. Currently hand-written in 2 scripts × 3 models × 6 fields = 36 hand-coded lines per script.

**Acceptance:**
- `articles` and `islamic-publications` use `extract_sentiment_block` and produce byte-identical sentiment columns to before (verify with a small dry-run that hashes the output dataframe).
- All 6 scripts use shared `get_value` / `extract_date`.

---

### Tier 1c — Extract HF Hub merge helper (~80 lines saved)

**Create `iwac_common/hub_merge.py`** with:

```python
def merge_with_hub_dataset(
    new_df: pd.DataFrame,
    repo: str,
    config_name: str,
    *,
    how: str = "left",
    suffixes: tuple = ("", "_old"),
    token: Optional[str] = None,
) -> pd.DataFrame:
    """Load existing HF dataset config, identify computed columns absent in
    new_df, and merge them onto new_df by 'o:id'. Returns the merged frame.
    Logs Rich-formatted stats."""
```

**Migration:**
- 5 scripts pass `how="left"` (default).
- `reference/upload_reference_hf.py` passes `how="outer", suffixes=("", "_old")` to preserve current behavior at L613.

**Acceptance:** dry-run merge against a snapshot of the live HF dataset produces the same column set and row count for every script.

---

### Tier 2 — Post-processing common module (~250 lines saved)

**Create `post-processing/_common.py`** with:
- `ensure_hf_token() -> str` — consolidates the 15-line auth dance across 5 scripts.
- `choose_dataset_config(repo_id, token, default=None) -> str` — Rich-table interactive picker, `default` lets `lemmatize_update_hf` skip the prompt.
- `load_subset(repo_id, config_name, token) -> Dataset` — wraps `load_dataset` with consistent error handling.
- `push_subset(ds, repo_id, config_name, commit_message, token, max_shard_size="1GB")` — single push helper.

**Then refactor `calculate_lexical_richness.py` and `calculate_word_count.py`** to share a thin "compute & merge a numeric column" scaffold. They may stay as separate scripts (different per-row functions) but lose ~150 lines each of duplicate plumbing.

**Acceptance:** each script's CLI surface is unchanged; output dataset has the same columns as before.

---

### Tier 2b — semantic_embedding.py split (~80 lines saved, organizational)

`semantic_embedding.py` is 802 lines. Split into:
- `post-processing/_embedding_utils.py` — chunk/cache/average helpers (~80 lines)
- `post-processing/_gemini_client.py` — retry + API wrapper (~30 lines)
- `semantic_embedding.py` — CLI + orchestration (stays the entry point)

Pure organizational move. No behavior change.

---

### Tier 3 — Smaller fixes

1. **`country_mapper.py`** — replace the long `elif` chain with a reverse dict built once at import. Optionally externalize the newspaper lists to `country_mapper.json`. Remove the unused empty `NEWSPAPER_TO_COUNTRY = {}`. ~50 lines saved, O(1) lookup.

2. **`requirements.txt`** — pin major versions for `datasets`, `huggingface_hub`, `aiohttp`, `pandas`, `pyarrow`, `spacy`, `rich`, `google-genai`. (Not migrating to `pyproject.toml` in this pass — out of scope.)

3. **`__init__.py` re-exports** — `post-processing/topic_modeling/__init__.py` should re-export `apply_all_patches` and the stopword constants so callers don't reach into module internals.

4. **Verify** the shared cache dir between `articles` and `islamic-publications` (both use `.cache_omk`) is intentional. If not, change `islamic-publications` to `.cache_omk_publications`.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Breaking the live upload pipeline | Each tier is import-only on first commit; no script gets removed code until imports verified. CI/manual smoke test before pushing. |
| Subtle field-shape changes | Tier 1b includes a hash-based comparison of pre/post DataFrame output for at least `articles` (the most-tested subset) before commit. |
| `reference`'s outer-merge behavior | Explicitly preserved in Tier 1c via parameter; called out in tests. |
| Cache invalidation | Cache dir names unchanged. Existing caches keep working. |
| Rich vs tqdm progress in `audiovisual`/`index` | This is a *visible* but non-functional change. Acceptable per CLAUDE.md's Rich mandate. Documented in Tier 1a notes. |

---

## Out of scope (not in this refactor)

- Migrating to `pyproject.toml` / `uv` / `poetry`.
- Adding tests where none exist (the project has no test suite today; not adding one as part of this refactor).
- Changing the dataset schema.
- Performance optimizations beyond what falls out of deduplication.
- Type-checker (`mypy`/`pyright`) integration.

---

## Progress tracker

- [x] Roadmap written
- [x] **Tier 1a** — `iwac_common/omeka_client.py` extracted, all 6 scripts migrated (−602 lines net)
- [x] **Tier 1b** — `iwac_common/field_mappers.py` extracted, all 6 scripts migrated (−79 lines net)
- [x] **Tier 1c** — `iwac_common/hub_merge.py` extracted, all 6 scripts migrated, reference's outer-merge preserved via parameters (−81 lines net)
- [x] **Tier 2** — `post-processing/_common.py` (auth + config picker + config-list lookup) extracted; 4 of 5 post-processing scripts migrated. `calculate_word_count.py` keeps its custom config picker (different UX); `semantic_embedding.py` keeps its custom config picker (per-config metadata table).
- [ ] Tier 2b — semantic_embedding split ← *next*
- [ ] Tier 3 — small fixes (country_mapper, requirements pin, __init__ re-exports, cache dir audit)

**Tier 1 cumulative:** 6 upload scripts shed 1,267 lines (621 → 393, 477 → 273, 571 → 343, 655 → 432, 759 → 568, 713 → 520). 505 lines added across 3 shared modules. Net **−762 lines** with no behavioral regressions.
