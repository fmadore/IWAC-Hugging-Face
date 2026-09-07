# Publication readiness review

Reviewed 2026-09-07. Scope: repository structure, installation and CI, shared
fetch/merge/write infrastructure, rights projection, enrichment and analysis
code, tests, README, contribution guidance, and citation metadata. This is a
source review plus offline validation, not a live audit of the published data
or a validation of the paper's scientific results.

## Assessment

The pipeline has a useful modular core: `iwac_common` centralizes the subset
registry, upload orchestration, metadata extraction, Hub merging, and verified
writes. Fail-closed fetches, per-row content masking, column allowlisting,
revision checks, and ID-based joins are strong foundations. A wholesale rewrite
before publication would add risk without addressing the most consequential
problems below.

The repository is not yet a fully reproducible publication artifact. Address
the high-priority findings, then archive one validated code/data/model/environment
snapshot. Passing tests alone does not establish that published enrichments
match the current source text.

## Open findings, ordered by priority

### 1. High: resumed enrichments are not tied to their source inputs

Evidence: `post-processing/semantic_embedding.py` constructs its cache filename
from a subset stem and model/dimension/task fingerprint. `lemmatize_update_hf.py`
uses subset/language/model/version and restores entries with `cache.get(row_ids[i])`.
Neither cache identity includes the source repository or a per-row text hash.

If a run is interrupted, the source text changes, and the same item is resumed,
the old cached result can be reused for the new text. Scratch and production
repositories can also share cache entries for identical IDs. An embedding
`all` run clears its cache by default, but `--resume` and incremental runs still
have this problem. A configuration fingerprint does not prove input identity.

Remediation: include repository, subset, source column, processing settings
(including chunking), and a hash of the actual input in cache identity. Reject
legacy entries that lack this evidence. Test unchanged-input reuse, changed text,
changed repository, and changed chunking independently. This can be done locally
without altering any published dataset columns.

### 2. High: authenticated Omeka URLs can appear in error logs

Evidence: `iwac_common/omeka_client.py` sends `key_identity` and `key_credential`
as query parameters, calls `raise_for_status()`, and interpolates exceptions in
`async_retry` and count-fetch failures. HTTP response exceptions can carry the
request URL. Additional callers print or log propagated exceptions.

A failed authenticated request can therefore expose credentials in terminal
output or a saved log. This review did not inspect credential files or establish
that an actual disclosure has occurred.

Remediation: convert transport errors to sanitized exceptions at the HTTP
boundary and redact sensitive query parameters from every retry/terminal error.
Check exception chaining and traceback logging too; changing one log line is
insufficient. Use synthetic credentials in tests and assert that neither plain
nor URL-encoded secrets appear for HTTP errors, retries, or final failures.

### 3. High for research accuracy: refreshed source text retains old computed columns

Evidence: `iwac_common/hub_merge.py` preserves Hub-only columns by `o:id`.
An existing item's changed OCR does not invalidate its embeddings, lemmas,
lexical metrics, or topic assignments. Incremental `empty` processing then
skips already-populated values.

This behavior protects expensive enrichments but can silently leave a row whose
metadata/text and derived values describe different versions of the document.
It is separate from the interrupted-cache problem above.

Remediation: record input fingerprints for derived outputs or maintain a
changed-input worklist and explicitly recompute affected stages. Define a
dependency graph: OCR changes affect different columns than table-of-contents
or image changes. Avoid blanket clearing of historical sentiment annotations,
which are a distinct upstream research artifact. A new published provenance
column would require deliberate schema and public-allowlist review.

### 4. Medium: analysis outputs do not carry a complete provenance record

Evidence: `_common.load_subset_dataframe` captures a Hub revision in dataframe
attributes, and the mirror has hashes, but the summary writers in
`analyses/topic_prevalence.py` and `analyses/keyness_bursts.py` do not serialize
that revision. Output names are fixed and successive runs replace them.
`modeling.save_model_parameters` also contains a hardcoded pipeline version
(`1.1.0`) distinct from project metadata (`1.0.0`).

Remediation: use one shared run-manifest helper recording code SHA and dirty
state, dataset repository/SHA, command arguments, software/model versions,
input/model hashes, UTC time, and output hashes. Write each run to a distinct
directory. Archive the exact artifacts used for figures and tables; do not rely
on a seed or a model-directory name alone.

### 5. Medium: packaging supports an editable checkout only

Evidence: `pyproject.toml` packages only `iwac_common`, `iwac_pipeline`, and
`country_mapper`, while `iwac_pipeline/cli.py` loads upload/mirror/publisher
scripts by paths outside those packages. Runtime dependencies are intentionally
not project metadata. `public_columns.json` has no explicit package-data rule.
The CLI test originally checked only whether files existed in the checkout.

A wheel cannot provide the same commands using this layout. The README now
states the editable-only installation contract explicitly. No wheel-install
test was performed in this review.

Remediation: move maintained implementations into importable modules, retain
historical paths as thin compatibility wrappers, explicitly include the
allowlist resource, and declare runtime dependencies from a single source.
Then test a built wheel in a clean environment from outside the checkout.
Until that work is complete, distribute and document the source checkout.

### 6. Local conflict resolved; publication environment still needs a lock

Observed: `pip check` reports that installed `numba 0.62.1` requires
`numpy<2.4,>=1.22`, but the environment has `numpy 2.5.1`. Numba is not a direct
entry in the repository requirements; this result does not establish that a
fresh install has the same conflict. Ruff was declared but absent; it was
installed to run the lint check.

Resolved for release 1.0.1: upgraded Numba to 0.67.0 and llvmlite to 0.49.0,
retaining NumPy 2.5.1. Upstream PyPI metadata confirms Numba's NumPy constraint
is `>=1.22,<2.6`. `pip check` and a compiled Numba array-sum smoke test pass.
These are local environment changes; Numba remains outside the pipeline's
direct runtime requirements.

Remaining remediation: create a clean publication environment from the declared
requirements, run `pip check` and the full suite, and archive the resulting
resolved dependency versions and spaCy model versions. Verify upstream versions
before changing dependency constraints. Do not freeze the current inconsistent
environment or resolve it by an untested NumPy downgrade.

### 7. Medium: write validation checks embedding length, not numeric validity

Evidence: `iwac_common/schema.py` checks `len(value)` against 768. A 768-character
string, nested vectors of outer length 768, or vectors containing NaN/infinity
can pass this dimensional check. ID validation also allows blank strings outside
the newly hardened merge input boundary.

Remediation: share one vector validator between dataframe and Dataset paths;
require one-dimensional finite numeric vectors while retaining intentional
null/empty handling. Share a nonblank ID contract across all write paths.
Tests should cover wrong types, nested arrays, nonfinite coordinates, and
valid Arrow/NumPy representations.

### 8. Medium: publication is sequential, not an atomic multi-subset transaction

Evidence: `publish_public.py` prepares all subsets, then writes each separately.
A later failure leaves earlier subsets updated. The local lock prevents local
interleaving, but cannot make several Hub commits transactional. The Hub writer
also explicitly documents the gap between its revision precheck and the
high-level push, which lacks an atomic parent-commit precondition.

Remediation: publish from an immutable source snapshot, record the source SHA
in durable publication metadata, and treat only the final validated Hub revision
as a release. Keep the operational single-writer rule across machines. Consider
staging all subset changes into one commit if atomic publication becomes a
requirement; this is a larger write-path change requiring dedicated tests.

## Efficiency and modularity

- Keep the shared upload runner and seven small subset adapters. Concentrate
  future extraction on script loading, run provenance, cache identity, and
  shared validation rather than adding abstractions around every mapper.
- The Omeka client bounds page concurrency and the Hub verification path reads
  IDs separately from full embedding columns. These are sensible cost controls.
- The public publisher holds every projected dataframe in `plans` at once.
  Stage validated projections to temporary Arrow/Parquet files if peak memory
  becomes material; retain the rule that all subsets pass privacy checks before
  the first push.
- Related-item search already computes cosine similarities in blocks, avoiding
  a full resident N-by-N matrix. Its compute cost is still quadratic. Benchmark
  before replacing it with approximate search, which changes retrieval results.
- Hub analysis column selection happens after `load_dataset`; it reduces pandas
  materialization but does not guarantee column-pruned downloads. Profile the
  download path before claiming network savings.
- Coverage is measured only over `iwac_common`, not the complete enrichment and
  analysis tree. Prioritize cache invalidation, transport failure handling, and
  real CLI behavior over raising the coverage percentage with shallow tests.

No runtime or peak-memory benchmark was performed; these are code-based
observations, not measured speedup claims.

## Changes made during this review

- Fixed `iwac-upload articles --help` being consumed by the top-level parser;
  subset options now reach the subset parser.
- Made merge inputs with missing, null, or blank IDs fail before a Hub read,
  rather than silently bypassing the merge or stringifying null identifiers.
- Rejected invalid embedding chunk windows. Overlap equal to or greater than
  chunk size could otherwise prevent the chunk loop from advancing.
- Added ten regression cases for these changes.
- Corrected README claims about upstream sentiment generation, enrichment
  completeness, source-class grouping, and derived-data disclosure guarantees.
  Removed undated text-availability proportions and the stale collection count.
- Documented editable-only installation and revision-pinned research reads;
  made activated-environment examples portable and added the missing contributor
  lint command.
- Clarified separate software/data citation and corrected sentiment provenance
  in `CITATION.cff`. Existing release identifiers and dates were preserved.

## Validation and limits

Baseline: **349 tests passed**, shared-library coverage **77.11%**.
After fixes: **359 tests passed**, shared-library coverage **77.35%**.
Ruff and compilation passed. Tests ran on the existing Windows Python 3.12.7
environment. The subsequent release preparation repaired the local dependency
conflict; `pip check` and a Numba JIT smoke test now pass.

The review itself performed no production uploads, paid inference, re-fitting,
DOI changes, or commits. Release 1.0.1 was subsequently prepared at the owner's
request, updating package/citation versions and the citation release date.
The Linux/Python 3.13 matrix, fresh installation, archived release
metadata, live dataset coverage, and current public projection contents were
not independently verified. Historical README campaign counts and empirical
reliability figures require their own dated evidence before reuse in a paper.
