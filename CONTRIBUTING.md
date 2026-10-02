# Contributing

## Local setup

Use Python 3.12 or newer and an editable checkout:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-lock.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

Before opening a pull request, run:

```powershell
.venv\Scripts\python -m compileall -q iwac_common iwac_pipeline post-processing articles audiovisual document images index islamic-publications reference data analyses country_mapper.py lemmatize_update_hf.py
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m pytest tests -q --cov=. --cov-report=json
.venv\Scripts\python scripts/check_coverage.py coverage.json --core-min 70
.venv\Scripts\python -m pip check
```

CI repeats the locked environment on Linux/Python 3.12, Linux/Python 3.13, and
Windows/Python 3.12. A separate Linux job tests the latest supported dependency
ranges. Coverage measures the complete production tree while retaining the
70% shared-core gate. A wheel job installs into a separate environment and runs
every command's help from an empty directory; no source-path imports can rescue
an incomplete distribution.

Regenerate the dependency lock deliberately, then validate it in a clean environment:

```bash
uv pip compile requirements-dev.txt --generate-hashes --universal --python-version 3.12 --output-file requirements-lock.txt
```

Runtime dependencies come from `requirements-core.txt`, `requirements-nlp.txt`
and `requirements-analysis.txt`; both package metadata and requirements files
reuse them. Keep install ranges separate from the exact research lock.

## Data-safety rules

- Never call `Dataset.push_to_hub` directly. Route writes through `iwac_common.hub.push_dataset_verified` / `push_datasets_verified` (or `post-processing/_common.py:push_dataset`). Full writes require private destinations; public mode revalidates the projection and commits data plus card atomically.
- Treat a failed Hub baseline read, missing Omeka total-count header, mapper exception, or media transport error as fatal by default. Overrides must preserve the affected existing Hub values.
- Join computed outputs on `o:id`; never assume two Hub reads have the same row order.
- Add stable subset/resource-class/embedding facts to `iwac_common/schema.py`, not another local list — including a new integer column (`int_columns`) and the source a new computed column derives from (`DERIVED_FROM`).
- Do not round-trip a whole subset through pandas to set one column's type: `to_pandas()` turns nullable ints into `float64`. Declare output types (`map_with_progress(..., output_types=...)`); the gateway's conform step is a backstop.
- Join content subsets to `index` on the `*_ids` columns, not on display titles.
- A new public column must be reviewed in `iwac_common/public_columns.json`. Full-text-like columns belong in the masked content contract, not merely the public allowlist.
- Run write-capable scripts against a scratch repo first via `IWAC_HF_PRIVATE_REPO` / `IWAC_HF_PUBLIC_REPO`. Tests must not require live Hub or Omeka access.

## Tests expected with changes

Bug fixes need a regression test that fails without the fix. Schema and pipeline changes should cover failure paths as well as the happy path: duplicate/missing IDs, truncated reads, revision conflicts, partial mapping/media failures, rights flags, and reload verification are all first-class behavior.

The GitHub branch should require `static-checks`, `Installed wheel`, and every
`Tests (…)` matrix job before merging. Branch protection is a repository setting
and therefore cannot be enforced by files in this checkout alone.

Read [the ingestion contract](docs/ingest-contract.md) and
[migration guide](docs/hardening-migration.md) when changing derived columns or
publication policy. New research analyses must state their observation unit,
inclusion rules, denominator and model/instrument identity, archive complete
outputs, and distinguish computational checks from expert validation.
