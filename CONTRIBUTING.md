# Contributing

## Local setup

Use Python 3.12 or newer and an editable checkout:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

Before opening a pull request, run:

```powershell
.venv\Scripts\python -m compileall -q iwac_common iwac_pipeline post-processing articles audiovisual document images index islamic-publications reference data
.venv\Scripts\python -m pytest tests -q --cov=iwac_common --cov-fail-under=70
.venv\Scripts\python -m pip check
```

CI repeats these checks on Linux/Python 3.12, Linux/Python 3.13, and Windows/Python 3.12.

## Data-safety rules

- Never call `Dataset.push_to_hub` directly. Route writes through `iwac_common.hub.push_dataset_verified` (or `post-processing/_common.py:push_dataset`).
- Treat a failed Hub baseline read, missing Omeka total-count header, mapper exception, or media transport error as fatal by default. Overrides must preserve the affected existing Hub values.
- Join computed outputs on `o:id`; never assume two Hub reads have the same row order.
- Add stable subset/resource-class/embedding facts to `iwac_common/schema.py`, not another local list.
- A new public column must be reviewed in `iwac_common/public_columns.json`. Full-text-like columns belong in the masked content contract, not merely the public allowlist.
- Run write-capable scripts against a scratch repo first via `IWAC_HF_PRIVATE_REPO` / `IWAC_HF_PUBLIC_REPO`. Tests must not require live Hub or Omeka access.

## Tests expected with changes

Bug fixes need a regression test that fails without the fix. Schema and pipeline changes should cover failure paths as well as the happy path: duplicate/missing IDs, truncated reads, revision conflicts, partial mapping/media failures, rights flags, and reload verification are all first-class behavior.

The GitHub branch should require the `static-checks` job and every `Tests (…)` matrix job before merging. Branch protection is a repository setting and therefore cannot be enforced by files in this checkout alone.
