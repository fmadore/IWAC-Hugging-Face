# v1.0.1 — Publication review and pipeline fixes

This patch release improves CLI behavior, input validation, and publication
documentation.

- Forward subset-specific help correctly (`iwac-upload articles --help`).
- Reject missing, null, and blank merge IDs before reading the Hub.
- Reject invalid embedding chunk windows that can prevent processing progress.
- Add ten regression cases.
- Correct sentiment provenance and enrichment-coverage claims, document
  editable-only installation and revision-pinned research reads, and clarify
  software/data citation guidance.
- Add a publication-readiness review with prioritized remediation steps.

Validation: 359 offline tests pass, with 77.35% coverage of `iwac_common`.
Ruff, compilation, and dependency consistency checks pass. The local Numba
environment was repaired with Numba 0.67.0 and llvmlite 0.49.0 alongside
NumPy 2.5.1; a compiled JIT smoke test passes.

This release does not modify published datasets or recompute enrichments.
The review identifies remaining work on cache input identity, credential-safe
HTTP error logging, stale computed columns, and reproducible publication
artifacts. See [the full review](https://github.com/fmadore/IWAC-Hugging-Face/blob/v1.0.1/docs/publication-review.md).
