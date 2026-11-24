"""Shared utilities for IWAC upload and post-processing scripts.

Modules:
    hub:           The single verified gateway for every Hub write, plus the
                   fail-closed baseline reads and the local repo write lock.
                   Nothing else in this repo may call ``push_to_hub``.
    schema:        Canonical subset registry (names, Omeka resource-class ids,
                   content and embedding columns) and the dataframe contracts
                   checked before a push.
    repos:         The two Hub repo IDs and the reviewed public-column
                   allowlist that separates them.
    omeka_client:  Async Omeka S API client with disk cache, retry, Rich
                   progress, and the shared IIIF thumbnail fetcher.
    field_mappers: Omeka JSON-LD → flat-column value extractors.
    text_utils:    Shared text normalisation helpers.
    hub_merge:     Merge fresh Omeka data with the existing HF Hub dataset,
                   preserving computed columns.
    upload_runner: The shared CLI/async driver behind the seven upload scripts;
                   each script contributes only an ``UploadSpec``.
    card_sync:     Read-only post-commit check: the gateway commits Parquet and
                   card metadata atomically in one create_commit; this verifies,
                   at the new revision, that the card's declared features match
                   the published Parquet and the prepared dataset.
    sentiment_panel: Canonical definition of the AI sentiment annotator panel
                   (model-keyed column prefixes, Omeka property mapping,
                   which models are frozen history).
    enrichment:    Persisted per-row provenance (input/config hashes) for
                   computed columns, and their invalidation.
    write_policy:  Publication policy shared by the public projection and the
                   write gateway (item/property visibility, neighbour lists).
    settings:      Startup .env loading (honours ``IWAC_ENV_FILE``) and the
                   lazy repository-id getters.
    paths:         The writable workspace root, kept apart from installed code.

Installable via ``pip install -e . --no-deps`` (see pyproject.toml), which
makes ``iwac_common`` and ``country_mapper`` importable without sys.path
manipulation.

Importing the package forces UTF-8 standard streams (``stdio``): every
script imports it, so a direct run with redirected output on Windows does not
fall back to cp1252 and lose its rich output.
"""

from .stdio import force_utf8_stdio

force_utf8_stdio()
