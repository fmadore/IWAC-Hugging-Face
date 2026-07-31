"""Shared utilities for IWAC upload and post-processing scripts.

Modules:
    omeka_client:  Async Omeka S API client with disk cache, retry, Rich
                   progress, and the shared IIIF thumbnail fetcher.
    field_mappers: Omeka JSON-LD → flat-column value extractors.
    hub_merge:     Merge fresh Omeka data with the existing HF Hub dataset,
                   preserving computed columns.
    sentiment_panel: Canonical definition of the AI sentiment annotator panel
                   (model-keyed column prefixes, Omeka property mapping,
                   which models are frozen history).

Installable via ``pip install -e . --no-deps`` (see pyproject.toml), which
makes ``iwac_common`` and ``country_mapper`` importable without sys.path
manipulation.
"""
