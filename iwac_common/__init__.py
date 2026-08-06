"""Shared utilities for IWAC upload and post-processing scripts.

Modules:
    omeka_client:  Async Omeka S API client with disk cache, retry, Rich
                   progress, and the shared IIIF thumbnail fetcher.
    field_mappers: Omeka JSON-LD → flat-column value extractors.
    hub_merge:     Merge fresh Omeka data with the existing HF Hub dataset,
                   preserving computed columns.
    card_sync:     Post-push guard: keep the Hub card's declared schema in step
                   with the parquet, because push_to_hub updates the card's byte
                   sizes but not its feature list.
    sentiment_panel: Canonical definition of the AI sentiment annotator panel
                   (model-keyed column prefixes, Omeka property mapping,
                   which models are frozen history).

Installable via ``pip install -e . --no-deps`` (see pyproject.toml), which
makes ``iwac_common`` and ``country_mapper`` importable without sys.path
manipulation.
"""
