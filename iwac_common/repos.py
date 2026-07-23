"""Canonical Hugging Face repo IDs for the IWAC dataset pipeline.

Since 2026-07 the collection lives in TWO Hub repos:

- ``PRIVATE_REPO_ID`` — ``fmadore/islam-west-africa-collection-full``, a
  private repo holding the complete dataset, including the full-text columns
  (``OCR``, ``lemma_text``, ``lemma_nostop``) that are not publicly readable
  on the Omeka S source. Every upload and post-processing script targets this
  repo; module build pipelines (IwacVisualizations) read it with a
  fine-grained token.
- ``PUBLIC_REPO_ID`` — ``fmadore/islam-west-africa-collection``, the public
  repo cited in publications. It is written ONLY by
  ``post-processing/publish_public.py``, which projects the private repo
  minus the private columns.

Override either via environment variables (useful for testing against a
scratch repo): ``IWAC_HF_PRIVATE_REPO`` / ``IWAC_HF_PUBLIC_REPO``.
"""

import os

PUBLIC_REPO_ID = os.getenv("IWAC_HF_PUBLIC_REPO", "fmadore/islam-west-africa-collection")
PRIVATE_REPO_ID = os.getenv("IWAC_HF_PRIVATE_REPO", "fmadore/islam-west-africa-collection-full")

# Full-text-derived columns, per subset. These are NOT dropped wholesale:
# publish_public.py masks them PER ROW by the ``OCR_is_public`` flag — the
# text stays public for items whose ``bibo:content`` is public on Omeka
# (verified: ~61% of articles, ~89% of publications, 25/26 documents, 7/867
# references) and is blanked only for items whose full text is private on
# the source. Lemmas derive from OCR, so they follow the same per-row mask.
# Subsets not listed have no full-text columns.
CONTENT_COLUMNS = {
    "articles": ["OCR", "lemma_text", "lemma_nostop"],
    "publications": ["OCR", "lemma_text", "lemma_nostop"],
    "documents": ["OCR"],
    "references": ["OCR", "lemma_text", "lemma_nostop"],
    "audiovisual": ["OCR"],
}

# Back-compat alias (older docs/scripts referenced PRIVATE_COLUMNS).
PRIVATE_COLUMNS = CONTENT_COLUMNS

# Path of the per-subset public-column allowlist consumed by
# publish_public.py (see the "_readme" key inside the file).
PUBLIC_COLUMNS_FILE = os.path.join(os.path.dirname(__file__), "public_columns.json")


def load_public_columns() -> dict:
    """Return the per-subset allowlist of publishable columns as
    ``{subset: set(columns)}`` (the ``_readme`` key is dropped)."""
    import json

    with open(PUBLIC_COLUMNS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {k: set(v) for k, v in data.items() if not k.startswith("_")}


__all__ = [
    "PUBLIC_REPO_ID",
    "PRIVATE_REPO_ID",
    "CONTENT_COLUMNS",
    "PRIVATE_COLUMNS",
    "PUBLIC_COLUMNS_FILE",
    "load_public_columns",
]
