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

from .schema import CONTENT_COLUMNS
from .settings import get_private_repo_id, get_public_repo_id

PUBLIC_REPO_ID = get_public_repo_id()
PRIVATE_REPO_ID = get_private_repo_id()

# Full-text-derived columns per subset live in ``schema.CONTENT_COLUMNS``
# (re-exported here). They are NOT dropped wholesale: the public projection
# masks them PER ROW by the ``OCR_is_public`` flag, so the text stays public
# for items whose ``bibo:content`` is public on Omeka and is blanked only for
# items whose full text is private on the source. Lemmas derive from OCR, so
# they follow the same per-row mask.

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
    "PUBLIC_COLUMNS_FILE",
    "load_public_columns",
    "get_private_repo_id",
    "get_public_repo_id",
]
