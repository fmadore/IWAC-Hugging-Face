"""Separate installed code/resources from writable research workspaces."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Use an explicit workspace, the source checkout, or the current directory.

    A wheel must never put datasets, credentials or results in site-packages.
    ``IWAC_WORK_DIR`` also permits several isolated runs from one installation.
    """
    configured = os.getenv("IWAC_WORK_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "pyproject.toml").is_file() and (candidate / "iwac_common").is_dir():
        return candidate
    return Path.cwd().resolve()


__all__ = ["workspace_root"]
