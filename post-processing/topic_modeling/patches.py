"""
patches.py
----------
Utility patches for consistent UTF-8 file I/O and robust JSON encoding
handling NumPy types and non-ASCII characters.
"""
from __future__ import annotations

import builtins
import functools
import json
from typing import Any

import numpy as np


def apply_utf8_open_patch() -> None:
    """Force UTF-8 encoding for all file operations on text modes.

    This globally wraps builtins.open to default to encoding='utf-8' and
    errors='replace' on text modes. Binary modes are left untouched.
    """
    original_open = builtins.open

    @functools.wraps(original_open)
    def patched_open(file, mode='r', *args, **kwargs):
        if 'b' not in mode:  # Only patch text modes
            if 'encoding' not in kwargs:
                kwargs['encoding'] = 'utf-8'
                kwargs.setdefault('errors', 'replace')
        return original_open(file, mode, *args, **kwargs)

    builtins.open = patched_open  # type: ignore[assignment]


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy types to native Python types.

    Prevents errors like "Object of type int64 is not JSON serializable".
    """

    def default(self, obj: Any):  # noqa: D401, N802  (keep signature to respect json API)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def apply_json_patches() -> None:
    """Patch json.dump(s) defaults and encoder globally.

    - Use _NumpyJSONEncoder as the default JSON encoder
    - Ensure ensure_ascii defaults to False to keep non-ASCII characters
    """
    json.JSONEncoder = _NumpyJSONEncoder  # type: ignore[assignment]
    json._default_encoder = _NumpyJSONEncoder()  # type: ignore[attr-defined]

    _original_json_dump = json.dump
    _original_json_dumps = json.dumps

    def _patched_dump(obj, fp, *args, **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        return _original_json_dump(obj, fp, *args, **kwargs)

    def _patched_dumps(obj, *args, **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        return _original_json_dumps(obj, *args, **kwargs)

    json.dump = _patched_dump  # type: ignore[assignment]
    json.dumps = _patched_dumps  # type: ignore[assignment]


def apply_all_patches() -> None:
    """Apply all global patches for the pipeline."""
    apply_utf8_open_patch()
    apply_json_patches()
