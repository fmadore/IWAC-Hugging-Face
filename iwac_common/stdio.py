"""UTF-8 standard streams for every IWAC program.

Called when ``iwac_common`` is imported. Every upload, stage and analysis
script imports it (and so does ``iwac_pipeline``), so direct runs such as
``python post-processing/x.py > file`` get UTF-8 output as well as the
installed commands and ``python -m`` runs.
"""

from __future__ import annotations

import sys


def force_utf8_stdio() -> None:
    """Render rich's box drawing and emoji on a non-UTF-8 stdout.

    Every command prints through rich, and ``_setup_console_logging`` routes
    logging through a ``RichHandler``. On Windows a redirected stdout defaults
    to cp1252, where rich's legacy renderer raises ``UnicodeEncodeError`` on the
    first emoji. Raised inside a logging handler, ``logging`` swallows it: the
    run loses the rest of its output and still exits 0, so a broken run looks
    exactly like a clean one. A subprocess started with ``python -m`` fails the
    same way, outright.

    Streams already in UTF-8 are left alone, so this does nothing on Linux,
    macOS or CI. Streams that are not ``TextIOWrapper``s (a notebook's), or are
    detached or closed, are skipped.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if encoding == "utf8":
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
