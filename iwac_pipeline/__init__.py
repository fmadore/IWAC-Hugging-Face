"""Installed command-line entry points for the IWAC data pipeline."""

from iwac_pipeline._stdio import force_utf8_stdio

# Every program the package runs imports this package first: the installed
# commands through cli.py, and ``python -m iwac_pipeline.<module>`` through the
# parent-package import. Doing it here covers all of them, including future
# scripts, without a call in each ``__main__`` guard.
force_utf8_stdio()
