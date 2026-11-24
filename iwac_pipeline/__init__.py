"""Installed command-line entry points for the IWAC data pipeline."""

from iwac_pipeline._stdio import force_utf8_stdio

# Importing iwac_common (above, through _stdio) already forced UTF-8 stdio;
# the explicit call keeps that guarantee visible at the package entry point
# for the installed commands and ``python -m iwac_pipeline.<module>`` runs.
force_utf8_stdio()
