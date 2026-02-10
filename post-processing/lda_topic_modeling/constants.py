"""
constants.py
-------------
Shared constants for LDA topic modeling pipeline.

Reuses the domain stopwords from the BERTopic pipeline and adds
LDA-specific defaults.

IMPORTANT: This is a collection about Islam in West Africa.
Islamic organizations (COSIM, FAIB, UIB, etc.) and religious events
(Ramadan, Tabaski, Maouloud, etc.) are CORE to the research and should
appear in topic labels. We only remove truly non-informative noise.
"""

# Import shared stopwords from BERTopic constants
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from topic_modeling.constants import DOMAIN_STOPWORDS, LABEL_ONLY_STOPWORDS  # noqa: E402

# LDA-specific defaults
# For a ~12 000-doc corpus across 5 countries, 50 topics is a reasonable
# starting point.  Use --optimize-topics to sweep a range and let C_v decide.
DEFAULT_NUM_TOPICS = 50
DEFAULT_PASSES = 20
DEFAULT_ITERATIONS = 500
DEFAULT_CHUNKSIZE = 2000
DEFAULT_RANDOM_STATE = 42
DEFAULT_MINIMUM_PROBABILITY = 0.01
DEFAULT_NO_BELOW = 5        # ignore tokens appearing in fewer than 5 docs
DEFAULT_NO_ABOVE = 0.5      # ignore tokens appearing in more than 50% of docs

# Topic-number optimisation grid (used by --optimize-topics)
DEFAULT_TOPIC_RANGE_START = 10
DEFAULT_TOPIC_RANGE_END = 100
DEFAULT_TOPIC_RANGE_STEP = 10
