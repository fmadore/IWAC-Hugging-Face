"""
constants.py
-------------
Shared constants for LDA topic modeling pipeline.

Reuses the shared domain stopwords and adds LDA-specific defaults,
custom collocations, and geographic / generic stopwords.

IMPORTANT: This is a collection about Islam in West Africa.
Islamic organizations (COSIM, FAIB, UIB, etc.) and religious events
(Ramadan, Tabaski, Maouloud, etc.) are CORE to the research and should
appear in topic labels. We only remove truly non-informative noise.
"""

# Import shared stopwords
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from topic_modeling.constants import DOMAIN_STOPWORDS, LABEL_ONLY_STOPWORDS  # noqa: E402

# ── Geographic stopwords ───────────────────────────────────────────
# The dataset already has a 'country' metadata field, so geographic
# tokens produce country-dominant topics instead of thematic ones.
# Both accented and unaccented forms are needed: accented for
# tokenization (direct string match), unaccented for label filtering
# (which normalises/strips accents before comparison).
LDA_GEO_STOPWORDS = {
    # West African countries
    "bénin", "benin",
    "burkina", "faso", "burkina_faso",
    "côte", "cote", "ivoire", "côte_ivoire", "cote_ivoire",
    "ghana",
    "mali",
    "niger",
    "nigéria", "nigeria",
    "sénégal", "senegal",
    "togo",
    # Major cities
    "abidjan",
    "ouagadougou", "ouaga",
    "lomé", "lome",
    "cotonou",
    "niamey",
    "bamako",
    "dakar",
    "bouaké", "bouake",
    "bobo_dioulasso", "bobo",
    "abobo",
    "adjamé", "adjame",
    "gagnoa",
    "yamoussoukro",
    "korhogo",
    "porto_novo",
    # Demonyms
    "ivoirien", "ivoirienne", "ivoiriens",
    "burkinabé", "burkinabe", "burkinabè",
    "togolais", "togolaise",
    "béninois", "beninois", "béninoise", "beninoise",
    "sénégalais", "senegalais", "sénégalaise", "senegalaise",
    "nigérien", "nigerien", "nigérienne", "nigerienne",
    "malien", "malienne",
    "ghanéen", "ghaneen", "ghanéenne", "ghaneenne",
}

# ── Generic functional stopwords ──────────────────────────────────
# Words that leak through lemmatisation and create junk / incoherent
# topics (e.g. "situation - permettre - beaucoup - problème").
LDA_GENERIC_STOPWORDS = {
    # Verbs with no topical signal
    "venir", "permettre", "prendre", "trouver", "arriver",
    "devenir", "passer", "commencer", "donner", "travailler",
    "penser", "mettre", "dire", "rester", "porter",
    "partir", "tenir", "suivre", "croire", "sembler",
    # Reporting verbs (newspaper boilerplate)
    "indiquer", "déclarer", "declarer", "ajouter",
    "préciser", "preciser", "souligner", "affirmer",
    "estimer", "noter", "rappeler", "annoncer",
    # Ceremony / discourse verbs
    "recevoir", "remercier", "inviter", "souhaiter",
    "appeler", "demander", "organiser",
    # Generic nouns
    "situation", "problème", "probleme", "moyen", "besoin",
    "niveau", "travail", "temps", "chose", "rien",
    "fois", "heure", "lieu", "monde", "nom",
    "moment", "part", "occasion", "difficulté", "difficulte",
    "argent", "ville",
    # Date / time
    "année", "annee", "an", "jour", "mois",
    "hier", "demain", "lendemain", "date", "période", "periode",
    # Generic adjectives
    "social", "local", "national", "régional", "regional",
    "dernier", "dernière", "derniere",
    "premier", "première", "premiere",
    "nouveau", "nouvelle", "nouveaux",
    "petit", "petite",
    "différent", "different",
    # Other
    "beaucoup", "non", "oui",
}

# ── Custom collocations ────────────────────────────────────────────
# Domain-specific multi-word expressions that should ALWAYS be joined,
# regardless of gensim's statistical phrase detection threshold.
# Each entry is a tuple of consecutive tokens → joined with "_".
# Applied after gensim's bigram/trigram phrase detection, so these
# act as a safety net for collocations the statistics miss.
CUSTOM_COLLOCATIONS: list[tuple[str, ...]] = [
    # Religious concepts
    ("nuit", "destin"),               # Laylat al-Qadr
    ("finance", "islamique"),         # Islamic finance
    ("école", "coranique"),           # Quranic school
    ("ecole", "coranique"),           # (unaccented variant)
    ("droit", "homme"),               # human rights
    ("société", "civile"),            # civil society
    ("societe", "civile"),            # (unaccented variant)
    ("conseil", "supérieur"),         # supreme council
    ("conseil", "superieur"),         # (unaccented variant)
    ("oeuvre", "sociale"),            # social work
    ("voile", "islamique"),           # Islamic veil
    ("état", "civil"),                # civil registry
    ("etat", "civil"),                # (unaccented variant)
    ("liberté", "religieuse"),        # religious freedom
    ("liberte", "religieuse"),        # (unaccented variant)
    ("dialogue", "interreligieux"),   # interfaith dialogue
    ("extrémisme", "violent"),        # violent extremism
    ("extremisme", "violent"),        # (unaccented variant)
    ("radicalisation", "religieuse"), # religious radicalisation
]

# LDA-specific defaults
# With geographic and generic stopwords removed, 30 topics avoids the
# junk / country-dominant topics that appeared at 40.
# Use --optimize-topics to sweep a range and let C_v decide.
DEFAULT_NUM_TOPICS = 30
DEFAULT_PASSES = 25          # more passes = better convergence on CPU
DEFAULT_ITERATIONS = 600     # allow more iterations per E-step
DEFAULT_CHUNKSIZE = 2000
DEFAULT_RANDOM_STATE = 42
DEFAULT_MINIMUM_PROBABILITY = 0.01
DEFAULT_NO_BELOW = 10        # ignore tokens in fewer than 10 docs (was 5; stricter removes OCR noise)
DEFAULT_NO_ABOVE = 0.40      # ignore tokens in more than 40% of docs (was 0.5; removes corpus-wide boilerplate)

# Topic-number optimisation grid (used by --optimize-topics)
DEFAULT_TOPIC_RANGE_START = 15
DEFAULT_TOPIC_RANGE_END = 80
DEFAULT_TOPIC_RANGE_STEP = 5
