"""
constants.py
-------------
Constants for the LDA topic modeling pipeline: stopword sets, custom
collocations, and training defaults.

IMPORTANT: This is a collection about Islam in West Africa.
Islamic organizations (COSIM, FAIB, UIB, etc.) and religious events
(Ramadan, Tabaski, Maouloud, etc.) are CORE to the research and should
appear in topic labels. We only remove truly non-informative noise.

Lower-case throughout — tokenization lowercases before matching.
"""

# ── Modeling stopwords ─────────────────────────────────────────────
# Removed from tokens BEFORE training/inference (affects the topics
# themselves). Be VERY conservative here — only words that truly add
# noise. (Historical note: this set was born as the BERTopic
# ``VECTORIZE_STOPWORDS``; the name DOMAIN_STOPWORDS is kept because
# every call site uses it.)
DOMAIN_STOPWORDS = {
    # Generic courtesy/boilerplate
    "monsieur", "madame", "excellence", "communiqué", "communique",
    "chers", "cher", "chère", "chere",

    # Generic words without topical signal
    "grand", "mondial", "international",

    # Functional words that leak through lemmatization
    "ledit", "faire", "devoir", "pouvoir",

    # OCR artifacts
    "lp", "bf", "wa", "adj", "octet", "sem", "at",

    # Generic French functional words that hurt clustering (no topical signal)
    "ensuite", "puis", "donc", "aussi", "toujours", "encore", "bien",
    "tout", "tres", "très", "avoir", "etre", "être", "plus", "entre",
    "apres", "après", "avant", "autre", "autres", "même", "meme",
    "selon", "ainsi", "car", "vers", "depuis", "pendant", "contre",
    "sans", "chez", "comme",
    "aller", "falloir", "vouloir", "savoir", "voir", "celer",
    "el",  # fragment from "El Hadj"

    # ENGLISH stopwords (critical for filtering non-French documents)
    "the", "of", "to", "and", "in", "for", "is", "on", "that", "by",
    "this", "with", "are", "from", "or", "an", "be", "as", "at", "was",
}

# ── Label-only stopwords ───────────────────────────────────────────
# Dropped from topic LABELS only (keeps labels readable); these words
# still participate in training.
LABEL_ONLY_STOPWORDS = {
    # A. Devotional formulae (abbreviations that are hard to interpret in labels)
    "psl", "saw", "swt",

    # B. ENGLISH stopwords (from OCR/non-French docs leaking through)
    "the", "of", "to", "and", "in", "for", "is", "on", "that", "by",
    "this", "with", "are", "from", "or", "an", "be", "as", "at", "was",
    "which", "have", "has", "their", "it", "its", "they", "will", "can",
    "all", "we", "been", "would", "were", "there", "who", "what", "more",
    "but", "if", "not", "so", "when", "other", "than", "no", "also", "into",

    # C. Dates & meeting filler (not topic-specific)
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin", "juillet",
    "août", "aout", "septembre", "octobre", "novembre", "décembre", "decembre",
    "année", "annee", "jour",
    "aujourd'hui", "hier", "demain",

    # D. Currency/quantities (numbers, not concepts)
    "fcfa", "franc", "francs", "cfa",
    "montant", "montants", "somme", "sommes",
    "millier", "milliers", "million", "millions", "milliard", "milliards",

    # E. Generic boilerplate (not domain-specific)
    "monsieur", "madame", "excellence",
    "communiqué", "communique",
    "chers", "cher", "chère", "chere",

    # F. OCR artifacts & abbreviations (garbage tokens)
    "lp", "bf", "dr", "mr", "mme", "mm", "wa",
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
    "octet", "sem", "at", "adj",

    # G. Generic location words (not specific places)
    "place", "rond", "point",

    # H. Very generic words that don't add meaning to labels
    "grand", "mondial", "international", "national", "régional", "regional",

    # I. Generic French functional/filler words that leak through lemmatization
    "ensuite", "puis", "donc", "aussi", "toujours", "encore", "bien",
    "tout", "tres", "très", "avoir", "etre", "être", "plus", "entre",
    "apres", "après", "avant", "autre", "autres", "même", "meme",
    "deja", "déjà", "selon", "lors", "ainsi", "car", "vers",
    "depuis", "pendant", "contre", "sous", "sans", "chez",
    "cela", "ceci", "celle", "ceux", "celui",
    "aller", "falloir", "vouloir", "savoir", "voir", "celer",
    "el",  # fragment from "El Hadj"
    "quelque", "quelques", "certain", "certains", "certaine", "certaines",
    "chaque", "tel", "telle", "tels", "telles",
    "beaucoup", "peu", "assez", "trop", "combien",
    "comme", "comment", "pourquoi", "quand",
    "premier", "première", "premiere", "deuxième", "deuxieme",
    "dernier", "dernière", "derniere",
    "nouveau", "nouvelle", "nouveaux",
    "seul", "seule", "seuls", "seules",
    "petit", "petite", "petits", "petites",
}

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

# Sweep models only need to be good enough for *relative* C_v comparison, so
# they train at reduced settings; the final model retrains at full
# DEFAULT_PASSES / DEFAULT_ITERATIONS. Cuts the sweep cost roughly 3-5x.
DEFAULT_SWEEP_PASSES = 10
DEFAULT_SWEEP_ITERATIONS = 200

# How many topics to keep in the per-document distribution column
# (lda_topic_topk): "id:prob|id:prob|..." sorted by descending probability.
DEFAULT_TOPIC_TOPK = 3
