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

    # Digitisation watermark. "Scanned by CamScanner" is stamped in the page
    # footer of 16 reference documents (2,220 occurrences in the raw OCR),
    # and phrase detection glued it onto whatever preceded it —
    # "mémoire_maîtrise_scanned_camscanner", "op_cit_scanned_camscanner",
    # "scanned_camscanner_organisation_pionnier". Stripping it here, BEFORE
    # phrase detection, is what frees the real words it swallowed.
    # ("scan"/"scanner" are deliberately absent: they never occur as OCR
    # noise, and French "scanner" is a legitimate medical term.)
    "camscanner", "scanned",

    # Corrupted PDF text layer, not Arabic-language content. Three reference
    # items (o:id 5315, 12709 — the same title twice — and 15722) carry
    # thousands of repetitions of an injected Arabic anti-virus notice
    # ("كورونا تصيب الفايروسات"; 4,730 hits in a single document), enough to
    # hijack an entire topic. The items themselves need re-OCR in Omeka.
    "فايروسات", "الفايروسات", "كورونا", "تصيب",

    # Generic French functional words that hurt clustering (no topical signal)
    "ensuite", "puis", "donc", "aussi", "toujours", "encore", "bien",
    "tout", "tres", "très", "avoir", "etre", "être", "plus", "entre",
    "apres", "après", "avant", "autre", "autres", "même", "meme",
    "selon", "ainsi", "car", "vers", "depuis", "pendant", "contre",
    "sans", "chez", "comme",
    "aller", "falloir", "vouloir", "savoir", "voir", "celer",
    # "el" / "al" live in FRAGMENT_STOPWORDS below — see the note there.

    # ENGLISH stopwords (critical for filtering non-French documents)
    "the", "of", "to", "and", "in", "for", "is", "on", "that", "by",
    "this", "with", "are", "from", "or", "an", "be", "as", "at", "was",

    # FRENCH function words (the mirror problem: English-language references
    # quote French sources, and the English spaCy stopword list keeps these —
    # without them the EN reference model grows French-residue junk topics).
    # Harmless for the French pipeline: spaCy fr already strips them.
    # "est" is safe despite the French "East" homograph: it survives in no
    # French dictionary (spaCy fr lemmatises the verb to "être"), only in the
    # English references model. Still deliberately NOT included: vol (theft),
    # son (sound).
    "la", "le", "les", "des", "du", "de", "un", "une", "et", "ou",
    "au", "aux", "dans", "sur", "par", "pour", "que", "qui",
    "sont", "ce", "cette", "ces", "sa", "ses", "leur", "leurs",
    "de_la", "de_l", "à", "a_la", "en", "ne", "est",
    "il", "elle", "ils", "pas", "nous",

    # GERMAN function words: the references corpus quotes German colonial
    # and missionary sources, and neither the French nor the English spaCy
    # stopword list strips them.
    "und", "der", "von", "das", "die",

    # Bibliographic apparatus abbreviations (reference lists, citations)
    "ed", "eds", "éd", "dir", "pp", "cit", "op_cit", "ibid", "idem", "fig",

    # Publishers and places of publication — footnote apparatus, never a
    # topic. Stripping them here, pre-phrase, is what dissolves the whole
    # citation family: "paris" alone anchors 32 compounds in the French
    # references model (paris_armand_colin, paris_cnrs, afrique_noir_paris),
    # and "london_hurst" is another publisher. "new"/"york" are listed
    # separately because matching happens before phrase detection, when
    # "new_york" does not yet exist as a token.
    #
    # The university-press cities below occur ONLY in the references
    # subsets, never in the press articles — the signature of apparatus
    # rather than content. Berlin, Londres and Genève are deliberately
    # absent for the opposite reason: they appear in the articles corpus as
    # real news subjects, and the Berlin Conference is part of the history
    # this collection studies.
    "karthala", "harmattan", "brill", "routledge",
    "paris", "london", "new", "york",
    "oxford", "cambridge", "leiden", "princeton", "stanford",
    "edinburgh", "chicago", "bloomington", "boulder",

    # University-press imprints. "press" is the head of every one of the 11
    # press compounds in the two references models (clarendon_press,
    # ohio_university_press, university_california_press …) and not one is
    # about the news media; it costs the French pipeline nothing, which says
    # "presse"/"média", and the English keeps "medium"/"media". The
    # institution names beside it occur ONLY in the references subsets.
    # "university" is deliberately absent — it carries real institutions
    # (university_abomey_calavi, university_medina, university_campus).
    "press", "indiana", "ohio", "evanston", "berkeley", "clarendon",
    "northwestern", "wisconsin", "california", "uk",

    # Rights notice stamped into the digitised documents ("licence accordée
    # Frédérick Madore") plus the curator's name in citations of his own
    # work — apparatus in both roles. Removing just these two names
    # dissolves the whole family (licence_accorder_frédérick_madore at 240
    # occurrences, page_/cité_licence_accorder_frédérick, frédérick_madore,
    # madore_frédérick) while sparing "licence", a real French word, and
    # "accorder", which anchors allah_accorder and dieu_accorder.
    "frédérick", "frederick", "madore",

    # Cited scholars. A bibliography names its authors far more often than
    # the text discusses them, so they crowd into labels as though they were
    # subjects. All of these are references-only — none appears in the
    # articles dictionary. Removing the given name and surname dissolves the
    # co-author strings too (leblanc_gomez_perez, savadogo_gomez_perez), so
    # "leblanc" and "savadogo" need not be listed and are better left out:
    # Savadogo is an everyday Burkinabè surname. For the same reason "souza"
    # is absent — it IS in the press corpus, naming the Afro-Brazilian
    # families of Ouidah (francisco_félix_souza, famille_souza).
    "muriel", "gomez", "perez", "sean", "hanretta", "weiss", "caleb",
    "playbook",  # from the report title weiss_aqim_imperial_playbook;
                 # "aqim" stays, it is a real organisation

    # Markup / URL residue surviving OCR and HTML extraction ("&amp;" → amp)
    "amp", "http", "https", "www", "url", "com", "pdf", "doi", "isbn",
}

# ── Fragment stopwords (removed AFTER phrase detection) ────────────
# Tokens that are noise on their own but carry meaning inside a compound.
# Removing them with DOMAIN_STOPWORDS would strip them before gensim's
# Phrases runs, so the compound could never form: a pre-phrase "al" costs
# 82 domain compounds across the three models — al_qadr (Laylat al-Qadr),
# al_qaïda / al_qaeda, al_azhar, al_hajj, al_houda, dar_al_hadith,
# abd_al_wahhab, aïd_al_fitr — exactly the religious events, organisations
# and titles this collection exists to study. Filtering them one pass later
# keeps every compound and drops only the orphaned fragment.
#
# Also included in the label stopwords below, so labels regenerated from an
# already-trained model lose the bare fragment without a full re-fit.
FRAGMENT_STOPWORDS = {
    "al",    # Arabic definite article: al-Azhar, Aïd al-Fitr, al-Qaïda
    "el",    # same, Maghrebi spelling — keeps the title "El Hadj" intact
    "page",  # "suite page 4" apparatus; keeps page_facebook
}

# ── Junk compounds (also removed AFTER phrase detection) ───────────
# The mirror case of FRAGMENT_STOPWORDS: whole phrases that are apparatus
# even though each part is a legitimate word on its own. They cannot be
# handled pre-phrase without collateral damage — removing "licence" or
# "accorder" would cost a real French noun and allah_accorder/dieu_accorder.
# This is a safety net rather than the primary defence: the pre-phrase
# stopwords above should stop most of these forming at all, but phrase
# detection can re-join what is left behind once a name is stripped out of
# the middle of a string.
JUNK_COMPOUND_STOPWORDS = {
    "licence_accorder",           # residue of the rights notice
    "university_press",
    "indiana_university_press",

    # Academic journal titles. Every part of these is core vocabulary —
    # "religion" alone runs to 5,957 occurrences in the articles corpus,
    # "journal" is the ordinary French word for a newspaper — so they can
    # only be caught whole, after phrasing. The whole family is listed
    # rather than one title: strip a single one and its siblings simply
    # take its place in the labels.
    "journal_religion_africa",
    "journal_africaniste",
    "journal_african",
    "journal_african_history",
    "africa_journal",
    "canadian_journal",
    "canadian_journal_african_studies",
    "canadian_journal_african_revue",
    "war_journal",                # Long War Journal, a source on jihadism
    # NOT listed: journal_officiel (the government gazette — a primary
    # source in this collection, not apparatus).
}

# What the post-phrase pass actually filters.
POST_PHRASE_STOPWORDS = FRAGMENT_STOPWORDS | JUNK_COMPOUND_STOPWORDS

# ── Artefact tokens rejected anywhere inside a label ───────────────
# The sets above match a label candidate as a whole, which is what you want
# almost everywhere — "al" must not veto "al_azhar". These tokens are the
# exception: they are pure digitisation damage, so a label is wrong the
# moment one appears in it, compound included. The distinction matters for
# models trained BEFORE these entered DOMAIN_STOPWORDS, whose dictionaries
# still hold "scanned_camscanner_organisation_pionnier" and friends — a
# re-fit removes them at the root, this keeps labels clean until then.
ARTIFACT_LABEL_STOPWORDS = {
    "camscanner", "scanned",
    "فايروسات", "الفايروسات", "كورونا", "تصيب",
    "amp",
    # Bibliographic apparatus: the rights notice, and cited scholars whose
    # names are named by the bibliography, not discussed by the text.
    "frédérick", "frederick", "madore",
    "gomez", "perez", "hanretta", "weiss",
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

# LDAvis-style relevance weighting for topic LABELS (Sievert & Shirley 2014):
#   relevance(w, k) = λ·log p(w|k) + (1−λ)·log(p(w|k) / p(w))
# λ=1.0 reproduces pure top-probability ranking; λ=0.6 (the LDAvis-recommended
# value) down-weights corpus-common words so labels are more distinctive.
# Pass lambda_relevance=None to get_topic_label for the old pure-probability
# behavior (--no-relevance-labels on the CLI).
DEFAULT_LAMBDA_RELEVANCE = 0.6

# Multi-seed k-sweep stability (--stability-seeds): 1 = single seed
# (exact legacy behavior). N>1 trains N reduced models per candidate k
# (seeds 42, 43, ...), selects k by MEAN C_v, and reports the sd plus a
# topic-stability score (mean best-match Jaccard of top-10 topic words
# between seed pairs).
DEFAULT_STABILITY_SEEDS = 1
STABILITY_TOPN_WORDS = 10   # top words per topic compared for the Jaccard score

# Held-out evaluation (--holdout): fraction of (chunked) training docs set
# aside with seed 42 during the k-sweep; sweep models train on the remainder
# and report held-out log-perplexity. 0.0 = off. The FINAL model always
# retrains on ALL documents.
DEFAULT_HOLDOUT_FRACTION = 0.0

# ── Per-config presets ─────────────────────────────────────────────
# Recommended settings per subset, applied automatically when the
# corresponding CLI flag is NOT given, so a plain
#   lda_topic_modeling.py --config references --mode fit -y
# runs the recommended recipe. Resolution order:
#   explicit CLI > training_parameters.json (predict mode) > preset > defaults.
#
# "num_topics" pins k. That matters more than it looks: k is baked into the
# meaning of the lda_topic_id column, so a preset that re-derives it lets an
# ordinary re-fit silently renumber every topic in the dataset.
# "optimize_topics" re-derives it on every fit and only kicks in when
# --num-topics is not given; --optimize-topics still forces a sweep.
#
# "language_overrides" lets one config serve several languages from separate
# models, keyed by the resolved language and merged over the base preset.
# Without it the English references pass had to carry --model-path by hand,
# and forgetting it overwrote the French model.
CONFIG_PRESETS: dict[str, dict] = {
    "articles": {
        # Whole-document model (press articles are short). k pinned, not
        # swept, to protect the existing column semantics on re-fit.
        "model_path": "lda_model",
        "language": "Français",
        "num_topics": 30,
    },
    "publications": {
        # Full periodical issues are long: chunked training/prediction.
        "model_path": "lda_model_publications",
        "chunk_words": 1000,
        "language": "Français",
        "topic_range": (15, 40, 5),
        "optimize_topics": True,
    },
    "references": {
        # Scholarly texts, one model per language.
        #
        # k is pinned rather than swept because on corpora this small C_v
        # cannot choose it. A 3-seed sweep put every k from 12 to 32 inside
        # 0.014 mean C_v while a single k varied by up to 0.035 across seeds
        # — the ranking is noise, and four successive re-fits duly picked
        # 24, 16, 24 and 32. What does vary systematically is topic
        # stability (mean best-match Jaccard between seeds), which falls as
        # k rises: 0.395 at k=8 against 0.274 at k=16 in the English model.
        # These values trade that stability off against granularity, giving
        # both models ~16 labelled documents per topic (280/16 and 123/8).
        # Re-check with --optimize-topics --stability-seeds 3 if the corpus
        # grows substantially; judge by stability, not by C_v.
        "model_path": "lda_model_references",
        "chunk_words": 1000,
        "language": "Français",
        "num_topics": 16,
        "topic_range": (12, 32, 4),
        "language_overrides": {
            "Anglais": {
                "model_path": "lda_model_references_en",
                "num_topics": 8,
                "topic_range": (8, 24, 4),
            },
        },
    },
}
