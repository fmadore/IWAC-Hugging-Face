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

    # "grand", "mondial", "international" moved to FRAGMENT_STOPWORDS: noise
    # alone, but they carry grand_imam, grand_mosquée, ligue_islamique_mondial.

    # Functional words that leak through lemmatization ("pouvoir" moved to
    # FRAGMENT_STOPWORDS: the noun anchors pouvoir_public, pouvoir_coutumier)
    "ledit", "faire", "devoir",

    # OCR artifacts ("wa" moved to FRAGMENT_STOPWORDS: the Arabic conjunction
    # inside ahl_sunna_wa_jamaa; "at" is in the English list below)
    "lp", "bf", "adj", "octet", "sem",

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
    "aller", "falloir", "vouloir", "voir", "celer",
    # "el" / "al" live in FRAGMENT_STOPWORDS below — see the note there, as
    # does "savoir" (the noun: savoir_islamique, transmission_savoir).

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
    # Known limitation: "un" and "au" also strip the English acronyms UN and
    # AU. The lemmatiser lowercases, so "the UN and the AU" and "un imam au
    # Mali" yield the same tokens; an English exemption would bring the French
    # residue back. Fixing it needs case-aware lemmas (a lemma-logic change).
    "la", "le", "les", "des", "du", "de", "un", "une", "et", "ou",
    "au", "aux", "dans", "sur", "par", "pour", "que", "qui",
    "sont", "ce", "cette", "ces", "sa", "ses", "leur", "leurs",
    "à", "en", "ne", "est",
    "il", "elle", "ils", "pas", "nous",
    # (de_la / de_l / a_la are compounds and live in JUNK_COMPOUND_STOPWORDS:
    # nothing underscore-joined exists yet when this set is applied.)

    # GERMAN function words: the references corpus quotes German colonial
    # and missionary sources, and neither the French nor the English spaCy
    # stopword list strips them. "die" is exempted for the English model
    # (LANGUAGE_PRE_PHRASE_EXEMPTIONS), where it is a verb.
    "und", "der", "von", "das", "die",

    # Bibliographic apparatus abbreviations (reference lists, citations).
    # "op" and "cit" are listed separately because matching happens before
    # phrase detection; op_cit itself is in JUNK_COMPOUND_STOPWORDS.
    "ed", "eds", "éd", "dir", "pp", "op", "cit", "ibid", "idem", "fig",

    # Publishers and places of publication — footnote apparatus, never a
    # topic. Stripping them here, pre-phrase, is what dissolves the whole
    # citation family: "paris" alone anchors 32 compounds in the French
    # references model (paris_armand_colin, paris_cnrs, afrique_noir_paris),
    # and "london_hurst" is another publisher. "york" alone is listed here;
    # "new" is a post-phrase fragment (it heads new_media, new_testament,
    # new_religious_movement), and new_york is caught whole after phrasing.
    #
    # The university-press cities below occur ONLY in the references
    # subsets, never in the press articles — the signature of apparatus
    # rather than content. Berlin, Londres and Genève are deliberately
    # absent for the opposite reason: they appear in the articles corpus as
    # real news subjects, and the Berlin Conference is part of the history
    # this collection studies.
    "karthala", "harmattan", "brill", "routledge",
    "paris", "london", "york",
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
    # co-author strings too (savadogo_gomez_perez), so "savadogo" need not be
    # listed and is better left out: Savadogo is an everyday Burkinabè
    # surname. For the same reason "souza" is absent — it IS in the press
    # corpus, naming the Afro-Brazilian families of Ouidah
    # (francisco_félix_souza, famille_souza).
    "muriel", "gomez", "perez", "sean", "hanretta", "weiss", "caleb",
    "playbook",  # from the report title weiss_aqim_imperial_playbook;
                 # "aqim" stays, it is a real organisation
    # Once gomez_perez was gone, the 2026-11 re-fit grew
    # marie_nathalie_leblanc (61 English chunks) into a topic label.
    # "leblanc" occurs in neither the articles nor the publications
    # dictionary; "marie" and "nathalie" do, so only the surname goes.
    "leblanc",

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
#
# The same holds for generic adjectives and nouns that head or modify an
# Islamic organisation, title, event, place of worship or calendar term:
# junk alone, the object of study inside a compound. They used to be
# pre-phrase stopwords and silently prevented every compound listed beside
# them. A word belongs in the pre-phrase sets only when its WHOLE compound
# family is noise (digitisation damage, citation apparatus, function words).
FRAGMENT_STOPWORDS = {
    "al",    # Arabic definite article: al-Azhar, Aïd al-Fitr, al-Qaïda
    "el",    # same, Maghrebi spelling — keeps the title "El Hadj" intact
    "wa",    # Arabic conjunction: ahl_sunna_wa_jamaa
    "page",  # "suite page 4" apparatus; keeps page_facebook

    # Titles, organisations, places of worship
    "grand",          # grand_imam, grand_mosquée, grand_marabout
    "mondial",        # ligue_islamique_mondial
    "international",  # conférence_islamique_international
    "national", "régional", "regional",  # conseil_national_islamique
    "local",          # comité_local, langue_local
    "social",         # oeuvre_social, action_social
    "premier", "première", "premiere",   # premier_ministre, premier_imam
    "dernier", "dernière", "derniere",   # jugement_dernier
    "nouveau", "nouvelle", "nouveaux",   # nouveau_an, nouveau_testament
    "petit", "petite",                   # petit_séminaire
    "lieu",           # lieu_culte
    "monde",          # monde_musulman, monde_arabe
    "nom",            # nom_allah, nom_dieu (basmala)
    "ville",          # ville_sainte
    "moyen",          # moyen_orient
    "non",            # non_musulman
    "pouvoir",        # pouvoir_public, pouvoir_coutumier
    "savoir",         # savoir_islamique, transmission_savoir
    "travail",        # travail_forcé (colonial forced labour)

    # Calendar and time: the lunar calendar is part of the subject
    "mois",           # mois_ramadan, mois_sacré
    "jour",           # jour_arafat, jour_jugement
    "an",             # nouveau_an, an_hégire (French only; English "an"
                      # stays pre-phrase, see LANGUAGE_PRE_PHRASE_EXEMPTIONS)
    "année", "annee",
    "heure",          # heure_prière
    "période", "periode",  # période_coloniale

    # English
    "new",            # new_testament, new_religious_movement, new_media
    "coast",          # gold_coast; "ivory" itself is a geo stopword
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

    "marie_nathalie",             # safety net once "leblanc" is stripped

    # Thesis apparatus: title pages and running headers of the mémoires and
    # theses in references. A handful of long theses carry them on every
    # page, so in the 2026-11 French re-fit they took over whole topics
    # (mémoire_maîtrise_kont_drissa, tabl_matiere - tome_ii - chapitre_ii).
    # Each compound has zero occurrences in the articles dictionary. NOT
    # listed: mémoire_collectif (collective memory is a subject), and
    # entretien_réaliser, which the press uses too (60 articles chunks).
    "mémoire_maîtrise",
    "mémoire_maîtrise_histoire",
    "mémoire_maîtrise_histoire_université",
    "mémoire_maîtrise_université",
    "mémoire_maîtrise_kont_drissa",
    "mémoire_master",
    "thèse_doctorat",
    "thèse_doctorat_cycle",
    "thèse_doctorat_etat",
    "thèse_doctorat_unique",
    "thèse_doctorat_université",
    "thèse_cycle",
    "thèse_unique",
    "thèse_mémoire",
    "département_histoire",
    # Document structure: front matter and chapter headings.
    "tabl_matiere",
    "table_matière",
    "tome_ii",
    "chapitre_ii",
    "chapitre_iii",
    "chapitre_iv",
    "chapitre_vi",
    "cadre_géographique",
    "source_oral",
    # Running headers and cited titles, as with the journal titles above.
    "arabisant_diffusion",
    "drissa_arabisant",
    "radicalisme_islamique_sud_sahara",
    # Archival shelfmarks in footnotes ("ANS, carton …, dossier …").
    "carton_dossier_archives_nationales",
    "archives_nationales",

    # Moved here from the pre-phrase sets, where an underscore entry can
    # never match (nothing is joined yet). Their components are stripped
    # pre-phrase, so these are a safety net for labels of older models.
    "op_cit", "de_la", "de_l", "a_la",
    "burkina_faso", "côte_ivoire", "cote_ivoire",
    "bobo_dioulasso", "porto_novo",
    "new_york", "ivory_coast",
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
    "leblanc", "muriel", "sean", "caleb",
    # porto_novo_république reached a references_en label: the city parts
    # are pre-phrase stopwords now, this keeps older models' labels clean.
    "porto", "novo",
}

# ── Label-only stopwords ───────────────────────────────────────────
# Dropped from topic LABELS only (keeps labels readable); these words
# still participate in training.
LABEL_ONLY_STOPWORDS = {
    # Words already in a modelling set (DOMAIN, GEO, GENERIC, FRAGMENT,
    # JUNK) are not repeated here: the label filter unions them all.
    # A. Devotional formulae (abbreviations that are hard to interpret in labels)
    "psl", "saw", "swt",

    # B. ENGLISH stopwords (from OCR/non-French docs leaking through)
    "which", "have", "has", "their", "it", "its", "they", "will", "can",
    "all", "we", "been", "would", "were", "there", "who", "what", "more",
    "but", "if", "not", "so", "when", "other", "than", "no", "also", "into",

    # C. Dates & meeting filler (not topic-specific)
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin", "juillet",
    "août", "aout", "septembre", "octobre", "novembre", "décembre", "decembre",
    "aujourd'hui",

    # D. Currency/quantities (numbers, not concepts)
    "fcfa", "franc", "francs", "cfa",
    "montant", "montants", "somme", "sommes",
    "millier", "milliers", "million", "millions", "milliard", "milliards",

    # F. OCR artifacts & abbreviations (garbage tokens)
    "dr", "mr", "mme", "mm",
    "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",

    # G. Generic location words (not specific places)
    "place", "rond", "point",

    # I. Generic French functional/filler words that leak through lemmatization
    "deja", "déjà", "lors",
    "sous",
    "cela", "ceci", "celle", "ceux", "celui",
    "quelque", "quelques", "certain", "certains", "certaine", "certaines",
    "chaque", "tel", "telle", "tels", "telles",
    "peu", "assez", "trop", "combien",
    "comment", "pourquoi", "quand",
    "deuxième", "deuxieme",
    "seul", "seule", "seuls", "seules",
    "petits", "petites",
}

# ── Geographic stopwords ───────────────────────────────────────────
# The dataset already has a 'country' metadata field, so geographic
# tokens produce country-dominant topics instead of thematic ones.
# Both accented and unaccented forms are needed: accented for
# tokenization (direct string match), unaccented for label filtering
# (which normalises/strips accents before comparison).
#
# Kept pre-phrase on purpose, unlike the generic adjectives in
# FRAGMENT_STOPWORDS: stripping the country still lets the generic part of an
# organisation's name form (conseil_supérieur_imam from "… imams du Bénin"),
# and only the country, which the metadata already carries, is lost.
#
# Hyphenated and apostrophe forms are single lemma tokens (lemmas never
# contain spaces), so they match here, before phrase detection, as written;
# both apostrophes are listed because matching is exact. Underscore compounds
# can only form after this pass and live in JUNK_COMPOUND_STOPWORDS.
LDA_GEO_STOPWORDS = {
    # West African countries
    "bénin", "benin",
    "burkina", "faso", "burkina-faso",
    "côte", "cote", "ivoire",
    "côte-d'ivoire", "côte-d’ivoire", "cote-d'ivoire", "cote-d’ivoire",
    "côte-ivoire", "cote-ivoire",
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
    "bobo", "dioulasso", "bobo-dioulasso",
    "abobo",
    "adjamé", "adjame",
    "gagnoa",
    "yamoussoukro",
    "korhogo",
    "porto", "novo", "porto-novo",
    # Demonyms
    "ivoirien", "ivoirienne", "ivoiriens",
    "burkinabé", "burkinabe", "burkinabè",
    "togolais", "togolaise",
    "béninois", "beninois", "béninoise", "beninoise",
    "sénégalais", "senegalais", "sénégalaise", "senegalaise",
    "nigérien", "nigerien", "nigérienne", "nigerienne",
    "malien", "malienne",
    "ghanéen", "ghaneen", "ghanéenne", "ghaneenne",
    "nigérian", "nigerian", "nigériane", "nigeriane", "nigérians", "nigerians",
    # English forms: English labels kept the geography the French ones lose
    # (references_en topic 0 read "… - dioula - ivorian - marriage").
    "ivorian", "ivorians", "beninese", "togolese", "burkinabes",
    "malian", "malians", "ghanaian", "ghanaians", "nigeriens",
    "senegalese",
    # "ivory" alone: "coast" heads gold_coast (colonial Ghana), so it is a
    # post-phrase fragment instead (FRAGMENT_STOPWORDS).
    "ivory",
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
    # Generic nouns (moyen, travail, heure, lieu, monde, nom, ville moved to
    # FRAGMENT_STOPWORDS: each heads domain compounds)
    "situation", "problème", "probleme", "besoin",
    "niveau", "temps", "chose", "rien",
    "fois",
    "moment", "part", "occasion", "difficulté", "difficulte",
    "argent",
    # Date / time (année, an, jour, mois, période moved to FRAGMENT_STOPWORDS:
    # mois_ramadan, jour_arafat, nouveau_an, période_coloniale)
    "hier", "demain", "lendemain", "date",
    # Generic adjectives (social, local, national, régional, dernier,
    # premier, nouveau, petit moved to FRAGMENT_STOPWORDS)
    "différent", "different",
    # Other ("non" moved to FRAGMENT_STOPWORDS: non_musulman)
    "beaucoup", "oui",
}

# ── Language-specific exemptions from the pre-phrase sets ─────────
# The pre-phrase sets are shared by every model, but a few entries are noise
# in one language and vocabulary in the other. Exempted words are not
# stripped before phrase detection for that model's language.
LANGUAGE_PRE_PHRASE_EXEMPTIONS: dict[str, frozenset[str]] = {
    # German article, English verb.
    "Anglais": frozenset({"die"}),
    # English article, French "year": exempt here and filtered post-phrase
    # via FRAGMENT_STOPWORDS, so nouveau_an survives and bare "an" does not.
    "Français": frozenset({"an"}),
}


def modeling_stopwords(language: str | None) -> set[str]:
    """The pre-phrase stopword set for one model language."""
    shared = set(DOMAIN_STOPWORDS) | LDA_GEO_STOPWORDS | LDA_GENERIC_STOPWORDS
    return shared - LANGUAGE_PRE_PHRASE_EXEMPTIONS.get(language or "", frozenset())

# ── Custom collocations ────────────────────────────────────────────
# Domain-specific multi-word expressions that should ALWAYS be joined,
# regardless of gensim's statistical phrase detection threshold.
# Each entry is a tuple of consecutive tokens → joined with "_".
# Applied after gensim's bigram/trigram phrase detection, so these
# act as a safety net for collocations the statistics miss — and BEFORE the
# post-phrase fragment filter, so a fragment word ("social") can still be
# joined here. ``lemma_nostop`` holds lemmas, so the masculine lemma form is
# the one that fires; the feminine surface forms are kept, harmlessly, for
# text that escaped lemmatisation.
CUSTOM_COLLOCATIONS: list[tuple[str, ...]] = [
    # Religious concepts
    ("nuit", "destin"),               # Laylat al-Qadr
    ("finance", "islamique"),         # Islamic finance
    ("école", "coranique"),           # Quranic school
    ("ecole", "coranique"),           # (unaccented variant)
    ("droit", "homme"),               # human rights
    ("société", "civil"),             # civil society (lemma)
    ("societe", "civil"),             # (unaccented variant)
    ("société", "civile"),            # (surface form)
    ("societe", "civile"),
    ("conseil", "supérieur"),         # supreme council
    ("conseil", "superieur"),         # (unaccented variant)
    ("oeuvre", "social"),             # social work (lemma; "œ" is folded to "oe")
    ("oeuvre", "sociale"),            # (surface form)
    ("voile", "islamique"),           # Islamic veil
    ("état", "civil"),                # civil registry
    ("etat", "civil"),                # (unaccented variant)
    ("liberté", "religieux"),         # religious freedom (lemma)
    ("liberte", "religieux"),         # (unaccented variant)
    ("liberté", "religieuse"),        # (surface form)
    ("liberte", "religieuse"),
    ("dialogue", "interreligieux"),   # interfaith dialogue
    ("extrémisme", "violent"),        # violent extremism
    ("extremisme", "violent"),        # (unaccented variant)
    ("radicalisation", "religieux"),  # religious radicalisation (lemma)
    ("radicalisation", "religieuse"), # (surface form)
]

# LDA-specific defaults
# With geographic and generic stopwords removed, 30 topics avoids the
# junk / country-dominant topics that appeared at 40.
# --optimize-topics --sweep-only reports a k sweep (C_v with seed sd, seed
# stability, held-out bound) without fitting or publishing anything; k
# stays an explicit, pinned choice.
DEFAULT_NUM_TOPICS = 30
DEFAULT_PASSES = 25          # more passes = better convergence on CPU
DEFAULT_ITERATIONS = 600     # allow more iterations per E-step
DEFAULT_CHUNKSIZE = 2000
DEFAULT_RANDOM_STATE = 42
DEFAULT_MINIMUM_PROBABILITY = 0.01
# Dictionary document-frequency bounds are counted in SOURCE documents
# (distinct values of the group column, o:id by default), not in training
# chunks: chunked, one book would otherwise pass no_below on its own.
DEFAULT_NO_BELOW = 10        # ignore tokens in fewer than 10 source documents
DEFAULT_NO_ABOVE = 0.40      # ignore tokens in more than 40% of source documents
# Phrase detection counts occurrences, so one thesis repeating a running
# header twenty times mints a compound (mémoire_maîtrise_kont_drissa). A
# learned compound is applied only if it occurs in at least this many
# distinct source documents. Training-only: the pruned phrase models are
# what the bundle saves, so prediction applies exactly the kept compounds.
DEFAULT_PHRASE_MIN_DOCS = 5
# Training drops documents with fewer whitespace tokens than this
# (--min-train-tokens); prediction leaves the topic columns null for a
# document with fewer IN-VOCABULARY tokens than its floor, which defaults to
# the model's training floor (--min-predict-tokens).
DEFAULT_MIN_TRAIN_TOKENS = 5

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

# Multi-seed k-sweep stability (--stability-seeds): 1 = single seed.
# N>1 trains N reduced models per candidate k (seeds 42, 43, ...) and reports
# mean C_v with its sd plus a topic-stability score (mean best-match Jaccard
# of top-10 topic words between seed pairs). The sweep is a diagnostic; it
# chooses k only with --select-k-from-sweep, and then says when the best and
# runner-up mean C_v lie within one seed sd of each other.
DEFAULT_STABILITY_SEEDS = 1
STABILITY_TOPN_WORDS = 10   # top words per topic compared for the Jaccard score

# Held-out evaluation (--holdout): fraction of independent source-document
# groups (--holdout-group-column, exact duplicates kept together) set aside
# with seed 42 BEFORE phrase and vocabulary fitting; sweep models train on
# the remainder and report gensim's held-out per-word log-perplexity bound.
# That bound infers each held-out document's topic mixture from the same
# words it scores — it is not a document-completion estimate. 0.0 = off.
DEFAULT_HOLDOUT_FRACTION = 0.0

# ── Per-config presets ─────────────────────────────────────────────
# Recommended settings per subset, applied automatically when the
# corresponding CLI flag is NOT given, so a plain
#   lda_topic_modeling.py --config references --mode fit -y
# runs the recommended recipe. Resolution order:
#   explicit CLI > training_parameters.json (predict mode) > preset > defaults.
#
# "num_topics" pins the intended granularity, not topic identity. Even at
# fixed k, re-fitting can reorder or change topics. Immutable artifact IDs
# disambiguate every fitted model; downstream keys are (model_id, topic_id).
# A pinned k (preset or --num-topics) always wins for the production fit:
# with one, --optimize-topics only reports the sweep and exits.
#
# "phrase_min_docs" overrides DEFAULT_PHRASE_MIN_DOCS per subset.
#
# "language_overrides" lets one config serve several languages from separate
# models, keyed by the resolved language and merged over the base preset.
# Without it the English references pass had to carry --model-path by hand,
# and forgetting it overwrote the French model.
CONFIG_PRESETS: dict[str, dict] = {
    "articles": {
        # Whole-document model (press articles are short). k pinned, not
        # swept, to retain the chosen granularity (not topic identity).
        "model_path": "lda_model",
        "language": "Français",
        "num_topics": 30,
    },
    "publications": {
        # Full periodical issues are long: chunked training/prediction.
        #
        # k pinned at 20 after the first fit (2026-08-11), for the same
        # reason articles and references pin theirs: k is baked into the
        # meaning of the lda_topic_id column, and re-deriving it on every
        # fit silently renumbers every topic in the published dataset.
        # This preset carried optimize_topics=True until that first fit
        # had a value to pin.
        #
        # Unlike references, the metrics here DO discriminate. The sweep
        # over k=15..40 (5,722 chunks from 1,451 of 1,501 issues, seed 42):
        #
        #     k   C_v      NPMI     U_Mass
        #     15  0.5549   0.0512   -1.4291
        #     20  0.5596   0.0513   -1.5886   <- selected
        #     25  0.5344   0.0208   -2.0130
        #     30  0.5416   0.0175   -2.4424
        #     35  0.5488   0.0202   -2.6847
        #     40  0.5554   0.0291   -2.3616
        #
        # C_v alone looks flat and even recovers at k=40 — read on its own
        # it would suggest the ranking is noise. NPMI contradicts that
        # outright: it holds at ~0.051 for k<=20 and HALVES to ~0.02 from
        # k=25 on, and U_Mass degrades monotonically over the same range.
        # Three metrics agreeing on "<=20, not >=25" is signal; C_v's
        # recovery at 40 is the outlier. 15 vs 20 is genuinely a coin flip
        # (NPMI 0.0512 vs 0.0513), so 20 wins on granularity: ~73 labelled
        # issues per topic against ~97 at k=15.
        #
        # Re-check with --optimize-topics --sweep-only --stability-seeds 3
        # if the corpus grows substantially (report only: no bundle, no
        # push), and judge by topic stability rather than C_v — but expect
        # to have to justify renumbering everything downstream before
        # changing this number.
        "model_path": "lda_model_publications",
        "chunk_words": 1000,
        "language": "Français",
        "num_topics": 20,
        "topic_range": (15, 40, 5),
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
        # Re-check with --optimize-topics --sweep-only --stability-seeds 3
        # if the corpus grows substantially (report only: no bundle, no
        # push); judge by stability, not by C_v.
        "model_path": "lda_model_references",
        "chunk_words": 1000,
        # Ten long books and theses supplied 30% of the French training
        # chunks; uncapped, each claimed a topic and 46% of documents fell
        # into one generic topic. At 16 chunks per document (2026-11, seeds
        # 42-44): largest topic 17%, no topic owned by a single book, shared
        # C_v 0.478 -> 0.520, stability 0.359 -> 0.320 (the uncapped figure
        # is inflated by book topics, which every seed recovers).
        "max_chunks_per_doc": 16,
        "language": "Français",
        "num_topics": 16,
        "topic_range": (12, 32, 4),
        "language_overrides": {
            "Anglais": {
                "model_path": "lda_model_references_en",
                "num_topics": 8,
                "topic_range": (8, 24, 4),
                # Already balanced uncapped (2-36 documents per topic); a cap
                # lowered both C_v and stability.
                "max_chunks_per_doc": None,
            },
        },
    },
}


def resolve_preset(config_name: str, language: str | None = None) -> tuple[str, dict]:
    """``(language, preset)`` for a subset, with that language's overrides folded in.

    The language defaults to the preset's own, else ``"Français"``. Shared by
    the LDA stage and the analyses that load its bundles, so both resolve the
    same model directory for a given subset and language.
    """
    preset = CONFIG_PRESETS.get(config_name, {})
    language = language or preset.get("language", "Français")
    merged = {**preset, **preset.get("language_overrides", {}).get(language, {})}
    return language, merged


# ── Topic-model exclusions (reviewed) ──────────────────────────────
# Rows that stay in the dataset (and public, where they are) but must never
# enter topic modelling, keyed by subset and o:id (as a string), each with the
# reviewed reason. An excluded row is dropped from training before phrase
# detection and dictionary building, and at prediction it gets null topic
# columns while keeping lda_model_name, the shape the --min-predict-tokens
# evidence floor gives a row the model declined.
#
# The registry is applied at prediction from current code, whatever bundle is
# loaded, so an older bundle (trained with these rows in) still nulls them; a
# fit records the exclusions it applied in training_parameters.json.
#
# Add an entry only after reading the item; remove one only when the reason no
# longer holds (for the Islam Info issues: once the lists are removed from
# their text).
_PILGRIM_LISTS = (
    "Islam Info issue whose text is mostly tabular lists of pilgrims' names; "
    "these lists form a spurious name-list topic (topic 18 of periodical bundle lda-sha256:bd68e625...)"
)
TOPIC_MODEL_EXCLUSIONS: dict[str, dict[str, str]] = {
    "publications": {
        oid: _PILGRIM_LISTS
        for oid in ("23757", "24190", "24233", "24234", "24635", "25286", "25287", "25289")
    },
}


def topic_model_exclusions(config_name: str) -> dict[str, str]:
    """``{o:id: reason}`` excluded from topic modelling for this subset."""
    return dict(TOPIC_MODEL_EXCLUSIONS.get(config_name, {}))
