"""
constants.py
-------------
Shared constants for topic modeling pipeline.

IMPORTANT: This is a collection about Islam in West Africa.
Islamic organizations (COSIM, FAIB, UIB, etc.) and religious events 
(Ramadan, Tabaski, Maouloud, etc.) are CORE to the research and should
appear in topic labels. We only remove truly non-informative noise.
"""

# Lower-case; CountVectorizer lowercases by default

# Tokens to drop from LABELS only (kept in vectorization/clustering unless also present in VECTORIZE_STOPWORDS)
# These make labels less readable but don't affect clustering
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

# Minimal set to drop during vectorization (impacts clustering). 
# Be VERY conservative here - only remove words that truly add noise to clustering.
VECTORIZE_STOPWORDS = {
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

# Backward-compat alias used in existing imports
DOMAIN_STOPWORDS = VECTORIZE_STOPWORDS
