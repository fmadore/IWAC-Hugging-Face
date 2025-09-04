"""
constants.py
-------------
Shared constants for topic modeling pipeline.
"""

# Lower-case; CountVectorizer lowercases by default

# Tokens to drop from LABELS only (kept in vectorization/clustering unless also present in VECTORIZE_STOPWORDS)
LABEL_ONLY_STOPWORDS = {
    # A. Devotional formulae & variants
    "psl", "saw", "swt", "bismillah", "inshallah",
    "muhammad", "mohamed", "mohammed", "mahomet",
    "prophète", "prophete", "taala", "taâla",
    # B. Titles & honorifics (label context)
    "imam", "grand imam", "imam ratib",
    "hadj", "hadja", "elhadj", "el-hadj", "el hadj", "alhaji",
    "cheikh", "cheick", "cheïkh", "cheik",
    "chérif", "cherif", "serigne",
    # C. Institutional boilerplate
    "islam", "islamique",
    "musulman", "musulmans", "musulmane", "musulmanes",
    "religieux", "religieuse",
    "union", "communauté", "communaute",
    "conseil", "centre",
    "supérieur", "superieur",
    "national", "régional", "regional",
    "association", "fédération", "federation",
    "ligue", "commission", "comité", "comite",
    "réseau", "reseau", "mouvement",
    "organisation", "organisations",
    "cadre", "bureau",
    "culture", "culturel", "culturelle", "culturels", "culturelles",
    # D. Dates & meeting filler
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin", "juillet",
    "août", "aout", "septembre", "octobre", "novembre", "décembre", "decembre",
    "année", "annee", "mois", "jour",
    "aujourd'hui", "aujourd’hui", "hier", "demain",
    "séance", "seance", "cérémonie", "ceremonie", "festivité", "festivite", "festivités", "festivites",
    # E. Currency/quantities & adminese
    "fcfa", "franc", "francs", "cfa",
    "montant", "montants", "somme", "sommes", "enveloppe", "enveloppes",
    "don", "dons", "millier", "milliers", "million", "millions", "milliard", "milliards",
    "budget", "budgets", "appui", "soutien",
    # F. Toponyms & demonyms (label-only, keep for clustering)
    # West Africa core (countries and common demonyms; accents + plain)
    "sénégal", "senegal", "sénégalais", "senegalais", "sénégalaise", "senegalaise", "sénégalaises", "senegalaises",
    "mali", "malien", "maliens", "malienne", "maliennes",
    "guinée", "guinee", "guinéen", "guineen", "guinéens", "guineens", "guinéenne", "guineenne",
    "benin", "bénin", "beninois", "béninois", "beninoise", "béninoise", "beninoises", "béninoises",
    "togo", "togolais", "togolaise", "togolaises",
    "ghana", "ghanéen", "ghaneen", "ghanéens", "ghaneens", "ghanéenne", "ghaneenne",
    "nigeria", "nigéria", "nigérian", "nigerian", "nigérians", "nigerians", "nigériane", "nigeriane",
    "niger", "nigérien", "nigerien", "nigériens", "nigeriens", "nigérienne", "nigerienne",
    "burkina", "burkina faso", "burkinabè", "burkinabe",
    "côte d'ivoire", "cote d'ivoire", "ivoire", "ivoirien", "ivoiriens", "ivoirienne", "ivoiriennes",
    "gambie", "gambien", "gambiens", "gambienne",
    "mauritanie", "mauritanien", "mauritaniens", "mauritanienne",
    # Capitals/major cities (common in press leads)
    "dakar", "bamako", "conakry", "cotonou", "lomé", "lome", "accra", "abuja", "niamey", "ouagadougou", "abidjan", "banjul", "nouakchott",
    # G. Places of worship and common locality names inflating labels
    "mosquée", "mosquees",
    "yopougon", "adjamé", "adjame", "abobo", "koumassi", "treichville", "riviera",
    "cadjèhoun", "cadjehoun", "korhogo", "porto-novo", "parakou", "yamoussoukro", "dédougou", "dedougou",
    # Unigram tokens for Côte d'Ivoire phrase coverage
    "côte", "cote",
    # H. Personal names (label-only; keep in clustering)
    "mamadou", "ibrahim", "ibrahima", "aboubacar", "aboubakar", "ousmane", "abdoul",
    "traoré", "traore", "koné", "kone", "coulibaly", "sidibé", "sidibe", "cissé", "cisse", "diaby", "doumbia", "konaté", "konate", "fofana", "ouédraogo", "ouedraogo",
    # I. Orthographic/term variants
    "oumma", "oummat", "ummah",
    # Event name variants (label-only if you prefer sub-topic emphasis)
    "mawlid", "maoulid", "maouloud",
    # H. Odd tokens & cleanup
    "allah", "paix", "bénédiction", "benediction",
    "frère", "frere", "soeur",
    "chers", "cher", "chère", "chere",
    "lumière", "lumiere", "luire", "prière", "priere", "sermon",
    # Places in devotional contexts
    "ramadan", "aid", "aïd", "oumra", "oumrah", "mecque", "médine", "medine",
    # Generic courtesy/political boilerplate
    "monsieur", "madame", "excellence", "président", "president", "ministre", "gouvernement", "communiqué", "communique",
    # Frequent generic words seen in labels
    "grand", "mondial", "international",
    # Special-case phrases and OCR strays
    "excellence monsieur", "sem", "at",
    # Contextual honorific: maître when tied to coranique (leave plain "maître" untouched)
    "maitre coranique", "maître coranique",
}

# Minimal set to drop during vectorization (impacts clustering). Keep conservative.
VECTORIZE_STOPWORDS = {
    # courtesy/boilerplate that adds little topical signal across the corpus
    "monsieur", "madame", "excellence", "communiqué", "communique",
    # very frequent domain words without discriminative value
    "allah",
    "paix", "bénédiction", "benediction", "frère", "frere", "soeur", "chers", "cher", "chère", "chere",
    "lumière", "lumiere", "luire", "prière", "priere", "sermon",
    # frequent generics in labels
    "grand", "national", "million", "milliard",
    "communauté", "communaute", "croyant", "coreligionnaire",
    "organiser", "organisation", "mondial", "international",
    # morphological/functional
    "ledit", "entrer", "faire", "devoir", "pouvoir",
    # legacy/ocr
    "el", "sem", "at",
    # phrase (rarely helpful)
    "excellence monsieur",
}

# Backward-compat alias used in existing imports
DOMAIN_STOPWORDS = VECTORIZE_STOPWORDS
