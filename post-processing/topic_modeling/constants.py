"""
constants.py
-------------
Shared constants for topic modeling pipeline.
"""

# Domain-specific stopwords to de-emphasize boilerplate in labels (c-TF-IDF)
# lower-case; CountVectorizer lowercases by default
DOMAIN_STOPWORDS = {
    "el",
    "cfa",
    "monsieur",
    "madame",
    "excellence",
    "président",
    "ministre",
    # phrase kept for clarity; unigram components above will remove it effectively
    "excellence monsieur",
}
