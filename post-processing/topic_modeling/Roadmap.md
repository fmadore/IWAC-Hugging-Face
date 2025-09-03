Two things stand out:

- Nearly half the documents are outliers (-1 = 5,465 / 11,664 ≈ 47%). That’s too high and indicates either the embedding space is not well organized for your data or the clustering is too strict.
- Topic labels are dominated by boilerplate/formulaic language and proper names (e.g., “Cher Frère…”, “Luire…”, “Hadj…”, “Laurent Gbagbo…”, “Ouédraogo…”, “Cfa…”). This masks real themes.

Below is a practical plan tailored for CPU-only and ~10k newspaper articles. It combines a few quick wins plus some structural improvements to your script to significantly reduce outliers and improve topic quality and labels.

What causes the high outlier rate and noisy topics

- Embedding content: You embed lemma_nostop. Sentence-transformers perform better on natural sentences, not lemmatized bag-of-words. Using lemma_nostop for embeddings degrades the neighborhood structure, creating more outliers.
- Clustering and UMAP: Default projections can create fragmented neighborhoods. HDBSCAN thresholds also impact outlier counts.
- Lexical representation: labels are pulled from c-TF-IDF on raw tokens (here lemma_nostop). If frequent boilerplate and proper names aren’t controlled, labels become noisy.
- Domain boilerplate: greetings, religious formulae, and political names appear in many docs; they overshadow topical words and produce weird labels (“Luire”, “Cher Frère”, “Blaise Compaoré” everywhere).
- Duplicate or near-duplicate news: a few large near-duplicate groups produce giant topics, and everything else becomes spiky noise.

High impact changes (CPU-friendly)

1. Use original text for embeddings, lemma_nostop for representation

- Keep using lemma_nostop for CountVectorizer/c-TF-IDF (labels), but feed the original French sentences to the embedding model.
- BERTopic supports docs_clean: pass docs (full French text) for embeddings and docs_clean (lemma_nostop) for vectorization. This alone typically reduces outliers substantially.

2. Add domain stopwords and frequency controls

- Curate a small domain stoplist to remove boilerplate and over-represented names/greetings: Examples to consider (adjust after inspecting top terms): allah, imam, hadj, ramadan, aid, bismillah, paix, bénédiction, frère, soeur, chers, cher, luire, priere, sermon, grand, national, million, milliard, cfa, monsieur, madame, excellence, excellence monsieur, président, ministre, gouvernement, communiqué, monde, organisation, islamique. For local names that show up in every article, consider frequency-based filters rather than hard stopwords, or add them to stopwords only if they dominate too many topics.
- In CountVectorizer set min_df to 10 and max_df to 0.9 to damp global boilerplate. Increase max_features to 20k–30k and ngrams to (1,3) to capture phrases.

3. Adjust UMAP/HDBSCAN for fewer outliers

- UMAP: increase n_neighbors and shrink n_components; use min_dist closer to 0 for tighter clusters. Good CPU defaults for news: n_neighbors=150, n_components=5, min_dist=0.0, metric='cosine'.
- HDBSCAN: make it more permissive so fewer outliers. Try min_samples=1 or 2, cluster_selection_method='leaf', and choose min_cluster_size based on the number of docs and desired topic count. As a rule of thumb, min_cluster_size ≈ len(texts) / desired_topics. If you want ~120 topics for 10k docs, min_cluster_size ≈ 80. For 6k–10k docs and 120–150 topics, something like 40–90 works well.

4. Reassign outliers more aggressively

- During training, use reduce_outliers with a lower threshold (0.05–0.15) to reassign many border points into clusters.
- At prediction time, your outlier reassign threshold of 0.35 is too strict. With CPU and HDBSCAN, set it to 0.05–0.10. It will reduce -1 dramatically without melting all topics together.

5. Better topic labels

- Use BERTopic’s ClassTfidfTransformer with reduce_frequent_words=True, and pass a tuned stopword list.
- Keep KeyBERTInspired if available, and consider increasing top_n_words and diversity.
- Filter out low-information parts of speech in labels by feeding cleaned docs (lemma_nostop that already removed stopwords) and using n-grams.
- Limit label length to 5–6 words to reduce noise and redundancy.

6. Optional but helpful: handle near-duplicates before training

- In news corpora, duplicates are common. They create disproportionally large clusters that distort the structure. At minimum, drop exact duplicates by hashing the cleaned text.
- If you can, also drop near-duplicates via MinHash LSH (datasketch) at Jaccard threshold 0.9–0.95; on 10k docs this is still feasible and greatly stabilizes topics.

7. If you have title + body, weight the title

- Concatenate title + ". " + body for embeddings, or duplicate title to weight it: doc_for_embeddings = f"{title}. {title}. {body}". Titles often anchor topics well.

Concrete code changes Below are safe modifications you can drop into your script. They keep your CLI but add:

- Use docs vs docs_clean
- Domain stopwords
- Tuned UMAP/HDBSCAN defaults for CPU
- c-TF-IDF reduce_frequent_words
- More permissive outlier reassignment

A) Build a domain stopword list and a function to assemble CountVectorizer stopwords

Add near the top:

- from bertopic._ctfidf import ClassTfidfTransformer
- Optional: from bertopic.representation import MaximalMarginalRelevance, PartOfSpeech (wrap in try/except like KeyBERT)

Add this helper:

def build_domain_stopwords(extra_stopwords_file: Optional[str] = None) -> set[str]: base = { # boilerplate religious greetings/formulae/common fillers (adapt to your corpus) "allah","imam","hadj","ramadan","aid","aïd","oumra","oumrah","mecque","médine", "paix","bénédiction","frère","soeur","chers","cher","chère","lumière","luire","prière","sermon", "grand","national","million","milliard","cfa","communauté","croyant","coreligionnaire","organiser","organisation", "monsieur","madame","excellence","président","ministre","gouvernement","communiqué","mondial","international", # forms seen in your labels "kebir","kébir","ledit","entrer","faire","devoir","pouvoir", } # add your own after inspecting top tokens extra = set() if extra_stopwords_file and Path(extra_stopwords_file).exists(): with open(extra_stopwords_file, "r") as f: for line in f: w = line.strip() if w: extra.add(w.lower()) return base | extra

B) Create CountVectorizer and c-TF-IDF with better defaults

In create_bertopic_model, build stopwords and set tighter vectorizer. Also pass ctfidf_model.

Add a new CLI argument:

- --domain-stopwords-file path/to/stopwords.txt

Change create_bertopic_model signature to accept domain_stopwords: Optional[set[str]] = None and desired_topics: Optional[int] = None.

Then inside create_bertopic_model:

- Compute min_cluster_size dynamically if desired_topics is given: if desired_topics and desired_topics > 0: min_cluster_size = max(20, len_for_min_cluster_size // desired_topics) # you'll pass len_for_min_cluster_size from main, see below
    
- Use UMAP tuned for CPU: umap_model = UMAP( n_neighbors=umap_n_neighbors, # set default to 150 n_components=umap_n_components, # set default to 5 min_dist=umap_min_dist, # set default to 0.0 metric=umap_metric, random_state=42, n_jobs=-1 )
    
- HDBSCAN tuned: hdbscan_model = HDBSCAN( min_cluster_size=min_topic_size, # after dynamic calc if provided min_samples=hdbscan_min_samples, # default 1 or 2 metric='euclidean', cluster_selection_method=hdbscan_selection_method, # 'leaf' cluster_selection_epsilon=hdbscan_epsilon, prediction_data=True )
    
- Build vectorizer: if domain_stopwords is None: domain_stopwords = set() vectorizer_model = CountVectorizer( ngram_range=(vectorizer_ngram_min, vectorizer_ngram_max), # default 1,3 stop_words=domain_stopwords, # now use domain stopwords max_features=vectorizer_max_features, # default 30000 min_df=vectorizer_min_df, # default 10 max_df=0.90, encoding='utf-8', decode_error='replace', strip_accents=None, lowercase=True, token_pattern=r'(?u)\b\w\w+\b' )
    
- Add ctfidf_model: ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    
- Representation model (keep KeyBERTInspired; optionally add MaximalMarginalRelevance if available)
    
- Construct BERTopic with new parts and set top_n_words smaller for cleaner labels (e.g., 6–8): topic_model = BERTopic( embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model, representation_model=representation_model, top_n_words=8, verbose=True, calculate_probabilities=True )
    

C) Use docs vs docs_clean in fit_topic_model

Change fit_topic_model to accept both docs (raw French sentences) and docs_clean (lemma_nostop). With BERTopic 0.15+, you can do: topics, probabilities = topic_model.fit_transform(docs, docs_clean=docs_clean)

In your current code you pass only one list. Modify the function to accept two lists:

def fit_topic_model( docs: List[str], docs_clean: List[str], ... ): ... topics, probabilities = topic_model.fit_transform(docs, docs_clean=docs_clean)

Then use reduce_outliers with a lower threshold, e.g. 0.1:

reduce_outliers_threshold=(args.reduce_outliers_train if args.reduce_outliers_train > 0 else None)

But update the default in the CLI to 0.10.

D) Reassign outliers more aggressively on predict

Lower outlier_reassign_threshold default to 0.08 or 0.10. Your predict code already supports this; just change the default CLI and recommend values.

## Topic Modeling Roadmap (BERTopic)

This roadmap tracks progress and next steps to reduce outliers and improve label quality for the IWAC corpus. It reflects the recent refactor and current pipeline behavior.

### Status overview
- Outliers were high (~47%); labels had boilerplate and proper names. The plan below addresses these with better embeddings, clustering, labeling, and light data cleaning.

### Implemented (Done)
- [x] Modularization and packaging (split into `patches.py`, `utils.py`, `modeling.py`, `constants.py`; added `__init__.py`)
- [x] Dual-text view: embeddings on `OCR`, vectorization/labels on `lemma_nostop`
- [x] CPU-friendly execution: `--cpu-only`, tunable embedding batch size, UMAP uses all CPU cores
- [x] Tunables via CLI: `--umap-*`, `--hdbscan-*`, `--vectorizer-*`
- [x] Topic labels: `KeyBERTInspired` if available + label de-duplication (max 8 words)
- [x] Outliers: training `reduce_outliers` supported; prediction reassignment threshold enabled
- [x] Domain stopwords (seed) in `constants.DOMAIN_STOPWORDS`
- [x] Robust I/O: global UTF-8; JSON patched for NumPy + non-ASCII

### Next steps (Planned)
1) Defaults tuned for news on CPU (fewer outliers)
	- [ ] Update defaults (overridable): UMAP `n_neighbors=150`, `n_components=5`, `min_dist=0.0`; HDBSCAN `min_samples=1–2`
	- [ ] Document profiles (fast CPU, balanced, quality)

2) Improve c-TF-IDF labeling
	- [ ] Use `ClassTfidfTransformer(reduce_frequent_words=True)` as `ctfidf_model`
	- [ ] CLI flag to toggle `reduce_frequent_words`

3) Domain stopwords configurability
	- [ ] Curate/expand list; document guidance
	- [ ] Add `--domain-stopwords-file` to merge with defaults

4) More permissive outlier reassignment
	- [ ] Lower training `--reduce-outliers-train` default to 0.05–0.15
	- [ ] Lower prediction `--outlier-reassign-threshold` default to 0.05–0.10
	- [ ] Add README guidance on tuning

5) Near-duplicate handling (optional)
	- [ ] Drop exact duplicates (hash `lemma_nostop`)
	- [ ] Optional MinHash LSH (`datasketch`) at Jaccard 0.9–0.95

6) Title weighting (optional, if title/body available)
	- [ ] `--title-col`/`--body-col` and scheme to weight title in embeddings

7) Additional label quality boosts (optional)
	- [ ] Try `PartOfSpeech`, `MaximalMarginalRelevance` representations with fallbacks
	- [ ] Consider reducing label length cap to 5–6 words

8) Testing and diagnostics
	- [ ] Add unit tests for utilities + small smoke test
	- [ ] Report metrics: outlier rate, topic size distribution, top labels

### Quick reference: current behaviors
- Inputs: `OCR` (embeddings), `lemma_nostop` (labels); only `language='Français'` rows are modeled
- Outputs: `topic_id`, `topic_prob`, `topic_label` (non-French rows remain `None`)
- CLI: tune UMAP/HDBSCAN/Vectorizer; CPU-only mode; outlier reduction + reassignment thresholds

### Milestones
- M1 (short-term): defaults (Step 1, 4), `ctfidf_model` (Step 2), stopwords file (Step 3)
- M2 (mid-term): dedup (Step 5), title weighting (Step 6), representation extras (Step 7)
- M3 (ongoing): tests & diagnostics (Step 8); iterate based on measured outlier rate and label quality
