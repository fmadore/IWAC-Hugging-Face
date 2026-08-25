# Research backlog

Analyses the dataset can already support but nobody has built. Each is grounded
in columns that exist today — none needs new computation on the Hub, and none is
blocked on anything.

Carried over from `CODE_AND_METHODS_AUDIT.md` (July 2026), which was deleted once
its bug list, safety rails, refactoring plan and methodology fixes had all landed.
This file is the only part of it that was still outstanding. The audit and the
completed `REFACTORING_ROADMAP.md` remain in git history:
`git show 3b44b98:CODE_AND_METHODS_AUDIT.md`.

Two of the original seven ideas were built and are not repeated here:
`analyses/topic_sentiment.py` (topic × sentiment × time × country) and
`analyses/entity_networks.py` (authority co-occurrence graph).

Report-only analyses write to `analyses/output/` (gitignored) and never add Hub
columns; shared statistics live in `analyses/_stats.py`.

---

## 1. Press ecology — how outlets differ

`newspaper` and `country` are on every article, so per-outlet topic profiles and
keyness are one `groupby` from the existing Dunning G² machinery in
`analyses/keyness_bursts.py`. The question worth asking is whether the state,
private and religious press treat Islam differently — which is measurable rather
than assumed, and is the kind of claim the collection was assembled to support.

Cheapest item here by a wide margin.

## 2. Reprint and wire-copy detection

`post-processing/related_articles.py` already computes near-duplicate cosine
pairs but keeps only each row's top-1 neighbour. Union-find over *all* pairs
≥ 0.95 would give reprint clusters: which stories crossed borders, and with what
lag.

Worth doing for a second reason — undetected reprints inflate topic-prevalence
counts, so every frequency-based analysis in the repo is slightly wrong in a way
nobody has measured.

## 3. Diachronic semantic drift

Word2vec per decade over `lemma_nostop`, aligned with Procrustes (HistWords).
How do the words around *islam*, *voile*, *terrorisme*, *confrérie* move across
decades? gensim is already a dependency and this is CPU-viable at this corpus
size.

Note the lemma columns cover `articles`, `publications`, `references` and
`documents` but **not** `audiovisual`, so this is a press-and-print analysis.

## 4. Geography of attention

`spatial` joined to `index.Coordonnées` gives place-mention frequency over time,
and with it the local-versus-foreign framing question: how much coverage of Islam
in Benin is actually about Nigeria, Mali, or the Middle East? Pairs naturally
with the burst detector for event geography.

`index.Coordonnées` is a `"lat, lng"` string, not two floats — split before use.

## 5. Agenda-setting: press versus academy

`references` (academic literature) against `articles` (press), as per-subject or
per-topic time series with lead–lag correlation. Does scholarship follow press
attention, or the reverse?

Few corpora can ask this at all; IWAC holds both sides. Two cautions: the LDA
topic ids are **not** comparable across the two subsets (different models,
unrelated numbering — always read `lda_model_name` beside a topic id), so join on
`subject`/authority terms rather than topics unless you fit a shared model. And
`references` has no lunar date and a far smaller n (867 rows), so annual bins
will be sparse.

---

## Settled — do not re-propose

**A per-document OCR-quality score.** Built as `calculate_ocr_quality.py` in July
2026 and reverted before its column ever reached the Hub. Two reasons, both still
true: OCR is now extracted with vision LLMs, which do not produce the
character-level garbling a dictionary hit-rate detects (dry run: 12,332 articles,
median 0.997, Q1 0.992 — no discriminating power); and it scored the 45
Ewé/Kabiyè/Dendi articles at 0.06–0.09 despite clean text, putting exactly the
West African-language sources the collection exists to document at the bottom of
the distribution. Any lexicon-keyed metric will do this. Score such rows `None`,
never "low quality".

**Flesch Reading Ease being English-calibrated.** Checked and false: under
`textstat.set_lang('fr')` the Kandel–Moles French constants apply, verified in
textstat's source. See the comment in
`post-processing/calculate_lexical_richness.py`.
