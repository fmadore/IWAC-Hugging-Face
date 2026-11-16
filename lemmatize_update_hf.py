#!/usr/bin/env python3
"""
lemmatize_update_hf.py
======================

Add lemmatisation and stop‑word–filtered text to an existing Hugging Face
Hub dataset, one language per pass (French by default, English with
``--language Anglais``).  It downloads the dataset, processes the chosen text
column with spaCy, adds two columns (lemmatised text and lemmatised text with
stop‑words removed) and pushes the updated dataset back to the repository.

``--update-mode missing`` (the default with ``-y`` or headless) recomputes only
rows whose persisted provenance does not match the current text, language and
configuration; ``all`` recomputes every row after a confirmation that ``-y``
skips. A run that would change nothing pushes nothing.

Usage
-----
    # Articles
    python lemmatize_update_hf.py --config articles -y

    # Publications (periodical issues) — same OCR text column as articles
    python lemmatize_update_hf.py --config publications --update-mode missing

    # Full form
    python lemmatize_update_hf.py \
        --repo fmadore/islam-west-africa-collection-full \
        --config articles \
        --text-column OCR \
        --lemma-column lemma_text \
        --clean-column lemma_nostop \
        --spacy-model fr_core_news_lg \
        --max-shard-size 1GB

Note: The default model (fr_core_news_lg) is optimized for CPU. For GPU users,
consider using fr_dep_news_trf for better accuracy (but much slower on CPU).

Environment variables
---------------------
HF_TOKEN   Personal access token for the Hugging Face Hub (alternatively you
           will be prompted to log‑in interactively).

Dependencies
------------
    pip install datasets huggingface_hub spacy rich python-dotenv
    python -m spacy download fr_core_news_lg

"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Iterable, Iterator, List
import re
import unicodedata

from datasets import Dataset
import spacy
from iwac_common.enrichment import compatible_rows, config_fingerprint, provenance_columns, invalidate_columns, config_json
from iwac_common.text_utils import language_labels
from iwac_common.paths import workspace_root

# Make ``post-processing/_common.py`` and ``_embedding_utils.py`` importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "post-processing"))
try:
    from iwac_pipeline.processing._common import (
        PRIVATE_REPO_ID,
        add_update_mode_args,
        ensure_hf_token,
        load_hub_dataset,
        push_dataset,
        resolve_config,
        resolve_update_mode,
    )
    from iwac_pipeline.processing._embedding_utils import (
        cached_value,
        delete_cache,
        input_fingerprint,
        load_cache,
        make_entry,
        repo_slug,
        save_cache,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"iwac_pipeline", "iwac_pipeline.processing"}:
        raise
    from _common import (
        PRIVATE_REPO_ID,
        add_update_mode_args,
        ensure_hf_token,
        load_hub_dataset,
        push_dataset,
        resolve_config,
        resolve_update_mode,
    )
    from _embedding_utils import (
        cached_value,
        delete_cache,
        input_fingerprint,
        load_cache,
        make_entry,
        repo_slug,
        save_cache,
    )

# Rich console imports for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.prompt import Confirm
from rich import box
from dotenv import load_dotenv

# Load .env so a non-interactive run finds HF_TOKEN instead of falling through
# to an interactive login that cannot read stdin.
load_dotenv(workspace_root() / ".env")

# Initialize Rich console
console = Console()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )


# Preprocessing constants and function
RE_DASH  = re.compile(r"[–—−]")
RE_SPACE = re.compile(r"\s{2,}")  # Fixed: was incorrectly escaped
# Fancy quotes to normalize (using Unicode codepoints for reliability). The
# transliteration letters U+02BC, U+02BE and U+02BF are letters, not quotes,
# and are deliberately left alone: they already sit inside one alphabetic token.
QUOTE_REPLACEMENTS = [
    ("′", "'"), ("＇", "'"),  # prime and fullwidth apostrophe (OCR)
    ("\u2018", "'"), ("\u2019", "'"), ("\u201a", "'"), ("\u201b", "'"),  # single quotes
    ("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'), ("\u201f", '"'),  # double quotes
    ("\u00ab", '"'), ("\u00bb", '"'),  # guillemets « »
]
MAP_LIG = str.maketrans({"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE"})
_APOSTROPHE_TRANSLATION = str.maketrans({old: new for old, new in QUOTE_REPLACEMENTS if new == "'"})

# Lemma vocabulary policy (LEMMA_LOGIC_VERSION 3).
#
# A token is kept when every hyphen/apostrophe-separated part of it is made of
# letters (combining marks allowed): ``sous-préfet``, ``Côte-d'Ivoire``,
# ``aujourd'hui`` and ``peut-être`` survive whole, with their internal ``-`` and
# ASCII ``'`` kept in the lemma. Tokens containing a digit or any other symbol
# (``1995``, ``3e``, ``M.``, ``U.S.A.``) are dropped, as before. A lemma never
# contains whitespace, because the LDA stage splits ``lemma_nostop`` on it.
#
# Bare elided clitics are dropped rather than emitted. The French tokenizer
# also splits a transliterated Arabic word at its apostrophe (``Jama'`` +
# ``at``, ``da'`` + ``wa``, ``Qur'`` + ``an``); when the head is not one of
# these clitics, the two tokens are rejoined into one lemma (``jama'at``).
FRENCH_ELIDED_CLITICS = frozenset({
    "l", "d", "j", "m", "n", "s", "t", "c",
    "qu", "jusqu", "lorsqu", "puisqu", "quoiqu", "quelqu",
})
NAME_PREFIXES = frozenset({"N", "M"})
ENGLISH_CLITICS = frozenset({"'s", "'re", "'ve", "'ll", "'d", "'m", "n't"})
_WORD_SEPARATORS = re.compile(r"[-']")


def _is_word_part(part: str) -> bool:
    categories = [unicodedata.category(char)[0] for char in part]
    return bool(part) and all(c in "LM" for c in categories) and "L" in categories


def is_lemma_word(text: str) -> bool:
    """True when ``text`` is letters joined by single internal ``-``/``'``."""
    return bool(text) and all(_is_word_part(part) for part in _WORD_SEPARATORS.split(text))


def is_bare_clitic(text: str) -> bool:
    lowered = text.lower()
    return lowered in ENGLISH_CLITICS or (
        lowered.endswith("'") and lowered[:-1] in FRENCH_ELIDED_CLITICS
    )


def _surface(token) -> str:
    return (getattr(token, "text", None) or token.lemma_).translate(_APOSTROPHE_TRANSLATION)


def token_lemmas(doc) -> List[tuple]:
    """``(lemma, is_stop)`` pairs for the kept tokens of one spaCy ``Doc``.

    Stop-word status is spaCy's ``token.is_stop`` on the SURFACE token, not a
    test of the lemma against the surface-form stop-word list. A blank
    pipeline has no lemmatizer, so an empty lemma falls back to the surface.
    """
    tokens = list(doc)
    pairs: List[tuple] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        text = _surface(token)
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        head = text[:-1]
        attached = following is not None and not getattr(token, "whitespace_", " ")
        if attached and text.lower() in FRENCH_ELIDED_CLITICS and _surface(following) == "'":
            i += 2  # ``l'`` followed by a space is tokenized ``l`` + ``'``
            continue
        # ``N'Guessan``, ``N'Djamena``, ``M'Bow``: a capital N/M before a
        # capitalised word is a West African name, not the clitic ne/me.
        is_name = head in NAME_PREFIXES and attached and _surface(following)[:1].isupper()
        if (text.endswith("'") and following is not None
                and not getattr(token, "whitespace_", " ")
                and is_lemma_word(head)
                and (is_name or head.lower() not in FRENCH_ELIDED_CLITICS)
                and is_lemma_word(_surface(following))):
            pairs.append((f"{text}{_surface(following)}".lower(), False))
            i += 2
            continue
        i += 1
        if is_bare_clitic(text) or not is_lemma_word(text):
            continue
        lemma = (token.lemma_ or text).translate(_APOSTROPHE_TRANSLATION).lower()
        if not is_lemma_word(lemma):
            lemma = text.lower()
        pairs.append((lemma, bool(token.is_stop)))
    return pairs

# Cap the characters handed to spaCy in a single call. A full periodical issue
# (publications) can top 1M chars; tagging it whole-hog blows up tok2vec memory
# on CPU, so long OCR is split into chunks of this size first.
SPACY_MAX_CHUNK_CHARS = 100_000

# Crash-resumable cache: freshly computed lemmas are checkpointed here (keyed
# by o:id) every CHECKPOINT_EVERY rows, reloaded on restart, and deleted after
# a successful push. Gitignored, like the other .cache_* dirs. A ``.sqlite3``
# cache takes incremental upserts, so each checkpoint writes only the rows
# completed since the previous one.
CACHE_DIR = workspace_root() / ".cache_lemmas"
CHECKPOINT_EVERY = 100

# Bump whenever the lemmatisation OUTPUT changes for the same input text —
# normalize(), the token filter, the stop-word rule, or the chunking. It is
# part of the resume-cache name and of the persisted provenance hash, so
# incremental mode recomputes every row. History: 2 = stop-word check on the
# surface token (token.is_stop); 3 = hyphenated/apostrophe words kept whole,
# bare clitics dropped, transliterations split at the apostrophe rejoined.
LEMMA_LOGIC_VERSION = 3


def cache_fingerprint(spacy_model: str, model_version: str) -> str:
    """Filesystem-safe slug identifying the lemmatisation configuration.

    Lemma resume caches MUST embed this in their filename. The cache is
    normally deleted after a successful push, so it only survives an
    interrupted run — but that is exactly the case that bites: if the
    spaCy model or the lemmatisation logic changes before the re-run, the
    leftover rows would be restored as if freshly computed, silently mixing
    two versions of the column. Mirrors ``_embedding_utils.cache_fingerprint``.
    """
    safe_model = spacy_model.replace("/", "-")
    return f"{safe_model}-{model_version}_spacy{spacy.__version__}_v{LEMMA_LOGIC_VERSION}"


def normalize(text: str) -> str:
    if not text:  # None or empty string (e.g. blank OCR rows) → nothing to lemmatise
        return ""
    text = unicodedata.normalize("NFC", text)
    for old, new in QUOTE_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.translate(MAP_LIG)
    text = RE_DASH.sub("-", text).replace("\u00a0", " ")
    return RE_SPACE.sub(" ", text).strip()


def load_spacy_model(name: str):
    """Load a spaCy model, downloading it the first time it is required."""
    try:
        return spacy.load(name, disable=["parser", "ner", "textcat"])
    except OSError:
        console.print(f"[yellow]⚠[/yellow] SpaCy model '{name}' not found – downloading…")
        from spacy.cli import download as spacy_download

        spacy_download(name)
        return spacy.load(name, disable=["parser", "ner", "textcat"])


def chunk_on_whitespace(text: str, max_chars: int) -> List[str]:
    """Split ``text`` into <= ``max_chars`` pieces, breaking at whitespace."""
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws
        chunks.append(text[start:end])
        start = end
    return chunks


def lemmatise_one(nlp, text: str) -> List[tuple]:
    """Lemmatise one already-normalized text, chunking long text to bound memory.

    Returns the :func:`token_lemmas` pairs of every chunk. This unbatched path
    is the reference the batched :func:`lemmatise_many` is tested against.

    A full periodical issue can top 1M chars; feeding that to spaCy in one go
    builds a huge tok2vec tensor and OOMs on CPU. Splitting into
    SPACY_MAX_CHUNK_CHARS pieces keeps each call small. Parser/NER are
    disabled, but the retained tagger/lemmatizer is contextual: boundaries
    can affect nearby lemmas. The chunk policy is persisted in provenance.
    """
    if not text:
        return []
    tokens: List[tuple] = []
    for chunk in chunk_on_whitespace(text, SPACY_MAX_CHUNK_CHARS):
        tokens.extend(token_lemmas(nlp(chunk)))
    return tokens


def lemmatise_many(
    nlp, texts: Iterable[str], *, batch_size: int = 32, n_process: int = 1
) -> Iterator[List[tuple]]:
    """Lemmatise many already-normalized texts; yield one token list per text,
    in input order (an empty text yields ``[]``).

    Same output as calling :func:`lemmatise_one` on each text, but through
    ``nlp.pipe``: spaCy batches the chunks through tok2vec/tagger instead of
    one call per chunk, which is the standard large CPU speed-up, and
    ``n_process`` > 1 spreads batches over worker processes. Chunking is
    unchanged, so the lemmas are identical.
    """
    total = [0]

    def stream():
        for idx, text in enumerate(texts):
            total[0] = idx + 1
            if not text:
                continue
            for chunk in chunk_on_whitespace(text, SPACY_MAX_CHUNK_CHARS):
                yield chunk, idx

    next_idx = 0
    current: int | None = None
    tokens: List[tuple] = []
    for doc, idx in nlp.pipe(
        stream(), as_tuples=True, batch_size=batch_size, n_process=n_process
    ):
        if current is not None and idx != current:
            while next_idx < current:
                yield []
                next_idx += 1
            yield tokens
            next_idx = current + 1
            tokens = []
        current = idx
        tokens.extend(token_lemmas(doc))
    if current is not None:
        while next_idx < current:
            yield []
            next_idx += 1
        yield tokens
        next_idx = current + 1
    while next_idx < total[0]:
        yield []
        next_idx += 1


def checkpoint_cache(cache: dict, pending: dict, cache_file: Path) -> None:
    """Persist the entries completed since the last checkpoint.

    A ``.sqlite3`` cache upserts only ``pending``; re-serialising the whole
    dict on every checkpoint is quadratic once rows reach 1M characters. A
    legacy gzip path cannot append, so it still receives the full cache.
    """
    save_cache(pending if cache_file.suffix == ".sqlite3" else cache, cache_file)
    pending.clear()


def print_lemma_plan(total: int, to_process: int, plan: dict) -> None:
    table = Table(title="Lemmatisation plan (dry run)", box=box.ROUNDED)
    table.add_column("Rows", style="cyan")
    table.add_column("Count", style="green", justify="right")
    for label, value in (
        ("Total", total),
        ("Would be lemmatised", to_process),
        ("Restored from the resume cache", plan["restored"]),
        ("Kept (current provenance)", plan["kept"]),
        ("Blank text (empty lemmas)", plan["blank"]),
        ("Other language (kept as-is)", plan["out_of_scope"]),
        ("Unknown/multilingual (cleared)", plan["cleared"]),
    ):
        table.add_row(label, f"{value:,}")
    console.print(table)


def lemmatise_dataset(
    ds,
    nlp,
    *,
    text_col: str,
    lemma_col: str,
    clean_col: str,
    process_choice: str,
    cache_file: Path,
    language_filter: str | None = None,
    batch_size: int = 32,
    n_process: int = 1,
    multilingual: str = "skip",
    dry_run: bool = False,
):
    """Add lemma columns to ``ds``, checkpointing to ``cache_file`` for resume.

    Returns the updated dataset, or ``None`` when no lemma, provenance or
    dependent LDA cell would change (nothing to push). With ``dry_run`` the
    plan is reported and ``None`` returned before spaCy runs or the cache is
    touched.

    ``process_choice`` is ``"all"`` or ``"missing"`` (``"empty"`` is the
    deprecated spelling of ``"missing"``). In ``missing`` mode a row is kept
    only when its persisted provenance matches the current text and
    configuration; blank-text rows get empty lemmas.

    Resilience: each freshly lemmatised row is written to an on-disk cache
    (keyed by ``o:id``) and flushed every CHECKPOINT_EVERY rows, so a crash
    loses at most that many rows. On restart, cached rows are reused and only
    the remainder is recomputed.

    ``language_filter`` skips rows whose PRIMARY language (the first
    pipe-separated component of ``language``, whitespace-stripped) is not the
    given label (e.g. "Français" or "Anglais"): a wrong-language pipeline
    would emit garbage lemmas. Matching on the primary language only makes
    bilingual rows ("Français|Anglais") deterministic — each row belongs to
    exactly one language pass, so the result does not depend on --mode or
    pass order. Skipped rows keep their existing values, so per-language
    passes compose (French pass with the French model, then English pass
    with the English model).
    """
    n = len(ds)
    row_ids = [str(x) for x in ds["o:id"]]
    texts = ds[text_col]
    existing_lemmas = ds[lemma_col] if lemma_col in ds.column_names else [None] * n
    existing_clean = ds[clean_col] if clean_col in ds.column_names else [None] * n
    row_languages = [language_labels(value) for value in ds["language"]] if "language" in ds.column_names else [[] for _ in range(n)]

    in_scope = [True] * n
    if language_filter:
        if "language" not in ds.column_names:
            raise ValueError(f"Cannot route --language {language_filter!r}: dataset has no language column")
        else:
            # A row is in scope only when its FIRST listed (primary) language
            # matches the filter: bilingual rows ("Français|Anglais") are
            # handled by exactly one pass, so output is deterministic
            # regardless of --mode or pass order. Components are stripped so
            # "Français | Anglais" parses correctly.
            in_scope = [
                bool(labels) and labels[0] == language_filter
                and (multilingual == "primary" or len(labels) == 1)
                for labels in row_languages
            ]
            console.print(
                f"[blue]→[/blue] Language filter '{language_filter}': [bold]{sum(in_scope):,}[/bold] of {n:,} rows eligible"
            )

    cache = load_cache(cache_file)
    fingerprints = [input_fingerprint(texts[i] or "") for i in range(n)]
    meta = getattr(nlp, "meta", {})
    settings = {
        "model_name": meta.get("name", "unknown"), "model_version": meta.get("version", "unknown"),
        "model_language": meta.get("lang", "unknown"), "spacy": spacy.__version__,
        "logic": LEMMA_LOGIC_VERSION, "language": language_filter,
        "multilingual": multilingual, "chunk_chars": SPACY_MAX_CHUNK_CHARS,
    }
    configuration = config_fingerprint(settings)
    description = config_json(settings)
    cache_inputs = [input_fingerprint(source, configuration) for source in fingerprints]
    compatible = compatible_rows(ds, lemma_col, fingerprints, configuration)
    input_col, config_col = provenance_columns(lemma_col)
    final_inputs = list(ds[input_col]) if input_col in ds.column_names else [None] * n
    final_configs = list(ds[config_col]) if config_col in ds.column_names else [None] * n
    json_col = f"{lemma_col}_config_json"
    final_descriptions = list(ds[json_col]) if json_col in ds.column_names else [None] * n

    if process_choice not in {"all", "missing", "empty"}:
        raise ValueError(f"Unknown process_choice {process_choice!r}")
    keep_compatible = process_choice in {"missing", "empty"}

    final_lemmas: List[str] = [""] * n
    final_clean: List[str] = [""] * n
    indices_to_process: List[int] = []
    plan = {"cleared": 0, "out_of_scope": 0, "blank": 0, "restored": 0, "kept": 0}
    for i in range(n):
        if not in_scope[i]:
            labels = row_languages[i]
            if not labels or (multilingual == "skip" and len(labels) > 1):
                # Unknown/mixed-language rows have no defensible monolingual
                # output, including one left by an older pipeline version.
                final_inputs[i] = final_configs[i] = final_descriptions[i] = None
                plan["cleared"] += 1
                continue
            # Out of scope: keep whatever the row already has.
            final_lemmas[i] = existing_lemmas[i] or ""
            final_clean[i] = existing_clean[i] or ""
            plan["out_of_scope"] += 1
            continue
        final_inputs[i] = None
        final_configs[i] = None
        final_descriptions[i] = None
        if not texts[i] or not str(texts[i]).strip():
            plan["blank"] += 1  # blank text -> empty lemmas (already "")
            continue
        # Reused only when computed from this exact text: an entry for text
        # that changed since (or a legacy entry without a hash) is redone.
        cached = cached_value(cache, row_ids[i], cache_inputs[i])
        if cached is not None:
            final_lemmas[i], final_clean[i] = cached[0], cached[1]
            final_inputs[i], final_configs[i], final_descriptions[i] = fingerprints[i], configuration, description
            plan["restored"] += 1
            continue
        if (keep_compatible and compatible[i] and existing_lemmas[i] is not None
                and existing_clean[i] is not None):
            final_lemmas[i] = existing_lemmas[i]
            final_clean[i] = existing_clean[i] or ""
            final_inputs[i], final_configs[i], final_descriptions[i] = fingerprints[i], configuration, description
            plan["kept"] += 1
            continue
        indices_to_process.append(i)

    console.print(
        f"[blue]→[/blue] [bold]{len(indices_to_process):,}[/bold] of {n:,} rows to lemmatise"
        + (f" ([green]{plan['restored']:,}[/green] restored from cache)" if plan["restored"] else "")
    )
    if dry_run:
        print_lemma_plan(n, len(indices_to_process), plan)
        return None

    since_ckpt = 0
    pending: dict = {}
    if indices_to_process:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]lemmatising", total=len(indices_to_process))
            token_lists = lemmatise_many(
                nlp,
                (normalize(texts[i]) for i in indices_to_process),
                batch_size=batch_size,
                n_process=n_process,
            )
            for i, tokens in zip(indices_to_process, token_lists, strict=True):
                lemma_text = " ".join(lemma for lemma, _ in tokens)
                # Stop-word filtering uses token.is_stop on the surface token
                # (not the lemma vs. the surface-form stop-word list).
                clean_text = " ".join(lemma for lemma, is_stop in tokens if not is_stop)
                final_lemmas[i] = lemma_text
                final_clean[i] = clean_text
                final_inputs[i], final_configs[i], final_descriptions[i] = fingerprints[i], configuration, description
                entry = make_entry([lemma_text, clean_text], cache_inputs[i])
                cache[row_ids[i]] = pending[row_ids[i]] = entry
                since_ckpt += 1
                progress.update(task, advance=1)
                if since_ckpt >= CHECKPOINT_EVERY:
                    checkpoint_cache(cache, pending, cache_file)
                    since_ckpt = 0
        if since_ckpt > 0:
            checkpoint_cache(cache, pending, cache_file)

    unchanged = all(
        before == after
        for old, new in ((existing_lemmas, final_lemmas), (existing_clean, final_clean))
        for before, after in zip(old, new, strict=True)
    ) and all(
        name in ds.column_names and list(ds[name]) == values
        for name, values in ((input_col, final_inputs), (config_col, final_configs),
                             (json_col, final_descriptions))
    ) and lemma_col in ds.column_names and clean_col in ds.column_names
    if unchanged:
        return None

    for col in (lemma_col, clean_col):
        if col in ds.column_names:
            ds = ds.remove_columns([col])
    ds = ds.add_column(lemma_col, final_lemmas)
    ds = ds.add_column(clean_col, final_clean)
    import pyarrow as pa
    for col, values in ((input_col, final_inputs), (config_col, final_configs), (json_col, final_descriptions)):
        if col in ds.column_names:
            ds = ds.remove_columns(col)
        ds = ds.add_column(col, pa.array(values, type=pa.string()))
    changed = [(before or "") != after for before, after in zip(existing_clean, final_clean, strict=True)]
    ds = invalidate_columns(ds, ["lda_topic_id", "lda_topic_prob", "lda_topic_label",
                                 "lda_topic_topk", "lda_model_name"], changed)
    return ds


# Subsets that carry an OCR text column worth lemmatising. Passing --config
# with another subset still works; this list only drives the interactive menu.
#
# 'audiovisual' is deliberately absent: only 5 of its 47 rows carry a
# transcription and 3 of those are Haoussa/Arabe, so a French pipeline would
# lemmatise mostly noise for the sake of two French rows.
LEMMATIZABLE_SUBSETS = ["articles", "publications", "references", "documents"]

# Default spaCy model per --language label (CPU-friendly large models).
LANGUAGE_MODEL_DEFAULTS = {
    "Français": "fr_core_news_lg",
    "Anglais": "en_core_web_lg",
}


# Every subset carrying an OCR column; accepted by --config.
OCR_SUBSETS = [*LEMMATIZABLE_SUBSETS, "audiovisual"]


def confirm_full_recompute(args, mode: str, what: str) -> bool:
    """Ask before ``all`` recomputes every row; ``-y``, a dry run or a headless
    run (whose explicit ``--update-mode all`` is the decision) skip it."""
    if mode != "all" or args.yes or args.dry_run or not sys.stdin.isatty():
        return True
    try:
        return Confirm.ask(f"Recompute {what} for every row?", default=False)
    except KeyboardInterrupt:
        return False


def main() -> int:
    configure_logging()

    parser = argparse.ArgumentParser(description="Add lemmatised columns to a Hugging Face dataset")
    parser.add_argument("--repo", default=PRIVATE_REPO_ID, help="Dataset repo on the Hugging Face Hub (default: private full mirror)")
    parser.add_argument("--config", choices=OCR_SUBSETS, default=None, help="Dataset subset/config to lemmatise. If omitted, you are prompted to choose among the subsets worth lemmatising.")
    parser.add_argument("--text-column", default="OCR", help="Name of the column containing the raw text to process")
    parser.add_argument("--lemma-column", default="lemma_text", help="Column name for the lemmatised text")
    parser.add_argument("--clean-column", default="lemma_nostop", help="Column name for the lemmatised text with stop-words removed")
    parser.add_argument("--spacy-model", default=None, help="spaCy model to use (default: chosen from --language via LANGUAGE_MODEL_DEFAULTS, else fr_core_news_lg)")
    parser.add_argument("--max-shard-size", default="1GB", help="Maximum Parquet shard size when pushing to the Hub")
    add_update_mode_args(parser)
    parser.add_argument("--mode", choices=["all", "empty"], default=None,
                        help="Deprecated alias for --update-mode ('empty' == 'missing').")
    parser.add_argument("--language", default=None, metavar="LABEL", help="Only lemmatise rows whose PRIMARY (first-listed) 'language' is LABEL (e.g. 'Français', 'Anglais'); picks the matching spaCy model unless --spacy-model is given; multilingual rows are skipped unless --multilingual primary is explicit. Other language rows keep their existing values — run one pass per language on mixed subsets like 'references'.")
    parser.add_argument("--french-only", action="store_true", help="Deprecated alias for --language Français")
    parser.add_argument("--multilingual", choices=["skip", "primary"], default="skip",
                        help="Skip multilingual records (default), or explicitly use the first metadata language")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report how many rows would be lemmatised, restored, kept or cleared, "
                             "without running spaCy, writing the cache or pushing")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Texts per spaCy nlp.pipe batch (default: 32)")
    parser.add_argument("--n-process", type=int, default=1,
                        help="spaCy worker processes (default: 1). More processes "
                             "use more RAM: each loads its own copy of the model.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.n_process < 1:
        parser.error("--batch-size and --n-process must be at least 1")

    # --update-mode wins; --mode is the deprecated alias (all|empty).
    if args.update_mode is None and args.mode is not None:
        console.print("[yellow]⚠[/yellow] --mode is deprecated; use --update-mode {all,missing} ('empty'→'missing').")
        args.update_mode = "missing" if args.mode == "empty" else args.mode

    inferred_language = {"fr": "Français", "en": "Anglais"}.get((args.spacy_model or "fr_").split("_")[0])
    language_filter = args.language or ("Français" if args.french_only else inferred_language)
    if language_filter is None:
        parser.error("A custom model needs an explicit --language label")
    spacy_model = args.spacy_model or LANGUAGE_MODEL_DEFAULTS.get(language_filter)
    if spacy_model is None:
        parser.error("This language requires an explicit compatible --spacy-model")

    token = ensure_hf_token(console=console)
    config_name = resolve_config(
        args.repo, token=token, cli_config=args.config,
        restrict_to=LEMMATIZABLE_SUBSETS, console=console,
    )

    # ------------------------------------------------------------------
    # Load dataset from the Hub
    # ------------------------------------------------------------------
    console.print(Panel(
        f"[bold]Repository:[/bold] {args.repo}\n"
        f"[bold]Subset:[/bold] {config_name}\n"
        f"[bold]Text column:[/bold] {args.text_column}\n"
        f"[bold]Language filter:[/bold] {language_filter or 'none (all rows)'}\n"
        f"[bold]spaCy model:[/bold] {spacy_model}",
        title="[bold blue]Lemmatization Configuration[/bold blue]",
        border_style="blue"
    ))

    ds: Dataset = load_hub_dataset(
        args.repo, config_name, token=token, console=console
    )
    source_revision = getattr(ds, "_iwac_source_revision", None)

    if args.text_column not in ds.column_names:
        console.print(f"[red]✗[/red] Column '{args.text_column}' not found in the dataset.")
        console.print(f"[yellow]ℹ[/yellow] Available columns: {', '.join(ds.column_names)}")
        raise ValueError(f"Column '{args.text_column}' not found in the dataset.")

    process_choice = resolve_update_mode(args, console=console)
    console.print(f"[green]✓[/green] Update mode: [cyan]{process_choice}[/cyan]")
    if not confirm_full_recompute(args, process_choice, f"'{args.lemma_column}'"):
        console.print("[yellow]ℹ[/yellow] Cancelled; existing lemmas kept.")
        return 0

    # ------------------------------------------------------------------
    # Load the spaCy model once
    # ------------------------------------------------------------------
    with console.status(f"[bold green]Loading spaCy model '{spacy_model}'...", spinner="dots"):
        nlp = load_spacy_model(spacy_model)
    # Long OCR is chunked to SPACY_MAX_CHUNK_CHARS before tagging (see
    # lemmatise_one), keeping each spaCy call well under the default 1M-char
    # limit, so nlp.max_length does not need raising.
    console.print(f"[green]✓[/green] Loaded spaCy model '{spacy_model}'")
    expected_lang = {"Français": "fr", "Anglais": "en"}.get(language_filter)
    if expected_lang and nlp.lang != expected_lang:
        parser.error(f"Model language {nlp.lang!r} conflicts with --language {language_filter!r}")

    # ------------------------------------------------------------------
    # Lemmatise — crash-resumable: checkpoints to .cache_lemmas, resumes on
    # restart, and the cache is deleted only after a successful push.
    # ------------------------------------------------------------------
    # Per-language cache files: a crashed English pass must not feed its
    # rows into a resumed French pass (and vice versa). The fingerprint adds
    # the spaCy model + logic version, so a cache left behind by an
    # interrupted run is ignored once either of those changes.
    cache_suffix = f"_{language_filter}_{args.multilingual}"
    model_version = nlp.meta.get("version", "unknown")
    fingerprint = cache_fingerprint(spacy_model, model_version)
    # The repository is part of the name (a scratch run must not feed a
    # production one), and each entry carries a hash of its input text.
    cache_file = (
        CACHE_DIR / f"{config_name}{cache_suffix}_{repo_slug(args.repo)}_{fingerprint}.sqlite3"
    )

    # Gzip caches predate the SQLite format (and older ones the fingerprint);
    # they are never read. Say so rather than leaving them to rot silently.
    for legacy in sorted(CACHE_DIR.glob(f"{config_name}{cache_suffix}*.json.gz")):
        console.print(
            f"[yellow]⚠[/yellow] Ignoring legacy resume cache [dim]{legacy.name}[/dim] "
            "(written by an earlier lemmatisation logic or cache format). "
            "Delete it once you no longer need it."
        )
    if not args.dry_run:
        console.print("[blue]→[/blue] Applying lemmatisation – this can take a while…")
    result = lemmatise_dataset(
        ds,
        nlp,
        text_col=args.text_column,
        lemma_col=args.lemma_column,
        clean_col=args.clean_column,
        process_choice=process_choice,
        cache_file=cache_file,
        language_filter=language_filter,
        batch_size=args.batch_size,
        n_process=args.n_process,
        multilingual=args.multilingual,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        console.print(Panel(
            "[yellow]Dry run — nothing lemmatised, cached or pushed.[/yellow]",
            border_style="yellow",
        ))
        return 0
    if result is None:
        delete_cache(cache_file)
        console.print("[green]✓[/green] Every lemma and its provenance are already current. Nothing to push.")
        return 0
    ds = result

    # ------------------------------------------------------------------
    # Push updated dataset to the Hub
    # ------------------------------------------------------------------
    console.print(f"[blue]→[/blue] Pushing updated dataset back to {args.repo} (subset: {config_name})…")
    if not push_dataset(
        ds,
        repo_id=args.repo,
        config_name=config_name,
        token=token,
        max_shard_size=args.max_shard_size,
        commit_message=(
            f"Add/update columns '{args.lemma_column}' and '{args.clean_column}' "
            f"for {config_name} (lemmatisation of {language_filter} rows, multilingual: "
            f"{args.multilingual}; spaCy model {spacy_model} {model_version}, spaCy "
            f"{spacy.__version__}, logic v{LEMMA_LOGIC_VERSION}; mode: {process_choice})"
        ),
        console=console,
        expected_revision=source_revision,
    ):
        console.print(
            f"[yellow]Computed lemmas remain cached in {cache_file}; re-run to resume.[/yellow]"
        )
        return 1

    # The cache only exists for resume; the push succeeded, so drop it.
    delete_cache(cache_file)

    # Summary table
    table = Table(title="Lemmatization Complete", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Repository", args.repo)
    table.add_row("Subset", config_name)
    table.add_row("Rows in subset", f"{len(ds):,}")
    table.add_row("Lemma column", args.lemma_column)
    table.add_row("Clean column", args.clean_column)
    table.add_row("Mode", process_choice)
    console.print(table)
    console.print("[green]✓[/green] Done!")
    return 0

    
if __name__ == "__main__":
    raise SystemExit(main())
