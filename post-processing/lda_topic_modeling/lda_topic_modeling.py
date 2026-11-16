#!/usr/bin/env python3
"""
lda_topic_modeling.py
=====================

Adds LDA-based topic modeling columns to a Hugging Face dataset.

Uses gensim LDA on the ``lemma_nostop`` column (already lemmatized,
stopwords removed) — no embeddings, no GPU required.

New columns added:
  - lda_topic_id    : dominant topic id
  - lda_topic_prob  : probability of the dominant topic
  - lda_topic_label : top words for the dominant topic
  - lda_topic_topk  : top-k topic distribution "id:prob|id:prob|..."
                      (descending probability; enables probability-weighted
                      analyses like topic prevalence over time)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from iwac_pipeline.processing._common import (  # type: ignore  # noqa: E402
    PRIVATE_REPO_ID,
    choose_config,
    ensure_hf_token,
    get_available_configs,
    load_hub_dataset,
    push_dataset,
    write_run_manifest,
)

from iwac_pipeline.processing.lda_topic_modeling.constants import (  # type: ignore
    CONFIG_PRESETS,
    modeling_stopwords,
    DEFAULT_MIN_TRAIN_TOKENS,
    DEFAULT_PHRASE_MIN_DOCS,
    DEFAULT_NUM_TOPICS,
    DEFAULT_PASSES,
    DEFAULT_ITERATIONS,
    DEFAULT_CHUNKSIZE,
    DEFAULT_NO_BELOW,
    DEFAULT_NO_ABOVE,
    DEFAULT_TOPIC_RANGE_START,
    DEFAULT_TOPIC_RANGE_END,
    DEFAULT_TOPIC_RANGE_STEP,
    DEFAULT_SWEEP_PASSES,
    DEFAULT_SWEEP_ITERATIONS,
    DEFAULT_TOPIC_TOPK,
    DEFAULT_LAMBDA_RELEVANCE,
    DEFAULT_STABILITY_SEEDS,
    DEFAULT_HOLDOUT_FRACTION,
)
from iwac_pipeline.processing.lda_topic_modeling.modeling import (  # type: ignore
    tokenize_documents,
    build_dictionary,
    build_corpus,
    cap_chunks,
    chunk_tokens,
    create_lda_model,
    save_lda_model,
    export_topic_table,
    load_lda_model,
    predict_batch,
    compute_coherence,
    compute_corpus_word_probs,
    save_model_parameters,
    get_topic_label,
    find_optimal_topics,
    sweep_ranking,
    tokenize_for_prediction,
    HOLDOUT_METRIC_NOTE,
)
from iwac_pipeline.processing.lda_topic_modeling.artifacts import (  # noqa: E402
    frozen_preprocessing, load_preprocessing, prediction_tokenizer_kwargs,
    publish_bundle, resolve_bundle, split_document_indices, digest_file, text_fingerprint,
    preprocessing_fingerprint, effective_model_identity,
)
from iwac_common.paths import workspace_root  # noqa: E402

console = Console(force_terminal=True)


# ── Helpers ─────────────────────────────────────────────────────────


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def choose_mode() -> str:
    console.print()
    panel = (
        "[cyan]1.[/cyan] Train a new LDA model\n"
        "[cyan]2.[/cyan] Load an existing LDA model"
    )
    console.print(Panel(panel, title="LDA Mode", border_style="blue"))
    while True:
        try:
            choice = Prompt.ask("[yellow]\u2192[/yellow] Choose", choices=["1", "2"], show_choices=False)
            return "fit" if choice == "1" else "predict"
        except KeyboardInterrupt:
            raise SystemExit(0) from None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add LDA topic columns to a Hugging Face dataset.")
    p.add_argument("--repo", default=PRIVATE_REPO_ID)
    p.add_argument("--config", type=str, default=None, help="Dataset config name (skip interactive prompt)")
    p.add_argument("--mode", type=str, choices=["fit", "predict"], default=None, help="Run mode (skip interactive prompt)")
    p.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language of documents to train/predict on (exact 'language' value, e.g. 'Français' or "
             "'Anglais'). Rows in other languages keep their existing topic values. "
             "Default: per-config preset, else 'Français'.",
    )
    p.add_argument("--num-topics", type=int, default=None,
                   help=f"Number of LDA topics; pins k (default: preset, else {DEFAULT_NUM_TOPICS})")
    p.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="Training passes over the corpus")
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Max iterations per pass")
    p.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help="Documents per training chunk")
    p.add_argument("--no-below", type=int, default=DEFAULT_NO_BELOW,
                   help="Min number of source documents (group column) a dictionary term must occur in")
    p.add_argument("--no-above", type=float, default=DEFAULT_NO_ABOVE,
                   help="Max share of source documents (group column) a dictionary term may occur in")
    p.add_argument(
        "--phrase-min-docs",
        type=int,
        default=None,
        help="Fit only: apply a learned compound only if it occurs in at least N distinct source "
             f"documents (group column). Default: per-config preset, else {DEFAULT_PHRASE_MIN_DOCS}.",
    )
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (1 = reproducible single-core)")
    p.add_argument("--model-path", default=None, help="Directory to save/load the LDA model (default: per-config preset, else 'lda_model')")
    p.add_argument("--max-shard-size", default="1GB")
    p.add_argument("--batch-size", type=int, default=500, help="Batch size for HF dataset map")
    p.add_argument("--max-documents", type=int, default=None, help="Limit training docs (for testing)")
    p.add_argument("--min-train-tokens", type=int, default=DEFAULT_MIN_TRAIN_TOKENS,
                   help="Min tokens to include a doc in training")
    p.add_argument(
        "--min-predict-tokens",
        type=int,
        default=None,
        help="Minimum-evidence floor: documents with fewer in-vocabulary tokens get null topic "
             "columns (still flagged in doc_topics.parquet). Default: the model's training floor "
             "(--min-train-tokens).",
    )
    p.add_argument(
        "--chunk-words",
        type=int,
        default=None,
        help="Train/predict on N-token chunks instead of whole documents "
             "(recommended for long-document subsets: references, publications). "
             "Prediction averages chunk distributions back to one mixture per document. "
             "In predict mode the value is read from the model's training_parameters.json "
             "when not given.",
    )
    p.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=None,
        help="Fit only: train on at most N chunks per document, evenly spaced, so a "
             "few long books cannot dominate the topics. Prediction still uses "
             "every chunk. Default: per-config preset, else no cap.",
    )
    p.add_argument("--skip-coherence", action="store_true", help="Skip coherence metric computation")
    p.add_argument(
        "--domain-stopwords-file",
        type=str,
        default=None,
        help="Extra stopwords file (one word per line, UTF-8)",
    )
    p.add_argument("--topic-label-words", type=int, default=6, help="Number of words in topic labels")
    p.add_argument(
        "--topic-topk",
        type=int,
        default=DEFAULT_TOPIC_TOPK,
        help="Number of topics kept in the lda_topic_topk distribution column",
    )
    # Topic-number sweep: a diagnostic. k stays a pinned, explicit choice.
    p.add_argument(
        "--optimize-topics",
        action="store_true",
        help="Run the k sweep (C_v with seed sd, stability, held-out bound), write its report "
             "and exit: no model bundle, no push. Same as --sweep-only unless "
             "--select-k-from-sweep is given.",
    )
    p.add_argument(
        "--sweep-only",
        action="store_true",
        help="Run the k sweep, report it and exit without fitting, publishing or pushing.",
    )
    p.add_argument(
        "--select-k-from-sweep",
        action="store_true",
        help="Explicit opt-in: fit the production model at the sweep's highest mean C_v. "
             "Refused when k is pinned by the preset or --num-topics; warns when the best and "
             "runner-up are within one seed sd.",
    )
    p.add_argument("--topic-range-start", type=int, default=None, help=f"Optimisation: first k to try (default: preset, else {DEFAULT_TOPIC_RANGE_START})")
    p.add_argument("--topic-range-end", type=int, default=None, help=f"Optimisation: last k to try (default: preset, else {DEFAULT_TOPIC_RANGE_END})")
    p.add_argument("--topic-range-step", type=int, default=None, help=f"Optimisation: step between k values (default: preset, else {DEFAULT_TOPIC_RANGE_STEP})")
    p.add_argument(
        "--sweep-passes",
        type=int,
        default=DEFAULT_SWEEP_PASSES,
        help="Optimisation: passes per sweep model (reduced; final model retrains at --passes)",
    )
    p.add_argument(
        "--sweep-iterations",
        type=int,
        default=DEFAULT_SWEEP_ITERATIONS,
        help="Optimisation: iterations per sweep model (reduced; final model retrains at --iterations)",
    )
    p.add_argument(
        "--stability-seeds",
        type=int,
        default=DEFAULT_STABILITY_SEEDS,
        help="Sweep: models per k (seeds 42..42+N-1); reports mean C_v, its sd and "
             "top-word Jaccard stability. 1 = single seed (no sd)",
    )
    p.add_argument(
        "--holdout",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="Sweep: fraction of independent document groups held out to report per-k held-out "
             "log-perplexity bound (not document-completion; report-only). 0 = off",
    )
    p.add_argument("--holdout-group-column", default="o:id",
                   help="Source-document unit: kept together during evaluation (exact duplicates also "
                        "stay together) and counted once for dictionary and compound document frequency")
    p.add_argument("--allow-legacy-preprocessing", action="store_true",
                   help="Explicitly allow old models without fully frozen preprocessing")
    p.add_argument("--include-unknown-language", action="store_true",
                   help="Explicitly include rows whose language is missing (excluded by default)")
    p.add_argument(
        "--no-relevance-labels",
        action="store_true",
        help="Use pure top-probability topic labels instead of LDAvis-style "
             "relevance-weighted labels (λ=%.2f)" % DEFAULT_LAMBDA_RELEVANCE,
    )
    p.add_argument(
        "--no-theta-export",
        action="store_true",
        help="Skip writing the full document-topic matrix to "
             "<model root>/predictions/<timestamp>/doc_topics.parquet",
    )
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    return p


def _export_theta(ds, theta_col, model_dir, model_name, num_topics, language, logger,
                  source_revision=None, repo_id=None, config_name=None, preprocessing=None,
                  evidence_col=None, min_vocab_tokens=None):
    """Write the full document-topic matrix (theta) for rows this pass
    computed to ``doc_topics.parquet`` in *model_dir* (the prediction run
    directory, ``<model root>/predictions/<timestamp>/``) so downstream
    analyses read exact distributions without re-running inference. One
    float column per topic (topic_0..topic_{k-1}), plus o:id,
    lda_model_name, the in-vocabulary token count and ``below_min_evidence``
    — True for rows whose published topic columns were left null by the
    minimum-evidence floor."""
    import pandas as pd

    ids = ds["o:id"] if "o:id" in ds.column_names else list(range(len(ds)))
    thetas = ds[theta_col]
    counts = ds[evidence_col] if evidence_col else [None] * len(ds)
    rows = []
    for oid, theta, source_text, n_vocab in zip(ids, thetas, ds["lemma_nostop"], counts):
        if theta is None:
            continue
        row = {"o:id": str(oid), "lda_model_name": model_name,
               "text_sha256": text_fingerprint(source_text),
               "n_vocab_tokens": n_vocab,
               "below_min_evidence": bool(min_vocab_tokens and n_vocab is not None
                                          and n_vocab < min_vocab_tokens)}
        for k in range(num_topics):
            row[f"topic_{k}"] = float(theta[k]) if k < len(theta) else 0.0
        rows.append(row)
    if not rows:
        logger.warning("No document-topic distributions to export (no rows computed this pass).")
        return
    out_path = model_dir / "doc_topics.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    (model_dir / "doc_topics.metadata.json").write_text(json.dumps({
        "model_id": model_name, "source_revision": source_revision,
        "repo_id": repo_id, "config_name": config_name,
        "language": language, "num_topics": num_topics,
        "sha256": digest_file(out_path),
        "preprocessing_sha256": preprocessing_fingerprint(preprocessing) if preprocessing is not None else None,
        "min_vocab_tokens": min_vocab_tokens,
        "n_below_min_evidence": int(sum(r["below_min_evidence"] for r in rows)),
    }, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Exported document-topic matrix: {out_path} ({len(rows)} docs this pass)")


def _document_chunks(doc, preprocessing):
    size = preprocessing["chunk_words"]
    cap = preprocessing.get("max_chunks_per_doc")
    return [part for part in cap_chunks(chunk_tokens(doc, size) if size else [doc], cap) if part]


def training_chunks(tokenized_docs, preprocessing):
    """Chunk each document, then apply the frozen per-document cap (if any)."""
    return [part for doc in tokenized_docs for part in _document_chunks(doc, preprocessing)]


def prepare_training_corpus(texts, preprocessing, no_below, no_above, groups=None):
    """Fit phrase/vocabulary transforms solely on the supplied training texts.

    *groups* names each text's source document (default: each text is its
    own). Settings with ``document_frequency_unit == "source_document"``
    count the dictionary's no_below/no_above in distinct source documents
    rather than chunks, and ``phrase_min_docs`` drops compounds found in
    fewer source documents; older settings keep the chunk-level counting
    they were fitted with.
    """
    if groups is None:
        groups = list(range(len(texts)))
    tokenizer = prediction_tokenizer_kwargs(preprocessing)
    tokenized, phraser = tokenize_documents(
        texts, **tokenizer,
        detect_phrases=preprocessing["detect_phrases"],
        phrase_min_count=preprocessing["phrase_min_count"],
        phrase_threshold=preprocessing["phrase_threshold"],
        phrase_min_docs=preprocessing.get("phrase_min_docs", 1),
        groups=groups,
    )
    chunks, chunk_groups = [], []
    for doc, group in zip(tokenized, groups):
        for part in _document_chunks(doc, preprocessing):
            chunks.append(part)
            chunk_groups.append(group)
    if not chunks:
        raise ValueError("No valid training tokens")
    by_source = preprocessing.get("document_frequency_unit") == "source_document"
    dictionary = build_dictionary(chunks, no_below=no_below, no_above=no_above,
                                  groups=chunk_groups if by_source else None)
    if not len(dictionary):
        raise ValueError("Dictionary is empty after frequency filtering")
    return chunks, phraser, dictionary, build_corpus(dictionary, chunks)


def prepare_evaluation(texts, groups, preprocessing, no_below, no_above, holdout):
    """Group split precedes every fitted transform; holdout only gets transform()."""
    train_idx, held_idx = split_document_indices(texts, groups, holdout)
    tokenized, phraser, dictionary, corpus = prepare_training_corpus(
        [texts[i] for i in train_idx], preprocessing, no_below, no_above,
        groups=[groups[i] for i in train_idx],
    )
    held_tokens = [tokenize_for_prediction(
        texts[i], phraser=phraser, **prediction_tokenizer_kwargs(preprocessing)
    ) for i in held_idx]
    held_tokens = training_chunks(held_tokens, preprocessing)
    held_corpus = [bow for bow in build_corpus(dictionary, held_tokens) if bow]
    if held_idx and not held_corpus:
        raise ValueError("Held-out documents contain no training-vocabulary tokens")
    return corpus, dictionary, tokenized, held_corpus, train_idx, held_idx


# ── Run settings ────────────────────────────────────────────────────

TEXT_COLUMN = "lemma_nostop"
TOPIC_ID_COL = "lda_topic_id"
TOPIC_PROB_COL = "lda_topic_prob"
TOPIC_LABEL_COL = "lda_topic_label"
TOPIC_TOPK_COL = "lda_topic_topk"
MODEL_NAME_COL = "lda_model_name"  # disambiguates FR vs EN per-language models
NEW_COLUMNS = [TOPIC_ID_COL, TOPIC_PROB_COL, TOPIC_LABEL_COL, TOPIC_TOPK_COL, MODEL_NAME_COL]
# Per-pass internal columns, extracted into the run directory and dropped
# before the push.
THETA_COL = "_lda_theta"
EVIDENCE_COL = "_lda_vocab_tokens"


@dataclass
class RunSettings:
    """Resolved settings: explicit CLI > frozen bundle (predict) > preset > defaults."""

    repo_id: str
    config_name: str
    mode: str
    preset: dict
    language: str
    model_root: Path
    model_dir: Path
    chunk_words: int | None
    max_chunks_per_doc: int | None
    phrase_min_docs: int
    num_topics: int
    k_source: str  # "--num-topics", "preset", "default" or "sweep"
    sweep: bool
    sweep_only: bool
    range_start: int
    range_end: int
    range_step: int
    lambda_relevance: float | None
    min_predict_tokens: int
    model_id: str | None = None
    saved_preprocessing: dict | None = None
    saved_params: dict | None = None

    @property
    def k_pinned(self) -> bool:
        return self.k_source in ("--num-topics", "preset")


@dataclass
class TrainingDocuments:
    texts: list[str]
    ids: list[str]
    groups: list[Any]


@dataclass
class SweepOutcome:
    results: list[dict]
    ranking: dict
    evaluation_split: dict


@dataclass
class LoadedModel:
    lda_model: Any
    dictionary: Any
    phraser: Any
    topic_labels: dict | None
    word_probs: Any
    lambda_relevance: float | None
    preprocessing: dict
    model_dir: Path
    model_id: str
    stopwords: set = field(default_factory=set)


def validate_args(args) -> None:
    if not 0 <= args.holdout < 1:
        raise SystemExit("--holdout must be in [0, 1)")
    if args.chunk_words is not None and args.chunk_words <= 0:
        raise SystemExit("--chunk-words must be positive")
    if args.max_chunks_per_doc is not None and args.max_chunks_per_doc <= 0:
        raise SystemExit("--max-chunks-per-doc must be positive")
    if args.phrase_min_docs is not None and args.phrase_min_docs < 1:
        raise SystemExit("--phrase-min-docs must be at least 1")
    if args.min_predict_tokens is not None and args.min_predict_tokens < 0:
        raise SystemExit("--min-predict-tokens must be non-negative")
    if args.sweep_only and args.select_k_from_sweep:
        raise SystemExit("--sweep-only and --select-k-from-sweep are mutually exclusive")


def resolve_settings(args, config_name: str, mode: str, logger) -> RunSettings:
    preset = CONFIG_PRESETS.get(config_name, {})
    language: str = args.language or preset.get("language", "Français")
    # One config can serve several languages from separate models, so fold
    # this language's overrides in before anything else reads the preset —
    # notably model_path, which otherwise defaults to the other language's
    # model and overwrites it.
    preset = {**preset, **preset.get("language_overrides", {}).get(language, {})}
    model_root = Path(args.model_path) if args.model_path else workspace_root() / preset.get("model_path", "lda_model")
    if mode == "fit" and (model_root / "bundle.json").exists():
        raise ValueError("Fit requires a model root, not an immutable bundle directory")
    chunk_words = args.chunk_words if args.chunk_words is not None else preset.get("chunk_words")
    max_chunks_per_doc = (
        args.max_chunks_per_doc if args.max_chunks_per_doc is not None
        else preset.get("max_chunks_per_doc")
    )
    if max_chunks_per_doc and not chunk_words:
        raise SystemExit("--max-chunks-per-doc requires --chunk-words")

    sweep = bool(args.optimize_topics or args.sweep_only or args.select_k_from_sweep)
    if mode == "predict" and sweep:
        raise SystemExit("The k sweep is fit-only (--optimize-topics / --sweep-only / --select-k-from-sweep)")
    if mode == "predict" and args.phrase_min_docs is not None:
        raise ValueError("--phrase-min-docs is a training setting; predict uses the model's own phrasers")
    if args.num_topics is not None:
        num_topics, k_source = args.num_topics, "--num-topics"
    elif "num_topics" in preset:
        num_topics, k_source = preset["num_topics"], "preset"
    else:
        num_topics, k_source = DEFAULT_NUM_TOPICS, "default"
    # An explicit or preset k always wins: the sweep can choose k only for
    # an unpinned config, and only on request.
    if args.select_k_from_sweep and k_source != "default":
        pin = "--num-topics" if k_source == "--num-topics" else f"the '{config_name}' preset"
        raise SystemExit(
            f"k is pinned at {num_topics} by {pin}; an explicit or preset k always wins. "
            "Review the sweep with --sweep-only and change the pin deliberately."
        )

    p_range = preset.get("topic_range")
    settings = RunSettings(
        repo_id=args.repo, config_name=config_name, mode=mode, preset=preset,
        language=language, model_root=model_root, model_dir=model_root,
        chunk_words=chunk_words, max_chunks_per_doc=max_chunks_per_doc,
        phrase_min_docs=(args.phrase_min_docs if args.phrase_min_docs is not None
                         else preset.get("phrase_min_docs", DEFAULT_PHRASE_MIN_DOCS)),
        num_topics=num_topics, k_source=k_source,
        sweep=sweep, sweep_only=sweep and not args.select_k_from_sweep,
        range_start=args.topic_range_start if args.topic_range_start is not None else (p_range[0] if p_range else DEFAULT_TOPIC_RANGE_START),
        range_end=args.topic_range_end if args.topic_range_end is not None else (p_range[1] if p_range else DEFAULT_TOPIC_RANGE_END),
        range_step=args.topic_range_step if args.topic_range_step is not None else (p_range[2] if p_range else DEFAULT_TOPIC_RANGE_STEP),
        # LDAvis-style relevance weight for labels (None = pure probability).
        lambda_relevance=None if args.no_relevance_labels else DEFAULT_LAMBDA_RELEVANCE,
        min_predict_tokens=args.min_predict_tokens if args.min_predict_tokens is not None else args.min_train_tokens,
    )
    if mode == "predict":
        _apply_frozen_settings(settings, args)
    _log_settings(settings, logger)
    return settings


def _apply_frozen_settings(settings: RunSettings, args) -> None:
    """Predict mode: the bundle's frozen preprocessing overrides presets."""
    model_dir, model_id = resolve_bundle(settings.model_root)
    saved_preprocessing, saved_params = load_preprocessing(
        model_dir, allow_legacy=args.allow_legacy_preprocessing,
    )
    if args.domain_stopwords_file:
        raise ValueError("Predict mode uses frozen stopwords; refit to change preprocessing")
    if args.language is not None and args.language != saved_preprocessing["language"]:
        raise ValueError("--language does not match the trained model")
    if args.chunk_words is not None and args.chunk_words != saved_preprocessing["chunk_words"]:
        raise ValueError("--chunk-words does not match the trained model")
    if args.max_chunks_per_doc is not None:
        raise ValueError("--max-chunks-per-doc is a training setting; predict uses the model's own")
    if args.no_relevance_labels and saved_params.get("lda", {}).get("label_lambda_relevance") is not None:
        raise ValueError("Predict mode uses frozen labels; refit to change label settings")
    settings.model_dir = model_dir
    settings.model_id = effective_model_identity(model_id, saved_preprocessing)
    settings.saved_preprocessing, settings.saved_params = saved_preprocessing, saved_params
    settings.language = saved_preprocessing["language"]
    settings.chunk_words = saved_preprocessing["chunk_words"]
    # Models fitted before the floor was recorded all used the default 5.
    training_floor = saved_params.get("extra", {}).get("min_train_tokens", DEFAULT_MIN_TRAIN_TOKENS)
    settings.min_predict_tokens = (
        args.min_predict_tokens if args.min_predict_tokens is not None else int(training_floor)
    )


def _log_settings(settings: RunSettings, logger) -> None:
    if not settings.preset:
        return
    if settings.sweep:
        k_text = (f", k-sweep {settings.range_start}-{settings.range_end} step {settings.range_step}"
                  + (" (report only)" if settings.sweep_only else " (selects k)"))
    else:
        k_text = f", k={settings.num_topics} ({'pinned' if settings.k_pinned else 'default'})"
    logger.info(
        f"Preset '{settings.config_name}': language={settings.language}, model_path={settings.model_dir}, "
        f"chunk_words={settings.chunk_words}"
        + (f", max_chunks_per_doc={settings.max_chunks_per_doc}"
           if settings.mode == "fit" and settings.max_chunks_per_doc else "")
        + (f", phrase_min_docs={settings.phrase_min_docs}" if settings.mode == "fit" else "")
        + k_text
        + " (explicit CLI flags override)"
    )


# ── Dataset ─────────────────────────────────────────────────────────


def load_source_dataset(args, settings: RunSettings, token, logger):
    """Load the subset and check it can serve this run; None means exit 1."""
    logger.info(f"Loading dataset '{settings.repo_id}' config '{settings.config_name}'...")
    ds = load_hub_dataset(settings.repo_id, settings.config_name, token=token, console=console)
    logger.info(f"Loaded {len(ds)} rows.")
    language = settings.language

    if "language" in ds.column_names:
        langs = ds["language"]
        lang_count = sum(1 for lang in langs if lang == language)
        other_count = sum(1 for lang in langs if lang and lang != language)
        logger.info(f"{language}: {lang_count} | Other: {other_count} | Total: {len(ds)}")
        unknown_count = sum(1 for lang in langs if lang is None or not str(lang).strip())
        if lang_count == 0 and not (args.include_unknown_language and unknown_count):
            logger.error(f"No '{language}' documents found.")
            return None
    else:
        if not args.include_unknown_language:
            logger.error("No language column; use --include-unknown-language to include unclassified texts explicitly.")
            return None
        logger.warning("No 'language' column — explicitly including unclassified texts.")

    if TEXT_COLUMN not in ds.column_names:
        logger.error(f"Column '{TEXT_COLUMN}' not found. Available: {ds.column_names}")
        return None
    return ds


def confirm_overwrite(ds, args, logger) -> bool:
    existing = [c for c in NEW_COLUMNS if c in ds.column_names]
    if not existing:
        return True
    logger.warning(f"Columns already exist and will be overwritten: {existing}")
    if args.yes:
        return True
    try:
        confirm = input("Continue and overwrite? (y/N): ").strip().lower()
    except KeyboardInterrupt:
        logger.info("\nCancelled.")
        return False
    if confirm not in ("y", "yes", "o", "oui"):
        logger.info("Cancelled.")
        return False
    return True


def build_stopwords(args, language: str, logger) -> set[str]:
    """Pre-phrase stopwords for this model's language, plus an optional file."""
    stopwords = modeling_stopwords(language)
    if args.domain_stopwords_file:
        try:
            sw_path = Path(args.domain_stopwords_file)
            if sw_path.exists():
                with sw_path.open("r", encoding="utf-8", errors="replace") as f:
                    extra = [line.strip().lower() for line in f if line.strip()]
                stopwords.update(extra)
                logger.info(f"Loaded {len(extra)} extra stopwords")
            else:
                logger.warning(f"Stopwords file not found: {sw_path}")
        except Exception as e:
            logger.warning(f"Could not load extra stopwords: {e}")
    return stopwords


def select_training_documents(ds, args, settings: RunSettings, logger) -> TrainingDocuments:
    """Eligible training texts with their ids and source-document groups.

    Reads only the four columns it needs: iterating whole rows would decode
    OCR text and embedding vectors for every row.
    """
    language = settings.language
    group_col = args.holdout_group_column
    logger.info(f"Extracting '{language}' texts from {TEXT_COLUMN}...")
    if group_col not in ds.column_names:
        raise ValueError(f"Missing holdout group column: {group_col}")
    if "o:id" not in ds.column_names:
        raise ValueError("Training requires an 'o:id' column")
    columns = list(dict.fromkeys(c for c in ("o:id", group_col, "language", TEXT_COLUMN) if c in ds.column_names))
    view = ds.select_columns(columns)
    languages = view["language"] if "language" in columns else [None] * len(view)
    texts, ids, groups = [], [], []
    for text, lang, oid, group in zip(view[TEXT_COLUMN], languages, view["o:id"], view[group_col]):
        unknown = not str(lang or "").strip()
        if not (lang == language or (args.include_unknown_language and unknown)):
            continue
        if not text or not str(text).strip() or len(str(text).split()) < args.min_train_tokens:
            continue
        texts.append(str(text))
        ids.append(str(oid))
        groups.append(group)
    logger.info(f"{language} docs for training: {len(texts)}")
    if args.max_documents and len(texts) > args.max_documents:
        logger.info(f"Limiting to {args.max_documents} docs")
        texts, ids, groups = (texts[: args.max_documents], ids[: args.max_documents],
                              groups[: args.max_documents])
    if not texts:
        raise ValueError("No eligible training documents")
    if any(g is None or not str(g).strip() for g in groups):
        raise ValueError("Holdout group values must be nonblank")
    return TrainingDocuments(texts, ids, groups)


# ── Sweep (diagnostic) ──────────────────────────────────────────────


def run_topic_sweep(args, settings: RunSettings, docs: TrainingDocuments, preprocessing, logger) -> SweepOutcome:
    """Evaluation models on the training split only; production fit is separate."""
    sweep_corpus, sweep_dictionary, sweep_tokens, held_corpus, train_idx, held_idx = prepare_evaluation(
        docs.texts, docs.groups, preprocessing, args.no_below, args.no_above, args.holdout,
    )
    evaluation_split = {
        "group_column": args.holdout_group_column,
        "train_document_ids": [docs.ids[i] for i in train_idx],
        "holdout_document_ids": [docs.ids[i] for i in held_idx],
        "heldout_nonempty_chunks": len(held_corpus),
        "actual_document_fraction": len(held_idx) / len(docs.texts),
        "holdout_metric": HOLDOUT_METRIC_NOTE if held_idx else None,
    }
    _, results = find_optimal_topics(
        sweep_corpus, sweep_dictionary, sweep_tokens,
        topic_range_start=settings.range_start, topic_range_end=settings.range_end,
        topic_range_step=settings.range_step, sweep_passes=args.sweep_passes,
        sweep_iterations=args.sweep_iterations, chunksize=args.chunksize,
        n_seeds=args.stability_seeds, holdout_corpus=held_corpus or None,
        logger=logger,
    )
    ranking = sweep_ranking(results)
    _display_optimization_results(results, ranking, settings)
    return SweepOutcome(results, ranking, evaluation_split)


def _sweep_record(args, settings: RunSettings, sweep: SweepOutcome) -> dict:
    return {
        "method": "C_v coherence grid search (diagnostic)",
        "range_tested": f"{settings.range_start}-{settings.range_end} step {settings.range_step}",
        "sweep_passes": args.sweep_passes,
        "sweep_iterations": args.sweep_iterations,
        "n_seeds": args.stability_seeds,
        "holdout_fraction": args.holdout,
        "holdout_metric": HOLDOUT_METRIC_NOTE,
        "ranking": sweep.ranking,
        "results": sweep.results,
    }


def write_sweep_report(args, settings: RunSettings, sweep: SweepOutcome, preprocessing,
                       docs: TrainingDocuments, source_revision, logger) -> Path:
    """Sweep-only output: a report and run manifest; no bundle, no push."""
    run_dir = settings.model_root / "sweeps" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    report = {
        "config_name": settings.config_name,
        "language": settings.language,
        "repo_id": settings.repo_id,
        "source_revision": source_revision,
        "k_pinned": settings.k_pinned,
        "pinned_k": settings.num_topics if settings.k_pinned else None,
        "k_source": settings.k_source,
        "num_training_docs": len(docs.texts),
        "no_below": args.no_below,
        "no_above": args.no_above,
        "preprocessing_sha256": preprocessing_fingerprint(preprocessing),
        "preprocessing": preprocessing,
        "evaluation_split": sweep.evaluation_split,
        **_sweep_record(args, settings, sweep),
    }
    path = run_dir / "sweep_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    write_run_manifest(
        run_dir, script="lda_topic_sweep", repo_id=settings.repo_id, revision=source_revision,
        args=args, inputs={"preprocessing": preprocessing}, outputs=[path],
    )
    console.print(f"[green]✓[/green] Sweep report: [cyan]{path}[/cyan]")
    console.print("[blue]ℹ[/blue] Sweep only: no model bundle written, nothing pushed.")
    logger.info("Sweep complete.")
    return path


# ── Fit / load ──────────────────────────────────────────────────────


def fit_and_publish(args, settings: RunSettings, docs: TrainingDocuments, preprocessing,
                    stopwords, sweep: SweepOutcome | None, source_revision, logger) -> LoadedModel:
    """Production fit on all eligible documents, sealed as an immutable bundle."""
    lambda_relevance = settings.lambda_relevance
    tokenized_valid, phraser, dictionary, corpus = prepare_training_corpus(
        docs.texts, preprocessing, args.no_below, args.no_above, groups=docs.groups,
    )
    logger.info(f"Dictionary: {len(dictionary)} terms")
    # Corpus-wide word probabilities p(w) for relevance-weighted labels.
    word_probs = compute_corpus_word_probs(dictionary, corpus)
    lda_model = create_lda_model(
        corpus, dictionary, num_topics=settings.num_topics, passes=args.passes,
        iterations=args.iterations, chunksize=args.chunksize, workers=args.workers,
        logger=logger,
    )
    topic_labels = {
        tid: get_topic_label(
            lda_model, tid, top_n=args.topic_label_words,
            lambda_relevance=lambda_relevance, word_probs=word_probs,
        ) for tid in range(lda_model.num_topics)
    }
    logger.info("Top topics:")
    for tid in range(min(10, lda_model.num_topics)):
        logger.info(f"  Topic {tid}: {topic_labels[tid]}")

    # Save model (including phrasers for prediction). The dictionary's
    # collection frequencies (cfs) persist with it, so predict mode
    # recovers p(w) for relevance labels without the training corpus.
    settings.model_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fit-", dir=settings.model_root))
    save_lda_model(lda_model, dictionary, staging, logger, phraser=phraser)

    coherence_metrics = None
    if not args.skip_coherence:
        logger.info("Computing coherence metrics...")
        coherence_metrics = compute_coherence(lda_model, tokenized_valid, dictionary, corpus, logger)
        _display_coherence(coherence_metrics)

    extra_info: dict = {
        "config_name": settings.config_name,
        "num_training_docs": len(docs.texts),
        "num_training_chunks": len(tokenized_valid),
        "dictionary_size": len(dictionary),
        "chunk_words": settings.chunk_words,
        "max_chunks_per_doc": settings.max_chunks_per_doc,
        "phrase_min_docs": settings.phrase_min_docs,
        "document_frequency_unit": f"source document ({args.holdout_group_column})",
        "min_train_tokens": args.min_train_tokens,
        "k_source": settings.k_source,
        "language": settings.language,
        "repo_id": settings.repo_id,
        "source_revision": source_revision,
        "training_document_ids": docs.ids,
        "include_unknown_language": args.include_unknown_language,
    }
    if sweep is not None:
        extra_info["topic_optimization"] = {
            **_sweep_record(args, settings, sweep),
            "method": "C_v coherence grid search (explicit --select-k-from-sweep)",
            "best_k": settings.num_topics,
        }
    save_model_parameters(
        staging,
        num_topics=settings.num_topics,
        passes=args.passes,
        iterations=args.iterations,
        chunksize=args.chunksize,
        no_below=args.no_below,
        no_above=args.no_above,
        stopwords_used=sorted(stopwords),
        coherence_metrics=coherence_metrics,
        extra_info=extra_info,
        logger=logger,
        alpha="asymmetric" if args.workers and args.workers > 1 else "auto",
        lambda_relevance=lambda_relevance,
        evaluation={
            "sweep_n_seeds": args.stability_seeds,
            "holdout_fraction": args.holdout,
            "performed": sweep is not None,
            "split": sweep.evaluation_split if sweep is not None else None,
        },
        preprocessing=preprocessing,
        topic_labels=topic_labels,
    )
    model_dir, model_id = publish_bundle(staging, settings.model_root)
    logger.info(f"Immutable model bundle: {model_id} ({model_dir})")
    return LoadedModel(lda_model, dictionary, phraser, topic_labels, word_probs, lambda_relevance,
                       preprocessing, model_dir, model_id, set(stopwords))


def load_for_prediction(settings: RunSettings, logger) -> LoadedModel:
    """Frozen bundle: its own stopwords, phrasers, labels and label weighting."""
    preprocessing, saved_params = settings.saved_preprocessing, settings.saved_params
    topic_labels = {int(k): v for k, v in saved_params.get("topic_labels", {}).items()} or None
    lambda_relevance = saved_params.get("lda", {}).get("label_lambda_relevance")
    lda_model, dictionary, phraser = load_lda_model(settings.model_dir, logger)
    # Recover p(w) from the dictionary's persisted collection frequencies
    # so predict labels match fit labels without the training corpus.
    word_probs = compute_corpus_word_probs(dictionary)
    if word_probs is None and lambda_relevance is not None:
        logger.warning(
            "Dictionary has no collection frequencies; falling back to "
            "pure top-probability labels for this run."
        )
        lambda_relevance = None
    return LoadedModel(lda_model, dictionary, phraser, topic_labels, word_probs, lambda_relevance,
                       preprocessing, settings.model_dir, settings.model_id,
                       set(preprocessing["stopwords"]))


# ── Predict / export / push ─────────────────────────────────────────


def predict_dataset(ds, args, settings: RunSettings, model: LoadedModel, logger):
    logger.info("Predicting topics for all documents...")
    logger.info(f"Minimum evidence: {settings.min_predict_tokens} in-vocabulary tokens per document")
    theta_col = None if args.no_theta_export else THETA_COL
    preprocessing = model.preprocessing
    # Declared output types: a first batch in which every row belongs to the
    # other language, has no text or falls below the evidence floor is all
    # None and would otherwise fix a new column's type as null, failing the
    # next batch.
    from datasets import Sequence as HFSequence, Value

    predict_features = ds.features.copy()
    predict_features.update({
        TOPIC_ID_COL: Value("int64"),
        TOPIC_PROB_COL: Value("float64"),
        TOPIC_LABEL_COL: Value("string"),
        TOPIC_TOPK_COL: Value("string"),
        MODEL_NAME_COL: Value("string"),
        EVIDENCE_COL: Value("int64"),
    })
    if theta_col is not None:
        predict_features[theta_col] = HFSequence(Value("float64"))
    return ds.map(
        lambda batch: predict_batch(
            model.lda_model,
            model.dictionary,
            batch,
            text_col=TEXT_COLUMN,
            topic_id_col=TOPIC_ID_COL,
            topic_prob_col=TOPIC_PROB_COL,
            topic_label_col=TOPIC_LABEL_COL,
            stopwords=model.stopwords,
            phraser=model.phraser,
            topic_topk_col=TOPIC_TOPK_COL,
            topk=args.topic_topk,
            chunk_words=settings.chunk_words,
            language=settings.language,
            model_name_col=MODEL_NAME_COL,
            model_name=model.model_id,
            theta_col=theta_col,
            lambda_relevance=model.lambda_relevance,
            word_probs=model.word_probs,
            min_token_length=preprocessing["min_token_length"],
            custom_collocations=[tuple(c) for c in preprocessing["custom_collocations"]],
            fragment_stopwords=set(preprocessing["fragment_stopwords"]),
            topic_labels=model.topic_labels,
            include_unknown_language=args.include_unknown_language,
            min_vocab_tokens=settings.min_predict_tokens,
            evidence_col=EVIDENCE_COL,
        ),
        batched=True,
        batch_size=args.batch_size,
        desc="LDA prediction",
        features=predict_features,
    )


def export_prediction_run(ds_processed, args, settings: RunSettings, model: LoadedModel,
                          source_revision, logger):
    """Write the run directory (theta, topic table, manifest); strip internal columns."""
    # Reports are mutable run products; model artifacts remain sealed.
    model_root = settings.model_root
    output_root = model_root.parent.parent if (model_root / "bundle.json").exists() else model_root
    run_dir = output_root / "predictions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    lda_model = model.lda_model

    if THETA_COL in ds_processed.column_names:
        _export_theta(
            ds_processed, THETA_COL, run_dir, model.model_id,
            lda_model.num_topics, settings.language, logger, source_revision, settings.repo_id,
            settings.config_name, model.preprocessing,
            evidence_col=EVIDENCE_COL, min_vocab_tokens=settings.min_predict_tokens,
        )

    topic_ids = ds_processed[TOPIC_ID_COL]
    topic_probs = ds_processed[TOPIC_PROB_COL]
    evidence = ds_processed[EVIDENCE_COL]
    processed = sum(1 for t in topic_ids if t is not None)
    skipped = len(topic_ids) - processed
    logger.info(f"With topics ({settings.language} + preserved): {processed} | Without: {skipped} | Total: {len(topic_ids)}")

    # Topic-level stats must cover only the rows THIS run predicted. Rows in
    # another language keep the ids written by that language's model, and
    # those do not index into this one (references: 16 French topics, 8
    # English): a foreign id reaching get_topic_label below raised
    # IndexError and killed the run just short of the push. The predicate
    # mirrors predict_batch, which also predicts rows with no language value.
    own_rows = [i for i, name in enumerate(ds_processed[MODEL_NAME_COL]) if name == model.model_id]
    floor = settings.min_predict_tokens
    suppressed = sum(
        1 for i in own_rows
        if topic_ids[i] is None and evidence[i] is not None and floor and evidence[i] < floor
    )
    if suppressed:
        logger.info(f"Below the evidence floor ({floor} in-vocabulary tokens), left without a topic: {suppressed}")

    valid_ids = [topic_ids[i] for i in own_rows if topic_ids[i] is not None]
    if valid_ids:
        logger.info(f"Unique topics assigned: {len(set(valid_ids))}")
        valid_probs_list = [
            topic_probs[i] for i in own_rows
            if topic_probs[i] is not None and topic_probs[i] > 0
        ]
        if valid_probs_list:
            logger.info(f"Mean probability: {np.mean(valid_probs_list):.3f}")
        counts = Counter(valid_ids)
        topics_path = export_topic_table(
            lda_model, run_dir, model_name=model.model_id, counts=counts,
            top_n=args.topic_label_words, lambda_relevance=model.lambda_relevance,
            word_probs=model.word_probs, labels=model.topic_labels,
        )
        logger.info(f"Topic lookup table written to {topics_path}")
        logger.info("Top 10 most frequent topics:")
        for tid, count in counts.most_common(10):
            label = model.topic_labels[tid] if model.topic_labels is not None else get_topic_label(
                lda_model, tid, top_n=args.topic_label_words,
                lambda_relevance=model.lambda_relevance, word_probs=model.word_probs,
            )
            logger.info(f"  Topic {tid}: {label} ({count} docs)")

    write_run_manifest(
        run_dir, script="lda_topic_modeling", repo_id=settings.repo_id,
        revision=source_revision, args=args,
        inputs={"model_id": model.model_id, "model_dir": str(model.model_dir),
                "preprocessing": model.preprocessing,
                "min_vocab_tokens": settings.min_predict_tokens},
        outputs=list(run_dir.glob("*.parquet")) + list(run_dir.glob("*.csv")) + list(run_dir.glob("*.metadata.json")),
    )
    ds_processed = ds_processed.remove_columns(
        [c for c in (THETA_COL, EVIDENCE_COL) if c in ds_processed.column_names]
    )
    ds_processed = _reorder_columns(ds_processed, logger)
    stats = {"processed": processed, "skipped": skipped, "suppressed": suppressed}
    return ds_processed, stats


def _reorder_columns(ds_processed, logger):
    cols = list(ds_processed.column_names)
    if TEXT_COLUMN not in cols:
        return ds_processed
    idx = cols.index(TEXT_COLUMN) + 1
    ordered = cols[:idx]
    for c in NEW_COLUMNS:
        if c in cols and c not in ordered:
            ordered.append(c)
    for c in cols[idx:]:
        if c not in ordered:
            ordered.append(c)
    logger.info("Columns reordered.")
    return ds_processed.select_columns(ordered)


def push_predictions(ds_processed, args, settings: RunSettings, model: LoadedModel,
                     token, source_revision, logger) -> bool:
    logger.info("Pushing dataset to Hub...")
    ok = push_dataset(
        ds_processed,
        repo_id=settings.repo_id,
        config_name=settings.config_name,
        commit_message=(
            f"Add LDA topic modeling columns ({', '.join(NEW_COLUMNS)}); "
            f"model {model.model_id}, min in-vocabulary tokens {settings.min_predict_tokens}"
        ),
        token=token,
        max_shard_size=args.max_shard_size,
        console=console,
        expected_revision=source_revision,
    )
    if ok:
        logger.info("Dataset saved successfully.")
    return bool(ok)


def print_summary(ds, args, settings: RunSettings, model: LoadedModel, stats: dict) -> None:
    table = Table(title="LDA Topic Modeling Summary", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Columns added", ", ".join(NEW_COLUMNS))
    table.add_row("Method", "LDA (gensim)")
    table.add_row("Language", settings.language)
    table.add_row("Chunk words", str(settings.chunk_words) if settings.chunk_words else "— (whole documents)")
    table.add_row("Num topics", f"{model.lda_model.num_topics} ({settings.k_source})"
                  if settings.mode == "fit" else str(model.lda_model.num_topics))
    table.add_row("Min in-vocabulary tokens", str(settings.min_predict_tokens))
    table.add_row("Docs with topics", str(stats["processed"]))
    table.add_row("Docs without topics", str(stats["skipped"]))
    table.add_row("  of which below evidence floor", str(stats["suppressed"]))
    table.add_row("Total docs", str(len(ds)))
    table.add_row("Model saved", str(model.model_dir))
    if settings.mode == "fit":
        table.add_row("Passes", str(args.passes))
        table.add_row("Iterations", str(args.iterations))
        table.add_row("Phrase min source docs", str(settings.phrase_min_docs))

    console.print()
    console.print(table)
    console.print()
    console.print("[green]✓[/green] Done!")
    if settings.mode == "fit":
        console.print(f"[blue]→[/blue] Parameters: [cyan]{model.model_dir / 'training_parameters.json'}[/cyan]")


# ── Main ────────────────────────────────────────────────────────────


def main() -> int:
    """Run the pipeline. Returns a process exit code (0 = success)."""
    configure_logging()
    logger = logging.getLogger(__name__)
    args = build_arg_parser().parse_args()
    validate_args(args)

    token = ensure_hf_token(console=console)
    if args.config:
        config_name = args.config
    else:
        available_configs = get_available_configs(
            args.repo, token=token, fallback=["articles", "publications"]
        )
        config_name = choose_config(available_configs, console=console)
    logger.info(f"Config: '{config_name}'")
    mode = args.mode or choose_mode()
    logger.info(f"Mode: '{mode}'")

    settings = resolve_settings(args, config_name, mode, logger)
    ds = load_source_dataset(args, settings, token, logger)
    if ds is None:
        return 1
    source_revision = getattr(ds, "_iwac_source_revision", None)
    if not settings.sweep_only and not confirm_overwrite(ds, args, logger):
        return 0

    if mode == "fit":
        stopwords = build_stopwords(args, settings.language, logger)
        preprocessing = frozen_preprocessing(
            stopwords, settings.language, settings.chunk_words, settings.max_chunks_per_doc,
            phrase_min_docs=settings.phrase_min_docs,
        )
        docs = select_training_documents(ds, args, settings, logger)
        sweep = None
        if settings.sweep:
            sweep = run_topic_sweep(args, settings, docs, preprocessing, logger)
            if settings.sweep_only:
                write_sweep_report(args, settings, sweep, preprocessing, docs, source_revision, logger)
                return 0
            settings.num_topics, settings.k_source = sweep.ranking["best_k"], "sweep"
        elif args.holdout:
            logger.info("Held-out evaluation runs only with the k sweep; this pinned-k fit is descriptive.")
        model = fit_and_publish(args, settings, docs, preprocessing, stopwords, sweep, source_revision, logger)
    else:
        model = load_for_prediction(settings, logger)

    ds_processed = predict_dataset(ds, args, settings, model, logger)
    logger.info("Prediction complete.")
    ds_processed, stats = export_prediction_run(ds_processed, args, settings, model, source_revision, logger)
    if not push_predictions(ds_processed, args, settings, model, token, source_revision, logger):
        return 1
    print_summary(ds, args, settings, model, stats)
    return 0


def _fmt(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def _display_optimization_results(results: list[dict], ranking: dict, settings: RunSettings) -> None:
    """Display the k sweep as a Rich table, with what the ranking can and cannot say."""
    table = Table(title="Topic Number Sweep (diagnostic)", box=box.ROUNDED)
    table.add_column("k", style="cyan", justify="right")
    table.add_column("C_v (mean)", style="green", justify="right")
    table.add_column("C_v sd", justify="right")
    table.add_column("Stability", justify="right")
    table.add_column("Held-out bound*", justify="right")
    table.add_column("NPMI", style="dim", justify="right")
    table.add_column("U_Mass", style="dim", justify="right")
    table.add_column("", justify="center")

    for r in results:
        notes = []
        if r["k"] == ranking["best_k"]:
            notes.append("highest C_v" + ("" if settings.sweep_only else " → selected"))
        if settings.k_pinned and r["k"] == settings.num_topics:
            notes.append("pinned")
        table.add_row(
            str(r["k"]), _fmt(r.get("c_v")), _fmt(r.get("c_v_sd")),
            _fmt(r.get("stability_jaccard"), 3), _fmt(r.get("holdout_log_perplexity")),
            _fmt(r.get("c_npmi")), _fmt(r.get("u_mass")), ", ".join(notes),
        )

    console.print()
    console.print(table)
    if any(r.get("holdout_log_perplexity") is not None for r in results):
        console.print(f"[blue]ℹ[/blue] *Held-out bound: {HOLDOUT_METRIC_NOTE}.")
    if ranking["runner_up_k"] is not None:
        if ranking["distinguishable"] is False:
            console.print(
                f"[yellow]⚠[/yellow] k={ranking['best_k']} and runner-up k={ranking['runner_up_k']} "
                f"differ by {ranking['gap']:.4f} mean C_v, within one seed sd ({ranking['seed_sd']:.4f}): "
                "the C_v ranking does not distinguish them."
            )
        elif ranking["distinguishable"] is None:
            console.print(
                "[yellow]⚠[/yellow] Single seed: no seed sd, so the C_v ranking cannot be judged. "
                "Use --stability-seeds 3 and compare stability."
            )
        else:
            console.print(
                f"[blue]ℹ[/blue] k={ranking['best_k']} leads k={ranking['runner_up_k']} by "
                f"{ranking['gap']:.4f} mean C_v (> seed sd {ranking['seed_sd']:.4f}); C_v remains a diagnostic."
            )
    if settings.k_pinned:
        console.print(
            f"[blue]ℹ[/blue] k is pinned at {settings.num_topics} ({settings.k_source}); "
            "the sweep does not change it."
        )
    console.print()


def _display_coherence(metrics: dict) -> None:
    """Display coherence metrics in a Rich table."""
    if "error" in metrics:
        console.print(f"[yellow]\u26a0[/yellow] Coherence error: {metrics['error']}")
        return

    table = Table(title="LDA Coherence Metrics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green", justify="right")

    for name in ("c_v", "c_npmi", "u_mass", "topic_diversity"):
        if name in metrics and "score" in metrics[name]:
            table.add_row(name.upper(), f"{metrics[name]['score']:.4f}")

    console.print(table)

    if "c_v" in metrics and "score" in metrics["c_v"]:
        cv = metrics["c_v"]["score"]
        console.print(
            f"[yellow]ℹ[/yellow] C_v={cv:.3f} is a corpus-dependent diagnostic; "
            "review representative documents and seed/preprocessing sensitivity before interpreting topics."
        )


if __name__ == "__main__":
    sys.exit(main())
