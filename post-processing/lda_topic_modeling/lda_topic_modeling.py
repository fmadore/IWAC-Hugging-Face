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
from datetime import datetime, timezone
from pathlib import Path

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
    DOMAIN_STOPWORDS,
    LDA_GEO_STOPWORDS,
    LDA_GENERIC_STOPWORDS,
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
    tokenize_for_prediction,
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
            raise SystemExit(0)


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
                   help=f"Number of LDA topics (default: preset/optimizer, else {DEFAULT_NUM_TOPICS})")
    p.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="Training passes over the corpus")
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Max iterations per pass")
    p.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help="Documents per training chunk")
    p.add_argument("--no-below", type=int, default=DEFAULT_NO_BELOW, help="Min document frequency for dictionary")
    p.add_argument("--no-above", type=float, default=DEFAULT_NO_ABOVE, help="Max document frequency ratio for dictionary")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (1 = reproducible single-core)")
    p.add_argument("--model-path", default=None, help="Directory to save/load the LDA model (default: per-config preset, else 'lda_model')")
    p.add_argument("--max-shard-size", default="1GB")
    p.add_argument("--batch-size", type=int, default=500, help="Batch size for HF dataset map")
    p.add_argument("--max-documents", type=int, default=None, help="Limit training docs (for testing)")
    p.add_argument("--min-train-tokens", type=int, default=5, help="Min tokens to include a doc in training")
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
    # Topic-number optimisation (DH best practice: sweep k, pick best C_v)
    p.add_argument(
        "--optimize-topics",
        action="store_true",
        help="Sweep a range of topic counts and pick the k with best C_v coherence "
             "(auto-enabled by the publications/references presets when --num-topics is not given)",
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
        help="Optimisation: models per k (seeds 42..42+N-1); select by mean C_v and "
             "report top-word Jaccard stability. 1 = legacy single-seed sweep",
    )
    p.add_argument(
        "--holdout",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="Optimisation: fraction of independent document groups held out to report per-k held-out "
             "log-perplexity (the winning k retrains on ALL docs). 0 = off",
    )
    p.add_argument("--holdout-group-column", default="o:id",
                   help="Keep a document/reprint group together during evaluation; exact duplicates also stay together")
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
             "<model_dir>/doc_topics.parquet during predict",
    )
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    return p


def _export_theta(ds, theta_col, model_dir, model_name, num_topics, language, logger,
                  source_revision=None, repo_id=None, config_name=None, preprocessing=None):
    """Write the full document-topic matrix (theta) for rows this pass
    computed to ``<model_dir>/doc_topics.parquet`` so downstream analyses read
    exact distributions without re-running inference. One float column per
    topic (topic_0..topic_{k-1}), plus o:id and lda_model_name."""
    import pandas as pd

    ids = ds["o:id"] if "o:id" in ds.column_names else list(range(len(ds)))
    thetas = ds[theta_col]
    rows = []
    for oid, theta, source_text in zip(ids, thetas, ds["lemma_nostop"]):
        if theta is None:
            continue
        row = {"o:id": str(oid), "lda_model_name": model_name,
               "text_sha256": text_fingerprint(source_text)}
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
    }, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Exported document-topic matrix: {out_path} ({len(rows)} docs this pass)")


def prepare_training_corpus(texts, preprocessing, no_below, no_above):
    """Fit phrase/vocabulary transforms solely on the supplied training texts."""
    tokenizer = prediction_tokenizer_kwargs(preprocessing)
    tokenized, phraser = tokenize_documents(
        texts, **tokenizer,
        detect_phrases=preprocessing["detect_phrases"],
        phrase_min_count=preprocessing["phrase_min_count"],
        phrase_threshold=preprocessing["phrase_threshold"],
    )
    chunks = preprocessing["chunk_words"]
    tokenized = [part for doc in tokenized
                 for part in (chunk_tokens(doc, chunks) if chunks else [doc]) if part]
    if not tokenized:
        raise ValueError("No valid training tokens")
    dictionary = build_dictionary(tokenized, no_below=no_below, no_above=no_above)
    if not len(dictionary):
        raise ValueError("Dictionary is empty after frequency filtering")
    return tokenized, phraser, dictionary, build_corpus(dictionary, tokenized)


def prepare_evaluation(texts, groups, preprocessing, no_below, no_above, holdout):
    """Group split precedes every fitted transform; holdout only gets transform()."""
    train_idx, held_idx = split_document_indices(texts, groups, holdout)
    tokenized, phraser, dictionary, corpus = prepare_training_corpus(
        [texts[i] for i in train_idx], preprocessing, no_below, no_above,
    )
    held_tokens = [tokenize_for_prediction(
        texts[i], phraser=phraser, **prediction_tokenizer_kwargs(preprocessing)
    ) for i in held_idx]
    chunks = preprocessing["chunk_words"]
    held_tokens = [part for doc in held_tokens
                   for part in (chunk_tokens(doc, chunks) if chunks else [doc]) if part]
    held_corpus = [bow for bow in build_corpus(dictionary, held_tokens) if bow]
    if held_idx and not held_corpus:
        raise ValueError("Held-out documents contain no training-vocabulary tokens")
    return corpus, dictionary, tokenized, held_corpus, train_idx, held_idx


# ── Main ────────────────────────────────────────────────────────────


def main() -> int:
    """Run the pipeline. Returns a process exit code (0 = success)."""
    configure_logging()
    logger = logging.getLogger(__name__)

    args = build_arg_parser().parse_args()
    if not 0 <= args.holdout < 1:
        raise SystemExit("--holdout must be in [0, 1)")
    if args.chunk_words is not None and args.chunk_words <= 0:
        raise SystemExit("--chunk-words must be positive")

    repo_id: str = args.repo
    text_column = "lemma_nostop"
    topic_id_col = "lda_topic_id"
    topic_prob_col = "lda_topic_prob"
    topic_label_col = "lda_topic_label"
    topic_topk_col = "lda_topic_topk"
    model_name_col = "lda_model_name"  # disambiguates FR vs EN per-language models
    new_columns = [topic_id_col, topic_prob_col, topic_label_col, topic_topk_col, model_name_col]
    # LDAvis-style relevance weight for labels (None = legacy pure-probability).
    lambda_relevance = None if args.no_relevance_labels else DEFAULT_LAMBDA_RELEVANCE

    # ── Auth ────────────────────────────────────────────────────────
    token = ensure_hf_token(console=console)

    # ── Config ──────────────────────────────────────────────────────
    if args.config:
        config_name = args.config
    else:
        available_configs = get_available_configs(
            repo_id, token=token, fallback=["articles", "publications"]
        )
        config_name = choose_config(available_configs, console=console)
    logger.info(f"Config: '{config_name}'")

    if args.mode:
        mode = args.mode
    else:
        mode = choose_mode()
    logger.info(f"Mode: '{mode}'")

    # ── Resolve settings: explicit CLI > params file (predict) > preset > defaults
    preset = CONFIG_PRESETS.get(config_name, {})
    language: str = args.language or preset.get("language", "Français")
    # One config can serve several languages from separate models, so fold
    # this language's overrides in before anything else reads the preset —
    # notably model_path, which otherwise defaults to the other language's
    # model and overwrites it.
    preset = {**preset, **preset.get("language_overrides", {}).get(language, {})}
    model_root = Path(args.model_path) if args.model_path else workspace_root() / preset.get("model_path", "lda_model")
    model_dir = model_root
    if mode == "fit" and (model_root / "bundle.json").exists():
        raise ValueError("Fit requires a model root, not an immutable bundle directory")
    chunk_words: int | None = (
        args.chunk_words if args.chunk_words is not None else preset.get("chunk_words")
    )
    saved_prediction_settings = None
    if mode == "predict":
        model_dir, model_id = resolve_bundle(model_root)
        saved_prediction_settings = load_preprocessing(
            model_dir, allow_legacy=args.allow_legacy_preprocessing,
        )
        saved_preprocessing, saved_params = saved_prediction_settings
        model_id = effective_model_identity(model_id, saved_preprocessing)
        if args.domain_stopwords_file:
            raise ValueError("Predict mode uses frozen stopwords; refit to change preprocessing")
        if args.language is not None and args.language != saved_preprocessing["language"]:
            raise ValueError("--language does not match the trained model")
        if args.chunk_words is not None and args.chunk_words != saved_preprocessing["chunk_words"]:
            raise ValueError("--chunk-words does not match the trained model")
        if args.no_relevance_labels and saved_params.get("lda", {}).get("label_lambda_relevance") is not None:
            raise ValueError("Predict mode uses frozen labels; refit to change label settings")
        language = saved_preprocessing["language"]
        chunk_words = saved_preprocessing["chunk_words"]
    p_range = preset.get("topic_range")
    range_start = args.topic_range_start if args.topic_range_start is not None else (p_range[0] if p_range else DEFAULT_TOPIC_RANGE_START)
    range_end = args.topic_range_end if args.topic_range_end is not None else (p_range[1] if p_range else DEFAULT_TOPIC_RANGE_END)
    range_step = args.topic_range_step if args.topic_range_step is not None else (p_range[2] if p_range else DEFAULT_TOPIC_RANGE_STEP)
    # Preset may auto-enable the k-sweep, but an explicit --num-topics wins.
    optimize_topics = args.optimize_topics or (
        mode == "fit" and args.num_topics is None and preset.get("optimize_topics", False)
    )
    if preset:
        logger.info(
            f"Preset '{config_name}': language={language}, model_path={model_dir}, "
            f"chunk_words={chunk_words}"
            + (
                f", k-sweep {range_start}-{range_end} step {range_step}"
                if optimize_topics
                else f", k={preset.get('num_topics', DEFAULT_NUM_TOPICS)} (pinned)"
            )
            + " (explicit CLI flags override)"
        )

    # ── Load dataset ────────────────────────────────────────────────
    logger.info(f"Loading dataset '{repo_id}' config '{config_name}'...")
    ds = load_hub_dataset(repo_id, config_name, token=token, console=console)
    source_revision = getattr(ds, "_iwac_source_revision", None)
    logger.info(f"Loaded {len(ds)} rows.")

    # Language stats
    if "language" in ds.column_names:
        langs = ds["language"]
        lang_count = sum(1 for l in langs if l == language)
        other_count = sum(1 for l in langs if l and l != language)
        logger.info(f"{language}: {lang_count} | Other: {other_count} | Total: {len(ds)}")
        unknown_count = sum(1 for lang in langs if lang is None or not str(lang).strip())
        if lang_count == 0 and not (args.include_unknown_language and unknown_count):
            logger.error(f"No '{language}' documents found.")
            return 1
    else:
        if not args.include_unknown_language:
            logger.error("No language column; use --include-unknown-language to include unclassified texts explicitly.")
            return 1
        logger.warning("No 'language' column — explicitly including unclassified texts.")

    if text_column not in ds.column_names:
        logger.error(f"Column '{text_column}' not found. Available: {ds.column_names}")
        return 1

    # Check existing columns
    existing = [c for c in new_columns if c in ds.column_names]
    if existing:
        logger.warning(f"Columns already exist and will be overwritten: {existing}")
        if not args.yes:
            try:
                confirm = input("Continue and overwrite? (y/N): ").strip().lower()
                if confirm not in ("y", "yes", "o", "oui"):
                    logger.info("Cancelled.")
                    return 0
            except KeyboardInterrupt:
                logger.info("\nCancelled.")
                return 0

    # ── Build stopwords ─────────────────────────────────────────────
    stopwords = set(DOMAIN_STOPWORDS) | LDA_GEO_STOPWORDS | LDA_GENERIC_STOPWORDS
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

    # ── Train or load ───────────────────────────────────────────────
    lda_model = None
    dictionary = None
    phraser = None
    topic_labels = None
    preprocessing = frozen_preprocessing(stopwords, language, chunk_words)

    if mode == "fit":
        # Extract training texts in the target language
        logger.info(f"Extracting '{language}' texts from lemma_nostop...")
        if args.holdout_group_column not in ds.column_names:
            raise ValueError(f"Missing holdout group column: {args.holdout_group_column}")
        records = [
            r for r in ds
            if (r.get("language") == language or (args.include_unknown_language and not str(r.get("language") or "").strip()))
            and r.get(text_column) and str(r[text_column]).strip()
            and len(str(r[text_column]).split()) >= args.min_train_tokens
        ]
        docs = [str(r[text_column]) for r in records]
        logger.info(f"{language} docs for training: {len(docs)}")

        if args.max_documents and len(docs) > args.max_documents:
            logger.info(f"Limiting to {args.max_documents} docs")
            docs = docs[: args.max_documents]
            records = records[: args.max_documents]

        if not docs:
            raise ValueError("No eligible training documents")
        num_topics = args.num_topics if args.num_topics is not None else preset.get("num_topics", DEFAULT_NUM_TOPICS)
        optimization_results = None
        evaluation_split = None
        if optimize_topics:
            groups = [r[args.holdout_group_column] for r in records]
            if any(g is None or not str(g).strip() for g in groups):
                raise ValueError("Holdout group values must be nonblank")
            sweep_corpus, sweep_dictionary, sweep_tokens, held_corpus, train_idx, held_idx = prepare_evaluation(
                docs, groups, preprocessing, args.no_below, args.no_above, args.holdout,
            )
            evaluation_split = {
                "group_column": args.holdout_group_column,
                "train_document_ids": [str(records[i]["o:id"]) for i in train_idx],
                "holdout_document_ids": [str(records[i]["o:id"]) for i in held_idx],
                "heldout_nonempty_chunks": len(held_corpus),
                "actual_document_fraction": len(held_idx) / len(docs),
            }
            num_topics, optimization_results = find_optimal_topics(
                sweep_corpus, sweep_dictionary, sweep_tokens,
                topic_range_start=range_start, topic_range_end=range_end,
                topic_range_step=range_step, sweep_passes=args.sweep_passes,
                sweep_iterations=args.sweep_iterations, chunksize=args.chunksize,
                n_seeds=args.stability_seeds, holdout_corpus=held_corpus or None,
                logger=logger,
            )
            _display_optimization_results(optimization_results, num_topics)
        elif args.holdout:
            logger.info("Held-out evaluation runs only with --optimize-topics; this pinned-k fit is descriptive.")

        # Production fit is explicitly separate from the evaluation models.
        tokenized_valid, phraser, dictionary, corpus = prepare_training_corpus(
            docs, preprocessing, args.no_below, args.no_above,
        )
        logger.info(f"Dictionary: {len(dictionary)} terms")
        # Corpus-wide word probabilities p(w) for relevance-weighted labels.
        word_probs = compute_corpus_word_probs(dictionary, corpus)

        # Train
        lda_model = create_lda_model(
            corpus,
            dictionary,
            num_topics=num_topics,
            passes=args.passes,
            iterations=args.iterations,
            chunksize=args.chunksize,
            workers=args.workers,
            logger=logger,
        )

        # Log top topics (relevance-weighted labels unless disabled)
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
        model_root.mkdir(parents=True, exist_ok=True)
        model_dir = Path(tempfile.mkdtemp(prefix=".fit-", dir=model_root))
        save_lda_model(lda_model, dictionary, model_dir, logger, phraser=phraser)

        # Coherence
        coherence_metrics = None
        if not args.skip_coherence:
            logger.info("Computing coherence metrics...")
            coherence_metrics = compute_coherence(
                lda_model, tokenized_valid, dictionary, corpus, logger
            )
            _display_coherence(coherence_metrics)

        # Save parameters
        extra_info: dict = {
            "config_name": config_name,
            "num_training_docs": len(docs),
            "num_training_chunks": len(tokenized_valid),
            "dictionary_size": len(dictionary),
            "chunk_words": chunk_words,
            "language": language,
            "repo_id": repo_id,
            "source_revision": source_revision,
            "training_document_ids": [str(r["o:id"]) for r in records],
            "include_unknown_language": args.include_unknown_language,
        }
        if optimization_results is not None:
            extra_info["topic_optimization"] = {
                "method": "C_v coherence grid search",
                "range_tested": f"{range_start}-{range_end} step {range_step}",
                "sweep_passes": args.sweep_passes,
                "sweep_iterations": args.sweep_iterations,
                "best_k": num_topics,
                "results": optimization_results,
            }
        save_model_parameters(
            model_dir,
            num_topics=num_topics,
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
                "performed": bool(optimize_topics),
                "split": evaluation_split,
            },
            preprocessing=preprocessing,
            topic_labels=topic_labels,
        )
        model_dir, model_id = publish_bundle(model_dir, model_root)
        logger.info(f"Immutable model bundle: {model_id} ({model_dir})")
        word_probs_for_predict = word_probs
    else:
        # Load existing model
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return 1
        preprocessing, saved_params = saved_prediction_settings
        language = preprocessing["language"]
        chunk_words = preprocessing["chunk_words"]
        stopwords = set(preprocessing["stopwords"])
        topic_labels = {int(k): v for k, v in saved_params.get("topic_labels", {}).items()} or None
        lambda_relevance = saved_params.get("lda", {}).get("label_lambda_relevance")
        lda_model, dictionary, phraser = load_lda_model(model_dir, logger)
        # Recover p(w) from the dictionary's persisted collection frequencies
        # so predict labels match fit labels without the training corpus.
        word_probs_for_predict = compute_corpus_word_probs(dictionary)
        if word_probs_for_predict is None and lambda_relevance is not None:
            logger.warning(
                "Dictionary has no collection frequencies; falling back to "
                "pure top-probability labels for this run."
            )
            lambda_relevance = None

    # ── Predict on full dataset ─────────────────────────────────────
    logger.info("Predicting topics for all documents...")

    theta_col = None if args.no_theta_export else "_lda_theta"
    # Declared output types: a first batch in which every row belongs to the
    # other language (or has no text) is all None and would otherwise fix a
    # new column's type as null, failing the next batch.
    from datasets import Sequence as HFSequence, Value

    predict_features = ds.features.copy()
    predict_features.update({
        topic_id_col: Value("int64"),
        topic_prob_col: Value("float64"),
        topic_label_col: Value("string"),
        topic_topk_col: Value("string"),
        model_name_col: Value("string"),
    })
    if theta_col is not None:
        predict_features[theta_col] = HFSequence(Value("float64"))
    ds_processed = ds.map(
        lambda batch: predict_batch(
            lda_model,
            dictionary,
            batch,
            text_col=text_column,
            topic_id_col=topic_id_col,
            topic_prob_col=topic_prob_col,
            topic_label_col=topic_label_col,
            stopwords=stopwords,
            phraser=phraser,
            topic_topk_col=topic_topk_col,
            topk=args.topic_topk,
            chunk_words=chunk_words,
            language=language,
            model_name_col=model_name_col,
            model_name=model_id,
            theta_col=theta_col,
            lambda_relevance=lambda_relevance,
            word_probs=word_probs_for_predict,
            min_token_length=preprocessing["min_token_length"],
            custom_collocations=[tuple(c) for c in preprocessing["custom_collocations"]],
            fragment_stopwords=set(preprocessing["fragment_stopwords"]),
            topic_labels=topic_labels,
            include_unknown_language=args.include_unknown_language,
        ),
        batched=True,
        batch_size=args.batch_size,
        desc="LDA prediction",
        features=predict_features,
    )

    logger.info("Prediction complete.")
    # Reports are mutable run products; model artifacts above remain sealed.
    output_root = model_root.parent.parent if (model_root / "bundle.json").exists() else model_root
    run_dir = output_root / "predictions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)

    # ── Export full document-topic matrix (theta) ───────────────────
    if theta_col is not None and theta_col in ds_processed.column_names:
        _export_theta(
            ds_processed, theta_col, run_dir, model_id,
            lda_model.num_topics, language, logger, source_revision, repo_id, config_name, preprocessing,
        )
        ds_processed = ds_processed.remove_columns([theta_col])

    # ── Statistics ──────────────────────────────────────────────────
    topic_ids = ds_processed[topic_id_col]
    topic_probs = ds_processed[topic_prob_col]

    processed = sum(1 for t in topic_ids if t is not None)
    skipped = sum(1 for t in topic_ids if t is None)
    logger.info(f"With topics ({language} + preserved): {processed} | Without: {skipped} | Total: {len(topic_ids)}")

    # Topic-level stats must cover only the rows THIS run predicted. Rows in
    # another language keep the ids written by that language's model, and
    # those do not index into this one: with references at 24 French topics
    # and 16 English ones, a French id of 18 reaching get_topic_label below
    # raised IndexError and killed the run just short of the push. The
    # predicate mirrors predict_batch, which also predicts rows with no
    # language value.
    own_rows = [i for i, name in enumerate(ds_processed[model_name_col]) if name == model_id]

    valid_ids = [topic_ids[i] for i in own_rows if topic_ids[i] is not None]
    if valid_ids:
        unique_topics = set(valid_ids)
        logger.info(f"Unique topics assigned: {len(unique_topics)}")

        valid_probs_list = [
            topic_probs[i] for i in own_rows
            if topic_probs[i] is not None and topic_probs[i] > 0
        ]
        if valid_probs_list:
            logger.info(f"Mean probability: {np.mean(valid_probs_list):.3f}")

        counts = Counter(valid_ids)
        topics_path = export_topic_table(
            lda_model, run_dir, model_name=model_id, counts=counts,
            top_n=args.topic_label_words, lambda_relevance=lambda_relevance,
            word_probs=word_probs_for_predict,
            labels=topic_labels,
        )
        logger.info(f"Topic lookup table written to {topics_path}")
        logger.info("Top 10 most frequent topics:")
        for tid, count in counts.most_common(10):
            label = topic_labels[tid] if topic_labels is not None else get_topic_label(
                lda_model, tid, top_n=args.topic_label_words,
                lambda_relevance=lambda_relevance, word_probs=word_probs_for_predict,
            )
            logger.info(f"  Topic {tid}: {label} ({count} docs)")

    # ── Reorder columns ────────────────────────────────────────────
    write_run_manifest(
        run_dir, script="lda_topic_modeling", repo_id=repo_id,
        revision=source_revision, args=args,
        inputs={"model_id": model_id, "model_dir": str(model_dir),
                "preprocessing": preprocessing},
        outputs=list(run_dir.glob("*.parquet")) + list(run_dir.glob("*.csv")) + list(run_dir.glob("*.metadata.json")),
    )
    insert_after = "lemma_nostop"
    cols = list(ds_processed.column_names)
    if insert_after in cols:
        idx = cols.index(insert_after) + 1
        ordered = cols[:idx]
        for c in new_columns:
            if c in cols and c not in ordered:
                ordered.append(c)
        for c in cols[idx:]:
            if c not in ordered:
                ordered.append(c)
        ds_processed = ds_processed.select_columns(ordered)
        logger.info("Columns reordered.")

    # ── Push to Hub ─────────────────────────────────────────────────
    logger.info("Pushing dataset to Hub...")
    if push_dataset(
        ds_processed,
        repo_id=repo_id,
        config_name=config_name,
        commit_message=f"Add LDA topic modeling columns ({', '.join(new_columns)})",
        token=token,
        max_shard_size=args.max_shard_size,
        console=console,
        expected_revision=source_revision,
    ):
        logger.info("Dataset saved successfully.")
    else:
        return 1

    # ── Summary ─────────────────────────────────────────────────────
    table = Table(title="LDA Topic Modeling Summary", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Columns added", ", ".join(new_columns))
    table.add_row("Method", "LDA (gensim)")
    table.add_row("Language", language)
    table.add_row("Chunk words", str(chunk_words) if chunk_words else "— (whole documents)")
    table.add_row("Num topics", str(lda_model.num_topics))
    table.add_row("Docs with topics", str(processed))
    table.add_row("Docs without topics", str(skipped))
    table.add_row("Total docs", str(len(ds)))
    table.add_row("Model saved", str(model_dir))
    if mode == "fit":
        table.add_row("Passes", str(args.passes))
        table.add_row("Iterations", str(args.iterations))

    console.print()
    console.print(table)
    console.print()
    console.print("[green]\u2713[/green] Done!")
    if mode == "fit":
        console.print(f"[blue]\u2192[/blue] Parameters: [cyan]{model_dir / 'training_parameters.json'}[/cyan]")
    return 0


def _display_optimization_results(results: list[dict], best_k: int) -> None:
    """Display the topic-number optimisation grid as a Rich table."""
    table = Table(title="Topic Number Optimisation (C_v)", box=box.ROUNDED)
    table.add_column("k", style="cyan", justify="right")
    table.add_column("C_v", style="green", justify="right")
    table.add_column("NPMI", style="dim", justify="right")
    table.add_column("U_Mass", style="dim", justify="right")
    table.add_column("", justify="center")

    for r in results:
        marker = "[bold green]<-- best[/bold green]" if r["k"] == best_k else ""
        cv = f"{r['c_v']:.4f}" if r.get("c_v") is not None else "—"
        npmi = f"{r['c_npmi']:.4f}" if r.get("c_npmi") is not None else "—"
        umass = f"{r['u_mass']:.4f}" if r.get("u_mass") is not None else "—"
        table.add_row(str(r["k"]), cv, npmi, umass, marker)

    console.print()
    console.print(table)
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
