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
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Ensure package imports work when running this file directly
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from _common import choose_config, ensure_hf_token, get_available_configs, PRIVATE_REPO_ID  # type: ignore  # noqa: E402

from lda_topic_modeling.constants import (  # type: ignore
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
)
from lda_topic_modeling.modeling import (  # type: ignore
    tokenize_documents,
    build_dictionary,
    build_corpus,
    chunk_tokens,
    create_lda_model,
    save_lda_model,
    load_lda_model,
    predict_batch,
    compute_coherence,
    save_model_parameters,
    get_topic_label,
    find_optimal_topics,
)

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
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    return p


# ── Main ────────────────────────────────────────────────────────────


def main() -> int:
    """Run the pipeline. Returns a process exit code (0 = success)."""
    configure_logging()
    logger = logging.getLogger(__name__)

    args = build_arg_parser().parse_args()

    repo_id: str = args.repo
    text_column = "lemma_nostop"
    topic_id_col = "lda_topic_id"
    topic_prob_col = "lda_topic_prob"
    topic_label_col = "lda_topic_label"
    topic_topk_col = "lda_topic_topk"
    new_columns = [topic_id_col, topic_prob_col, topic_label_col, topic_topk_col]

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
    model_dir = Path(args.model_path or preset.get("model_path", "lda_model"))
    chunk_words: int | None = (
        args.chunk_words if args.chunk_words is not None else preset.get("chunk_words")
    )
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
            + (f", k-sweep {range_start}-{range_end} step {range_step}" if optimize_topics else "")
            + " (explicit CLI flags override)"
        )

    # ── Load dataset ────────────────────────────────────────────────
    logger.info(f"Loading dataset '{repo_id}' config '{config_name}'...")
    try:
        ds = load_dataset(repo_id, name=config_name, split="train", token=token)
    except Exception as e:
        logger.error(f"Dataset load error: {e}")
        return 1
    logger.info(f"Loaded {len(ds)} rows.")

    # Language stats
    if "language" in ds.column_names:
        langs = ds["language"]
        lang_count = sum(1 for l in langs if l == language)
        other_count = sum(1 for l in langs if l and l != language)
        logger.info(f"{language}: {lang_count} | Other: {other_count} | Total: {len(ds)}")
        if lang_count == 0:
            logger.error(f"No '{language}' documents found.")
            return 1
    else:
        logger.warning("No 'language' column — all texts will be processed.")

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

    if mode == "fit":
        # Extract training texts in the target language
        logger.info(f"Extracting '{language}' texts from lemma_nostop...")
        if "language" in ds.column_names:
            docs = [
                str(t)
                for t, lang in zip(ds[text_column], ds["language"])
                if lang == language
                and t
                and str(t).strip()
                and len(str(t).split()) >= args.min_train_tokens
            ]
        else:
            docs = [str(t) for t in ds[text_column] if t and str(t).strip()]
        logger.info(f"{language} docs for training: {len(docs)}")

        if args.max_documents and len(docs) > args.max_documents:
            logger.info(f"Limiting to {args.max_documents} docs")
            docs = docs[: args.max_documents]

        # Tokenize (with bigram/trigram phrase detection)
        logger.info("Tokenizing (with phrase detection)...")
        tokenized, phraser = tokenize_documents(docs, stopwords=stopwords)
        valid = [(d, t) for d, t in zip(docs, tokenized) if t]
        if not valid:
            logger.error("No valid tokenized documents.")
            return 1
        docs_valid = [d for d, _ in valid]
        tokenized_valid = [t for _, t in valid]
        logger.info(f"Valid tokenized docs: {len(tokenized_valid)}")

        # Long-document subsets train on fixed-size chunks: phrase
        # detection above ran on whole documents, so phrase tokens
        # survive the split intact.
        if chunk_words:
            tokenized_valid = [
                chunk
                for doc_tokens in tokenized_valid
                for chunk in chunk_tokens(doc_tokens, chunk_words)
                if chunk
            ]
            logger.info(
                f"Chunking at {chunk_words} tokens: "
                f"{len(valid)} documents -> {len(tokenized_valid)} training chunks"
            )

        # Dictionary + corpus
        logger.info("Building dictionary and corpus...")
        dictionary = build_dictionary(tokenized_valid, no_below=args.no_below, no_above=args.no_above)
        logger.info(f"Dictionary: {len(dictionary)} terms")
        corpus = build_corpus(dictionary, tokenized_valid)

        # Optimise num_topics if requested (DH best practice)
        num_topics = args.num_topics if args.num_topics is not None else DEFAULT_NUM_TOPICS
        optimization_results = None
        if optimize_topics:
            logger.info(
                "Running topic-number optimisation "
                f"(sweep models at passes={args.sweep_passes}, iterations={args.sweep_iterations}; "
                "the winning k retrains at full settings)..."
            )
            best_k, optimization_results = find_optimal_topics(
                corpus,
                dictionary,
                tokenized_valid,
                topic_range_start=range_start,
                topic_range_end=range_end,
                topic_range_step=range_step,
                sweep_passes=args.sweep_passes,
                sweep_iterations=args.sweep_iterations,
                chunksize=args.chunksize,
                logger=logger,
            )
            _display_optimization_results(optimization_results, best_k)
            num_topics = best_k
            logger.info(f"Using optimal num_topics={num_topics}")

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

        # Log top topics
        logger.info("Top topics:")
        for tid in range(min(10, lda_model.num_topics)):
            label = get_topic_label(lda_model, tid, top_n=args.topic_label_words)
            logger.info(f"  Topic {tid}: {label}")

        # Save model (including phrasers for prediction)
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
            "num_training_docs": len(tokenized_valid),
            "dictionary_size": len(dictionary),
            "chunk_words": chunk_words,
            "language": language,
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
        )
    else:
        # Load existing model
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return 1
        lda_model, dictionary, phraser = load_lda_model(model_dir, logger)

        # A chunk-trained model must predict with the same chunking and
        # language filter. The params file reflects how THIS model was
        # trained, so it overrides the preset (but not explicit CLI flags).
        params_path = model_dir / "training_parameters.json"
        if params_path.exists():
            try:
                import json

                saved_extra = json.loads(params_path.read_text(encoding="utf-8")).get("extra", {})
                if args.chunk_words is None and saved_extra.get("chunk_words"):
                    chunk_words = int(saved_extra["chunk_words"])
                    logger.info(f"Using chunk_words={chunk_words} from training_parameters.json")
                if args.language is None and saved_extra.get("language"):
                    language = str(saved_extra["language"])
                    logger.info(f"Using language={language} from training_parameters.json")
            except Exception as e:
                logger.warning(f"Could not read settings from {params_path}: {e}")

    # ── Predict on full dataset ─────────────────────────────────────
    logger.info("Predicting topics for all documents...")

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
        ),
        batched=True,
        batch_size=args.batch_size,
        desc="LDA prediction",
    )

    logger.info("Prediction complete.")

    # ── Statistics ──────────────────────────────────────────────────
    topic_ids = ds_processed[topic_id_col]
    topic_probs = ds_processed[topic_prob_col]

    processed = sum(1 for t in topic_ids if t is not None)
    skipped = sum(1 for t in topic_ids if t is None)
    logger.info(f"With topics ({language} + preserved): {processed} | Without: {skipped} | Total: {len(topic_ids)}")

    valid_ids = [t for t in topic_ids if t is not None]
    if valid_ids:
        unique_topics = set(valid_ids)
        logger.info(f"Unique topics assigned: {len(unique_topics)}")

        valid_probs_list = [p for p in topic_probs if p is not None and p > 0]
        if valid_probs_list:
            logger.info(f"Mean probability: {np.mean(valid_probs_list):.3f}")

        counts = Counter(valid_ids)
        logger.info("Top 10 most frequent topics:")
        for tid, count in counts.most_common(10):
            label = get_topic_label(lda_model, tid, top_n=args.topic_label_words)
            logger.info(f"  Topic {tid}: {label} ({count} docs)")

    # ── Reorder columns ────────────────────────────────────────────
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
    try:
        ds_processed.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
            commit_message=f"Add LDA topic modeling columns ({', '.join(new_columns)})",
            token=token,
            max_shard_size=args.max_shard_size,
        )
        logger.info("Dataset saved successfully.")
    except Exception as e:
        logger.error(f"Push error: {e}")
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
        if cv >= 0.5:
            console.print("[green]\u2713[/green] Good coherence (C_v >= 0.5)")
        elif cv >= 0.4:
            console.print("[yellow]i[/yellow] Acceptable coherence (C_v 0.4-0.5)")
        else:
            console.print("[yellow]\u26a0[/yellow] Low coherence (C_v < 0.4) — consider adjusting num_topics")


if __name__ == "__main__":
    sys.exit(main())
