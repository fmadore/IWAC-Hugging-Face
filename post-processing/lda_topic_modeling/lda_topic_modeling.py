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
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import get_token, login
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from tqdm import tqdm

# Ensure package imports work when running this file directly
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from topic_modeling.patches import apply_all_patches  # type: ignore

from lda_topic_modeling.constants import (  # type: ignore
    DOMAIN_STOPWORDS,
    LABEL_ONLY_STOPWORDS,
    LDA_GEO_STOPWORDS,
    LDA_GENERIC_STOPWORDS,
    CUSTOM_COLLOCATIONS,
    DEFAULT_NUM_TOPICS,
    DEFAULT_PASSES,
    DEFAULT_ITERATIONS,
    DEFAULT_CHUNKSIZE,
    DEFAULT_NO_BELOW,
    DEFAULT_NO_ABOVE,
    DEFAULT_TOPIC_RANGE_START,
    DEFAULT_TOPIC_RANGE_END,
    DEFAULT_TOPIC_RANGE_STEP,
)
from lda_topic_modeling.modeling import (  # type: ignore
    tokenize_documents,
    apply_custom_collocations,
    apply_phraser,
    build_dictionary,
    build_corpus,
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


def get_available_configs(repo_id: str, token: str) -> list[str]:
    try:
        from huggingface_hub import dataset_info

        info = dataset_info(repo_id, token=token)
        if hasattr(info, "config_names") and info.config_names:
            return list(info.config_names)
        return ["articles", "publications"]
    except Exception:
        return ["articles", "publications"]


def choose_config(available: list[str]) -> str:
    if len(available) == 1:
        console.print(f"[green]\u2713[/green] Single config: [cyan]{available[0]}[/cyan]")
        return available[0]

    table = Table(title="Available configurations", box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Configuration", style="green")
    for i, cfg in enumerate(available, 1):
        table.add_row(str(i), cfg)
    console.print(table)

    while True:
        try:
            choice = IntPrompt.ask(
                "[yellow]\u2192[/yellow] Choose a configuration",
                choices=[str(i) for i in range(1, len(available) + 1)],
                show_choices=False,
            )
            return available[choice - 1]
        except KeyboardInterrupt:
            raise SystemExit(0)


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
    p.add_argument("--repo", default="fmadore/islam-west-africa-collection")
    p.add_argument("--config", type=str, default=None, help="Dataset config name (skip interactive prompt)")
    p.add_argument("--mode", type=str, choices=["fit", "predict"], default=None, help="Run mode (skip interactive prompt)")
    p.add_argument("--num-topics", type=int, default=DEFAULT_NUM_TOPICS, help="Number of LDA topics")
    p.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="Training passes over the corpus")
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Max iterations per pass")
    p.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help="Documents per training chunk")
    p.add_argument("--no-below", type=int, default=DEFAULT_NO_BELOW, help="Min document frequency for dictionary")
    p.add_argument("--no-above", type=float, default=DEFAULT_NO_ABOVE, help="Max document frequency ratio for dictionary")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (1 = reproducible single-core)")
    p.add_argument("--model-path", default="lda_model", help="Directory to save/load the LDA model")
    p.add_argument("--max-shard-size", default="1GB")
    p.add_argument("--batch-size", type=int, default=500, help="Batch size for HF dataset map")
    p.add_argument("--max-documents", type=int, default=None, help="Limit training docs (for testing)")
    p.add_argument("--min-train-tokens", type=int, default=5, help="Min tokens to include a doc in training")
    p.add_argument("--skip-coherence", action="store_true", help="Skip coherence metric computation")
    p.add_argument(
        "--domain-stopwords-file",
        type=str,
        default=None,
        help="Extra stopwords file (one word per line, UTF-8)",
    )
    p.add_argument("--topic-label-words", type=int, default=6, help="Number of words in topic labels")
    # Topic-number optimisation (DH best practice: sweep k, pick best C_v)
    p.add_argument(
        "--optimize-topics",
        action="store_true",
        help="Sweep a range of topic counts and pick the k with best C_v coherence (recommended for first run)",
    )
    p.add_argument("--topic-range-start", type=int, default=DEFAULT_TOPIC_RANGE_START, help="Optimisation: first k to try")
    p.add_argument("--topic-range-end", type=int, default=DEFAULT_TOPIC_RANGE_END, help="Optimisation: last k to try")
    p.add_argument("--topic-range-step", type=int, default=DEFAULT_TOPIC_RANGE_STEP, help="Optimisation: step between k values")
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    return p


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    apply_all_patches()
    configure_logging()
    logger = logging.getLogger(__name__)

    args = build_arg_parser().parse_args()

    repo_id: str = args.repo
    text_column = "lemma_nostop"
    topic_id_col = "lda_topic_id"
    topic_prob_col = "lda_topic_prob"
    topic_label_col = "lda_topic_label"
    new_columns = [topic_id_col, topic_prob_col, topic_label_col]
    model_dir = Path(args.model_path)

    # ── Auth ────────────────────────────────────────────────────────
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        logger.info("HF token not found. Trying interactive login.")
        try:
            login()
            token = get_token()
            if not token:
                logger.error("Login failed.")
                return
        except Exception as e:
            logger.error(f"Login error: {e}")
            return

    # ── Config ──────────────────────────────────────────────────────
    if args.config:
        config_name = args.config
    else:
        available_configs = get_available_configs(repo_id, token)
        config_name = choose_config(available_configs)
    logger.info(f"Config: '{config_name}'")

    if args.mode:
        mode = args.mode
    else:
        mode = choose_mode()
    logger.info(f"Mode: '{mode}'")

    # ── Load dataset ────────────────────────────────────────────────
    logger.info(f"Loading dataset '{repo_id}' config '{config_name}'...")
    try:
        ds = load_dataset(repo_id, name=config_name, split="train", token=token)
    except Exception as e:
        logger.error(f"Dataset load error: {e}")
        return
    logger.info(f"Loaded {len(ds)} rows.")

    # Language stats
    if "language" in ds.column_names:
        langs = ds["language"]
        fr_count = sum(1 for l in langs if l == "Français")
        other_count = sum(1 for l in langs if l and l != "Français")
        logger.info(f"French: {fr_count} | Other: {other_count} | Total: {len(ds)}")
        if fr_count == 0:
            logger.error("No French documents found.")
            return
    else:
        logger.warning("No 'language' column — all texts will be processed.")

    if text_column not in ds.column_names:
        logger.error(f"Column '{text_column}' not found. Available: {ds.column_names}")
        return

    # Check existing columns
    existing = [c for c in new_columns if c in ds.column_names]
    if existing:
        logger.warning(f"Columns already exist and will be overwritten: {existing}")
        if not args.yes:
            try:
                confirm = input("Continue and overwrite? (y/N): ").strip().lower()
                if confirm not in ("y", "yes", "o", "oui"):
                    logger.info("Cancelled.")
                    return
            except KeyboardInterrupt:
                logger.info("\nCancelled.")
                return

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
        # Extract French texts
        logger.info("Extracting French texts from lemma_nostop...")
        if "language" in ds.column_names:
            docs = [
                str(t)
                for t, lang in zip(ds[text_column], ds["language"])
                if lang == "Français"
                and t
                and str(t).strip()
                and len(str(t).split()) >= args.min_train_tokens
            ]
        else:
            docs = [str(t) for t in ds[text_column] if t and str(t).strip()]
        logger.info(f"French docs for training: {len(docs)}")

        if args.max_documents and len(docs) > args.max_documents:
            logger.info(f"Limiting to {args.max_documents} docs")
            docs = docs[: args.max_documents]

        # Tokenize (with bigram/trigram phrase detection)
        logger.info("Tokenizing (with phrase detection)...")
        tokenized, phraser = tokenize_documents(docs, stopwords=stopwords)
        valid = [(d, t) for d, t in zip(docs, tokenized) if t]
        if not valid:
            logger.error("No valid tokenized documents.")
            return
        docs_valid = [d for d, _ in valid]
        tokenized_valid = [t for _, t in valid]
        logger.info(f"Valid tokenized docs: {len(tokenized_valid)}")

        # Dictionary + corpus
        logger.info("Building dictionary and corpus...")
        dictionary = build_dictionary(tokenized_valid, no_below=args.no_below, no_above=args.no_above)
        logger.info(f"Dictionary: {len(dictionary)} terms")
        corpus = build_corpus(dictionary, tokenized_valid)

        # Optimise num_topics if requested (DH best practice)
        num_topics = args.num_topics
        optimization_results = None
        if args.optimize_topics:
            logger.info("Running topic-number optimisation (this may take a while)...")
            best_k, optimization_results = find_optimal_topics(
                corpus,
                dictionary,
                tokenized_valid,
                topic_range_start=args.topic_range_start,
                topic_range_end=args.topic_range_end,
                topic_range_step=args.topic_range_step,
                passes=args.passes,
                iterations=args.iterations,
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
        }
        if optimization_results is not None:
            extra_info["topic_optimization"] = {
                "method": "C_v coherence grid search",
                "range_tested": f"{args.topic_range_start}-{args.topic_range_end} step {args.topic_range_step}",
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
        )
    else:
        # Load existing model
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return
        lda_model, dictionary, phraser = load_lda_model(model_dir, logger)

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
    logger.info(f"Processed (French): {processed} | Skipped: {skipped} | Total: {len(topic_ids)}")

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
        return

    # ── Summary ─────────────────────────────────────────────────────
    table = Table(title="LDA Topic Modeling Summary", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Columns added", ", ".join(new_columns))
    table.add_row("Method", "LDA (gensim)")
    table.add_row("Num topics", str(lda_model.num_topics))
    table.add_row("French docs processed", str(processed))
    table.add_row("Docs skipped", str(skipped))
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
    main()
