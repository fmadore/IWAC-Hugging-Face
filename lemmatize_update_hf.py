#!/usr/bin/env python3
"""
lemmatize_update_hf.py
======================

Add French lemmatisation and stop‑word–filtered text to an existing Hugging Face
Hub dataset.  It downloads the dataset, processes the chosen text column with
spaCy, adds two new columns (lemmatised text and lemmatised text with French
stop‑words removed) and pushes the updated dataset back to the repository.

Usage
-----
    python lemmatize_update_hf.py \
        --repo fmadore/iwac-newspaper-articles \
        --text-column OCR \
        --lemma-column lemma_text \
        --clean-column lemma_nostop \
        --spacy-model fr_core_news_sm \
        --max-shard-size 1GB

Environment variables
---------------------
HF_TOKEN   Personal access token for the Hugging Face Hub (alternatively you
           will be prompted to log‑in interactively).

Dependencies
------------
    pip install datasets huggingface_hub spacy tqdm python-dotenv
    python -m spacy download fr_core_news_sm

"""

import os
import argparse
import logging
from typing import List

import datasets
from datasets import Dataset
from huggingface_hub import login, HfFolder
import spacy
from tqdm import tqdm


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_spacy_model(name: str):
    """Load a spaCy model, downloading it the first time it is required."""
    try:
        return spacy.load(name, disable=["parser", "ner", "textcat"])
    except OSError:
        logging.info("SpaCy model '%s' not found – downloading…", name)
        from spacy.cli import download as spacy_download

        spacy_download(name)
        return spacy.load(name, disable=["parser", "ner", "textcat"])


def lemmatise_batch(
    batch: dict,
    nlp,
    text_col: str,
    lemma_col: str,
    clean_col: str,
) -> dict:
    """Apply lemmatisation and stop‑word removal to one batch of examples."""

    texts: List[str] = batch[text_col]
    lemmas, clean = [], []

    for doc in nlp.pipe(texts, batch_size=32, n_process=1):
        # Full lemmatised text (punctuation kept out, case‑folded)
        lemma_tokens = [tok.lemma_.lower() for tok in doc if tok.is_alpha]
        lemmas.append(" ".join(lemma_tokens))

        # Lemmas without French stop‑words
        nostop_tokens = [tok for tok in lemma_tokens if tok not in nlp.Defaults.stop_words]
        clean.append(" ".join(nostop_tokens))

    return {lemma_col: lemmas, clean_col: clean}


def main():
    configure_logging()

    parser = argparse.ArgumentParser(description="Add lemmatised columns to a Hugging Face dataset")
    parser.add_argument("--repo", required=True, help="Dataset repo on the Hugging Face Hub (e.g. fmadore/iwac-newspaper-articles)")
    parser.add_argument("--text-column", default="OCR", help="Name of the column containing the raw French text to process")
    parser.add_argument("--lemma-column", default="lemma_text", help="Column name for the lemmatised text")
    parser.add_argument("--clean-column", default="lemma_nostop", help="Column name for the lemmatised text with stop‑words removed")
    parser.add_argument("--spacy-model", default="fr_dep_news_trf", help="spaCy model to use for French lemmatisation")
    parser.add_argument("--max-shard-size", default="1GB", help="Maximum Parquet shard size when pushing to the Hub")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Authenticate with the Hub
    # ------------------------------------------------------------------
    if os.getenv("HF_TOKEN") is None and HfFolder.get_token() is None:
        login()

    token = os.getenv("HF_TOKEN") or HfFolder.get_token()

    # ------------------------------------------------------------------
    # Load dataset from the Hub
    # ------------------------------------------------------------------
    logging.info("Loading dataset '%s'…", args.repo)
    ds: Dataset = datasets.load_dataset(args.repo, split="train", token=token)

    if args.text_column not in ds.column_names:
        raise ValueError(
            f"Column '{args.text_column}' not found in the dataset. Available columns: {ds.column_names}"
        )

    # ------------------------------------------------------------------
    # Load the spaCy model once
    # ------------------------------------------------------------------
    logging.info("Loading spaCy model '%s'…", args.spacy_model)
    nlp = load_spacy_model(args.spacy_model)

    # ------------------------------------------------------------------
    # Map lemmatisation over the dataset
    # ------------------------------------------------------------------
    logging.info("Applying lemmatisation – this can take a while…")
    ds = ds.map(
        lemmatise_batch,
        batched=True,
        batch_size=1000,
        fn_kwargs={
            "nlp": nlp,
            "text_col": args.text_column,
            "lemma_col": args.lemma_column,
            "clean_col": args.clean_column,
        },
        desc="lemmatising",
    )

    # ------------------------------------------------------------------
    # Push updated dataset to the Hub
    # ------------------------------------------------------------------
    logging.info("Pushing updated dataset back to %s…", args.repo)
    ds.push_to_hub(
        args.repo,
        token=token,
        max_shard_size=args.max_shard_size,
        commit_message=f"Add columns '{args.lemma_column}' and '{args.clean_column}' (French lemmatisation)",
    )

    logging.info("Done. New columns: %s, %s", args.lemma_column, args.clean_column)


if __name__ == "__main__":
    main()
