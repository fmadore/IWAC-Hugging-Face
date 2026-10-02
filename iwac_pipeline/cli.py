"""Installed commands dispatch to importable modules, in a wheel or checkout."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence

from iwac_common.schema import ALL_CONFIGS
from iwac_common.upload_runner import run_upload

UPLOAD_MODULES = {
    "articles": "iwac_pipeline.uploads.articles.upload_newspaper_hf",
    "publications": "iwac_pipeline.uploads.publications.upload_Islamic_publications_hf",
    "documents": "iwac_pipeline.uploads.documents.upload_documents_hf",
    "references": "iwac_pipeline.uploads.references.upload_reference_hf",
    "index": "iwac_pipeline.uploads.index.upload_index_hf",
    "images": "iwac_pipeline.uploads.images.upload_image_hf",
    "audiovisual": "iwac_pipeline.uploads.audiovisual.upload_audiovisual_hf",
}
PROCESS_MODULES = {
    "embeddings": "iwac_pipeline.processing.semantic_embedding",
    "image-embeddings": "iwac_pipeline.processing.semantic_embedding_images",
    "lemmas": "lemmatize_update_hf",
    "lexical": "iwac_pipeline.processing.calculate_lexical_richness",
    "word-count": "iwac_pipeline.processing.calculate_word_count",
    "hijri": "iwac_pipeline.processing.calculate_hijri_dates",
    "lda": "iwac_pipeline.processing.lda_topic_modeling.lda_topic_modeling",
    "sentiment-agreement": "iwac_pipeline.processing.sentiment_agreement",
    "related": "iwac_pipeline.processing.related_articles",
}
ANALYSIS_MODULES = {
    "topics": "iwac_pipeline.analyses.topic_prevalence",
    "topic-sentiment": "iwac_pipeline.analyses.topic_sentiment",
    "keyness": "iwac_pipeline.analyses.keyness_bursts",
    "entities": "iwac_pipeline.analyses.entity_networks",
    "coverage": "iwac_pipeline.analyses.corpus_coverage",
    "annotation-review": "iwac_pipeline.analyses.annotation_review",
    "reprints": "iwac_pipeline.analyses.reprint_candidates",
    "topic-review": "iwac_pipeline.analyses.topic_review",
}


def _dispatch_args(prog: str, choices, argv: Sequence[str] | None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("command", choices=choices)
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Subcommand --help belongs to the subcommand's parser.
    args = parser.parse_args(arguments[:1])
    return args.command, arguments[1:]


def _run_module(module_name: str, argv: Sequence[str]) -> int:
    module = importlib.import_module(module_name)
    previous = sys.argv
    try:
        sys.argv = [module_name, *argv]
        return int(module.main() or 0)
    finally:
        sys.argv = previous


def upload_main(argv: Sequence[str] | None = None) -> int:
    """Refresh one subset from Omeka into a verified private destination."""
    command, remaining = _dispatch_args("iwac-upload", ALL_CONFIGS, argv)
    module = importlib.import_module(UPLOAD_MODULES[command])
    return run_upload(module.SPEC, remaining)


def process_main(argv: Sequence[str] | None = None) -> int:
    """Run an enrichment; install the nlp/analysis extras for its dependencies."""
    command, remaining = _dispatch_args("iwac-process", PROCESS_MODULES, argv)
    return _run_module(PROCESS_MODULES[command], remaining)


def analyze_main(argv: Sequence[str] | None = None) -> int:
    """Run an analysis or prepare a source-linked research validation report."""
    command, remaining = _dispatch_args("iwac-analyze", ANALYSIS_MODULES, argv)
    return _run_module(ANALYSIS_MODULES[command], remaining)


def mirror_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="iwac-mirror")
    parser.add_argument("--dataset", choices=["private", "public"], default=None)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args(argv)
    module = importlib.import_module("iwac_pipeline.mirror.fetch_datasets")
    repo_id, label = module.choose_dataset(args.dataset)
    return int(module.main(dataset_id=repo_id, label=label, fmt=args.format))


def publish_public_main(argv: Sequence[str] | None = None) -> int:
    return _run_module("iwac_pipeline.processing.publish_public", sys.argv[1:] if argv is None else argv)
