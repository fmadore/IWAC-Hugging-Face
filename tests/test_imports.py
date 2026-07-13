"""Import-smoke: every pipeline script must at least import cleanly.

Catches dead-import breakage, syntax errors, and module-level crashes (the
publications empty-branch NameError would have been caught at review time by
exercising the module). Scripts live in hyphenated directories, so they are
loaded by file path rather than as packages.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "articles/upload_newspaper_hf.py",
    "audiovisual/upload_audiovisual_hf.py",
    "document/upload_documents_hf.py",
    "islamic-publications/upload_Islamic_publications_hf.py",
    "reference/upload_reference_hf.py",
    "index/upload_index_hf.py",
    "images/upload_image_hf.py",
    "lemmatize_update_hf.py",
    "post-processing/publish_public.py",
    "post-processing/calculate_lexical_richness.py",
    "post-processing/calculate_word_count.py",
    "post-processing/calculate_ocr_quality.py",
    "post-processing/semantic_embedding.py",
    "post-processing/semantic_embedding_images.py",
    "post-processing/sentiment_agreement.py",
    "post-processing/related_articles.py",
    "post-processing/lda_topic_modeling/lda_topic_modeling.py",
    # modeling.py uses relative imports; tests/test_analyses.py imports it as
    # a package member instead of by file path.
    "post-processing/lda_topic_modeling/constants.py",
    "analyses/topic_prevalence.py",
    "analyses/keyness_bursts.py",
    "analyses/topic_sentiment.py",
    "analyses/entity_networks.py",
    "data/fetch_datasets.py",
    "country_mapper.py",
]


@pytest.mark.parametrize("rel_path", SCRIPTS)
def test_script_imports(rel_path):
    path = REPO_ROOT / rel_path
    assert path.exists(), f"{rel_path} vanished — update SCRIPTS in this test"
    name = "smoke_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
