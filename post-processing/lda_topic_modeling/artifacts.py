"""Frozen LDA preprocessing, grouped evaluation, and content-addressed bundles.

Model identities describe an exact set of artifacts, not a mutable directory
name or a topic count. Prediction outputs live outside the immutable bundle.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .constants import CUSTOM_COLLOCATIONS, POST_PHRASE_STOPWORDS


def text_fingerprint(text) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def preprocessing_fingerprint(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()


def effective_model_identity(artifact_id: str, preprocessing: dict) -> str:
    """Legacy artifacts do not contain every transform; include reconstructed settings."""
    if artifact_id.startswith("lda-legacy-sha256:"):
        digest = hashlib.sha256((artifact_id + preprocessing_fingerprint(preprocessing)).encode()).hexdigest()
        return f"lda-legacy-sha256:{digest}"
    return artifact_id


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_preprocessing(stopwords, language, chunk_words) -> dict:
    return {
        "version": 1, "stopwords": sorted(stopwords),
        "fragment_stopwords": sorted(POST_PHRASE_STOPWORDS),
        "custom_collocations": [list(c) for c in CUSTOM_COLLOCATIONS],
        "min_token_length": 2, "detect_phrases": True,
        "phrase_min_count": 20, "phrase_threshold": 10.0,
        "language": language, "chunk_words": chunk_words,
        "inference_initialization": "per-document-token-sha256-v1",
    }


def prediction_tokenizer_kwargs(preprocessing: dict) -> dict:
    return {
        "stopwords": set(preprocessing["stopwords"]),
        "min_token_length": preprocessing["min_token_length"],
        "custom_collocations": [tuple(c) for c in preprocessing["custom_collocations"]],
        "fragment_stopwords": set(preprocessing["fragment_stopwords"]),
    }


def split_document_indices(texts, groups, fraction: float, seed: int = 42):
    """Split original documents; groups and exact duplicate texts stay together.

Splitting before phrase learning/chunking prevents both vocabulary leakage and
chunks of one source appearing on opposite sides of evaluation.
"""
    if not 0 <= fraction < 1:
        raise ValueError("holdout must be in [0, 1)")
    if len(texts) != len(groups):
        raise ValueError("Every document needs a holdout group")
    if not fraction:
        return list(range(len(texts))), []
    parent = list(range(len(texts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen_group, seen_text = {}, {}
    for i, (text, group) in enumerate(zip(texts, groups)):
        for table, key in ((seen_group, str(group)), (seen_text, " ".join(text.split()))):
            if key in table:
                parent[root(i)] = root(table[key])
            else:
                table[key] = i
    unique = sorted({root(i) for i in range(len(texts))})
    if len(unique) < 2:
        raise ValueError("Held-out evaluation requires at least two independent document groups")
    n_hold = min(len(unique) - 1, max(1, round(len(unique) * fraction)))
    selected = set(np.random.default_rng(seed).choice(unique, n_hold, replace=False).tolist())
    train = [i for i in range(len(texts)) if root(i) not in selected]
    holdout = [i for i in range(len(texts)) if root(i) in selected]
    return train, holdout


def _artifact_hashes(directory: Path) -> dict:
    return {
        p.relative_to(directory).as_posix(): digest_file(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "bundle.json"
    }


def _identity(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def publish_bundle(staging: Path, root: Path) -> tuple[Path, str]:
    """Seal new model artifacts, then atomically switch the current pointer."""
    files = _artifact_hashes(staging)
    digest = _identity(files)
    model_id = f"lda-sha256:{digest}"
    (staging / "bundle.json").write_text(
        json.dumps({"model_id": model_id, "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )
    destination = root / "bundles" / digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        resolve_bundle(destination)  # Do not silently reuse damaged artifacts.
        shutil.rmtree(staging)
    else:
        staging.rename(destination)
    pointer = root / ".current.json.tmp"
    pointer.write_text(json.dumps({"bundle": f"bundles/{digest}"}) + "\n", encoding="utf-8")
    os.replace(pointer, root / "current.json")
    return destination, model_id


def resolve_bundle(path: Path) -> tuple[Path, str]:
    """Resolve and verify new bundles; legacy directories remain readable."""
    path = Path(path)
    if (path / "current.json").exists():
        relative = json.loads((path / "current.json").read_text(encoding="utf-8"))["bundle"]
        resolved = (path / relative).resolve()
        if not resolved.is_relative_to(path.resolve()):
            raise ValueError("Model pointer must remain inside its model directory")
        path = resolved
    manifest_path = path / "bundle.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        if manifest["model_id"] != f"lda-sha256:{_identity(files)}":
            raise ValueError("Invalid model bundle identity")
        actual_files = {p.relative_to(path).as_posix() for p in path.rglob("*")
                        if p.is_file() and p.name != "bundle.json"}
        if actual_files != set(files):
            raise ValueError("Model bundle files changed")
        for name, expected in files.items():
            candidate = (path / name).resolve()
            if not candidate.is_relative_to(path.resolve()) or not candidate.is_file():
                raise ValueError(f"Missing or invalid model artifact: {name}")
            if digest_file(candidate) != expected:
                raise ValueError(f"Model artifact changed: {name}")
        return path, manifest["model_id"]
    # Hash only legacy model/preprocessing artifacts, never mutable reports.
    files = {
        p.name: digest_file(p) for p in sorted(path.glob("*"))
        if p.is_file() and (
            p.name.startswith(("lda_model", "dictionary", "bigram_phraser", "trigram_phraser"))
            or p.name == "training_parameters.json"
        )
    }
    if not files:
        raise FileNotFoundError(f"No LDA artifacts in {path}")
    return path, f"lda-legacy-sha256:{_identity(files)}"


def load_preprocessing(directory: Path, *, allow_legacy: bool = False) -> tuple[dict, dict]:
    """Require frozen settings; explicit legacy opt-in reconstructs known fields."""
    params = json.loads((directory / "training_parameters.json").read_text(encoding="utf-8"))
    if "preprocessing" in params:
        return params["preprocessing"], params
    if not allow_legacy:
        raise ValueError(
            "Legacy model lacks frozen preprocessing. Refit it or explicitly use "
            "--allow-legacy-preprocessing (collocations then come from current code)."
        )
    extra = params.get("extra", {})
    settings = frozen_preprocessing(
        params["stopwords"]["words"], extra.get("language", "Français"), extra.get("chunk_words")
    )
    settings["fragment_stopwords"] = params["stopwords"].get("fragments", settings["fragment_stopwords"])
    return settings, params
