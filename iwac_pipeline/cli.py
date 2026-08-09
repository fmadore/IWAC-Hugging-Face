"""Thin installed CLI wrappers around the repository's maintained scripts."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

from iwac_common.schema import ALL_CONFIGS
from iwac_common.upload_runner import run_upload

REPO_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_SCRIPTS = {
    "articles": "articles/upload_newspaper_hf.py",
    "publications": "islamic-publications/upload_Islamic_publications_hf.py",
    "documents": "document/upload_documents_hf.py",
    "references": "reference/upload_reference_hf.py",
    "index": "index/upload_index_hf.py",
    "audiovisual": "audiovisual/upload_audiovisual_hf.py",
    "images": "images/upload_image_hf.py",
}


def _load_script(relative_path: str, module_name: str) -> ModuleType:
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"Pipeline script is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def upload_main(argv: Sequence[str] | None = None) -> int:
    """Run one of the seven Omeka-to-Hub upload pipelines."""
    parser = argparse.ArgumentParser(
        prog="iwac-upload",
        description="Refresh one IWAC subset from Omeka into the private Hub mirror.",
    )
    parser.add_argument("subset", choices=ALL_CONFIGS)
    args, remaining = parser.parse_known_args(argv)
    module = _load_script(
        UPLOAD_SCRIPTS[args.subset], f"_iwac_upload_{args.subset}"
    )
    spec = getattr(module, "SPEC", None)
    if spec is None:
        raise RuntimeError(f"{UPLOAD_SCRIPTS[args.subset]} does not declare SPEC")
    return run_upload(spec, remaining)


def mirror_main(argv: Sequence[str] | None = None) -> int:
    """Create a verified local CSV mirror."""
    parser = argparse.ArgumentParser(prog="iwac-mirror")
    parser.add_argument("--dataset", choices=["private", "public"], default=None)
    args = parser.parse_args(argv)
    module = _load_script("data/fetch_datasets.py", "_iwac_mirror")
    repo_id, label = module.choose_dataset(args.dataset)
    return int(module.main(dataset_id=repo_id, label=label))


def publish_public_main() -> int:
    """Project the private mirror to the rights-filtered public dataset."""
    module = _load_script("post-processing/publish_public.py", "_iwac_publish_public")
    result = module.main()
    return int(result or 0)


__all__ = ["UPLOAD_SCRIPTS", "upload_main", "mirror_main", "publish_public_main"]
