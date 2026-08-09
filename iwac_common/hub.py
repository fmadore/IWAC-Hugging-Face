"""Fail-closed Hugging Face Hub access and the single verified write gateway."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from huggingface_hub import HfApi, get_token


class HubBaselineUnavailableError(RuntimeError):
    """The current Hub state could not be read safely."""


class ConcurrentHubWriteError(RuntimeError):
    """The Hub repository changed after the caller loaded its input."""


class HubWriteError(RuntimeError):
    """A push landed incompletely or failed post-write verification."""


class HubWriteLockedError(RuntimeError):
    """Another local process currently owns this repository's write lock."""


@dataclass(frozen=True)
class HubWriteResult:
    before_revision: str
    after_revision: str
    card_already_matched: bool


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve an explicit, environment, or locally stored HF token."""
    if explicit:
        return explicit
    return os.getenv("HF_TOKEN") or get_token()


def get_repo_revision(repo_id: str, *, token: Optional[str] = None) -> str:
    """Return the current dataset-repository SHA or fail closed."""
    token = resolve_hf_token(token)
    try:
        info = HfApi(token=token).dataset_info(repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001 - converted to a typed boundary error
        raise HubBaselineUnavailableError(
            f"Cannot read the current revision of dataset repository {repo_id!r}: "
            f"{exc}. Refusing to write without a verified baseline."
        ) from exc
    revision = getattr(info, "sha", None)
    if not revision:
        raise HubBaselineUnavailableError(
            f"Dataset repository {repo_id!r} returned no revision SHA; refusing to write."
        )
    return str(revision)


def get_repo_configs(repo_id: str, *, token: Optional[str] = None) -> set[str]:
    """Return declared config names, raising when the repository is unreadable."""
    token = resolve_hf_token(token)
    api = HfApi(token=token)
    try:
        info = api.dataset_info(repo_id=repo_id)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        raise HubBaselineUnavailableError(
            f"Cannot inspect configs in dataset repository {repo_id!r}: {exc}"
        ) from exc
    names = set(getattr(info, "config_names", None) or ())
    # A damaged/metadata-light card may omit config_names even though parquet
    # exists. Include top-level parquet directories so --initialize cannot
    # mistake an unreadable existing config for a new one.
    names.update(
        path.split("/", 1)[0]
        for path in files
        if "/" in path and path.endswith(".parquet")
    )
    return names


def _lock_root() -> Path:
    configured = os.getenv("IWAC_LOCK_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / ".iwac_locks"


@contextmanager
def hub_write_lock(repo_id: str):
    """Process-local-machine lock preventing overlapping writes to one repo."""
    root = _lock_root()
    root.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:16]
    path = root / f"{slug}.lock"
    payload = (
        f"repo={repo_id}\npid={os.getpid()}\n"
        f"started={datetime.now(timezone.utc).isoformat()}\n"
    )
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            owner = path.read_text(encoding="utf-8").strip()
        except OSError:
            owner = "owner details unavailable"
        raise HubWriteLockedError(
            f"A write to {repo_id!r} is already locked at {path}: {owner}. "
            "If the owning process crashed, remove this one lock file manually."
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _dataset_ids(ds) -> list[str]:
    if "o:id" not in ds.column_names:
        raise HubWriteError("Pushed dataset has no 'o:id' column")
    return [str(value) for value in ds["o:id"]]


def push_dataset_verified(
    ds,
    *,
    repo_id: str,
    config_name: str,
    token: Optional[str],
    commit_message: str,
    max_shard_size: str = "1GB",
    expected_revision: Optional[str] = None,
    expected_columns: Optional[Sequence[str]] = None,
    expected_ids: Optional[Iterable[object]] = None,
    console=None,
    verify_reload: bool = True,
    acquire_lock: bool = True,
) -> HubWriteResult:
    """Push one config, repair its card schema, and reload-verify the result.

    ``expected_revision`` is the SHA observed when computation started.  A
    mismatch aborts before the push, preventing the common lost-update case.
    Hugging Face's high-level ``Dataset.push_to_hub`` has no parent-commit
    precondition, so the local lock plus this immediate recheck is the strongest
    safe guard available without reimplementing its parquet commit builder.
    """
    from datasets import load_dataset
    from rich.console import Console

    from .card_sync import CardSchemaError, sync_card_features
    from .schema import DataContractError, validate_dataset

    console = console or Console()
    token = resolve_hf_token(token)
    try:
        validate_dataset(ds, config_name)
    except DataContractError as exc:
        raise HubWriteError(f"Refusing invalid {config_name!r} dataset: {exc}") from exc
    columns = list(expected_columns or ds.column_names)
    ids = [str(v) for v in (expected_ids if expected_ids is not None else _dataset_ids(ds))]
    if len(ids) != len(set(ids)):
        raise HubWriteError("Refusing to push duplicated 'o:id' values")

    lock_context = hub_write_lock(repo_id) if acquire_lock else nullcontext()
    with lock_context:
        before = get_repo_revision(repo_id, token=token)
        if expected_revision is not None and before != expected_revision:
            raise ConcurrentHubWriteError(
                f"{repo_id} changed from {expected_revision} to {before} while "
                f"'{config_name}' was being computed. Reload and recompute instead "
                "of overwriting the newer revision."
            )
        try:
            ds.push_to_hub(
                repo_id=repo_id,
                config_name=config_name,
                token=token,
                max_shard_size=max_shard_size,
                commit_message=commit_message,
            )
            card_matched = sync_card_features(
                repo_id,
                config_name,
                token=token,
                console=console,
                expected_columns=columns,
            )
            published_revision = get_repo_revision(repo_id, token=token)
            if verify_reload:
                reloaded = load_dataset(
                    repo_id,
                    name=config_name,
                    split="train",
                    token=token,
                    revision=published_revision,
                    download_mode="force_redownload",
                )
                try:
                    validate_dataset(reloaded, config_name)
                except DataContractError as exc:
                    raise HubWriteError(
                        f"Reloaded '{config_name}' violates its data contract: {exc}"
                    ) from exc
                if list(reloaded.column_names) != columns:
                    raise HubWriteError(
                        f"Reloaded '{config_name}' columns differ from the pushed frame"
                    )
                reloaded_ids = _dataset_ids(reloaded)
                if len(reloaded_ids) != len(set(reloaded_ids)):
                    raise HubWriteError(
                        f"Reloaded '{config_name}' contains duplicate 'o:id' values"
                    )
                if set(reloaded_ids) != set(ids):
                    missing = sorted(set(ids) - set(reloaded_ids))[:5]
                    extra = sorted(set(reloaded_ids) - set(ids))[:5]
                    raise HubWriteError(
                        f"Reloaded '{config_name}' id set differs (missing={missing}, "
                        f"unexpected={extra})"
                    )
        except (CardSchemaError, HubWriteError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise HubWriteError(
                f"Failed to publish or verify {repo_id!r}/{config_name!r}: {exc}"
            ) from exc
        after = get_repo_revision(repo_id, token=token)
        if after != published_revision:
            raise ConcurrentHubWriteError(
                f"{repo_id} changed from {published_revision} to {after} while "
                f"the '{config_name}' write was being verified. The pushed revision "
                "was valid, but it is no longer the repository head; inspect the "
                "other writer before continuing."
            )
    return HubWriteResult(before, after, card_matched)


__all__ = [
    "HubBaselineUnavailableError",
    "ConcurrentHubWriteError",
    "HubWriteError",
    "HubWriteLockedError",
    "HubWriteResult",
    "resolve_hf_token",
    "get_repo_revision",
    "get_repo_configs",
    "hub_write_lock",
    "push_dataset_verified",
]
