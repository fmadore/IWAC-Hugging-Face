"""Fail-closed Hugging Face Hub access and the single verified write gateway."""

from __future__ import annotations

import hashlib
import os
import socket
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


def read_hub_columns(
    repo_id: str,
    config_name: str,
    *,
    revision: str,
    columns: Sequence[str],
    token: Optional[str] = None,
    fs=None,
):
    """Read only ``columns`` of one config's parquet at a pinned revision.

    Parquet is columnar and ``HfFileSystem`` serves byte ranges, so only the
    requested column chunks travel: the index frequency pass needs five short
    string columns of ``articles``, not its full text and embeddings. Columns a
    shard does not carry are skipped (the caller checks what it needs).
    Nullable integers stay ``Int64``. Raises on any problem, so callers can fall
    back to a full load rather than proceed without data.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from .schema import arrow_to_pandas

    fs = fs or HfFileSystem(token=token)
    shards = sorted(fs.glob(f"datasets/{repo_id}@{revision}/{config_name}/*.parquet"))
    if not shards:
        raise HubBaselineUnavailableError(
            f"No parquet found for '{config_name}' in {repo_id} at {revision}"
        )
    tables = []
    for shard in shards:
        with fs.open(shard, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            available = [c for c in columns if c in parquet.schema_arrow.names]
            tables.append(parquet.read(columns=available))
    table = pa.concat_tables(tables, promote_options="default")
    return arrow_to_pandas(table)


def load_hub_columns(
    repo_id: str,
    config_name: str,
    *,
    revision: str,
    columns: Sequence[str],
    token: Optional[str] = None,
    console=None,
):
    """:func:`read_hub_columns`, falling back to a full ``load_dataset``.

    The fallback reads the same pinned revision and keeps the same column
    selection and nullable integers; it only costs more bandwidth. A failure of
    both raises :class:`HubBaselineUnavailableError`.
    """
    try:
        return read_hub_columns(
            repo_id, config_name, revision=revision, columns=columns, token=token
        )
    except Exception as exc:  # noqa: BLE001 - never proceed without the data
        if console is not None:
            console.print(
                f"[dim]ℹ Column-pruned read of '{config_name}' unavailable ({exc}); "
                "falling back to a full load.[/dim]"
            )
    from datasets import load_dataset

    from .schema import dataset_to_pandas

    try:
        ds = load_dataset(
            repo_id, name=config_name, split="train", token=token, revision=revision,
        )
    except Exception as exc:  # noqa: BLE001
        raise HubBaselineUnavailableError(
            f"Cannot load '{config_name}' from {repo_id} at {revision}: {exc}"
        ) from exc
    keep = [c for c in columns if c in ds.column_names]
    return dataset_to_pandas(ds.select_columns(keep))


def _lock_root() -> Path:
    configured = os.getenv("IWAC_LOCK_DIR")
    if configured:
        return Path(configured)
    from .paths import workspace_root

    return workspace_root() / ".iwac_locks"


def _process_alive(pid: int) -> bool:
    """Best-effort liveness check, biased towards reporting "alive".

    A PID can be reused, so a true answer never proves it is *our* writer. That
    is the safe direction: an unrelated live process keeps the lock held (the
    operator investigates), while only a confirmed-dead PID lets it be
    reclaimed automatically.
    """
    if pid <= 0:
        return True
    if os.name == "nt":  # pragma: no cover - exercised on the Windows CI leg
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError (alive, another user) and anything unexpected.
        return True
    return True


def _lock_owner(path: Path) -> dict[str, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    fields = {}
    for line in raw.splitlines():
        key, _, value = line.partition("=")
        if value:
            fields[key.strip()] = value.strip()
    return fields


def _reclaim_if_dead(path: Path, repo_id: str, console=None) -> bool:
    """Remove a lock whose owning process is gone. Returns True if reclaimed.

    Only a lock written by *this* host is ever reclaimed: the lock directory can
    sit on a shared filesystem, where a remote PID says nothing about the owner.
    """
    fields = _lock_owner(path)
    if fields.get("host") != socket.gethostname():
        return False
    try:
        pid = int(fields.get("pid", ""))
    except ValueError:
        return False
    if _process_alive(pid):
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return True
    if console is not None:
        console.print(
            f"[yellow]⚠[/yellow] Reclaimed a stale write lock for {repo_id} "
            f"(pid {pid} on this host is gone; started {fields.get('started', '?')})."
        )
    return True


@contextmanager
def hub_write_lock(repo_id: str, *, console=None, root_dir: Optional[Path] = None):
    """Process-local-machine lock preventing overlapping writes to one repo.

    A lock left behind by a crashed local process is reclaimed automatically;
    one held by a live process, or written by another host, still fails closed.
    """
    root = Path(root_dir) if root_dir is not None else _lock_root()
    root.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:16]
    path = root / f"{slug}.lock"
    payload = (
        f"repo={repo_id}\npid={os.getpid()}\nhost={socket.gethostname()}\n"
        f"started={datetime.now(timezone.utc).isoformat()}\n"
    )

    def acquire():
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    try:
        fd = acquire()
    except FileExistsError as exc:
        reclaimed = _reclaim_if_dead(path, repo_id, console)
        try:
            fd = acquire() if reclaimed else None
        except FileExistsError:
            fd = None  # Another process won the reclaim race.
        if fd is None:
            owner = _lock_owner(path)
            detail = ", ".join(f"{k}={v}" for k, v in owner.items()) or "details unavailable"
            raise HubWriteLockedError(
                f"A write to {repo_id!r} is already locked at {path} ({detail}). "
                "The owning process is still running (or the lock belongs to "
                "another host). Wait for it to finish rather than deleting the "
                "lock — two concurrent pushes lose each other's columns."
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


def _published_ids_columnar(
    repo_id: str, config_name: str, revision: str, token: Optional[str]
) -> list[str]:
    """Read only the ``o:id`` column of the published parquet.

    Parquet is columnar, so this transfers one narrow column instead of the
    whole subset — the difference between a few MB and re-downloading every
    768-dim embedding on each push. Raises on any problem so the caller can
    fall back to the exhaustive reload rather than skip verification.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    shards = sorted(fs.glob(f"datasets/{repo_id}@{revision}/{config_name}/*.parquet"))
    if not shards:
        raise HubWriteError(
            f"No parquet found for '{config_name}' in {repo_id} at {revision}"
        )
    ids: list[str] = []
    for shard in shards:
        with fs.open(shard, "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=["o:id"])
        ids.extend(str(value) for value in table.column("o:id").to_pylist())
    return ids


def _published_ids(
    repo_id: str,
    config_name: str,
    revision: str,
    token: Optional[str],
    expected_columns: Sequence[str],
    console,
) -> list[str]:
    """Return the published row ids, preferring the cheap columnar read.

    Column-level verification is *not* done here: ``sync_card_features`` has
    already compared the card's declared features against the parquet footer on
    the Hub and against ``expected_columns``, which is what the CastError guard
    needs. What remains is the row-level question — did every row land, exactly
    once — and that needs one column, not all of them.
    """
    try:
        return _published_ids_columnar(repo_id, config_name, revision, token)
    except Exception as exc:  # noqa: BLE001 - never let a fast path skip verification
        console.print(
            f"[dim]ℹ Columnar id verification unavailable ({exc}); "
            f"falling back to a full reload.[/dim]"
        )
    from datasets import load_dataset

    from .schema import DataContractError, validate_dataset

    reloaded = load_dataset(
        repo_id,
        name=config_name,
        split="train",
        token=token,
        revision=revision,
        download_mode="force_redownload",
    )
    try:
        validate_dataset(reloaded, config_name)
    except DataContractError as contract_exc:
        raise HubWriteError(
            f"Reloaded '{config_name}' violates its data contract: {contract_exc}"
        ) from contract_exc
    if list(reloaded.column_names) != list(expected_columns):
        raise HubWriteError(
            f"Reloaded '{config_name}' columns differ from the pushed frame"
        )
    return _dataset_ids(reloaded)


def _assert_destination(api, repo_id: str, mode: str):
    from .repos import get_private_repo_id, get_public_repo_id

    if mode not in {"full", "public_projection"}:
        raise HubWriteError(f"Unknown write mode: {mode!r}")
    try:
        info = api.dataset_info(repo_id=repo_id)
    except Exception as exc:
        raise HubBaselineUnavailableError(f"Cannot verify destination {repo_id!r}: {exc}") from exc
    if not getattr(info, "sha", None):
        raise HubBaselineUnavailableError(f"Destination {repo_id!r} has no revision")
    if mode == "full":
        public_names = {get_public_repo_id(), "fmadore/islam-west-africa-collection"}
        if repo_id in public_names or getattr(info, "private", None) is not True:
            raise HubWriteError(
                f"Full-data writes require a verified private destination; refused {repo_id!r}. "
                "Use the public projection workflow for public repositories."
            )
    elif repo_id in {get_private_repo_id(), "fmadore/islam-west-africa-collection-full"}:
        raise HubWriteError("Refusing to overwrite the full mirror with a public projection")
    return info


def _shard_bytes(value: str) -> int:
    import re

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)?\s*", value, re.I)
    if not match:
        raise ValueError(f"Invalid max_shard_size: {value!r}")
    unit = (match[2] or "B").upper()
    power = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4,
             "KIB": 1, "MIB": 2, "GIB": 3, "TIB": 4}[unit]
    size = int(float(match[1]) * (1024 if "I" in unit else 1000) ** power)
    if size < 1:
        raise ValueError("max_shard_size must be positive")
    return size


def _card_for_write(repo_id: str, token, revision: str, files: Sequence[str]):
    from huggingface_hub import DatasetCard
    from .card_sync import _load_card

    if "README.md" in files:
        return _load_card(repo_id, token, revision)
    return DatasetCard("---\n{}\n---\n\n# Dataset\n")


def _replace_card_entry(card, key: str, config_name: str, replacement: dict) -> None:
    import copy

    entries = copy.deepcopy(card.data.get(key) or [])
    if isinstance(entries, dict):
        entries = [entries]
    for index, entry in enumerate(entries):
        if entry.get("config_name", "default") == config_name:
            entries[index] = {**entry, **replacement}
            break
    else:
        entries.append(replacement)
    card.data[key] = entries


def _stage_datasets(prepared, root: Path, card, max_bytes: int):
    """Serialize all configs before submitting one atomic Hub commit."""
    import math
    from huggingface_hub import CommitOperationAdd
    from .card_sync import _features_yaml

    operations = []
    paths = set()
    for config_name, ds in prepared.items():
        table = ds.with_format("arrow")[:]
        count = max(1, min(len(ds), math.ceil(table.nbytes / max_bytes)))
        download_size = 0
        for index in range(count):
            part = ds.shard(num_shards=count, index=index, contiguous=True) if len(ds) else ds
            relative = f"{config_name}/train-{index:05d}-of-{count:05d}.parquet"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(path)
            download_size += path.stat().st_size
            paths.add(relative)
            operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(path)))
        _replace_card_entry(card, "dataset_info", config_name, {
            "config_name": config_name,
            "features": _features_yaml(ds.features.arrow_schema),
            "splits": [{"name": "train", "num_bytes": table.nbytes, "num_examples": len(ds)}],
            "download_size": download_size,
            "dataset_size": table.nbytes,
        })
        _replace_card_entry(card, "configs", config_name, {
            "config_name": config_name,
            "data_files": [{"split": "train", "path": f"{config_name}/train-*.parquet"}],
        })
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.content.encode("utf-8")))
    return operations, paths


def _verify_committed_dataset(ds, *, repo_id, config_name, token, revision, console):
    from .card_sync import sync_card_features

    sync_card_features(
        repo_id, config_name, token=token, console=console, repair=False,
        expected_columns=ds.column_names, expected_schema=ds.features.arrow_schema,
        revision=revision,
    )
    actual = _published_ids(repo_id, config_name, revision, token, ds.column_names, console)
    expected = _dataset_ids(ds)
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise HubWriteError(f"Reloaded '{config_name}' id set differs from the committed dataset")


def push_datasets_verified(
    datasets_by_config,
    *,
    repo_id: str,
    token: Optional[str],
    commit_message: str,
    max_shard_size: str = "1GB",
    expected_revision: Optional[str] = None,
    console=None,
    verify_reload: bool = True,
    acquire_lock: bool = True,
    mode: str = "full",
) -> HubWriteResult:
    """Commit all selected configs and their exact card metadata atomically.

    Every Parquet shard is prepared and validated before one ``create_commit``.
    ``parent_commit`` is enforced by the Hub, closing the check-then-push race
    across machines. Full writes require a verified private destination; public
    projection writes independently revalidate item/property visibility.
    """
    import tempfile
    from datasets import Dataset
    from huggingface_hub import CommitOperationDelete
    from rich.console import Console
    from .schema import DataContractError, conform_dataset, validate_dataset
    from .write_policy import PublicationPolicyError, validate_public_projection

    if not datasets_by_config:
        raise HubWriteError("Refusing an empty publication plan")
    console = console or Console()
    token = resolve_hf_token(token)
    prepared = {}
    try:
        for config_name, ds in datasets_by_config.items():
            ds = conform_dataset(ds, config_name)
            validate_dataset(ds, config_name)
            if not isinstance(ds, Dataset):
                raise DataContractError("The write gateway requires an Arrow-backed Dataset")
            if mode == "public_projection":
                validate_public_projection(ds, config_name)
            prepared[config_name] = ds
    except (DataContractError, PublicationPolicyError) as exc:
        raise HubWriteError(f"Refusing invalid publication: {exc}") from exc
    max_bytes = _shard_bytes(max_shard_size)
    context = hub_write_lock(repo_id, console=console) if acquire_lock else nullcontext()
    with context:
        api = HfApi(token=token)
        info = _assert_destination(api, repo_id, mode)
        before = str(info.sha)
        if expected_revision is not None and before != expected_revision:
            raise ConcurrentHubWriteError(
                f"{repo_id} changed from {expected_revision} to {before}; reload and recompute."
            )
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=before)
        card = _card_for_write(repo_id, token, before, files)
        with tempfile.TemporaryDirectory(prefix="iwac-publish-") as directory:
            operations, staged_paths = _stage_datasets(prepared, Path(directory), card, max_bytes)
            for path in files:
                if path.split("/", 1)[0] in prepared and path.endswith(".parquet") and path not in staged_paths:
                    operations.append(CommitOperationDelete(path_in_repo=path))
            # Recheck destination visibility after potentially lengthy serialization.
            latest = _assert_destination(api, repo_id, mode)
            if str(latest.sha) != before:
                raise ConcurrentHubWriteError(f"{repo_id} changed while its publication was staged")
            try:
                committed = api.create_commit(
                    repo_id=repo_id, repo_type="dataset", operations=operations,
                    commit_message=commit_message, parent_commit=before,
                )
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (409, 412):
                    raise ConcurrentHubWriteError(
                        f"The Hub rejected the stale parent revision {before}; reload and recompute."
                    ) from exc
                raise HubWriteError(f"Atomic publication failed: {exc}") from exc
        revision = getattr(committed, "oid", None)
        if not revision:
            raise HubWriteError("The commit returned no revision; inspect the repository before retrying")
        if verify_reload:
            try:
                for config_name, ds in prepared.items():
                    _verify_committed_dataset(
                        ds, repo_id=repo_id, config_name=config_name, token=token,
                        revision=revision, console=console,
                    )
            except Exception as exc:
                raise HubWriteError(
                    f"Commit {revision} landed atomically, but verification failed: {exc}. "
                    "Inspect this revision before retrying."
                ) from exc
        after = get_repo_revision(repo_id, token=token)
        if after != revision:
            raise ConcurrentHubWriteError(
                f"Committed and verified {revision}, but {repo_id} is now at {after}."
            )
    return HubWriteResult(before, str(revision), True)


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
    mode: str = "full",
) -> HubWriteResult:
    """Compatibility entry point for one config, using the atomic gateway."""
    if expected_columns is not None and list(expected_columns) != list(ds.column_names):
        raise HubWriteError("Prepared columns differ from expected_columns")
    if expected_ids is not None:
        ids = [str(v) for v in expected_ids]
        if len(ids) != len(set(ids)) or set(ids) != set(_dataset_ids(ds)):
            raise HubWriteError("Prepared ids differ from expected_ids")
    expected_revision = expected_revision or getattr(ds, "_iwac_source_revision", None)
    return push_datasets_verified(
        {config_name: ds}, repo_id=repo_id, token=token, commit_message=commit_message,
        max_shard_size=max_shard_size, expected_revision=expected_revision,
        console=console, verify_reload=verify_reload, acquire_lock=acquire_lock, mode=mode,
    )


__all__ = [
    "HubBaselineUnavailableError", "ConcurrentHubWriteError", "HubWriteError",
    "HubWriteLockedError", "HubWriteResult", "resolve_hf_token", "get_repo_revision",
    "get_repo_configs", "read_hub_columns", "load_hub_columns", "hub_write_lock",
    "push_dataset_verified", "push_datasets_verified",
]
