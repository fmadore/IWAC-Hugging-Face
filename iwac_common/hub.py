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


def _lock_root() -> Path:
    configured = os.getenv("IWAC_LOCK_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / ".iwac_locks"


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
def hub_write_lock(repo_id: str, *, console=None):
    """Process-local-machine lock preventing overlapping writes to one repo.

    A lock left behind by a crashed local process is reclaimed automatically;
    one held by a live process, or written by another host, still fails closed.
    """
    root = _lock_root()
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

    Verification after the push is split by cost: ``sync_card_features`` reads
    the parquet footer to check the schema (cheap, and the CastError guard),
    then ``verify_reload`` checks the published row ids through one column.
    """
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

    lock_context = (
        hub_write_lock(repo_id, console=console) if acquire_lock else nullcontext()
    )
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
                reloaded_ids = _published_ids(
                    repo_id, config_name, published_revision, token, columns, console
                )
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
