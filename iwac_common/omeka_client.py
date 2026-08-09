"""Async Omeka S API client shared by every IWAC upload script.

Extracted from the (formerly duplicated) infrastructure in:
    articles/upload_newspaper_hf.py
    audiovisual/upload_audiovisual_hf.py
    document/upload_documents_hf.py
    islamic-publications/upload_Islamic_publications_hf.py
    index/upload_index_hf.py
    reference/upload_reference_hf.py

Each subset script builds its own ``Config`` (with the appropriate
``CACHE_DIR``) and either uses ``OmekaApiClient`` directly or subclasses it.
"""

from __future__ import annotations

import asyncio
import functools
import gzip
import hashlib
import io
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import aiofiles
import aiohttp
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Omeka API + cache settings loaded from environment variables.

    Each subset script instantiates this with its own ``CACHE_DIR`` so the
    on-disk caches stay separated. ``API_URL``, ``API_KEY_IDENTITY`` and
    ``API_KEY_CREDENTIAL`` are read from the environment by default.
    """

    API_URL: str = field(
        default_factory=lambda: os.getenv("OMEKA_BASE_URL", "https://islam.zmo.de/api")
    )
    API_KEY_IDENTITY: str = field(
        default_factory=lambda: os.getenv("OMEKA_KEY_IDENTITY", "")
    )
    API_KEY_CREDENTIAL: str = field(
        default_factory=lambda: os.getenv("OMEKA_KEY_CREDENTIAL", "")
    )
    CACHE_DIR: str = ".cache_omk"
    CACHE_HOURS: int = 24


# ---------------------------------------------------------------------------
# Disk cache (gzipped JSON)
# ---------------------------------------------------------------------------


class Cache:
    """Async on-disk cache, one gzipped JSON file per request key."""

    def __init__(self, directory: str, hours: int = 24):
        self.dir = directory
        self.duration = timedelta(hours=hours)
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        name = hashlib.md5(key.encode()).hexdigest() + ".json.gz"
        return os.path.join(self.dir, name)

    async def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        if datetime.now() - mtime > self.duration:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            return None
        try:
            async with aiofiles.open(path, "rb") as f:
                data = await f.read()
            with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as gz:
                return json.load(gz)
        except (OSError, EOFError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring corrupt cache entry %s: %s", path, exc)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            return None

    async def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        buf = io.BytesIO()
        with gzip.open(buf, "wt", encoding="utf-8") as gz:
            json.dump(value, gz)
        try:
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(buf.getvalue())
            os.replace(tmp_path, path)
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# HTTP connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Process-wide aiohttp session, lazily created and reused."""

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None

    async def get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=20),
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


conn_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Async retry decorator
# ---------------------------------------------------------------------------


def async_retry(
    max_tries: int = 5,
    exceptions: Union[Type[Exception], tuple] = (
        aiohttp.ClientError,
        asyncio.TimeoutError,
    ),
):
    """Retry an async callable with exponential backoff (1, 2, 4, … seconds).

    The last failed attempt re-raises the original exception (no trailing
    backoff sleep).
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_tries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_tries - 1:
                        raise
                    logger.warning(
                        f"{func.__name__}: tentative {attempt + 1}/{max_tries} échouée ({exc})"
                    )
                    await asyncio.sleep(2**attempt)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Omeka API client
# ---------------------------------------------------------------------------


class TruncatedFetchError(RuntimeError):
    """The paginated fetch returned fewer items than the API's
    ``Omeka-S-Total-Results`` header reports — a partial response or a stale
    cache. Aborting protects the Hub dataset from a silent mass-delete."""


class MediaFetchGuardError(RuntimeError):
    """At least one per-item media lookup failed without explicit override.

    A lookup error is different from a genuine absence. By default even one
    fails the run; with the explicit override, item-aware field preservation
    carries the affected values forward from the Hub baseline.

    Observed 2026-07-27: a VPN made islam.zmo.de unreachable mid-run and
    1,501 publications were on course to lose every PDF URL."""


class MediaFetchStats:
    """Counts per-item media lookups and the fields each failure affects.

    Both primary-media and IIIF helpers report here. Absences are never counted
    — an item with no primary media makes no attempt — so the rate measures
    transport/parse failures rather than sparse media coverage.
    """

    def __init__(self) -> None:
        self.attempted = 0
        self.failed = 0
        self.failed_fields_by_item: dict[str, set[str]] = {}

    def reset(self) -> None:
        self.attempted = 0
        self.failed = 0
        self.failed_fields_by_item = {}

    def record_attempt(self) -> None:
        self.attempted += 1

    def record_failure(
        self, item_id: Union[str, int, None] = None, fields: Iterable[str] = ()
    ) -> None:
        self.failed += 1
        if item_id is not None:
            self.failed_fields_by_item.setdefault(str(item_id), set()).update(fields)

    @property
    def failure_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0

    def check(self, *, allow_failures: bool = False) -> Optional[str]:
        """Raise :class:`MediaFetchGuardError` on any media failure.

        Returns a human-readable summary when failures are explicitly allowed,
        else ``None`` when every attempted lookup succeeded.
        """
        if not self.failed:
            return None
        summary = (
            f"{self.failed:,}/{self.attempted:,} media lookups failed "
            f"({self.failure_rate:.0%})"
        )
        if not allow_failures:
            raise MediaFetchGuardError(
                f"{summary}. Even one transient failure can blank a good "
                "PDF/thumbnail URL or drop a mapper row, so the run fails closed. "
                "Check connectivity to the Omeka host (a VPN or DNS change is "
                "the usual cause) and re-run. Use "
                "--allow-media-failures only if the media really are gone."
            )
        return summary


#: Process-wide media-lookup counters; ``upload_runner`` resets per run.
media_stats = MediaFetchStats()


class OmekaApiClient:
    """Minimal async client for the Omeka S REST API.

    Subset scripts that need to fetch a single resource class call
    :meth:`fetch_items`. Scripts that need multiple classes (``reference``,
    ``index``) loop over them at the call site.
    """

    def __init__(
        self,
        cfg: Config,
        use_cache: bool = True,
        console: Optional[Console] = None,
    ):
        self.cfg = cfg
        self.cache = Cache(cfg.CACHE_DIR, cfg.CACHE_HOURS) if use_cache else None
        self.console = console or Console()

    @async_retry()
    async def _get(self, endpoint: str, params: Dict[str, Any]) -> Any:
        request_params = dict(params)
        request_params.update(
            {
                "key_identity": self.cfg.API_KEY_IDENTITY,
                "key_credential": self.cfg.API_KEY_CREDENTIAL,
            }
        )
        url = f"{self.cfg.API_URL}/{endpoint}"
        sess = await conn_manager.get()
        async with sess.get(url, params=request_params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        # Host + identity are part of the cache namespace. Without them, a run
        # pointed at staging (or using a less privileged identity) can silently
        # reuse production/private response bodies from the same cache folder.
        key = (
            f"{self.cfg.API_URL}|{self.cfg.API_KEY_IDENTITY}|{endpoint}:"
            f"{json.dumps(params, sort_keys=True)}"
        )
        if self.cache:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached
        data = await self._get(endpoint, params)
        if self.cache:
            await self.cache.set(key, data)
        return data

    async def fetch_items_page(
        self, rcid: int, page: int, per: int = 100
    ) -> List[Dict[str, Any]]:
        return await self.request(
            "items",
            {
                "resource_class_id": rcid,
                "page": page,
                "per_page": per,
                "sort_by": "o:id",
                "sort_order": "asc",
            },
        )

    @async_retry()
    async def fetch_total_results(self, rcid: int) -> Optional[int]:
        """Read the ``Omeka-S-Total-Results`` header for a resource class.

        Deliberately bypasses the cache (the JSON cache stores bodies, not
        headers) — one cheap per_page=1 request per class.
        """
        params = {
            "resource_class_id": rcid,
            "page": 1,
            "per_page": 1,
            "key_identity": self.cfg.API_KEY_IDENTITY,
            "key_credential": self.cfg.API_KEY_CREDENTIAL,
        }
        sess = await conn_manager.get()
        async with sess.get(f"{self.cfg.API_URL}/items", params=params) as resp:
            resp.raise_for_status()
            total = resp.headers.get("Omeka-S-Total-Results")
            return int(total) if total is not None else None

    async def fetch_items(
        self, rcid: int, *, verify_total: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch every item with the given resource_class_id, paginated.

        With ``verify_total`` (default), the fetched count is reconciled
        against the API's ``Omeka-S-Total-Results`` header: pagination stops
        on the first short page, so a transient short-but-200 response would
        otherwise truncate the fetch silently — and a truncated fetch flowing
        into the Hub merge deletes rows. A shortfall raises
        :class:`TruncatedFetchError`; note a stale local cache (item added on
        Omeka within the cache TTL) triggers the same signal — re-run with
        the cache disabled.
        """
        per = 100
        expected: Optional[int] = None
        if verify_total:
            try:
                expected = await self.fetch_total_results(rcid)
            except Exception as exc:  # noqa: BLE001
                raise TruncatedFetchError(
                    f"Could not read Omeka-S-Total-Results for class {rcid}: "
                    f"{exc}. Count reconciliation is required for a safe upload."
                ) from exc
            if expected is None:
                raise TruncatedFetchError(
                    f"Omeka returned no Omeka-S-Total-Results header for class {rcid}; "
                    "refusing an unreconciled upload."
                )

        first = await self.fetch_items_page(rcid, 1, per)
        items = list(first)
        if expected is not None:
            page_count = max(1, (expected + per - 1) // per)
        else:
            page_count = 1 if len(first) < per else None

        if page_count and page_count > 1:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Fetching pages..."),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Fetching item pages", total=page_count - 1
                )
                semaphore = asyncio.Semaphore(4)

                async def fetch_page(page: int):
                    async with semaphore:
                        batch = await self.fetch_items_page(rcid, page, per)
                    progress.update(
                        task, advance=1, description=f"[cyan]Page {page} fetched"
                    )
                    return page, batch

                batches = await asyncio.gather(
                    *(fetch_page(page) for page in range(2, page_count + 1))
                )
                for page, batch in sorted(batches):
                    items.extend(batch)
        elif page_count is None:
            page = 2
            while True:
                batch = await self.fetch_items_page(rcid, page, per)
                if not batch:
                    break
                items.extend(batch)
                if len(batch) < per:
                    break
                page += 1

        if verify_total:
            if len(items) != expected:
                raise TruncatedFetchError(
                    f"Fetched {len(items)} items for class {rcid} but the API "
                    f"reports {expected}. Either a page response was truncated "
                    f"mid-run, or the local cache is stale (items added on Omeka "
                    f"since it was written). Re-run — with --no-cache if it "
                    f"persists — instead of pushing a shrunken dataset."
                )

        self.console.print(
            f"[green]✓[/green] {len(items)} items retrieved for class {rcid}"
        )
        return items

    async def fetch_media_data(self, media_id: str) -> Any:
        return await self.request(f"media/{media_id}", {})


async def fetch_primary_media_url(
    item: Dict[str, Any],
    api: OmekaApiClient,
    *,
    affected_fields: Iterable[str],
) -> str:
    """Return a primary media URL while recording item-aware failures."""
    primary = item.get("o:primary_media")
    if not primary:
        return ""
    media_stats.record_attempt()
    try:
        media_id = str(primary["@id"]).rstrip("/").split("/")[-1]
        media = await api.fetch_media_data(media_id)
        return str(media.get("o:original_url", ""))
    except Exception as exc:  # noqa: BLE001
        media_stats.record_failure(item.get("o:id"), affected_fields)
        logger.warning(
            "Primary media lookup failed for item %s: %s", item.get("o:id"), exc
        )
        return ""


# ---------------------------------------------------------------------------
# IIIF helpers
# ---------------------------------------------------------------------------

IIIF_BASE_URL = "https://islam.zmo.de/iiif/3"


async def fetch_iiif_thumbnail_url(
    omeka_id: Union[str, int], session: aiohttp.ClientSession
) -> str:
    """Fetch the thumbnail URL from an item's IIIF manifest.

    Returns ``""`` for a genuine absence (404/410). Transport, parse, auth,
    rate-limit, and server failures are recorded for the upload runner's
    fail-closed guard while still allowing all in-flight mappers to finish.
    """
    manifest_url = f"{IIIF_BASE_URL}/{omeka_id}/manifest"
    thumbnail_url = ""
    media_stats.record_attempt()
    try:
        # Shorter timeout: this request runs once per item.
        async with session.get(manifest_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                try:
                    manifest = await resp.json()
                    thumbnails = manifest.get("thumbnail")
                    if isinstance(thumbnails, list) and thumbnails:
                        thumbnail_info = thumbnails[0]
                        if isinstance(thumbnail_info, dict):
                            thumbnail_url = thumbnail_info.get("id", "")
                except json.JSONDecodeError as e_json:
                    media_stats.record_failure(
                        omeka_id, ("thumbnail", "iiif_manifest")
                    )
                    logger.warning(
                        f"JSON decoding error for IIIF manifest {omeka_id}: {e_json}. URL: {manifest_url}"
                    )
            elif resp.status in (404, 410):
                logger.info(
                    "No IIIF manifest for item %s (HTTP %s)", omeka_id, resp.status
                )
            else:
                media_stats.record_failure(
                    omeka_id, ("thumbnail", "iiif_manifest")
                )
                logger.warning(
                    f"IIIF request failed with HTTP {resp.status} for {omeka_id}. "
                    f"URL: {manifest_url}"
                )
    # Only transport-level errors count as failures. A 200 carrying no
    # thumbnail, or a 404 for an item with no manifest, is a genuine absence
    # and must not inflate the failure rate.
    except asyncio.TimeoutError:
        media_stats.record_failure(omeka_id, ("thumbnail", "iiif_manifest"))
        logger.warning(f"Timeout fetching IIIF manifest for {omeka_id}. URL: {manifest_url}")
    except aiohttp.ClientError as e_client:
        media_stats.record_failure(omeka_id, ("thumbnail", "iiif_manifest"))
        logger.warning(
            f"Client error fetching IIIF manifest for {omeka_id}: {e_client}. URL: {manifest_url}"
        )
    except Exception as e_general:
        media_stats.record_failure(omeka_id, ("thumbnail", "iiif_manifest"))
        logger.error(
            f"Unexpected error fetching IIIF manifest for {omeka_id}: {e_general}. URL: {manifest_url}"
        )
    return thumbnail_url


__all__ = [
    "Config",
    "Cache",
    "ConnectionManager",
    "conn_manager",
    "async_retry",
    "OmekaApiClient",
    "TruncatedFetchError",
    "IIIF_BASE_URL",
    "fetch_iiif_thumbnail_url",
    "fetch_primary_media_url",
    "MediaFetchGuardError",
    "media_stats",
]
