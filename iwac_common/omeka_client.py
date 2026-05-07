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
import gzip
import hashlib
import io
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, Union

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
            return None
        async with aiofiles.open(path, "rb") as f:
            data = await f.read()
        with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as gz:
            return json.load(gz)

    async def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        buf = io.BytesIO()
        with gzip.open(buf, "wt", encoding="utf-8") as gz:
            json.dump(value, gz)
        async with aiofiles.open(path, "wb") as f:
            await f.write(buf.getvalue())


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
                connector=aiohttp.TCPConnector(limit=20, ssl=False),
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
    """Retry an async callable with exponential backoff (1, 2, 4, … seconds)."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_tries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    logger.warning(
                        f"{func.__name__}: tentative {attempt + 1}/{max_tries} échouée ({exc})"
                    )
                    await asyncio.sleep(2**attempt)
            raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Omeka API client
# ---------------------------------------------------------------------------


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
        params.update(
            {
                "key_identity": self.cfg.API_KEY_IDENTITY,
                "key_credential": self.cfg.API_KEY_CREDENTIAL,
            }
        )
        url = f"{self.cfg.API_URL}/{endpoint}"
        sess = await conn_manager.get()
        async with sess.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
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
            {"resource_class_id": rcid, "page": page, "per_page": per},
        )

    async def fetch_items(self, rcid: int) -> List[Dict[str, Any]]:
        """Fetch every item with the given resource_class_id, paginated."""
        first = await self.fetch_items_page(rcid, 1)
        items = list(first)
        per = 100
        if len(first) == per:
            page = 2
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Fetching pages..."),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("[cyan]Fetching item pages", total=None)
                while True:
                    batch = await self.fetch_items_page(rcid, page)
                    if not batch:
                        break
                    items.extend(batch)
                    progress.update(
                        task, advance=1, description=f"[cyan]Page {page} fetched"
                    )
                    if len(batch) < per:
                        break
                    page += 1
        self.console.print(
            f"[green]✓[/green] {len(items)} items retrieved for class {rcid}"
        )
        return items

    async def fetch_media_data(self, media_id: str) -> Any:
        return await self.request(f"media/{media_id}", {})


__all__ = [
    "Config",
    "Cache",
    "ConnectionManager",
    "conn_manager",
    "async_retry",
    "OmekaApiClient",
]
