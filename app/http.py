"""Shared HTTP client for external APIs: polite User-Agent, per-host rate limits, disk cache."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

# Minimum seconds between requests per host. arXiv asks for one request every 3 seconds.
HOST_INTERVALS = {
    "arxiv.org": 3.0,
    "export.arxiv.org": 3.0,
    "api.openalex.org": 0.12,
    "api.crossref.org": 0.1,
    "api.semanticscholar.org": 1.0,
}
MAX_ATTEMPTS = 4


class _HostLimiter:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    async def wait(self, host: str) -> None:
        interval = HOST_INTERVALS.get(host, 0.0)
        if not interval:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            delay = self._last.get(host, 0.0) + interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last[host] = time.monotonic()


_limiter = _HostLimiter()
_client: Optional[httpx.AsyncClient] = None


def user_agent() -> str:
    email = get_settings().contact_email
    return f"CitationChecker/0.1 (mailto:{email})" if email else "CitationChecker/0.1"


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            headers={"User-Agent": user_agent()},
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=15.0),
        )
    return _client


async def fetch(url: str, params: Optional[dict] = None) -> httpx.Response:
    """GET with per-host rate limiting and retries on 429/5xx/network errors."""
    host = urlparse(url).hostname or ""
    for attempt in range(MAX_ATTEMPTS):
        await _limiter.wait(host)
        try:
            resp = await get_client().get(url, params=params)
        except httpx.TransportError as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            log.warning("GET %s failed (%s); retrying", url, type(exc).__name__)
            await asyncio.sleep(2**attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == MAX_ATTEMPTS - 1:
                resp.raise_for_status()
            retry_after = resp.headers.get("retry-after", "")
            delay = float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
            log.warning("GET %s -> %d; retry in %.0fs", url, resp.status_code, delay)
            await asyncio.sleep(min(delay, 60))
            continue
        return resp
    raise AssertionError("unreachable")


def cache_path(namespace: str, key: str, suffix: str = "") -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    path = get_settings().cache_dir / namespace / f"{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def fetch_cached(url: str, namespace: str, params: Optional[dict] = None, suffix: str = "") -> bytes:
    """Download once; later calls read from disk. Raises httpx.HTTPStatusError on 4xx."""
    key = url + ("?" + "&".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "")
    path = cache_path(namespace, key, suffix)
    if path.exists():
        return path.read_bytes()
    resp = await fetch(url, params)
    resp.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return resp.content
