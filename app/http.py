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

# Minimum seconds between requests per host. arXiv asks for one request every 3 seconds
# (configurable via ARXIV_MIN_INTERVAL).
HOST_INTERVALS = {
    "api.openalex.org": 0.12,
    "api.crossref.org": 0.1,
    "api.semanticscholar.org": 1.0,
}
ARXIV_HOSTS = ("arxiv.org", "export.arxiv.org")
MAX_ATTEMPTS = 4
DEFAULT_MAX_BYTES = 60 * 1024 * 1024
# Status codes whose outcome we cache as "not available" instead of retrying every run.
NEGATIVE_STATUSES = (401, 403, 404, 410, 451)


class DownloadTooLarge(Exception):
    pass


class _HostLimiter:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    async def wait(self, host: str) -> None:
        interval = get_settings().arxiv_min_interval if host in ARXIV_HOSTS else HOST_INTERVALS.get(host, 0.0)
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


def set_client(client: Optional[httpx.AsyncClient]) -> None:
    """Override the shared client (tests use httpx.MockTransport)."""
    global _client, _limiter
    _client = client
    _limiter = _HostLimiter()


async def fetch(url: str, params: Optional[dict] = None, max_bytes: int = DEFAULT_MAX_BYTES) -> httpx.Response:
    """GET with per-host rate limiting, retries on 429/5xx/network errors, and a size cap."""
    host = urlparse(url).hostname or ""
    for attempt in range(MAX_ATTEMPTS):
        await _limiter.wait(host)
        try:
            async with get_client().stream("GET", url, params=params) as resp:
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise DownloadTooLarge(f"{url}: {int(declared) / 1e6:.0f} MB")
                chunks, size = [], 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise DownloadTooLarge(f"{url}: over {max_bytes / 1e6:.0f} MB")
                    chunks.append(chunk)
                resp._content = b"".join(chunks)  # make .content/.json() work after the stream closes
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


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


async def fetch_cached(
    url: str,
    namespace: str,
    params: Optional[dict] = None,
    suffix: str = "",
    allow_missing: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Optional[bytes]:
    """Download once; later calls read from disk.

    With allow_missing=True, 401/403/404/410/451 return None (and that outcome is cached);
    otherwise they raise httpx.HTTPStatusError.
    """
    key = url + ("?" + "&".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "")
    path = cache_path(namespace, key, suffix)
    missing_marker = path.with_suffix(path.suffix + ".missing")
    if path.exists():
        return path.read_bytes()
    if allow_missing and missing_marker.exists():
        return None
    resp = await fetch(url, params, max_bytes=max_bytes)
    if allow_missing and resp.status_code in NEGATIVE_STATUSES:
        missing_marker.write_text(str(resp.status_code))
        return None
    resp.raise_for_status()
    _write_atomic(path, resp.content)
    return resp.content
