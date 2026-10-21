"""Abuse protection for a publicly reachable instance (brief section 10.4).

Rate limits are in memory: they reset when the server restarts, which is fine for a
single-process hackathon deployment.
"""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from typing import Callable, Optional

from fastapi import Request

from app.config import Settings

WINDOW_SECONDS = 3600


class RateLimiter:
    """Sliding one-hour window of events per key (client IP)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def _window(self, key: str) -> deque[float]:
        events = self._events[key]
        cutoff = self.clock() - WINDOW_SECONDS
        while events and events[0] <= cutoff:
            events.popleft()
        return events

    def retry_after(self, key: str, per_hour: int) -> Optional[int]:
        """Seconds until `key` may act again, or None if it is under the limit (0 = unlimited)."""
        if per_hour <= 0:
            return None
        events = self._window(key)
        if len(events) < per_hour:
            return None
        return max(1, int(events[0] + WINDOW_SECONDS - self.clock()) + 1)

    def record(self, key: str) -> None:
        self._window(key).append(self.clock())


def client_ip(request: Request, settings: Settings) -> str:
    """The caller's IP. Proxy headers are trusted only when configured (TRUST_PROXY_HEADERS).

    Each proxy appends the address it received the request from, so the client's address is the
    entry added by the outermost trusted proxy: TRUSTED_PROXY_HOPS entries from the end. Entries
    further left come from the client itself and could be forged to dodge the rate limits."""
    if settings.trust_proxy_headers:
        entries = [e.strip() for e in request.headers.get("x-forwarded-for", "").split(",") if e.strip()]
        if entries:
            return entries[-min(settings.trusted_proxy_hops, len(entries))]
    return request.client.host if request.client else "unknown"


def passcode_ok(settings: Settings, given: Optional[str]) -> bool:
    """Full checks need JUDGE_MODE_PASSWORD when it is set; quick checks and the demo stay open."""
    expected = settings.judge_mode_password
    if not expected:
        return True
    return hmac.compare_digest((given or "").strip().encode(), expected.encode())


def minutes_text(seconds: int) -> str:
    minutes = max(1, round(seconds / 60))
    return "about a minute" if minutes == 1 else f"about {minutes} minutes"
