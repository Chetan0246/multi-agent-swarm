"""Search/fetch provider interface used by research workers' client-side tools."""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SearchFilters:
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)
    recency_days: int | None = None
    domain_type: str = "web"  # web | news | research_paper
    language: str | None = None
    location: str | None = None


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str
    site: str = ""
    date: str | None = None


@dataclass
class Page:
    url: str
    title: str
    text: str  # markdown/plain text; may be truncated by the caller for the model
    error: str | None = None


class SearchProvider(Protocol):
    name: str
    free: bool  # True when search/fetch cost no money (token costs still apply)

    async def search(self, query: str, filters: SearchFilters | None = None,
                     purpose: str | None = None) -> list[SearchHit]: ...

    async def fetch(self, urls: list[str], purpose: str | None = None) -> list[Page]: ...


class MinuteLimiter:
    """Client-side sliding-window limiter, shared by all workers (e.g. 30 searches/min)."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.per_minute = per_minute
        self.clock = clock
        self.stamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, n: int = 1) -> None:
        for _ in range(n):
            async with self._lock:
                while True:
                    now = self.clock()
                    while self.stamps and now - self.stamps[0] >= 60.0:
                        self.stamps.popleft()
                    if len(self.stamps) < self.per_minute:
                        self.stamps.append(now)
                        break
                    await asyncio.sleep(60.0 - (now - self.stamps[0]) + 0.01)


_WS = re.compile(r"\s+")
_MD = re.compile(r"[*_`#>\[\]()|~]")


def normalize(text: str) -> str:
    """Normalisation for quote grounding: case, markdown markup, quotes and whitespace."""
    t = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"')
    t = t.replace("”", '"').replace("–", "-").replace("—", "-").replace(" ", " ")
    t = _MD.sub(" ", t)
    return _WS.sub(" ", t).strip()


def quote_is_grounded(quote: str, haystack_norm: str, min_part: int = 20) -> bool:
    """True when the quote appears in the (normalised) source text.

    Ellipses are allowed: every part longer than `min_part` characters must appear.
    """
    parts = [normalize(p) for p in re.split(r"\.\.\.|…", quote)]
    parts = [p.strip(" .,;:\"'") for p in parts]
    parts = [p for p in parts if len(p) >= min_part]
    return bool(parts) and all(p in haystack_norm for p in parts)
