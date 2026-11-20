"""TinyFish Search and Fetch (free within a daily allowance).

Docs: https://docs.tinyfish.ai/tinysearch/reference and https://docs.tinyfish.ai/tinyfetch/reference
(read 2026-11-21). Free allowance: 12,000 searches/day at 30/min and 1,000 fetched URLs/day at
150 URLs/min, resetting 00:00 UTC. Key: https://agent.tinyfish.ai/api-keys -> TINYFISH_API_KEY.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any

import httpx2

from ..errors import SearchAuthError, SearchQuotaExhausted, SearchUnavailable, missing_tinyfish_key
from .base import MinuteLimiter, Page, SearchFilters, SearchHit

SEARCH_URL = "https://api.search.tinyfish.ai"
FETCH_URL = "https://api.fetch.tinyfish.ai"


class TinyFish:
    name = "tinyfish"
    free = True

    def __init__(self, api_key: str | None = None, searches_per_min: int = 30,
                 fetches_per_min: int = 150, max_retries: int = 3,
                 http: httpx2.AsyncClient | None = None):
        key = api_key or os.environ.get("TINYFISH_API_KEY")
        if not key:
            raise missing_tinyfish_key()
        self.api_key: str = key
        self.http = http or httpx2.AsyncClient(timeout=150.0)  # docs: client timeout >= 150 s
        self.search_limit = MinuteLimiter(searches_per_min)
        self.fetch_limit = MinuteLimiter(fetches_per_min)
        self.max_retries = max_retries
        self.stats = {"searches": 0, "fetched_urls": 0, "fetch_errors": 0}

    async def _call(self, method: str, url: str, **kw) -> dict[str, Any]:
        headers = {"X-API-Key": self.api_key}
        for attempt in range(self.max_retries + 1):
            try:
                r = await self.http.request(method, url, headers=headers, **kw)
            except httpx2.TransportError as e:
                if attempt == self.max_retries:
                    raise SearchUnavailable(f"TinyFish unreachable: {type(e).__name__}") from e
                await asyncio.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.json()
            body: dict[str, Any] = {}
            with contextlib.suppress(ValueError):
                body = r.json()
            err = body.get("error", {}) if isinstance(body.get("error"), dict) else {}
            details = err.get("details") or {}
            if r.status_code == 401:
                raise SearchAuthError("TinyFish rejected the API key (401)")
            if r.status_code == 402 or details.get("reason") == "free_allowance_exhausted":
                raise SearchQuotaExhausted("TinyFish daily free allowance exhausted")
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                try:
                    wait = float(r.headers.get("retry-after", ""))
                except ValueError:
                    wait = 2.0 ** (attempt + 1)
                await asyncio.sleep(min(wait, 60.0))
                continue
            if r.status_code >= 500 or r.status_code == 429:
                raise SearchUnavailable(f"TinyFish returned {r.status_code}")
            raise SearchUnavailable(f"TinyFish request failed ({r.status_code}): {err.get('message', '')}")
        raise SearchUnavailable("TinyFish: retries exhausted")

    async def search(self, query: str, filters: SearchFilters | None = None,
                     purpose: str | None = None) -> list[SearchHit]:
        f = filters or SearchFilters()
        params: dict[str, Any] = {"query": query}
        if purpose:
            params["purpose"] = purpose[:2000]
        if f.include_domains:
            params["include_domains"] = ",".join(f.include_domains)
        if f.exclude_domains:
            params["exclude_domains"] = ",".join(f.exclude_domains)
        if f.domain_type != "web":
            params["domain_type"] = f.domain_type
        if f.recency_days and f.domain_type != "research_paper":
            params["recency_minutes"] = min(5_256_000, f.recency_days * 1440)
        if f.language:
            params["language"] = f.language
        if f.location:
            params["location"] = f.location
        await self.search_limit.acquire()
        data = await self._call("GET", SEARCH_URL, params=params)
        self.stats["searches"] += 1
        return [SearchHit(str(r.get("url", "")), r.get("title") or "", r.get("snippet") or "",
                          r.get("site_name") or "", r.get("date"))
                for r in data.get("results", []) if r.get("url")]

    async def fetch(self, urls: list[str], purpose: str | None = None) -> list[Page]:
        pages: list[Page] = []
        for i in range(0, len(urls), 10):  # max 10 URLs per request
            chunk = urls[i:i + 10]
            await self.fetch_limit.acquire(len(chunk))
            body: dict[str, Any] = {"urls": chunk, "format": "markdown", "per_url_timeout_ms": 45_000,
                                    "ttl": 86_400}  # accept TinyFish's cache up to a day old
            if purpose:
                body["purpose"] = purpose[:2000]
            data = await self._call("POST", FETCH_URL, json=body)
            for r in data.get("results", []):
                text = r.get("text") if isinstance(r.get("text"), str) else ""
                pages.append(Page(r.get("url", ""), r.get("title") or "", text or ""))
                self.stats["fetched_urls"] += 1
            for e in data.get("errors", []):
                pages.append(Page(e.get("url", ""), "", "", error=str(e.get("error", "error"))))
                self.stats["fetch_errors"] += 1
        return pages

    async def aclose(self) -> None:
        await self.http.aclose()
