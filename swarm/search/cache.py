"""Disk cache in front of any provider: repeated questions cost no quota and no time."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from .base import Page, SearchFilters, SearchHit, SearchProvider


class CachedProvider:
    def __init__(self, inner: SearchProvider, path: str | Path, search_ttl_s: int = 86_400,
                 fetch_ttl_s: int = 7 * 86_400):
        self.inner = inner
        self.name = inner.name
        self.free = inner.free
        self.search_ttl_s = search_ttl_s
        self.fetch_ttl_s = fetch_ttl_s
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, t REAL)")
        self.hits = 0
        self.misses = 0

    async def aclose(self) -> None:
        self.db.close()
        if (close := getattr(self.inner, "aclose", None)) is not None:
            await close()

    @property
    def stats(self) -> dict:
        return {**getattr(self.inner, "stats", {}), "cache_hits": self.hits, "cache_misses": self.misses}

    def _get(self, key: str, ttl: int):
        row = self.db.execute("SELECT v, t FROM kv WHERE k = ?", (key,)).fetchone()
        if row and time.time() - row[1] < ttl:
            self.hits += 1
            return json.loads(row[0])
        self.misses += 1
        return None

    def _put(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?, ?)", (key, json.dumps(value), time.time()))
        self.db.commit()

    @staticmethod
    def _key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    async def search(self, query, filters: SearchFilters | None = None, purpose=None) -> list[SearchHit]:
        k = self._key(self.name, "search", query, asdict(filters) if filters else None)
        if (v := self._get(k, self.search_ttl_s)) is not None:
            return [SearchHit(**h) for h in v]
        hits = await self.inner.search(query, filters, purpose)
        self._put(k, [asdict(h) for h in hits])
        return hits

    async def fetch(self, urls, purpose=None) -> list[Page]:
        out: dict[str, Page] = {}
        todo = []
        for u in urls:
            if (v := self._get(self._key(self.name, "fetch", u), self.fetch_ttl_s)) is not None:
                out[u] = Page(**v)
            else:
                todo.append(u)
        if todo:
            for p in await self.inner.fetch(todo, purpose):
                out[p.url] = p
                if not p.error:  # never cache failures
                    self._put(self._key(self.name, "fetch", p.url), asdict(p))
        return [out[u] for u in urls if u in out] + [p for u, p in out.items() if u not in urls]
