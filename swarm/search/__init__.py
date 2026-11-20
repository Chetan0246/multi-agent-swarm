"""Pluggable web search/fetch for research workers.

* `tinyfish` (default): TinyFish Search + Fetch, free within a daily allowance.
* `anthropic`: Claude's server-side web search/fetch ($10 per 1,000 searches); handled
  inside the model call, so it has no provider object here.
* `LocalDocs`: your own files, searchable offline; combined with web search via `Combined`.
* `MockSearch`: offline, deterministic.
"""
from __future__ import annotations

from .base import MinuteLimiter, Page, SearchFilters, SearchHit, SearchProvider, normalize, quote_is_grounded
from .cache import CachedProvider
from .local import LocalDocs
from .mock import MockSearch

__all__ = ["CachedProvider", "LocalDocs", "MinuteLimiter", "MockSearch", "Page", "SearchFilters",
           "SearchHit", "SearchProvider", "normalize", "quote_is_grounded", "make_provider"]


def make_provider(kind: str, cache_path=None, api_key: str | None = None) -> SearchProvider:
    if kind == "mock":
        p: SearchProvider = MockSearch()
    elif kind == "tinyfish":
        from .tinyfish import TinyFish

        p = TinyFish(api_key)
    else:
        raise ValueError(f"unknown search provider {kind!r} (expected tinyfish, anthropic or mock)")
    return CachedProvider(p, cache_path) if cache_path else p
