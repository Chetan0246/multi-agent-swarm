"""One place that turns user options into a configured research run (CLI and server share it)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import cache_dir, env_overrides, load_config
from .research import (
    AnthropicResearchLLM,
    MockResearchLLM,
    ResearchConfig,
    ResearchLLM,
    ResearchReport,
    config_from,
    deep_research,
)
from .search import CachedProvider, LocalDocs, SearchFilters, make_provider

DEFAULT_EXCLUDE = ["pinterest.com", "quora.com"]  # low-signal sources; override in config


@dataclass
class ResearchOptions:
    question: str
    preset: str | None = None
    search: str | None = None  # tinyfish (free) | anthropic (paid server-side) | none
    mock: bool = False  # fully offline: no Claude, no web
    docs: list[str] = field(default_factory=list)
    docs_only: bool = False
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    recency_days: int | None = None
    kind: str | None = None  # web | news | research_paper
    language: str | None = None
    model: str | None = None
    worker_model: str | None = None
    max_usd: float | None = None
    max_subquestions: int | None = None
    max_agents: int | None = None
    max_searches: int | None = None
    check_claims: bool | None = None
    use_cache: bool = True
    config_path: str | None = None


def resolve(opts: ResearchOptions) -> tuple[ResearchLLM, ResearchConfig, dict[str, Any]]:
    """Build the LLM client and config: preset < config file < environment < options."""
    file_cfg, _ = load_config(opts.config_path)
    r = {**file_cfg.get("research", {}), **env_overrides()}
    pick = lambda name, default=None: v if (v := getattr(opts, name, None)) is not None else r.get(name, default)  # noqa: E731
    preset = str(pick("preset", "standard"))
    search_kind = "none" if opts.docs_only else pick("search", "tinyfish")
    max_usd = pick("max_usd")
    cfg = config_from(
        preset, model=pick("model"), max_usd=(None if max_usd == 0 else max_usd),
        max_subquestions=opts.max_subquestions, max_agents=opts.max_agents,
        max_searches=opts.max_searches, check_claims=opts.check_claims)
    info: dict[str, Any] = {"preset": preset, "search": "mock" if opts.mock else search_kind,
                            "documents": len(opts.docs)}
    if opts.mock:
        cfg.free_search = True
        return MockResearchLLM(), cfg, info
    filters = SearchFilters(
        include_domains=list(pick("include_domains", []) or []),
        exclude_domains=list(pick("exclude_domains", DEFAULT_EXCLUDE) or []),
        recency_days=opts.recency_days, domain_type=opts.kind or "web", language=pick("language"))
    web = None
    if search_kind == "tinyfish":
        path = cache_dir() / "web.sqlite" if opts.use_cache else None
        web = make_provider("tinyfish", cache_path=path)
    elif search_kind not in ("anthropic", "none"):
        raise ValueError(f"unknown search provider {search_kind!r} (tinyfish, anthropic)")
    docs = LocalDocs(opts.docs) if opts.docs else None
    if search_kind == "anthropic" and docs is not None:
        raise ValueError("--docs needs client-side search: use --search tinyfish or --docs-only")
    cfg.free_search = search_kind != "anthropic"
    if web is not None and isinstance(web, CachedProvider):
        info["cache"] = str(cache_dir() / "web.sqlite")
    llm = AnthropicResearchLLM(max_searches=cfg.max_searches, search=web, docs=docs, filters=filters,
                               max_tool_turns=cfg.max_tool_turns, worker_model=pick("worker_model"),
                               language=pick("language"))
    return llm, cfg, info


async def run_research(opts: ResearchOptions, on_event=None, prior: ResearchReport | None = None
                       ) -> ResearchReport:
    llm, cfg, info = resolve(opts)
    try:
        report = await deep_research(opts.question, llm, cfg, on_event=on_event, prior=prior)
    finally:
        await aclose(llm)
    report.stats.update({f"run_{k}": v for k, v in info.items()})
    return report


async def aclose(llm: ResearchLLM) -> None:
    """Release network clients and the cache database held by a resolved LLM (if any)."""
    if (close := getattr(llm, "aclose", None)) is not None:
        await close()
