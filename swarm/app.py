"""One place that turns user options into a configured research run (CLI and server share it)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .config import cache_dir, env_overrides, load_config
from .errors import ConfigError
from .pricing import PRICES, Price, register_price
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
    provider: str | None = None  # anthropic (Claude) | openai (any OpenAI-compatible API)
    base_url: str | None = None  # OpenAI-compatible endpoint, e.g. https://host/v1
    price: str | None = None  # "IN,OUT[,CACHED]" USD per million tokens, for unlisted models
    api_key_env: str | None = None  # variable holding the provider's key (default OPENAI_API_KEY)


def register_prices(file_cfg: dict[str, Any], models: list[str], price: str | None = None) -> None:
    """Prices for models the built-in table doesn't know (e.g. an OpenAI-compatible provider's).

    From `--price IN,OUT[,CACHED]` / SWARM_PRICE (applies to `models`) or swarm.toml:
        [prices]
        "deepseek-ai/DeepSeek-V4.1-Flash-1017" = [0.10, 0.40]        # or [in, out, cached]
    Without a cached price, cached prompt tokens are billed as ordinary input (never understated).
    """
    table: dict[str, Any] = dict(file_cfg.get("prices", {}))
    if price:
        try:
            nums = [float(x) for x in price.split(",")]
        except ValueError as e:
            raise ConfigError(f"--price {price!r}: expected IN,OUT or IN,OUT,CACHED (USD per million)") from e
        for m in models:
            table[m] = nums
    for model, nums in table.items():
        if not isinstance(nums, (list, tuple)) or len(nums) not in (2, 3) or min(nums) < 0:
            raise ConfigError(f"price for {model!r} must be [in, out] or [in, out, cached] in USD per million")
        inp, out = float(nums[0]), float(nums[1])
        cached = float(nums[2]) / inp if len(nums) == 3 and inp > 0 else 1.0
        register_price(model, Price(inp, out, cache_read_mult=cached, cache_write_mult=1.0))


def resolve(opts: ResearchOptions) -> tuple[ResearchLLM, ResearchConfig, dict[str, Any]]:
    """Build the LLM client and config: preset < config file < environment < options."""
    file_cfg, _ = load_config(opts.config_path)
    r = {**file_cfg.get("research", {}), **env_overrides()}
    pick = lambda name, default=None: v if (v := getattr(opts, name, None)) is not None else r.get(name, default)  # noqa: E731
    preset = str(pick("preset", "standard"))
    search_kind = "none" if opts.docs_only else pick("search", "tinyfish")
    max_usd = pick("max_usd")
    base_url = pick("base_url") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    # An explicit base URL, or a model that isn't Claude, implies the OpenAI-compatible provider.
    # (An endpoint only in the environment doesn't: other tools set OPENAI_BASE_URL too.)
    model_name = str(pick("model") or "")
    not_claude = bool(model_name) and not model_name.startswith("claude")
    provider = str(pick("provider") or ("openai" if pick("base_url") or not_claude else "anthropic"))
    if provider not in ("anthropic", "openai"):
        raise ConfigError(f"unknown provider {provider!r} (anthropic or openai)")
    if provider == "openai" and not opts.mock:
        if not pick("model"):
            raise ConfigError("the OpenAI-compatible provider needs a model name",
                              "pass --model <provider's model id>, e.g. deepseek-ai/DeepSeek-V4.1-Flash-1017")
        if search_kind == "anthropic":
            raise ConfigError("--search anthropic works only with Claude",
                              "use --search tinyfish (free) with the OpenAI-compatible provider")
    models = [m for m in (pick("model"), pick("worker_model")) if m]
    register_prices(file_cfg, models, pick("price"))
    cfg = config_from(
        preset, model=pick("model"), max_usd=(None if max_usd == 0 else max_usd),
        max_subquestions=opts.max_subquestions, max_agents=opts.max_agents,
        max_searches=opts.max_searches, check_claims=opts.check_claims)
    info: dict[str, Any] = {"preset": preset, "search": "mock" if opts.mock else search_kind,
                            "documents": len(opts.docs), "provider": "mock" if opts.mock else provider,
                            "model": cfg.model}
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
    llm: ResearchLLM
    if provider == "openai":
        from .openai_compat import OpenAICompatLLM

        cached_known = all(PRICES[m].cache_read_mult < 1.0 for m in models if m in PRICES)
        try:
            llm = OpenAICompatLLM(base_url=base_url, search=web, docs=docs, filters=filters,
                                  max_tool_turns=cfg.max_tool_turns, worker_model=pick("worker_model"),
                                  language=pick("language"), cached_price_known=cached_known,
                                  api_key_env=str(pick("api_key_env") or "OPENAI_API_KEY"))
        except Exception:
            if isinstance(web, CachedProvider):
                web.db.close()  # don't leak the cache connection when the provider can't start
            raise
        info["base_url"] = llm.base_url
    else:
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
