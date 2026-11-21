"""One small model interface for every workflow (build, write, analyze, review, doc).

    chat = make_chat(model="deepseek-ai/DeepSeek-V4.1-Flash-1017", base_url=..., max_usd=1.0)
    text = await chat.complete(system, user, schema=SCHEMA, role="build.plan")

* Claude or any OpenAI-compatible provider, chosen like everywhere else in the project.
* Every call is priced as it returns; once the run's dollar cap is reached, the next call
  raises `RunCapReached` (fatal: the engine stops queued work instead of retrying it).
* `parallel()` runs many model jobs through the swarm engine: AIMD back-off on 429s, retries
  on 5xx and timeouts, circuit breaker, effort escalation when a job's check fails.
* `MockChat` answers from a function, for offline tests and `--mock` demos.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..backends import BatchResponse, CallUsage
from ..errors import SwarmError
from ..orchestrator import Swarm
from ..planner import PlanConfig, Rung
from ..pricing import price_of, usage_usd
from ..task import Task


class RunCapReached(SwarmError):
    fatal = True
    hint = "raise --max-usd to allow more spend"


@dataclass
class Chat:
    """Base class: subclasses implement `_call` and return (text, usage)."""

    model: str
    max_usd: float | None = None
    usd: float = 0.0
    calls: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)  # role, usd per call (no content)
    stopped: Any = None  # the fatal error (e.g. RunCapReached) that cut a parallel step short

    async def complete(self, system: str, user: str, *, schema: dict | None = None, effort: str = "medium",
                       max_tokens: int = 8000, role: str = "") -> str:
        if self.max_usd is not None and self.usd >= self.max_usd:
            raise RunCapReached(f"the run's ${self.max_usd:.2f} cap is reached (spent ${self.usd:.4f})")
        text, usage = await self._call(system, user, schema, effort, max_tokens, role)
        cost = usage_usd(self.model, usage)
        self.usd += cost
        self.calls += 1
        self.log.append({"role": role, "usd": round(cost, 6)})
        return text

    async def json(self, system: str, user: str, schema: dict, **kw: Any) -> dict:
        from ..research import _parse_json_obj

        return _parse_json_obj(await self.complete(system, user, schema=schema, **kw))

    async def _call(self, system: str, user: str, schema: dict | None, effort: str, max_tokens: int,
                    role: str) -> tuple[str, CallUsage]:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


class AnthropicChat(Chat):
    def __init__(self, model: str, max_usd: float | None = None, use_fallbacks: bool = True):
        super().__init__(model, max_usd)
        from ..research import AnthropicResearchLLM

        self._llm = AnthropicResearchLLM(use_fallbacks=use_fallbacks)  # reuses its client and error mapping

    async def _call(self, system, user, schema, effort, max_tokens, role):
        from ..research import _usage

        oc: dict[str, Any] = {"effort": effort}
        if schema is not None:
            oc["format"] = {"type": "json_schema", "schema": schema}
        msg = await self._llm._create(model=self.model, max_tokens=max_tokens, system=system,
                                      messages=[{"role": "user", "content": user}], output_config=oc)
        usage, _ = _usage(msg)
        if msg.stop_reason == "refusal":
            return "", usage
        return "".join(b.text for b in msg.content if b.type == "text"), usage

    async def aclose(self) -> None:
        await self._llm.client.close()


class OpenAIChat(Chat):
    def __init__(self, model: str, max_usd: float | None = None, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", cached_price_known: bool = False, http: Any = None):
        super().__init__(model, max_usd)
        from ..openai_compat import OpenAICompatLLM

        self._llm = OpenAICompatLLM(base_url=base_url, api_key_env=api_key_env, require_search=False,
                                    cached_price_known=cached_price_known, http=http, max_tokens=16000)

    async def _call(self, system, user, schema, effort, max_tokens, role):
        from ..openai_compat import _json_rule

        sys_prompt = system + (_json_rule(schema) if schema is not None else "")
        data = await self._llm._chat(self.model, [{"role": "system", "content": sys_prompt},
                                                  {"role": "user", "content": user}],
                                     effort, json_out=schema is not None, max_tokens=max_tokens)
        msg, _ = self._llm._message(data)
        return msg.get("content") or "", self._llm._usage(data)

    async def aclose(self) -> None:
        await self._llm.aclose()


class MockChat(Chat):
    """Offline: `respond(role, system, user, schema) -> str` decides every answer."""

    def __init__(self, respond: Callable[[str, str, str, dict | None], str], model: str = "claude-haiku-5-5",
                 max_usd: float | None = None):
        super().__init__(model, max_usd)
        self.respond = respond

    async def _call(self, system, user, schema, effort, max_tokens, role):
        text = self.respond(role, system, user, schema)
        return text, CallUsage(input_tokens=(len(system) + len(user)) // 4, output_tokens=len(text) // 4)


def make_chat(model: str | None = None, *, provider: str | None = None, base_url: str | None = None,
              api_key_env: str | None = None, price: str | None = None, max_usd: float | None = None,
              config_path: str | None = None) -> Chat:
    """Resolve provider, model and prices like `swarm research` does (flags > env > swarm.toml)."""
    import os

    from ..app import register_prices
    from ..config import env_overrides, load_config
    from ..pricing import PRICES

    file_cfg, _ = load_config(config_path)
    conf = {**file_cfg.get("research", {}), **file_cfg.get("run", {}), **env_overrides()}
    model = model or conf.get("model") or "claude-opus-5-5"
    base_url = base_url or conf.get("base_url")
    provider = provider or conf.get("provider") or ("openai" if base_url or not model.startswith("claude")
                                                    else "anthropic")
    register_prices(file_cfg, [model], price or conf.get("price"))
    price_of(model)  # unknown model: fail before any spend
    if provider == "openai":
        cached = PRICES[model].cache_read_mult < 1.0
        return OpenAIChat(model, max_usd, base_url or os.environ.get("OPENAI_BASE_URL")
                          or os.environ.get("OPENAI_API_BASE"),
                          api_key_env or conf.get("api_key_env") or "OPENAI_API_KEY", cached)
    return AnthropicChat(model, max_usd)


# ----------------------------------------------------------------------------- parallel jobs


class _FnBackend:
    """Engine backend whose 'call' is an arbitrary async job (one task per call)."""

    bulk = False

    def __init__(self, jobs: dict[str, Callable[[str], Awaitable[str]]]):
        self.jobs = jobs

    async def run_batch(self, tasks: list[Task], effort: str, model: str) -> BatchResponse:
        answers = {t.id: await self.jobs[t.id](effort) for t in tasks}
        return BatchResponse(answers, CallUsage())  # cost is tracked by the Chat, per call


async def parallel(chat: Chat, jobs: dict[str, Callable[[str], Awaitable[str]]],
                   check: Callable[[str, str], bool] | None = None, efforts: tuple[str, ...] = ("medium", "high"),
                   agents: int = 6, on_event: Callable[[str, dict], None] | None = None) -> dict[str, str | None]:
    """Run jobs concurrently; a job whose answer fails `check(id, answer)` is retried at the next
    effort (then once more). Returns id -> answer (None when it never passed)."""
    if not jobs:
        return {}
    def validator(jid: str) -> Callable[[str], bool]:
        return (lambda a: check(jid, a)) if check is not None else (lambda a: bool(a))

    tasks = [Task(jid, f"job:{jid}", 2, validator=validator(jid)) for jid in jobs]
    cfg = PlanConfig(ladder=tuple(Rung(chat.model, e) for e in efforts), adaptive=False, static_batch=1,
                     max_batch=1, hedging=False, top_rung_retries=1)
    engine = Swarm(_FnBackend(jobs), cfg, agents=max(1, min(agents, len(jobs))), on_event=on_event)
    run = await engine.arun(tasks)
    if engine.fatal is not None:
        chat.stopped = engine.fatal
    if engine.fatal is not None and not any(r.ok for r in run.results.values()):
        raise engine.fatal
    return {jid: (run.results[jid].answer if run.results[jid].ok else None) for jid in jobs}


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)
