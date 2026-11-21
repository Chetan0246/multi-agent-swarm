"""Task kinds for `swarm run`: many independent tasks, each checked, retried and escalated.

One JSONL line per task. Fields common to every kind: `id`, `prompt`, optional `difficulty`
(1-3), `priority`, and a check: `expected` (exact answer) or `pattern` (regex).

* `text` (default): classification, extraction, short answers, rewriting, translation,
  summaries. Many tasks share one call (batched JSON answers).
* `code`: `prompt` asks for code; `tests` holds Python asserts. The answer passes when the
  tests run cleanly in the sandbox (needs --allow-code-execution; see sandbox.py).
* `web`: a fact to look up on the web. A tool-using worker searches (free TinyFish) and
  reads pages; the answer comes with verbatim quotes checked against those pages. Passes
  when it has `min_sources` grounded sources (default 1) and matches `expected`/`pattern`
  if given.

Kinds run side by side, each through its own engine, so a slow web lookup never holds up a
batch of text tasks.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends import CODE_SYSTEM_PROMPT, SYSTEM_PROMPT, BatchResponse, CallUsage
from .errors import ConfigError
from .orchestrator import RunReport, Swarm
from .planner import PlanConfig, Rung
from .research import Citation, _add
from .sandbox import code_validator
from .task import Task, _norm

KINDS = ("text", "code", "web")
SYSTEMS = {"text": SYSTEM_PROMPT, "code": CODE_SYSTEM_PROMPT}


# ----------------------------------------------------------------------------- loading


def web_validator(expected: str | None, pattern: str | None, min_sources: int = 1) -> Callable[[str], bool]:
    def check(answer: str) -> bool:
        try:
            d = json.loads(answer)
        except json.JSONDecodeError:
            return False
        text = str(d.get("answer", "")).strip()
        if not text or len({s.get("url") for s in d.get("sources", [])}) < min_sources:
            return False
        if expected is not None and _norm(expected) not in _norm(text):
            return False
        return pattern is None or re.search(pattern, text, re.S) is not None

    return check


def load_tasks(path: str | Path, allow_code: bool = False, code_timeout: float = 10.0) -> list[Task]:
    tasks: list[Task] = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            t = Task.from_dict(d)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise ConfigError(f"{path}:{n}: not a valid task ({e})") from e
        kind = str(d.get("kind", "text"))
        if kind not in KINDS:
            raise ConfigError(f"{path}:{n}: unknown kind {kind!r} (expected {', '.join(KINDS)})")
        t.meta["kind"] = kind
        if kind == "code" and d.get("tests"):
            if not allow_code:
                raise ConfigError(f"{path}:{n}: code task with tests, but running model-written code is off",
                                  "add --allow-code-execution (read the sandbox notes in the docs first)")
            t.validator = code_validator(str(d["tests"]), code_timeout)
        elif kind == "web":
            t.validator = web_validator(t.expected, t.pattern, int(d.get("min_sources", 1)))
            t.expected = t.pattern = None  # checked inside the validator, against the answer text
        tasks.append(t)
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ConfigError(f"{path}: task ids must be unique")
    return tasks


# ----------------------------------------------------------------------------- web backend


class WebTaskBackend:
    """Engine adapter: one web lookup per task through a tool-using research worker."""

    bulk = False

    def __init__(self, llm: Any, price_model: str | None = None):
        self.llm = llm  # AnthropicResearchLLM / OpenAICompatLLM with a search provider, or MockResearchLLM
        self.price_model = price_model
        self.searches = 0

    async def aclose(self) -> None:
        if (close := getattr(self.llm, "aclose", None)) is not None:
            await close()

    async def run_batch(self, tasks: list[Task], effort: str, model: str) -> BatchResponse:
        results = await asyncio.gather(*(self.llm.investigate("(a standalone lookup task)", t.prompt, model, effort)
                                         for t in tasks))
        usage = CallUsage()
        answers: dict[str, str] = {}
        for t, (text, cites, u, searches) in zip(tasks, results):
            _add(usage, u)
            self.searches += searches
            answers[t.id] = json.dumps({"answer": text, "sources": [_src(c) for c in cites]}, ensure_ascii=False)
        return BatchResponse(answers, usage)


def _src(c: Citation) -> dict[str, str]:
    return {"url": c.url, "title": c.title, "quote": c.quote}


# ----------------------------------------------------------------------------- running


@dataclass
class KindRun:
    kind: str
    report: RunReport
    extra: dict[str, Any] = field(default_factory=dict)


def plan_for(kind: str, base: PlanConfig) -> PlanConfig:
    """Batching suited to each kind: web is one lookup per call; code batches stay small."""
    from dataclasses import replace

    if kind == "web":
        return replace(base, adaptive=False, static_batch=1, max_batch=1, hedging=False)
    if kind == "code":
        return replace(base, max_batch=min(base.max_batch, 4), static_batch=min(base.static_batch, 4),
                       max_batch_input_tokens=max(base.max_batch_input_tokens, 8000))
    return base


async def run_kinds(tasks: list[Task], backends: dict[str, Any], base: PlanConfig,
                    state_dir: Path | None = None, agents: int | None = None,
                    on_event: Callable[[str, dict], None] | None = None) -> list[KindRun]:
    """Run each kind's tasks through its own engine, concurrently. Each kind keeps its own
    journal and learned policy (their costs and accuracy differ)."""
    by_kind: dict[str, list[Task]] = {}
    for t in tasks:
        by_kind.setdefault(t.meta.get("kind", "text"), []).append(t)

    async def one(kind: str, ts: list[Task]) -> KindRun:
        journal = str(state_dir / f"journal-{kind}.jsonl") if state_dir else None
        policy = str(state_dir / f"policy-{kind}.json") if state_dir else None
        sink = (lambda k, d: on_event(k, {"kind": kind, **d})) if on_event else None
        engine = Swarm(backends[kind], plan_for(kind, base), agents=agents, journal_path=journal,
                       policy_path=policy, on_event=sink)
        report = await engine.arun(ts)
        extra = {"web_searches": backends[kind].searches} if kind == "web" else {}
        if engine.fatal is not None:
            extra["stopped_early"] = f"{type(engine.fatal).__name__}: {str(engine.fatal).splitlines()[0]}"
        return KindRun(kind, report, extra)

    try:
        return list(await asyncio.gather(*(one(k, ts) for k, ts in by_kind.items())))
    finally:
        for b in {id(b): b for b in backends.values()}.values():
            if (close := getattr(b, "aclose", None)) is not None:
                await close()


def ladder(model: str, efforts: str) -> tuple[Rung, ...]:
    return tuple(Rung(model, e.strip()) for e in efforts.split(",") if e.strip())


def build_backends(kinds: set[str], provider: str, model: str, *, base_url: str | None = None,
                   api_key_env: str = "OPENAI_API_KEY", shared_context: str | None = None,
                   batch_api: bool = False, mock: bool = False, search: str = "tinyfish",
                   use_fallbacks: bool = True, cached_price_known: bool = False,
                   use_cache: bool = True) -> dict[str, Any]:
    """One backend per kind that the workload uses."""
    out: dict[str, Any] = {}
    if mock:
        from .backends import MockBackend
        from .research import MockResearchLLM

        for k in kinds - {"web"}:
            out[k] = MockBackend(seed=1, outage_at=None)
        if "web" in kinds:
            out["web"] = WebTaskBackend(MockResearchLLM(fail_low=False))
        return out

    def web_search():
        if search == "mock":
            from .search import MockSearch

            return MockSearch()
        from .config import cache_dir
        from .search import make_provider

        return make_provider("tinyfish", cache_path=cache_dir() / "web.sqlite" if use_cache else None)

    if provider == "openai":
        from .openai_compat import OpenAICompatBackend, OpenAICompatLLM

        def client(**kw):
            return OpenAICompatLLM(base_url=base_url, api_key_env=api_key_env,
                                   cached_price_known=cached_price_known, **kw)

        for k in kinds - {"web"}:
            out[k] = OpenAICompatBackend(client(require_search=False), SYSTEMS[k], shared_context)
        if "web" in kinds:
            out["web"] = WebTaskBackend(client(search=web_search()))
        return out

    from .backends import AnthropicBackend, AnthropicBatchBackend
    from .research import AnthropicResearchLLM

    for k in kinds - {"web"}:
        out[k] = (AnthropicBatchBackend(shared_context=shared_context, system=SYSTEMS[k]) if batch_api
                  else AnthropicBackend(shared_context=shared_context, use_fallbacks=use_fallbacks,
                                        system=SYSTEMS[k]))
    if "web" in kinds:
        out["web"] = WebTaskBackend(AnthropicResearchLLM(search=web_search(), use_fallbacks=use_fallbacks))
    return out
