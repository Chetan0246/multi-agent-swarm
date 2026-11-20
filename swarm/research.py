"""Deep research on top of the swarm engine.

    question ─► plan (1 call) ─► sub-questions ─► swarm engine: parallel web-search
                                                  workers, citation check, effort escalation
                                                       │ verified findings + sources
                                                       ▼
                               synthesis (1 call) ─► citation-integrity check ─► report.md

Why this shape (see docs/research):
* Breadth-first research is where multi-agent has the strongest evidence (Anthropic's
  orchestrator-worker research system, +90.2% over a single agent), because each worker
  searches with a fresh context.
* Code, not an LLM, schedules the workers. They pull from the engine's queue, so the run
  never waits on one slow wave, and they never talk to each other (no misalignment).
* Every finding passes a deterministic gate before synthesis: enough citations from enough
  distinct domains. A weak finding is re-researched at higher effort.
* After synthesis, every [n] reference is checked against the source list; any reference
  to a non-existent source is removed and counted.
"""
from __future__ import annotations

import asyncio
import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from .backends import BatchResponse, CallUsage, RateLimited, TransientError, classify_429
from .orchestrator import RunReport, Swarm
from .planner import PlanConfig, Rung
from .task import Task

# ----------------------------------------------------------------------------- types


@dataclass
class Citation:
    url: str
    title: str
    quote: str = ""

    @property
    def domain(self) -> str:
        host = urlparse(self.url).netloc.lower()
        return host[4:] if host.startswith("www.") else host


@dataclass
class Finding:
    sub_id: str
    question: str
    text: str
    citations: list[Citation]
    verified: bool
    effort: str


@dataclass
class ResearchReport:
    question: str
    markdown: str
    findings: list[Finding]
    sources: list[Citation]
    stats: dict[str, Any] = field(default_factory=dict)


class ResearchLLM(Protocol):
    """The three model calls deep research needs. Real and mock implementations below."""

    async def plan(self, question: str, max_subquestions: int, model: str,
                   effort: str) -> tuple[list[dict], CallUsage]: ...

    async def investigate(self, question: str, sub_question: str, model: str,
                          effort: str) -> tuple[str, list[Citation], CallUsage, int]: ...

    async def synthesize(self, question: str, findings_block: str, n_sources: int, model: str,
                         effort: str) -> tuple[str, CallUsage]: ...


# ----------------------------------------------------------------------------- prompts

PLAN_SYSTEM = (
    "You plan web research for a team of parallel research workers. Break the user's question "
    "into independent sub-questions that can each be answered by separate web searches, with no "
    "worker needing another's result. Scale the number to the question: 1 for a simple fact, "
    "2-4 for a comparison or a question with a few distinct parts, more only for genuinely broad "
    "questions. Do not create overlapping sub-questions. Each sub-question must be specific and "
    "self-contained (name the entities and the time frame). Rate each 1 (a single fact), "
    "2 (needs a few sources) or 3 (contested, technical or needs synthesis across many sources)."
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subquestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "difficulty": {"type": "integer"}},
                "required": ["question", "difficulty"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subquestions"],
    "additionalProperties": False,
}

WORKER_SYSTEM = (
    "You are one research worker in a team. Answer only your assigned sub-question; other "
    "workers cover the rest of the overall question, which is given for context only. Search "
    "the web, prefer primary and recent sources, and cross-check key facts across at least two "
    "independent sources. Write a dense factual answer of 80-250 words with specific numbers, "
    "names and dates. Every factual claim must be supported by a source you found. If sources "
    "disagree, say so. If you cannot find reliable information, say exactly what is missing "
    "rather than guessing."
)

SYNTH_SYSTEM = (
    "You write a research report from findings gathered by research workers. Use only the "
    "findings provided: add no facts of your own. Cite sources inline with their numbers in "
    "square brackets, e.g. [3] or [2][5], using only the numbers listed. Every paragraph that "
    "states a fact needs at least one citation. Structure: a 2-4 sentence direct answer first, "
    "then sections with headings, then a short 'Limitations' section noting unverified findings, "
    "disagreements between sources and gaps. Write in Markdown. Do not add a sources list; it "
    "is appended automatically."
)

# ----------------------------------------------------------------------------- verification


def citation_validator(min_citations: int = 2, min_domains: int = 2, min_words: int = 40):
    """Deterministic acceptance test for one finding (the worker's JSON-encoded answer)."""

    def check(answer: str) -> bool:
        try:
            d = json.loads(answer)
        except (json.JSONDecodeError, TypeError):
            return False
        cites = [Citation(**c) for c in d.get("citations", []) if c.get("url")]
        words = len(str(d.get("text", "")).split())
        domains = {c.domain for c in cites}
        return len({c.url for c in cites}) >= min_citations and len(domains) >= min_domains \
            and words >= min_words

    return check


REF = re.compile(r"\[(\d{1,3})\]")


def check_references(markdown: str, n_sources: int) -> tuple[str, dict[str, Any]]:
    """Drop references to sources that don't exist; measure how well the report is cited."""
    invalid = 0

    def fix(m: re.Match) -> str:
        nonlocal invalid
        k = int(m.group(1))
        if 1 <= k <= n_sources:
            return m.group(0)
        invalid += 1
        return ""

    clean = REF.sub(fix, markdown)
    used = {int(k) for k in REF.findall(clean)}
    paras = [p for p in re.split(r"\n\s*\n", clean) if p.strip() and not p.lstrip().startswith("#")]
    fact_paras = [p for p in paras if len(p.split()) >= 25]
    uncited = sum(1 for p in fact_paras if not REF.search(p))
    return clean, {
        "invalid_refs_removed": invalid,
        "sources_cited": len(used),
        "sources_total": n_sources,
        "uncited_paragraphs": uncited,
        "paragraphs": len(fact_paras),
    }


# ----------------------------------------------------------------------------- engine adapter


class ResearchBackend:
    """Adapts a ResearchLLM to the swarm engine: one engine task = one sub-question."""

    bulk = False

    def __init__(self, llm: ResearchLLM, question: str):
        self.llm = llm
        self.question = question
        self.searches = 0

    async def run_batch(self, tasks: list[Task], effort: str, model: str) -> BatchResponse:
        results = await asyncio.gather(
            *(self.llm.investigate(self.question, t.prompt, model, effort) for t in tasks))
        usage = CallUsage()
        answers: dict[str, str] = {}
        for t, (text, cites, u, searches) in zip(tasks, results):
            self.searches += searches
            usage.input_tokens += u.input_tokens
            usage.cache_read_tokens += u.cache_read_tokens
            usage.cache_write_tokens += u.cache_write_tokens
            usage.output_tokens += u.output_tokens
            answers[t.id] = json.dumps({"text": text, "citations": [c.__dict__ for c in cites]})
        return BatchResponse(answers, usage)


# ----------------------------------------------------------------------------- orchestration


@dataclass
class ResearchConfig:
    model: str = "claude-opus-5-5"
    efforts: tuple[str, ...] = ("low", "medium", "high")  # worker escalation ladder
    plan_effort: str = "medium"
    synth_effort: str = "medium"
    max_subquestions: int = 6
    max_agents: int = 6
    min_citations: int = 2
    min_domains: int = 2
    token_budget: float | None = None


async def deep_research(question: str, llm: ResearchLLM, cfg: ResearchConfig | None = None,
                        on_event=None) -> ResearchReport:
    cfg = cfg or ResearchConfig()
    loop = asyncio.get_running_loop()
    t0 = loop.time()

    # 1. plan --------------------------------------------------------------------------
    subs, plan_usage = await llm.plan(question, cfg.max_subquestions, cfg.model, cfg.plan_effort)
    subs = [s for s in subs if str(s.get("question", "")).strip()][: cfg.max_subquestions]
    if not subs:
        subs = [{"question": question, "difficulty": 2}]
    check = citation_validator(cfg.min_citations, cfg.min_domains)
    tasks = [Task(f"q{i + 1}", s["question"], max(1, min(3, int(s.get("difficulty", 2)))),
                  validator=check) for i, s in enumerate(subs)]

    # 2. parallel research through the swarm engine -------------------------------------
    backend = ResearchBackend(llm, question)
    pcfg = PlanConfig(
        ladder=tuple(Rung(cfg.model, e) for e in cfg.efforts),
        adaptive=False, static_batch=1, max_batch=1,  # one web-research call per sub-question
        max_agents=cfg.max_agents, knee_slack=0.0,  # research is latency-bound: one agent each
        hedging=False,  # duplicating a multi-search call doubles a large cost
        top_rung_retries=1, token_budget=cfg.token_budget,
    )
    agents = max(1, min(cfg.max_agents, len(tasks)))  # one worker per sub-question, no idle extras
    run: RunReport = await Swarm(backend, pcfg, agents=agents, on_event=on_event).arun(tasks)

    # 3. collect findings and number the sources ---------------------------------------
    findings: list[Finding] = []
    sources: list[Citation] = []
    index: dict[str, int] = {}
    for t in tasks:
        r = run.results[t.id]
        try:
            d = json.loads(r.answer) if r.answer else {}
        except json.JSONDecodeError:
            d = {}
        cites = [Citation(**c) for c in d.get("citations", []) if c.get("url")]
        for c in cites:
            if c.url not in index:
                sources.append(c)
                index[c.url] = len(sources)
        if d.get("text"):
            findings.append(Finding(t.id, t.prompt, d["text"], cites, r.ok, r.effort))

    blocks = []
    for f in findings:
        refs = sorted({index[c.url] for c in f.citations})
        tag = "verified" if f.verified else "UNVERIFIED (too few independent sources)"
        blocks.append(f"### {f.sub_id}: {f.question}\nStatus: {tag}\nSources: "
                      f"{', '.join(f'[{k}]' for k in refs) or 'none'}\n\n{f.text}")
    listing = "\n".join(f"[{i}] {c.title} ({c.domain})" for i, c in enumerate(sources, 1))
    findings_block = "\n\n".join(blocks) + "\n\n## Source list\n" + listing

    # 4. synthesis + integrity check ------------------------------------------------------
    md, synth_usage = await llm.synthesize(question, findings_block, len(sources), cfg.model,
                                           cfg.synth_effort)
    md, integrity = check_references(md, len(sources))
    md = md.rstrip() + "\n\n## Sources\n\n" + "\n".join(
        f"{i}. [{c.title or c.url}]({c.url})" for i, c in enumerate(sources, 1)) + "\n"

    billed = run.billed_equiv_tokens + plan_usage.billed_equiv + synth_usage.billed_equiv
    stats = {
        "subquestions": len(tasks),
        "verified_findings": sum(f.verified for f in findings),
        "verified_%": round(100 * sum(f.verified for f in findings) / max(1, len(tasks)), 1),
        "agents": run.agents,
        "worker_calls": run.calls,
        "web_searches": backend.searches,
        "sources": len(sources),
        "total_tokens": run.total_tokens + plan_usage.total + synth_usage.total,
        "billed_equiv_tokens": round(billed),
        "wall_time_s": round(loop.time() - t0, 2),
        "escalated": sum(1 for f in findings if f.effort != cfg.efforts[0]),
        **integrity,
    }
    return ResearchReport(question, md, findings, sources, stats)


# ----------------------------------------------------------------------------- Claude


def _usage(msg) -> tuple[CallUsage, int]:
    u = msg.usage
    stu = getattr(u, "server_tool_use", None)
    searches = int(getattr(stu, "web_search_requests", 0) or 0) if stu else 0
    return CallUsage(
        input_tokens=u.input_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        output_tokens=u.output_tokens or 0,
    ), searches


def _add(a: CallUsage, b: CallUsage) -> None:
    a.input_tokens += b.input_tokens
    a.cache_read_tokens += b.cache_read_tokens
    a.cache_write_tokens += b.cache_write_tokens
    a.output_tokens += b.output_tokens


class AnthropicResearchLLM:
    """Claude with server-side web search/fetch. No client tool loop: searches run on
    Anthropic's servers inside one request (resumed on `pause_turn`)."""

    def __init__(self, max_searches: int = 5, max_fetches: int = 3, use_fallbacks: bool = True,
                 timeout: float = 600.0, max_continuations: int = 4):
        import anthropic

        self._a = anthropic
        self.client = anthropic.AsyncAnthropic(max_retries=0, timeout=timeout)
        self.max_searches = max_searches
        self.max_fetches = max_fetches
        self.use_fallbacks = use_fallbacks
        self.max_continuations = max_continuations

    async def _create(self, **kw):
        a = self._a
        try:
            if self.use_fallbacks:
                kw["extra_body"] = {"fallbacks": "default"}
                return await self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], **kw)
            return await self.client.messages.create(**kw)
        except a.RateLimitError as e:
            raise classify_429(e.body, getattr(e.response, "headers", None)) from e
        except a.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientError(f"HTTP {e.status_code}") from e
            raise
        except (a.APIConnectionError, a.APITimeoutError) as e:
            raise TransientError(type(e).__name__) from e

    async def plan(self, question, max_subquestions, model, effort):
        msg = await self._create(
            model=model, max_tokens=4000,
            system=PLAN_SYSTEM + f" Never return more than {max_subquestions} sub-questions.",
            messages=[{"role": "user", "content": question}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
        )
        usage, _ = _usage(msg)
        if msg.stop_reason == "refusal":
            return [], usage
        text = next((b.text for b in msg.content if b.type == "text"), "{}")
        try:
            return json.loads(text).get("subquestions", []), usage
        except json.JSONDecodeError:
            return [], usage

    async def investigate(self, question, sub_question, model, effort):
        tools = [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": self.max_searches},
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": self.max_fetches,
             "citations": {"enabled": True}},
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content":
                                           f"Overall question (context only): {question}\n\n"
                                           f"Your sub-question: {sub_question}"}]
        usage, searches = CallUsage(), 0
        blocks: list[Any] = []  # everything the paused turn has produced so far
        for _ in range(self.max_continuations + 1):
            msg = await self._create(model=model, max_tokens=16000, system=WORKER_SYSTEM,
                                     messages=messages, tools=tools,
                                     output_config={"effort": effort})
            u, s = _usage(msg)
            _add(usage, u)
            searches += s
            blocks.extend(msg.content)
            if msg.stop_reason != "pause_turn":
                break
            # Resend the turn so far; the server resumes from the trailing server_tool_use.
            messages = messages[:1] + [{"role": "assistant", "content": list(blocks)}]
        if msg.stop_reason == "refusal":
            return "", [], usage, searches
        text_parts, cites = [], []
        for b in blocks:
            if b.type != "text":
                continue
            text_parts.append(b.text)
            for c in getattr(b, "citations", None) or []:
                url = getattr(c, "url", None)
                if url:
                    cites.append(Citation(url, getattr(c, "title", "") or "",
                                          (getattr(c, "cited_text", "") or "")[:300]))
        return "".join(text_parts).strip(), cites, usage, searches

    async def synthesize(self, question, findings_block, n_sources, model, effort):
        msg = await self._create(
            model=model, max_tokens=16000, system=SYNTH_SYSTEM,
            messages=[{"role": "user", "content": f"Question: {question}\n\nFindings:\n\n"
                       f"{findings_block}\n\nValid citation numbers: 1-{n_sources}."}],
            output_config={"effort": effort},
        )
        usage, _ = _usage(msg)
        text = "".join(b.text for b in msg.content if b.type == "text")
        return text or "_The synthesis step returned no text._", usage


# ----------------------------------------------------------------------------- offline mock


class MockResearchLLM:
    """Deterministic offline stand-in, for demos and tests (no network, no spend).

    Low effort sometimes finds only one source, which exercises verification and
    escalation; synthesis cites one non-existent source, which exercises the
    reference-integrity check.
    """

    DOMAINS = ["arxiv.org", "nature.com", "acm.org", "ieee.org", "nist.gov", "who.int",
               "oecd.org", "nber.org", "mit.edu", "stanford.edu"]

    def __init__(self, seed: int = 0, latency: float = 0.02, fail_low: bool = True,
                 throttle_first: int = 0):
        import random

        self.rng = random.Random(seed)
        self.latency = latency
        self.fail_low = fail_low
        self.throttle_left = throttle_first

    async def plan(self, question, max_subquestions, model, effort):
        aspects = ["current state and key numbers", "main approaches and how they compare",
                   "evidence for and against", "risks and limitations", "recent developments",
                   "practical recommendations"]
        n = min(max_subquestions, 4)
        return ([{"question": f"{question} - {a}", "difficulty": 1 + i % 3}
                 for i, a in enumerate(aspects[:n])], CallUsage(input_tokens=300, output_tokens=150))

    async def investigate(self, question, sub_question, model, effort):
        await asyncio.sleep(self.latency)
        if self.throttle_left > 0:
            self.throttle_left -= 1
            raise RateLimited("simulated 429")
        weak = self.fail_low and effort == "low" and self.rng.random() < 0.4
        k = 1 if weak else 3
        doms = self.rng.sample(self.DOMAINS, k)
        cites = [Citation(f"https://{d}/doc/{zlib.crc32((sub_question + d).encode()) % 10000}",
                          f"{d} source on {sub_question[-40:]}", "quoted evidence") for d in doms]
        text = (f"Finding for '{sub_question}'. " + "Specific evidence with figures and dates. " * 12)
        out = {"low": 400, "medium": 900, "high": 1600}.get(effort, 900)
        return text, cites, CallUsage(input_tokens=6000, output_tokens=out), 2 + k

    async def synthesize(self, question, findings_block, n_sources, model, effort):
        cited = " ".join(f"[{i}]" for i in range(1, n_sources + 1))
        md = (f"# {question}\n\nDirect answer drawn from the verified findings {cited}.\n\n"
              f"## Details\n\n" + "The findings agree on the main points and differ on details "
              "that the sources report differently, as noted below. " * 3 + "[1] [99]\n\n"
              "## Limitations\n\nSome findings rest on fewer sources than others.\n")
        return md, CallUsage(input_tokens=4000, output_tokens=1200)
