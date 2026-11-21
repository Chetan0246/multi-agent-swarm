"""Evaluate deep research against a single-agent baseline on questions with known answers.

For each question, both systems produce an answer:

* **swarm**: the full deep-research pipeline (plan → parallel cited workers → checks).
* **single**: one Claude call with web search on the whole question, the same model, and
  the same total search allowance as the swarm's workers (compute-matched on searches,
  as cost-aware evaluation practice recommends).

Scoring is deterministic: an answer is correct when any accepted alias appears in its
direct-answer part (the text before the first `## ` heading), so a report gets no credit for
merely listing the right answer among alternatives further down. Each result records
dollars, wall time and, for the swarm, verified % and claim support. Runs stop when the
total dollar cap is reached.
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .pricing import estimate_research, usage_usd
from .research import ResearchConfig, ResearchLLM, deep_research


@dataclass
class EvalItem:
    id: str
    question: str
    answers: list[str]


@dataclass
class EvalResult:
    item_id: str
    system: str
    correct: bool
    usd: float
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class _Meter:
    """Wraps a ResearchLLM and adds up the dollars of every call it makes.

    A deep-research run that fails partway (quota gone, a 4xx at synthesis) raises before it
    can report its cost, yet its earlier calls were billed. Metering the calls themselves
    lets the eval's total cap count that spend instead of treating it as $0.
    """

    def __init__(self, llm: ResearchLLM):
        self._llm = llm
        self.usd = 0.0

    def __getattr__(self, name: str) -> Any:  # worker_model, take_lost_usage, search_stats...
        return getattr(self._llm, name)

    def _price(self, model: str) -> str:
        return getattr(self._llm, "worker_model", None) or model

    async def plan(self, question, max_subquestions, model, effort):
        subs, usage = await self._llm.plan(question, max_subquestions, model, effort)
        self.usd += usage_usd(model, usage)
        return subs, usage

    async def investigate(self, question, sub_question, model, effort):
        text, cites, usage, searches = await self._llm.investigate(question, sub_question, model, effort)
        self.usd += usage_usd(self._price(model), usage, searches)
        return text, cites, usage, searches

    async def synthesize(self, question, findings_block, n_sources, model, effort):
        md, usage = await self._llm.synthesize(question, findings_block, n_sources, model, effort)
        self.usd += usage_usd(model, usage)
        return md, usage

    async def judge(self, claims, model, effort):
        verdicts, usage = await self._llm.judge(claims, model, effort)
        self.usd += usage_usd(model, usage)
        return verdicts, usage


def load_items(path: str | Path) -> list[EvalItem]:
    items = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(EvalItem(str(d["id"]), d["question"], [str(a) for a in d["answers"]]))
    return items


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"(?<=\d)[,   ](?=\d{3}\b)", "", s)  # 299,792,458 -> 299792458
    return re.sub(r"\s+", " ", s)


def direct_answer(markdown: str) -> str:
    """The part of a report before its first second-level heading (title line excluded)."""
    body = re.sub(r"^\s*#\s[^\n]*\n", "", markdown, count=1)
    return re.split(r"\n##\s", body, maxsplit=1)[0]


def is_correct(markdown: str, answers: list[str]) -> bool:
    """Whole-word match in the direct answer, so "Rust" doesn't match "trust" or "5700" "57000"."""
    head = _norm(direct_answer(markdown))
    return any(re.search(rf"(?<!\w){re.escape(_norm(a).strip())}(?!\w)", head) for a in answers)


async def single_agent(question: str, llm: ResearchLLM, cfg: ResearchConfig) -> tuple[str, float]:
    text, cites, usage, searches = await llm.investigate(question, question, cfg.model,
                                                        cfg.efforts[min(1, len(cfg.efforts) - 1)])
    urls = list(dict.fromkeys(c.url for c in cites))
    md = text + ("\n\n## Sources\n\n" + "\n".join(f"{i}. {u}" for i, u in enumerate(urls, 1)) if urls else "")
    return md, usage_usd(cfg.model, usage, searches)


async def run_eval(items: list[EvalItem], swarm_llm: ResearchLLM, single_llm: ResearchLLM | None,
                   cfg: ResearchConfig, max_total_usd: float | None = None,
                   concurrency: int = 2, progress=None) -> list[EvalResult]:
    results: list[EvalResult] = []
    spent = 0.0
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()

    async def one(item: EvalItem) -> None:
        nonlocal spent
        async with sem:
            if stop.is_set():
                return
            t0 = time.perf_counter()
            meter = _Meter(swarm_llm)  # per question, so concurrent questions don't mix
            try:
                rep = await deep_research(item.question, meter, cfg)
                s = rep.stats
                r = EvalResult(item.id, "swarm", is_correct(rep.markdown, item.answers), s["usd"],
                               time.perf_counter() - t0,
                               {k: s.get(k) for k in ("verified_%", "claim_support_%", "sources",
                                                      "web_searches", "agents", "injection_excluded")})
            except Exception as e:  # keep evaluating the rest; its spend still counts
                r = EvalResult(item.id, "swarm", False, round(meter.usd, 4), time.perf_counter() - t0,
                               error=repr(e)[:200])
            results.append(r)
            spent += r.usd
            if single_llm is not None:
                t0 = time.perf_counter()
                try:
                    md, usd = await single_agent(item.question, single_llm, cfg)
                    r2 = EvalResult(item.id, "single", is_correct(md, item.answers), usd,
                                    time.perf_counter() - t0)
                except Exception as e:
                    r2 = EvalResult(item.id, "single", False, 0.0, time.perf_counter() - t0,
                                    error=repr(e)[:200])
                results.append(r2)
                spent += r2.usd
            if progress:
                progress(item, results, spent)
            if max_total_usd is not None and spent >= max_total_usd:
                stop.set()

    await asyncio.gather(*(one(it) for it in items))
    return results


def summarize(results: list[EvalResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for system in sorted({r.system for r in results}):
        rs = [r for r in results if r.system == system]
        cs = [float(v) for r in rs if (v := r.extra.get("claim_support_%")) is not None]
        out[system] = {
            "questions": len(rs),
            "accuracy_%": round(100 * sum(r.correct for r in rs) / max(1, len(rs)), 1),
            "usd_total": round(sum(r.usd for r in rs), 3),
            "usd_per_question": round(statistics.mean(r.usd for r in rs), 4) if rs else 0.0,
            "usd_per_correct": round(sum(r.usd for r in rs) / max(1, sum(r.correct for r in rs)), 4),
            "seconds_median": round(statistics.median(r.seconds for r in rs), 2) if rs else 0.0,
            "claim_support_%": round(statistics.mean(cs), 1) if cs else None,
            "errors": sum(1 for r in rs if r.error),
        }
    return out


def estimate_eval(n_items: int, cfg: ResearchConfig, with_single: bool) -> tuple[float, float]:
    e = estimate_research(cfg.model, cfg.max_subquestions, cfg.max_searches, cfg.efforts)
    factor = 1.3 if with_single else 1.0  # single agent ≈ one worker call with a larger search budget
    return n_items * e.low_usd * factor, n_items * e.high_usd * factor


def to_markdown(summary: dict[str, dict[str, Any]], results: list[EvalResult]) -> str:
    cols = ["accuracy_%", "usd_per_question", "usd_per_correct", "seconds_median", "claim_support_%",
            "errors"]
    lines = ["| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for s, row in summary.items():
        lines.append(f"| {s} | " + " | ".join(str(row[c]) for c in cols) + " |")
    lines += ["", "| question | system | correct | usd | seconds |", "|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r.item_id, r.system)):
        lines.append(f"| {r.item_id} | {r.system} | {'yes' if r.correct else 'no'} | "
                     f"{r.usd:.3f} | {r.seconds:.1f} |")
    return "\n".join(lines) + "\n"


def results_json(results: list[EvalResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)
