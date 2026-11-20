"""CLI.

  python -m swarm bench [--n 600 --seeds 3]   offline benchmark (simulated backend)
  python -m swarm run tasks.jsonl [options]   process a JSONL workload on Claude
  python -m swarm research "question" --out report.md   deep research with cited report
  python -m swarm gen out.jsonl --n 50        write a sample checkable workload
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

from .backends import AnthropicBackend, AnthropicBatchBackend, Backend, MockBackend
from .orchestrator import Swarm, jsonl_event_sink
from .planner import PlanConfig, Rung
from .task import Task
from .workloads import mixed_workload

KEYS = ["success_%", "wall_time_s", "agents", "llm_calls", "tokens_per_task", "cost_per_task",
        "call_p99_s", "hedges", "breaker_trips", "transient_errors", "throttles"]


PROFILES = {
    # name: (dilution per extra task in a batch, description)
    "v1-tuned": 0.004,   # the endpoint v1's hand-set batch rules were tuned on
    "long-ctx": 0.0008,  # big batches nearly free: static rules leave savings unused
    "fragile": 0.012,    # big batches hurt: static rules lose accuracy
}


def _mock(seed: int, dilution: float = 0.004, **kw) -> MockBackend:
    # A hostile-but-realistic endpoint: 4% stragglers at 10x latency and a 0.4 s 5xx incident.
    return MockBackend(seed=seed, base_latency=0.03, sec_per_out_token=0.0001, straggler_p=0.04,
                       straggler_x=10.0, outage_at=0.25, outage_for=0.4, dilution=dilution, **kw)


def _scenarios(tasks: list[Task], seed: int, dil: float, policy_file: str) -> dict[str, dict]:
    out = {}
    # naive: one task per call, high effort, 32 agents, no cache/dedupe/hedging/breaker
    uniq = [Task(t.id, f"{t.prompt}\u200b{i}", t.difficulty, expected=t.expected)
            for i, t in enumerate(tasks)]
    cfg = PlanConfig(ladder=(Rung("claude-opus-5-5", "high"),), adaptive=False, static_batch=1,
                     hedging=False, breaker_threshold=10**9, max_transient_retries=50, seed=seed)
    out["naive"] = Swarm(_mock(seed, dil, cache=False), cfg, agents=32).run(uniq).summary()

    # v1: static batching + effort escalation (the previous version of this project)
    cfg = PlanConfig(adaptive=False, static_batch=12, hedging=False, breaker_threshold=10**9,
                     max_transient_retries=50, seed=seed)
    out["v1-static"] = Swarm(_mock(seed, dil), cfg).run(tasks).summary()

    # v2 cold: adaptive policy + hedging + breaker, starting from priors only
    out["v2-cold"] = Swarm(_mock(seed, dil), PlanConfig(seed=seed)).run(tasks).summary()

    # Single-agent baseline with the same pipeline and token budget, as recommended by
    # cost-matched evaluation practice: isolates what the extra agents buy (wall time).
    out["v2-1agent"] = Swarm(_mock(seed, dil), PlanConfig(seed=seed), agents=1).run(tasks).summary()

    # v2 warm: policy state learned on two *different* workloads from the same endpoint
    for k in (1, 2):
        Swarm(_mock(seed + 100 * k, dil), PlanConfig(seed=seed + k), policy_path=policy_file).run(
            mixed_workload(len(tasks), seed=seed + 100 * k, dup_frac=0))
    out["v2-warm"] = Swarm(_mock(seed, dil), PlanConfig(seed=seed),
                           policy_path=policy_file).run(tasks).summary()
    return out


def cmd_bench(a: argparse.Namespace) -> None:
    names = ["naive", "v1-static", "v2-1agent", "v2-cold", "v2-warm"]
    w = max(map(len, KEYS)) + 2
    export: dict = {"n": a.n, "seeds": a.seeds, "profiles": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for prof, dil in PROFILES.items():
            if a.profile and prof != a.profile:
                continue
            runs: dict[str, list[dict]] = {}
            for s in range(a.seeds):
                tasks = mixed_workload(a.n, seed=a.seed + s, dup_frac=a.dup)
                pf = os.path.join(tmp, f"{prof}{s}.json")
                for name, summ in _scenarios(tasks, a.seed + s, dil, pf).items():
                    runs.setdefault(name, []).append(summ)
            mean = {n: {k: statistics.mean(r[k] for r in runs[n]) for k in KEYS} for n in names}
            print(f"\n== endpoint profile '{prof}' (dilution {dil}) - {a.n} tasks x {a.seeds} seeds, mean ==")
            print("metric".ljust(w) + "".join(n.rjust(11) for n in names))
            for k in KEYS:
                print(k.ljust(w) + "".join(f"{round(mean[n][k], 3):>11g}" for n in names))
            n, v1, v2 = mean["naive"], mean["v1-static"], mean["v2-warm"]
            print(f"v2-warm vs naive: {n['cost_per_task'] / v2['cost_per_task']:.1f}x cheaper, "
                  f"{n['wall_time_s'] / v2['wall_time_s']:.1f}x faster, "
                  f"{n['agents'] / v2['agents']:.1f}x fewer agents | vs v1: "
                  f"{v1['cost_per_task'] / v2['cost_per_task']:.2f}x cheaper, "
                  f"{v1['wall_time_s'] / v2['wall_time_s']:.2f}x faster, "
                  f"success {v1['success_%']:.2f}% -> {v2['success_%']:.2f}%")
            sd = {n: {k: statistics.stdev(r[k] for r in runs[n]) if a.seeds > 1 else 0.0 for k in KEYS}
                  for n in names}
            # 95% CI half-width of the mean (normal approx.; few seeds → treat as indicative)
            ci = {n: {k: 1.96 * sd[n][k] / math.sqrt(a.seeds) for k in KEYS} for n in names}
            export["profiles"][prof] = {"dilution": dil, "mean": mean, "stdev": sd, "ci95": ci,
                                        "runs": runs}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(export, indent=2))
        print(f"\nwrote {a.out}")


def cmd_run(a: argparse.Namespace) -> None:
    with open(a.path) as f:
        tasks = [Task.from_dict(json.loads(line)) for line in f if line.strip()]
    shared = Path(a.shared_context).read_text() if a.shared_context else None
    backend: Backend
    if a.mock:
        backend = _mock(1)
    elif a.batch_api:
        backend = AnthropicBatchBackend(shared_context=shared)
    else:
        backend = AnthropicBackend(shared_context=shared, use_fallbacks=not a.no_fallbacks)
    cfg = PlanConfig(
        ladder=tuple(Rung(a.model, e) for e in a.efforts.split(",")),
        max_batch=a.max_batch, max_agents=a.max_agents, adaptive=not a.static,
        hedging=not a.no_hedge, min_confidence=a.min_confidence, token_budget=a.budget,
    )
    state = Path(a.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    swarm = Swarm(backend, cfg, agents=a.agents,
                  journal_path=str(state / "journal.jsonl"),
                  policy_path=str(state / "policy.json"),
                  on_event=jsonl_event_sink(str(state / "events.jsonl")) if a.events else None)
    report = swarm.run(tasks)
    out = open(a.out, "w") if a.out else sys.stdout
    for t in tasks:
        out.write(report.results[t.id].to_json() + "\n")
    if a.out:
        out.close()
    print(json.dumps(report.summary(), indent=2), file=sys.stderr)


def cmd_research(a: argparse.Namespace) -> None:
    import asyncio

    from .research import AnthropicResearchLLM, MockResearchLLM, ResearchConfig, ResearchLLM, deep_research

    llm: ResearchLLM = MockResearchLLM() if a.mock else AnthropicResearchLLM(
        max_searches=a.max_searches, use_fallbacks=not a.no_fallbacks)
    cfg = ResearchConfig(model=a.model, efforts=tuple(a.efforts.split(",")),
                         max_subquestions=a.max_subquestions, max_agents=a.max_agents,
                         min_citations=a.min_citations, min_domains=a.min_domains, token_budget=a.budget)

    def progress(kind: str, data: dict) -> None:
        if kind == "plan":
            print(f"planned {data['unique']} sub-questions, {data['agents']} parallel workers",
                  file=sys.stderr)
        elif kind == "batch":
            print(f"  worker done: effort rung {data['rung']}, verified {data['ok']}/{data['size']}, "
                  f"{data['latency']}s", file=sys.stderr)

    report = asyncio.run(deep_research(a.question, llm, cfg, on_event=progress))
    if a.out:
        Path(a.out).write_text(report.markdown)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(report.markdown)
    print(json.dumps(report.stats, indent=2), file=sys.stderr)


def cmd_gen(a: argparse.Namespace) -> None:
    with open(a.path, "w") as f:
        for t in mixed_workload(a.n, seed=a.seed):
            f.write(json.dumps({"id": t.id, "prompt": t.prompt, "difficulty": t.difficulty,
                                "expected": t.expected}) + "\n")
    print(f"wrote {a.n} tasks to {a.path}")


def main() -> None:
    p = argparse.ArgumentParser(prog="swarm")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="offline benchmark on the simulator")
    b.add_argument("--n", type=int, default=500)
    b.add_argument("--seeds", type=int, default=2)
    b.add_argument("--profile", choices=list(PROFILES), default=None)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--dup", type=float, default=0.0, help="fraction of duplicate prompts")
    b.add_argument("--out", default=None, help="write means, stdevs and raw runs as JSON")
    b.set_defaults(fn=cmd_bench)

    r = sub.add_parser("run", help="process a JSONL workload")
    r.add_argument("path")
    r.add_argument("--mock", action="store_true", help="offline simulator instead of Claude")
    r.add_argument("--batch-api", action="store_true", help="Message Batches API: 50%% price, slow")
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--efforts", default="low,medium,high", help="escalation ladder")
    r.add_argument("--max-batch", type=int, default=24)
    r.add_argument("--max-agents", type=int, default=8)
    r.add_argument("--agents", type=int, default=None, help="override auto-sizing")
    r.add_argument("--static", action="store_true", help="disable the adaptive policy")
    r.add_argument("--no-hedge", action="store_true")
    r.add_argument("--min-confidence", type=float, default=0.6)
    r.add_argument("--budget", type=float, default=None, help="max billed-equivalent tokens")
    r.add_argument("--shared-context", default=None, help="file with context common to all tasks")
    r.add_argument("--state-dir", default=".swarm", help="journal, learned policy, events")
    r.add_argument("--events", action="store_true", help="write events.jsonl to the state dir")
    r.add_argument("--out", default=None)
    r.add_argument("--no-fallbacks", action="store_true")
    r.set_defaults(fn=cmd_run)

    q = sub.add_parser("research", help="deep research: parallel cited web research -> report")
    q.add_argument("question")
    q.add_argument("--out", default=None, help="write the Markdown report here")
    q.add_argument("--mock", action="store_true", help="offline demo, no API calls")
    q.add_argument("--model", default="claude-opus-5-5")
    q.add_argument("--efforts", default="low,medium,high", help="worker escalation ladder")
    q.add_argument("--max-subquestions", type=int, default=6)
    q.add_argument("--max-agents", type=int, default=6)
    q.add_argument("--max-searches", type=int, default=5, help="web searches per worker call")
    q.add_argument("--min-citations", type=int, default=2)
    q.add_argument("--min-domains", type=int, default=2)
    q.add_argument("--budget", type=float, default=None, help="max billed-equivalent tokens for workers")
    q.add_argument("--no-fallbacks", action="store_true")
    q.set_defaults(fn=cmd_research)

    g = sub.add_parser("gen", help="write a sample workload")
    g.add_argument("path")
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--seed", type=int, default=1)
    g.set_defaults(fn=cmd_gen)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
