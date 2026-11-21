"""CLI.

  python -m swarm bench [--n 600 --seeds 3]   offline benchmark (simulated backend)
  python -m swarm run tasks.jsonl [options]   process a JSONL workload on Claude
  python -m swarm research "question" --out report.html  deep research with cited report
  python -m swarm doctor [--online]                     check setup, keys and config
  python -m swarm serve                                 web UI + REST API on localhost:8000
  python -m swarm eval [questions.jsonl] [--mock]       evaluate vs a single-agent baseline
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

    from .app import ResearchOptions, aclose, resolve, run_research
    from .export import cost_label, load_report, write_report
    from .pricing import estimate_research

    opts = ResearchOptions(
        question=a.question, preset=a.preset, search=a.search, mock=a.mock, docs=a.docs or [],
        docs_only=a.docs_only, include_domains=_csv(a.include_domains), exclude_domains=_csv(a.exclude_domains),
        recency_days=a.recency_days, kind="news" if a.news else "research_paper" if a.papers else None,
        language=a.language, model=a.model, worker_model=a.worker_model, max_usd=a.max_usd,
        max_subquestions=a.max_subquestions, max_agents=a.max_agents, max_searches=a.max_searches,
        check_claims=False if a.no_claim_check else None, use_cache=not a.no_cache, config_path=a.config)
    llm, cfg, info = resolve(opts)
    asyncio.run(aclose(llm))  # resolved only for the estimate; run_research builds its own
    est = estimate_research(cfg.model, cfg.max_subquestions, cfg.max_searches, cfg.efforts,
                            free_search=cfg.free_search)
    cap = f"cap ${cfg.max_usd:.2f}" if cfg.max_usd else "no cap"
    print(f"preset {info['preset']}, search {info['search']}, estimate {est} ({cap})"
          f"{' [mock: no real spend]' if a.mock else ''}", file=sys.stderr)
    if a.estimate:
        return
    prior = load_report(a.follow_up) if a.follow_up else None

    def progress(kind: str, data: dict) -> None:
        if kind == "plan":
            print(f"planned {data['unique']} sub-questions, {data['agents']} parallel workers", file=sys.stderr)
        elif kind == "batch":
            print(f"  worker done: effort rung {data['rung']}, verified {data['ok']}/{data['size']}, "
                  f"{data['latency']}s", file=sys.stderr)

    report = asyncio.run(run_research(opts, on_event=progress, prior=prior))
    if a.out:
        fmt = write_report(report, a.out, a.format)
        print(f"wrote {a.out} ({fmt})", file=sys.stderr)
        if fmt != "json":  # keep a JSON copy so the report can be followed up later
            json_path = str(Path(a.out).with_suffix(".json"))
            write_report(report, json_path, "json")
            print(f"wrote {json_path} (for --follow-up)", file=sys.stderr)
    else:
        print(report.markdown)
    if a.verbose:
        print(json.dumps(report.stats, indent=2), file=sys.stderr)
    s = report.stats
    print(f"cost {cost_label(s)} (estimate {s['usd_estimate']}), {s['wall_time_s']}s, "
          f"{s['verified_%']}% findings verified, claim support {s.get('claim_support_%', 'n/a')}%",
          file=sys.stderr)


def _csv(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def cmd_doctor(a: argparse.Namespace) -> None:
    import asyncio
    import importlib.util
    import os
    import platform

    from . import __version__
    from .config import cache_dir, load_config

    ok = lambda b: "ok " if b else "-- "  # noqa: E731
    print(f"multi-agent-swarm {__version__} on Python {platform.python_version()}")
    try:
        import anthropic
        print(f"{ok(True)}anthropic SDK {anthropic.__version__}")
    except ModuleNotFoundError:
        print(f"{ok(False)}anthropic SDK not installed (pip install multi-agent-swarm)")
    cfg, path = load_config(a.config)
    print(f"{ok(path is not None)}config file: {path or 'none (defaults)'}")
    has_claude = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))
    print(f"{ok(has_claude)}Anthropic credentials in environment"
          f"{'' if has_claude else ' (set ANTHROPIC_API_KEY, or rely on `ant auth login`)'}")
    has_tf = bool(os.environ.get("TINYFISH_API_KEY"))
    print(f"{ok(has_tf)}TINYFISH_API_KEY{'' if has_tf else ' missing: free key at https://agent.tinyfish.ai/api-keys'}")
    print(f"{ok(True)}cache dir: {cache_dir()}")
    for mod, why in [("pypdf", "PDF documents (--docs)"), ("fastapi", "web UI / API (swarm serve)"),
                     ("uvicorn", "web UI / API (swarm serve)")]:
        print(f"{ok(importlib.util.find_spec(mod) is not None)}{mod}: {why}")
    if a.online and has_tf:
        from .search.tinyfish import TinyFish

        async def probe():
            tf = TinyFish()
            try:
                hits = await tf.search("TinyFish search API")
                return f"TinyFish search works ({len(hits)} results)"
            finally:
                await tf.aclose()
        try:
            print(f"{ok(True)}{asyncio.run(probe())}")
        except Exception as e:
            print(f"{ok(False)}TinyFish search failed: {e}")


def cmd_serve(a: argparse.Namespace) -> None:
    from .server import serve

    serve(host=a.host, port=a.port, config_path=a.config)


def cmd_eval(a: argparse.Namespace) -> None:
    import asyncio

    from .evaluate import estimate_eval, load_items, results_json, run_eval, summarize, to_markdown
    from .research import AnthropicResearchLLM, MockResearchLLM, ResearchConfig, ResearchLLM

    items = load_items(a.path)[: a.limit or None]
    cfg = ResearchConfig(model=a.model, max_subquestions=a.max_subquestions,
                         max_searches=a.max_searches, max_usd=a.max_usd_per_question or None)
    with_single = not a.no_single
    lo, hi = estimate_eval(len(items), cfg, with_single)
    cap = f"cap ${a.max_usd_total:.2f}" if a.max_usd_total else "no total cap"
    print(f"{len(items)} questions, systems: swarm{' + single' if with_single else ''}; "
          f"estimate ~${lo:.2f}-${hi:.2f} ({cap}){' [mock: no real spend]' if a.mock else ''}",
          file=sys.stderr)
    if a.estimate:
        return
    swarm_llm: ResearchLLM
    single_llm: ResearchLLM | None
    if a.mock:
        # Offline plumbing check: the mock "knows" some answers so scoring is exercised.
        swarm_llm = MockResearchLLM(answers={it.question: it.answers[0]
                                             for i, it in enumerate(items) if i % 3})
        single_llm = MockResearchLLM(answers={it.question: it.answers[0] for i, it in enumerate(items)
                                              if i % 2}) if with_single else None
    else:
        swarm_llm = AnthropicResearchLLM(max_searches=a.max_searches)
        # Compute-matched on searches: the single agent may use as many as all workers together.
        single_llm = AnthropicResearchLLM(max_searches=a.max_searches * a.max_subquestions) \
            if with_single else None

    def progress(item, results, spent):
        done = {r.item_id for r in results}
        print(f"  [{len(done)}/{len(items)}] {item.id}: " + ", ".join(
            f"{r.system} {'ok' if r.correct else 'miss'}" for r in results if r.item_id == item.id)
            + f"  (spent ${spent:.2f})", file=sys.stderr)

    results = asyncio.run(run_eval(items, swarm_llm, single_llm, cfg, a.max_usd_total or None,
                                   a.concurrency, progress))
    summary = summarize(results)
    md = to_markdown(summary, results)
    print(md)
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "results": json.loads(results_json(results)),
                                           "mock": a.mock, "model": a.model}, indent=2))
        print(f"wrote {a.out}", file=sys.stderr)


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
    q.add_argument("--preset", choices=["quick", "standard", "deep"], default=None,
                   help="depth/cost preset (default standard)")
    q.add_argument("--search", choices=["tinyfish", "anthropic"], default=None,
                   help="tinyfish: free TinyFish search+fetch (default); anthropic: Claude server-side ($10/1k)")
    q.add_argument("--docs", nargs="+", metavar="PATH", help="your own files/folders as sources (txt, md, html, pdf)")
    q.add_argument("--docs-only", action="store_true", help="research only --docs, no web")
    q.add_argument("--include-domains", help="comma-separated domains to search only")
    q.add_argument("--exclude-domains", help="comma-separated domains to skip")
    q.add_argument("--recency-days", type=int, help="only results from the last N days")
    q.add_argument("--news", action="store_true", help="search news sources")
    q.add_argument("--papers", action="store_true", help="search research papers")
    q.add_argument("--language", help="report and search language, e.g. de, fr, ja")
    q.add_argument("--out", default=None, help="write the report here (.md, .html or .json)")
    q.add_argument("--format", choices=["md", "html", "json"], default=None, help="override format from --out")
    q.add_argument("--follow-up", metavar="REPORT.json", help="extend an earlier report instead of starting over")
    q.add_argument("--mock", action="store_true", help="offline demo, no API calls")
    q.add_argument("--model", default=None, help="model for planning/synthesis (default claude-opus-5-5)")
    q.add_argument("--worker-model", default=None, help="model for research workers (default: --model)")
    q.add_argument("--max-subquestions", type=int, default=None)
    q.add_argument("--max-agents", type=int, default=None)
    q.add_argument("--max-searches", type=int, default=None, help="searches per worker call")
    q.add_argument("--max-usd", type=float, default=None, help="hard dollar cap for the run (0 = none)")
    q.add_argument("--estimate", action="store_true", help="print the cost estimate and exit")
    q.add_argument("--no-claim-check", action="store_true", help="skip checking claims against sources")
    q.add_argument("--no-cache", action="store_true", help="bypass the local search/fetch cache")
    q.add_argument("--config", default=None, help="config file (default: swarm.toml or ~/.config/swarm)")
    q.add_argument("-v", "--verbose", action="store_true", help="print full run stats")
    q.set_defaults(fn=cmd_research)

    d = sub.add_parser("doctor", help="check installation, keys and configuration")
    d.add_argument("--online", action="store_true", help="also make one free TinyFish search")
    d.add_argument("--config", default=None)
    d.set_defaults(fn=cmd_doctor)

    sv = sub.add_parser("serve", help="run the web UI and REST API (needs swarm[server])")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--config", default=None)
    sv.set_defaults(fn=cmd_serve)

    e = sub.add_parser("eval", help="evaluate deep research vs a single agent on known-answer questions")
    e.add_argument("path", nargs="?", default="evals/questions.jsonl")
    e.add_argument("--mock", action="store_true", help="offline plumbing check, no API calls")
    e.add_argument("--model", default="claude-opus-5-5")
    e.add_argument("--limit", type=int, default=0, help="only the first N questions")
    e.add_argument("--max-subquestions", type=int, default=4)
    e.add_argument("--max-searches", type=int, default=5)
    e.add_argument("--max-usd-per-question", type=float, default=2.0)
    e.add_argument("--max-usd-total", type=float, default=10.0,
                   help="stop once this much is spent (0 = none)")
    e.add_argument("--concurrency", type=int, default=2, help="questions in flight at once")
    e.add_argument("--no-single", action="store_true", help="skip the single-agent baseline")
    e.add_argument("--estimate", action="store_true", help="print the cost estimate and exit")
    e.add_argument("--out", default=None, help="write summary and per-question results as JSON")
    e.set_defaults(fn=cmd_eval)

    g = sub.add_parser("gen", help="write a sample workload")
    g.add_argument("path")
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--seed", type=int, default=1)
    g.set_defaults(fn=cmd_gen)

    a = p.parse_args()
    from .errors import SwarmError
    from .research import OverBudget

    try:
        a.fn(a)
    except (SwarmError, OverBudget, ValueError) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit("interrupted")


if __name__ == "__main__":
    main()
