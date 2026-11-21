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
from typing import Any

from .backends import MockBackend
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
    """Many independent tasks (text, code or web lookups), batched, checked and escalated."""
    import asyncio

    from .app import register_prices
    from .config import env_overrides, load_config
    from .pricing import PRICES, price_of, usd_to_billed_tokens
    from .tasks import build_backends, ladder, load_tasks, run_kinds

    file_cfg, _ = load_config(a.config)
    # [run] settings, falling back to [research] (provider, model, base_url, api_key_env, price)
    conf = {**file_cfg.get("research", {}), **file_cfg.get("run", {}), **env_overrides()}
    pick = lambda k, d=None: v if (v := getattr(a, k, None)) is not None else conf.get(k, d)  # noqa: E731
    model = str(pick("model", "claude-opus-5-5"))
    provider = str(pick("provider") or ("openai" if pick("base_url") or not model.startswith("claude")
                                        else "anthropic"))
    register_prices(file_cfg, [model], pick("price"))
    if not a.mock:
        price_of(model)  # unknown model: fail before any spend
    tasks = load_tasks(a.path, allow_code=a.allow_code_execution, code_timeout=a.code_timeout)
    kinds = {t.meta["kind"] for t in tasks}
    budget = a.budget
    if a.max_usd:
        budget = min(budget or float("inf"), usd_to_billed_tokens(model, a.max_usd))
    shared = Path(a.shared_context).read_text() if a.shared_context else None
    import os

    base_url = pick("base_url") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    backends = build_backends(
        kinds, provider, model, base_url=base_url, api_key_env=str(pick("api_key_env") or "OPENAI_API_KEY"),
        shared_context=shared, batch_api=a.batch_api, mock=a.mock, search=a.search,
        use_fallbacks=not a.no_fallbacks, cached_price_known=PRICES.get(model) is not None
        and PRICES[model].cache_read_mult < 1.0)
    cfg = PlanConfig(ladder=ladder(model, a.efforts), max_batch=a.max_batch, max_agents=a.max_agents,
                     adaptive=not a.static, hedging=not a.no_hedge, min_confidence=a.min_confidence,
                     token_budget=budget)
    state = Path(a.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    counts = ", ".join(f"{sum(t.meta['kind'] == k for t in tasks)} {k}" for k in sorted(kinds))
    print(f"{'mock' if a.mock else provider} {model}: {len(tasks)} tasks ({counts})"
          f"{f', cap ${a.max_usd:.2f}' if a.max_usd else ''}", file=sys.stderr)
    runs = asyncio.run(run_kinds(tasks, backends, cfg, state_dir=state, agents=a.agents,
                                 on_event=jsonl_event_sink(str(state / "events.jsonl")) if a.events else None))
    results = {tid: r for run in runs for tid, r in run.report.results.items()}
    out = open(a.out, "w") if a.out else sys.stdout
    for t in tasks:
        out.write(json.dumps({**results[t.id].to_dict(), "kind": t.meta["kind"]}, ensure_ascii=False) + "\n")
    if a.out:
        out.close()
    summary = {r.kind: {**r.report.summary(), **r.extra} for r in runs}
    if not a.mock:
        price = price_of(model)  # billed-equivalent tokens are in input-token units
        for r in runs:
            summary[r.kind]["usd_approx"] = round(r.report.billed_equiv_tokens * price.input / 1e6, 4)
    print(json.dumps(summary, indent=2), file=sys.stderr)


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
        check_claims=False if a.no_claim_check else None, use_cache=not a.no_cache, config_path=a.config,
        provider=a.provider, base_url=a.base_url, price=a.price, api_key_env=a.api_key_env)
    llm, cfg, info = resolve(opts)
    asyncio.run(aclose(llm))  # resolved only for the estimate; run_research builds its own
    est = estimate_research(cfg.model, cfg.max_subquestions, cfg.max_searches, cfg.efforts,
                            free_search=cfg.free_search)
    cap = f"cap ${cfg.max_usd:.2f}" if cfg.max_usd else "no cap"
    print(f"{info['provider']} {info['model']}, preset {info['preset']}, search {info['search']}, "
          f"estimate {est} ({cap})"
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
    claims = (f"claim support {s['claim_support_%']}%" if s.get("claim_support_%") is not None
              else "claim check off")
    missing = f", {s['unanswered_subquestions']} sub-question(s) unanswered" if s.get("unanswered_subquestions") else ""
    print(f"cost {cost_label(s)} (estimate {s['usd_estimate']}), {s['wall_time_s']}s, "
          f"{s['verified_%']}% findings verified{missing}, {claims}", file=sys.stderr)


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
    env_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    r_cfg = {**cfg.get("research", {}),
             **{k: v for k, v in (("provider", os.environ.get("SWARM_PROVIDER")), ("base_url", env_url)) if v}}
    if r_cfg.get("provider") == "openai" or r_cfg.get("base_url"):
        has_oa = bool(os.environ.get("OPENAI_API_KEY"))
        print(f"{ok(bool(r_cfg.get('base_url')))}OpenAI-compatible base URL: {r_cfg.get('base_url') or 'not set'}")
        print(f"{ok(has_oa)}OPENAI_API_KEY{'' if has_oa else ' missing'}")
    else:
        print("-- OpenAI-compatible provider: not configured (optional: --provider openai)")
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

    serve(host=a.host, port=a.port, config_path=a.config, embedded_worker=not a.no_worker, insecure=a.insecure)


def cmd_worker(a: argparse.Namespace) -> None:
    from .server import state_dir
    from .worker import run_worker

    run_worker(a.db or str(state_dir() / "jobs.sqlite"), a.concurrency, a.config, a.timeout)


def cmd_keys(a: argparse.Namespace) -> None:
    from .auth import create_key, list_keys, revoke_key
    from .jobs import JobDB
    from .server import state_dir

    db = JobDB(a.db or state_dir() / "jobs.sqlite")
    if a.action == "create":
        if not a.name:
            raise ValueError("keys create needs --name")
        kid, key = create_key(db, a.name, a.daily_usd, a.rpm, a.admin)
        print(key)
        print(f"key id {kid} for {a.name!r}: store it now, it is not shown again"
              f"{f'; daily budget ${a.daily_usd:.2f}' if a.daily_usd else ''}"
              f"{f'; {a.rpm} requests/min' if a.rpm else ''}{'; admin' if a.admin else ''}", file=sys.stderr)
    elif a.action == "list":
        import datetime

        for k in list_keys(db):
            created = datetime.datetime.fromtimestamp(k["created"]).strftime("%Y-%m-%d")
            state = "revoked" if k["revoked"] else "active"
            print(f"{k['id']}  {k['name']:<20} {state:<8} created {created}  "
                  f"daily ${k['daily_usd'] if k['daily_usd'] is not None else '-'}  rpm {k['rpm'] or '-'}"
                  f"{'  admin' if k['admin'] else ''}")
    else:
        if not a.id:
            raise ValueError("keys revoke needs a key id (see `swarm keys list`)")
        print("revoked" if revoke_key(db, a.id) else "no active key with that id")


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


# ----------------------------------------------------------------------------- workflows


def _provider_args(p: argparse.ArgumentParser, max_usd: float) -> None:
    p.add_argument("--model", default=None, help="default from swarm.toml, else claude-opus-5-5")
    p.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (implies --provider openai)")
    p.add_argument("--api-key-env", default=None, metavar="NAME", help="variable holding the provider key")
    p.add_argument("--price", default=None, metavar="IN,OUT[,CACHED]", help="USD per million tokens")
    p.add_argument("--max-usd", type=float, default=max_usd, help=f"hard dollar cap (default {max_usd})")
    p.add_argument("--agents", type=int, default=6, help="parallel workers")
    p.add_argument("--config", default=None)


def _chat(a: argparse.Namespace):
    from .workflows.chat import make_chat

    return make_chat(a.model, provider=a.provider, base_url=a.base_url, api_key_env=a.api_key_env,
                     price=a.price, max_usd=a.max_usd or None, config_path=a.config)


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _run_workflow(a: argparse.Namespace, coro_fn) -> object:
    import asyncio

    chat = _chat(a)
    _say(f"{type(chat).__name__.removesuffix('Chat').lower()} {chat.model}, cap "
         f"{f'${chat.max_usd:.2f}' if chat.max_usd else 'none'}")

    async def go():
        try:
            return await coro_fn(chat)
        finally:
            await chat.aclose()

    res = asyncio.run(go())
    if chat.stopped is not None:
        _say(f"note: stopped early: {str(chat.stopped).splitlines()[0]}")
    return res


def cmd_build(a: argparse.Namespace) -> None:
    from .workflows.build import build

    res: Any = _run_workflow(a, lambda chat: build(a.description, chat, Path(a.out), allow_execution=a.allow_execution,
                                                   max_files=a.max_files, fix_rounds=a.fix_rounds, agents=a.agents,
                                                   force=a.force, log=_say))
    status = {True: "tests pass", False: "tests FAIL", None: "not run"}[res.tests_passed]
    _say(f"wrote {len(res.files_written)} files to {res.project}; {status}; fix rounds {res.fix_rounds}; "
         f"${res.usd:.4f} over {res.calls} calls")
    for n in res.notes:
        if n != "setup changed" and not n.startswith("fixed:"):
            _say(f"note: {n}")
    if res.spec.get("start"):
        _say(f"start it with: cd {a.out} && {res.spec['start']}")


def cmd_write(a: argparse.Namespace) -> None:
    from .workflows.write import write

    res: Any = _run_workflow(a, lambda chat: write(a.brief, chat, words=a.words, max_sections=a.sections,
                                                   context=[Path(c) for c in a.context or []], agents=a.agents,
                                                   log=_say))
    _write_out(a.out, res.markdown)
    _say(f"{res.words} words in {res.sections} sections{', edited' if res.edited else ''}; "
         f"${res.usd:.4f} over {res.calls} calls")


def cmd_analyze(a: argparse.Namespace) -> None:
    from .workflows.analyze import analyze

    out = Path(a.out).parent if a.out else Path("analysis-report")
    res: Any = _run_workflow(a, lambda chat: analyze(Path(a.data), a.question, chat, out, max_questions=a.questions,
                                                     allow_code=a.allow_code_execution, agents=a.agents, log=_say))
    _write_out(a.out or str(out / "report.md"), res.markdown)
    _say(f"{len(res.findings)} analyses, {len(res.failed)} failed, {res.unverified_numbers} unverified numbers; "
         f"${res.usd:.4f} over {res.calls} calls")


def cmd_review(a: argparse.Namespace) -> None:
    from .workflows.review import review, to_json

    res: Any = _run_workflow(a, lambda chat: review(Path(a.path), chat, focus=a.focus or "",
                                                    max_files=a.max_files, agents=a.agents,
                                                    verify_findings=not a.no_verify, log=_say))
    _write_out(a.out, res.markdown)
    if a.out:
        Path(a.out).with_suffix(".json").write_text(json.dumps(to_json(res), indent=1))
    _say(f"{res.files_reviewed} files, {len(res.findings)} findings, {res.dropped_ungrounded} dropped as ungrounded, "
         f"{res.refuted} refuted on a second check; "
         f"${res.usd:.4f} over {res.calls} calls")


def cmd_doc(a: argparse.Namespace) -> None:
    from .workflows.doc import read_document, summarize, transform

    text = read_document(Path(a.file))
    if a.action == "translate":
        if not a.to:
            raise ValueError("translate needs --to LANGUAGE")
        res: Any = _run_workflow(a, lambda chat: transform(
            text, chat, f"translate it into {a.to}", f"their {a.to} translation", agents=a.agents, log=_say))
    elif a.action == "rewrite":
        if not a.style:
            raise ValueError('rewrite needs --style, e.g. --style "plain English for customers"')
        res = _run_workflow(a, lambda chat: transform(
            text, chat, f"rewrite it as: {a.style}", "how each is written in the new style", agents=a.agents,
            log=_say))
    else:
        res = _run_workflow(a, lambda chat: summarize(text, chat, words=a.words, agents=a.agents, log=_say))
    _write_out(a.out, res.text)
    _say(f"{res.chunks} chunks, {res.failed_chunks} failed; ${res.usd:.4f} over {res.calls} calls")


def _write_out(path: str | None, text: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        _say(f"wrote {path}")
    else:
        print(text)


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

    r = sub.add_parser("run", help="process many tasks from JSONL: text, code (tested) or web lookups (cited)")
    r.add_argument("path")
    r.add_argument("--mock", action="store_true", help="offline simulator instead of a model")
    r.add_argument("--batch-api", action="store_true", help="Message Batches API: 50%% price, slow (Claude)")
    r.add_argument("--model", default=None, help="default claude-opus-5-5; a non-Claude id selects --provider openai")
    r.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    r.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (implies --provider openai)")
    r.add_argument("--api-key-env", default=None, metavar="NAME", help="variable holding the provider key")
    r.add_argument("--price", default=None, metavar="IN,OUT[,CACHED]", help="USD per million tokens")
    r.add_argument("--max-usd", type=float, default=None, help="hard dollar cap for the run")
    r.add_argument("--allow-code-execution", action="store_true",
                   help="run model-written code against code tasks' tests (sandboxed subprocess)")
    r.add_argument("--code-timeout", type=float, default=10.0, help="seconds per test run")
    r.add_argument("--search", choices=["tinyfish", "mock"], default="tinyfish", help="search for web tasks")
    r.add_argument("--config", default=None)
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
    q.add_argument("--provider", choices=["anthropic", "openai"], default=None,
                   help="anthropic (Claude, default) or openai: any OpenAI-compatible API "
                        "(OPENAI_API_KEY, OPENAI_BASE_URL)")
    q.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. https://host/v1 "
                                                    "(implies --provider openai)")
    q.add_argument("--api-key-env", default=None, metavar="NAME",
                   help="environment variable holding the provider's key (default OPENAI_API_KEY), "
                        "e.g. RIVER_API_KEY")
    q.add_argument("--price", default=None, metavar="IN,OUT[,CACHED]",
                   help="USD per million tokens for a model with no built-in price")
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
    sv.add_argument("--no-worker", action="store_true", help="API only; run `swarm worker` processes for jobs")
    sv.add_argument("--insecure", action="store_true", help="allow a public interface without any API key")
    sv.set_defaults(fn=cmd_serve)

    wk = sub.add_parser("worker", help="run queued server jobs (scale these out beside `swarm serve --no-worker`)")
    wk.add_argument("--concurrency", type=int, default=2, help="jobs at once in this process")
    wk.add_argument("--timeout", type=float, default=1800.0, help="seconds before a job is stopped")
    wk.add_argument("--db", default=None, help="job database (default $SWARM_STATE_DIR/jobs.sqlite)")
    wk.add_argument("--config", default=None)
    wk.set_defaults(fn=cmd_worker)

    ks = sub.add_parser("keys", help="manage server API keys: create, list, revoke")
    ks.add_argument("action", choices=["create", "list", "revoke"])
    ks.add_argument("id", nargs="?", help="key id (revoke)")
    ks.add_argument("--name", default=None)
    ks.add_argument("--daily-usd", type=float, default=None, help="daily dollar budget")
    ks.add_argument("--rpm", type=int, default=None, help="requests per minute")
    ks.add_argument("--admin", action="store_true", help="sees all jobs and /metrics")
    ks.add_argument("--db", default=None)
    ks.set_defaults(fn=cmd_keys)

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

    b = sub.add_parser("build", help="build a new app from a description: plan, parallel files, test, fix")
    b.add_argument("description")
    b.add_argument("--out", required=True, help="new project folder")
    b.add_argument("--allow-execution", action="store_true",
                   help="install dependencies and run the tests (allowlisted tools, no shell, no API keys)")
    b.add_argument("--max-files", type=int, default=25)
    b.add_argument("--fix-rounds", type=int, default=3)
    b.add_argument("--force", action="store_true", help="write into a non-empty --out folder")
    _provider_args(b, 3.0)
    b.set_defaults(fn=cmd_build)

    w = sub.add_parser("write", help="long-form writing: outline, sections in parallel, edit pass")
    w.add_argument("brief")
    w.add_argument("--words", type=int, default=1500)
    w.add_argument("--sections", type=int, default=8)
    w.add_argument("--context", nargs="+", default=None, help="files with facts to use (notes, a research report)")
    w.add_argument("--out", default=None)
    _provider_args(w, 1.0)
    w.set_defaults(fn=cmd_write)

    an = sub.add_parser("analyze", help="data analysis: questions answered by code run on your CSV")
    an.add_argument("data")
    an.add_argument("question")
    an.add_argument("--questions", type=int, default=5, help="analyses to run in parallel")
    an.add_argument("--allow-code-execution", action="store_true", help="required: runs analysis code in the sandbox")
    an.add_argument("--out", default=None, help="report path (charts go in analysis/ next to it)")
    _provider_args(an, 1.0)
    an.set_defaults(fn=cmd_analyze)

    rv = sub.add_parser("review", help="code review: files in parallel, findings grounded in quoted code")
    rv.add_argument("path")
    rv.add_argument("--focus", default=None, help='e.g. "security" or "error handling"')
    rv.add_argument("--max-files", type=int, default=80)
    rv.add_argument("--no-verify", action="store_true",
                    help="skip the second check of each finding against the code it imports (faster, less precise)")
    rv.add_argument("--out", default=None, help="Markdown report (a .json with the findings is written too)")
    _provider_args(rv, 2.0)
    rv.set_defaults(fn=cmd_review)

    dc = sub.add_parser("doc", help="long documents in parallel chunks: translate, summarize, rewrite")
    dc.add_argument("action", choices=["translate", "summarize", "rewrite"])
    dc.add_argument("file")
    dc.add_argument("--to", default=None, help="target language (translate)")
    dc.add_argument("--style", default=None, help="target style (rewrite)")
    dc.add_argument("--words", type=int, default=400, help="summary length (summarize)")
    dc.add_argument("--out", default=None)
    _provider_args(dc, 2.0)
    dc.set_defaults(fn=cmd_doc)

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
