"""The swarm engine.

A small pool of agents pulls work from a scheduler. Batches are formed lazily at
dispatch time, so every decision uses what the run has learned so far.

    dedupe/resume ─► pools[(min_rung, difficulty)] ─► scheduler ─► agent ─► verify
          ▲                       ▲                       │            │
     journal (WAL)        failed → next rung       policy picks    breaker / limiter /
                                                  (rung, batch)    hedger / budget

* Scheduler: priority first; prefers pools that can fill the batch the policy wants,
  and dispatches a partial batch only when nothing full is ready (no idle agents).
* Policy: Thompson-sampled cost per success (see policy.py), or static v1 rules.
* Reliability: 429 → AIMD limiter; 5xx/timeouts → circuit breaker reroutes or waits;
  stragglers → hedged duplicate; budget → hard ceiling; WAL → crash-safe resume.
  None of these failure modes spend one of a task's escalation attempts.
"""
from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .backends import Backend, BatchResponse, RateLimited, SpendCapReached, TransientError
from .limiter import AdaptiveLimiter
from .planner import PlanConfig, plan_agents
from .policy import AdaptivePolicy, StaticPolicy
from .reliability import CircuitBreaker, Hedger, Journal, TokenBudget
from .task import Result, Task, _norm


@dataclass
class _TState:
    attempts: int = 0
    throttles: int = 0
    transients: int = 0
    tokens: float = 0.0
    best: str | None = None
    best_conf: float = -1.0
    last_rung: int = 0
    seen: list[str] = field(default_factory=list)  # normalised answers, for agreement


@dataclass
class _Batch:
    rung: int
    arm: int  # batch size the policy chose (for learning)
    min_rung: int
    difficulty: int
    tasks: list[Task]


@dataclass
class _Wait:
    seconds: float


@dataclass
class RunReport:
    results: dict[str, Result]
    wall_time: float
    agents: int
    calls: int
    total_tokens: int
    billed_equiv_tokens: float
    latencies: list[float] = field(default_factory=list)
    per_rung_success: dict[str, int] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        return 100.0 * sum(r.ok for r in self.results.values()) / max(1, self.n)

    @property
    def tokens_per_task(self) -> float:
        return self.total_tokens / max(1, self.n)

    @property
    def billed_per_task(self) -> float:
        return self.billed_equiv_tokens / max(1, self.n)

    def pct(self, q: float) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[min(len(s) - 1, int(q * len(s)))]

    def summary(self) -> dict:
        return {
            "tasks": self.n,
            "success_%": round(self.success_rate, 2),
            "wall_time_s": round(self.wall_time, 3),
            "agents": self.agents,
            "llm_calls": self.calls,
            "tokens_per_task": round(self.tokens_per_task, 1),
            "cost_per_task": round(self.billed_per_task, 1),
            "throughput_tasks_per_s": round(self.n / max(1e-9, self.wall_time), 1),
            "call_p50_s": round(self.pct(0.5), 3),
            "call_p99_s": round(self.pct(0.99), 3),
            **self.counters,
            "solved_at": self.per_rung_success,
        }


class Swarm:
    def __init__(
        self,
        backend: Backend,
        cfg: PlanConfig | None = None,
        agents: int | None = None,
        journal_path: str | None = None,
        policy_path: str | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.backend = backend
        self.cfg = cfg or PlanConfig()
        self.fixed_agents = agents
        self.journal_path = journal_path
        self.policy_path = policy_path
        self.on_event = on_event
        self.fatal: BaseException | None = None  # first fatal error of the last run, if any
        c = self.cfg
        self.policy: AdaptivePolicy | StaticPolicy
        if c.adaptive:
            self.policy = AdaptivePolicy(c.ladder, max_batch=c.max_batch, seed=c.seed)
            if policy_path:
                self.policy.load(policy_path)
        else:
            self.policy = StaticPolicy(c.ladder, max_batch=c.static_batch)

    def _emit(self, kind: str, **data) -> None:
        if self.on_event:
            self.on_event(kind, data)

    # ------------------------------------------------------------------ public
    def run(self, tasks: list[Task]) -> RunReport:
        return asyncio.run(self.arun(tasks))

    async def arun(self, tasks: list[Task]) -> RunReport:
        cfg, ladder, policy = self.cfg, self.cfg.ladder, self.policy
        t0 = time.perf_counter()
        self.fatal = None
        journal = Journal(self.journal_path)
        results: dict[str, Result] = {}
        counters = dict(throttles=0, transient_errors=0, hedges=0, hedge_wins=0,
                        breaker_trips=0, budget_stops=0, resumed=0, cache_hits=0)

        # 1. resume + dedupe + content cache --------------------------------
        groups: dict[str, list[Task]] = {}
        for t in tasks:
            rec = journal.done.get(t.id)
            # Resume only if the journalled task had the same prompt: a reused state dir with a
            # different workload must not hand back another task's answer under the same id.
            if rec is not None and rec.get("ok") and rec.get("fp") == t.fingerprint:
                results[t.id] = Result(**{k: rec[k] for k in Result.__dataclass_fields__ if k in rec})
                counters["resumed"] += 1
                continue
            groups.setdefault(t.fingerprint, []).append(t)
        todo: list[Task] = []
        for fp, members in groups.items():
            ans = journal.by_fp.get(fp)
            if ans is not None and all(m.check(ans) for m in members if m.has_checker):
                for m in members:
                    results[m.id] = Result(m.id, ans, True, 0, "cache", 0, cached=True)
                counters["cache_hits"] += len(members)
            else:
                todo.append(members[0])

        # 2. pools + sizing ---------------------------------------------------
        seq = itertools.count()
        shuffler = random.Random(cfg.seed)
        pools: dict[tuple[int, int], list] = {}

        def push(t: Task, min_rung: int) -> None:
            heapq.heappush(pools.setdefault((min_rung, t.difficulty), []), (t.priority, next(seq), t))

        for t in todo:
            push(t, 0)

        est_batches = 0
        for (mr, d), heap in pools.items():
            _, b = policy.choose(d, mr, len(heap))
            est_batches += math.ceil(len(heap) / b)
        if self.fixed_agents:
            n_agents = self.fixed_agents
        elif getattr(self.backend, "bulk", False):
            n_agents = max(1, min(cfg.bulk_max_agents, est_batches))
        else:
            n_agents = plan_agents(est_batches, cfg.max_agents, cfg.knee_slack)
        hedging = cfg.hedging and not getattr(self.backend, "bulk", False)

        # Headroom above the agent count is only reachable by hedged duplicates.
        hedge_slots = max(1, n_agents // 8) if hedging else 0
        limiter = AdaptiveLimiter(start=n_agents + hedge_slots, ceiling=n_agents + hedge_slots)
        breaker = CircuitBreaker(cfg.breaker_threshold, cfg.breaker_cooldown)
        hedger = Hedger()
        budget = TokenBudget(cfg.token_budget)
        state: dict[str, _TState] = {t.id: _TState() for t in todo}
        tot: dict[str, Any] = {"calls": 0, "tokens": 0, "billed": 0.0, "outstanding": len(todo)}
        latencies: list[float] = []
        per_rung: dict[str, int] = {}
        observed_cost: dict[tuple[int, int], float] = {}  # EMA of billed tokens per task
        changed = asyncio.Event()
        self._emit("plan", tasks=len(tasks), unique=len(todo), agents=n_agents, est_batches=est_batches)

        # 3. scheduling -------------------------------------------------------
        def allowed_rungs() -> set[int]:
            return {i for i, r in enumerate(ladder) if breaker.available(r.model)}

        def next_item():
            live = [(k, h) for k, h in pools.items() if h]
            if not live:
                return None
            ok_rungs = allowed_rungs()
            best = None
            for (mr, d), heap in live:
                cand = ok_rungs & set(range(mr, len(ladder)))
                if not cand:
                    continue
                r, b = policy.choose(d, mr, len(heap), allowed=cand)
                prio, s, _ = heap[0]
                key = (prio, 0 if len(heap) >= b else 1, -d, -mr, s)
                if best is None or key < best[0]:
                    best = (key, mr, d, r, b)
            if best is None:  # every usable model's breaker is open
                wait = min(breaker.retry_after(r.model) for r in ladder)
                return _Wait(wait)
            _, mr, d, r, b = best
            heap = pools[(mr, d)]
            out, tok = [], 0
            while heap and len(out) < b:
                t = heap[0][2]
                if out and tok + t.est_tokens > cfg.max_batch_input_tokens:
                    break
                heapq.heappop(heap)
                out.append(t)
                tok += t.est_tokens
            # Shuffle within the batch: answers can depend on their position (later items
            # condition on earlier ones), so a retried task should not keep the same slot.
            shuffler.shuffle(out)
            return _Batch(r, b, mr, d, out)

        # 4. finalisation -----------------------------------------------------
        def finalize(t: Task, ok: bool, err: str | None = None) -> None:
            st = state[t.id]
            rung = ladder[st.last_rung]
            for m in groups[t.fingerprint]:
                # Unchecked duplicates inherit the representative's verdict (same prompt).
                m_ok = ok and (m.check(st.best) if m.has_checker else True)
                r = Result(m.id, st.best, m_ok, st.attempts, rung.effort, round(st.tokens),
                           confidence=None if st.best_conf < 0 else st.best_conf,
                           error=None if m_ok else (err or "failed_validation"))
                results[m.id] = r
                journal.write({**r.to_dict(), "fp": t.fingerprint})
            if ok:
                key = f"{rung.model}:{rung.effort}"
                per_rung[key] = per_rung.get(key, 0) + 1
            tot["outstanding"] -= 1

        def drain(err: str) -> None:
            """Finalize everything still queued (fatal condition such as a spend cap)."""
            for heap in pools.values():
                while heap:
                    finalize(heapq.heappop(heap)[2], False, err)

        def requeue(batch: _Batch) -> None:
            for t in batch.tasks:
                push(t, batch.min_rung)

        # 5. calls with hedging ----------------------------------------------
        async def one(batch: _Batch) -> tuple[BatchResponse, float]:
            r = ladder[batch.rung]
            async with limiter:
                s = time.perf_counter()
                resp = await self.backend.run_batch(batch.tasks, r.effort, r.model)
                return resp, time.perf_counter() - s

        async def call(batch: _Batch) -> tuple[BatchResponse, float, float]:
            """Returns (response, latency, extra billed tokens spent on a cancelled hedge)."""
            primary = asyncio.create_task(one(batch))
            delay = hedger.delay() if hedging else None
            hedger.calls += 1
            if delay is None:
                resp, lat = await primary
                hedger.record(lat)
                return resp, lat, 0.0
            done, _ = await asyncio.wait({primary}, timeout=delay)
            if done or not hedger.may_hedge() or not limiter.has_free_slot():
                resp, lat = await primary
                hedger.record(lat)
                return resp, lat, 0.0
            hedger.hedges += 1
            counters["hedges"] += 1
            secondary = asyncio.create_task(one(batch))
            pending = {primary, secondary}
            first_exc: BaseException | None = None
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for d_ in done:
                    if d_.exception() is None:
                        for p in pending:
                            p.cancel()
                        resp, lat = d_.result()
                        if d_ is secondary:
                            hedger.wins += 1
                            counters["hedge_wins"] += 1
                            lat += delay  # caller-observed latency includes the wait
                        hedger.record(lat)
                        # The cancelled twin was billed for (at least) its input tokens.
                        u = resp.usage
                        extra = u.price_mult * (u.input_tokens + 0.1 * u.cache_read_tokens
                                                + 1.25 * u.cache_write_tokens) if pending else 0.0
                        return resp, lat, extra
                    first_exc = first_exc or d_.exception()
            assert first_exc is not None
            raise first_exc  # both attempts failed

        # 6. agents -----------------------------------------------------------
        async def agent(idx: int) -> None:
            while True:
                item = next_item()
                if item is None:
                    if tot["outstanding"] <= 0:
                        changed.set()
                        return
                    changed.clear()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(changed.wait(), timeout=0.25)
                    continue
                if isinstance(item, _Wait):
                    await asyncio.sleep(item.seconds)
                    continue

                batch = item
                model = ladder[batch.rung].model
                n = len(batch.tasks)
                seen = observed_cost.get((batch.difficulty, batch.rung))
                per = seen if seen is not None else policy.per_task_cost(batch.difficulty, batch.rung, n)
                est = per * n * 1.5  # reserve with a safety margin; settle to actual after
                if not budget.try_reserve(est):
                    counters["budget_stops"] += 1
                    for t in batch.tasks:
                        finalize(t, False, "budget_exhausted")
                    changed.set()
                    continue

                breaker.acquire(model)
                try:
                    resp, lat, extra = await call(batch)
                except SpendCapReached:
                    breaker.release(model)
                    budget.settle(est, 0.0)
                    counters["spend_cap"] = 1
                    for t in batch.tasks:
                        finalize(t, False, "spend_cap")
                    drain("spend_cap")
                    changed.set()
                    continue
                except RateLimited as rl:
                    breaker.release(model)
                    budget.settle(est, 0.0)
                    if rl.kind == "ramp":
                        limiter.on_ramp()
                    else:
                        limiter.on_throttle()
                    counters["throttles"] += 1
                    worst = 0
                    for t in batch.tasks:
                        state[t.id].throttles += 1
                        worst = max(worst, state[t.id].throttles)
                    if worst > cfg.max_throttle_retries:
                        for t in batch.tasks:
                            finalize(t, False, "rate_limited")
                    else:
                        backoff = min(8.0, 0.05 * 2 ** worst)
                        await asyncio.sleep(max(backoff, rl.retry_after or 0.0))
                        requeue(batch)
                    changed.set()
                    continue
                except TransientError as e:
                    budget.settle(est, 0.0)
                    before = breaker.trips
                    breaker.failure(model)
                    counters["breaker_trips"] += breaker.trips - before
                    counters["transient_errors"] += 1
                    self._emit("transient", model=model, error=str(e))
                    worst = 0
                    for t in batch.tasks:
                        state[t.id].transients += 1
                        worst = max(worst, state[t.id].transients)
                    if worst > cfg.max_transient_retries:
                        for t in batch.tasks:
                            finalize(t, False, f"transient: {e}")
                    else:
                        await asyncio.sleep(min(4.0, 0.02 * 2 ** worst))
                        requeue(batch)
                    changed.set()
                    continue
                except Exception as e:  # non-retryable (4xx, bugs): fail fast, keep going
                    breaker.release(model)
                    budget.settle(est, 0.0)
                    if getattr(e, "fatal", False):  # bad key, quota gone: no task can succeed
                        self.fatal = self.fatal or e
                        counters["fatal_errors"] = counters.get("fatal_errors", 0) + 1
                        for t in batch.tasks:
                            finalize(t, False, f"fatal: {type(e).__name__}")
                        drain(f"fatal: {type(e).__name__}")
                        changed.set()
                        continue
                    for t in batch.tasks:
                        finalize(t, False, f"{type(e).__name__}: {e}")
                    changed.set()
                    continue

                breaker.success(model)
                limiter.on_success()
                billed = resp.usage.billed_equiv + extra
                budget.settle(est, billed)
                tot["calls"] += 1
                tot["tokens"] += resp.usage.total
                tot["billed"] += billed
                latencies.append(lat)
                conf = resp.confidence or {}
                wins = 0
                for t in batch.tasks:
                    st = state[t.id]
                    st.attempts += 1
                    st.last_rung = batch.rung
                    st.tokens += billed / n
                    ans = resp.answers.get(t.id)
                    c = conf.get(t.id, 1.0 if t.has_checker else 0.0)
                    if ans is not None and c >= st.best_conf:  # keep the most confident answer
                        st.best, st.best_conf = ans, c
                    if t.has_checker:
                        accepted = t.check(ans)
                    elif cfg.unchecked_mode == "agreement":
                        norm = _norm(ans) if ans else None
                        accepted = norm is not None and norm in st.seen
                        if norm:
                            st.seen.append(norm)
                            if accepted:
                                st.best_conf = max(st.best_conf, c)
                    else:
                        accepted = t.check(ans, c, cfg.min_confidence)
                    if accepted:
                        st.best, st.best_conf = ans, c
                        wins += 1
                        finalize(t, True)
                    elif batch.rung + 1 < len(ladder):
                        push(t, batch.rung + 1)
                    elif st.attempts < len(ladder) + cfg.top_rung_retries and not resp.refused:
                        push(t, batch.rung)  # retry in place at the top rung
                    else:
                        err = ("refused" if resp.refused else "truncated" if resp.truncated
                               else "failed_validation" if t.has_checker
                               else "no_agreement" if cfg.unchecked_mode == "agreement"
                               else "low_confidence")
                        finalize(t, False, err)
                policy.observe(batch.difficulty, batch.rung, batch.arm, n, wins, billed)
                k = (batch.difficulty, batch.rung)
                observed_cost[k] = billed / n if k not in observed_cost else (
                    0.8 * observed_cost[k] + 0.2 * billed / n)
                self._emit("batch", rung=batch.rung, size=n, ok=wins, latency=round(lat, 4),
                           billed=round(billed, 1))
                changed.set()

        try:
            if todo:
                await asyncio.gather(*(agent(i) for i in range(n_agents)))
        finally:
            journal.close()
            if self.policy_path:
                policy.save(self.policy_path)

        counters["breaker_trips"] = breaker.trips
        return RunReport(
            results=results,
            wall_time=time.perf_counter() - t0,
            agents=n_agents if todo else 0,
            calls=int(tot["calls"]),
            total_tokens=int(tot["tokens"]),
            billed_equiv_tokens=tot["billed"],
            latencies=latencies,
            per_rung_success=per_rung,
            counters=counters,
        )


def jsonl_event_sink(path: str) -> Callable[[str, dict], None]:
    fh = open(path, "a", encoding="utf-8")

    def sink(kind: str, data: dict) -> None:
        fh.write(json.dumps({"t": round(time.time(), 3), "event": kind, **data}) + "\n")
        fh.flush()

    return sink
