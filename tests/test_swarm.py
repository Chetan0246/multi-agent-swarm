import asyncio
import json

import httpx

from swarm.backends import (
    AnthropicBackend,
    AnthropicBatchBackend,
    BatchResponse,
    CallUsage,
    MockBackend,
    RateLimited,
    SpendCapReached,
    TransientError,
    classify_429,
)
from swarm.limiter import AdaptiveLimiter
from swarm.orchestrator import Swarm
from swarm.planner import PlanConfig, Rung, plan_agents
from swarm.policy import AdaptivePolicy, _BayesLogit
from swarm.reliability import CircuitBreaker, Hedger, Journal, TokenBudget
from swarm.task import Task
from swarm.workloads import mixed_workload

LADDER = (Rung("m", "low"), Rung("m", "medium"), Rung("m", "high"))


def T(i, d=1, exp="x", prio=0):
    return Task(f"t{i}", f"prompt {i}", d, expected=exp, priority=prio)


class Echo:
    """Answers correctly; records every call."""

    bulk = False

    def __init__(self, delay=0.0):
        self.calls = []
        self.delay = delay

    async def run_batch(self, tasks, effort, model):
        self.calls.append((model, effort, [t.id for t in tasks]))
        await asyncio.sleep(self.delay)
        return BatchResponse({t.id: t.expected for t in tasks}, CallUsage(input_tokens=10, output_tokens=2),
                             {t.id: 0.9 for t in tasks})


# ---------------------------------------------------------------- planning / policy
def test_plan_agents_knee():
    assert plan_agents(0, 8) == 1
    assert plan_agents(2, 8) == 3
    assert plan_agents(1000, 8) == 7  # within 15% of 8 agents' makespan
    assert plan_agents(1000, 8, knee_slack=0.0) == 8
    assert plan_agents(46, 8) < 8


def test_bayes_logit_learns_curve():
    m = _BayesLogit(0.0, 0.0)
    for _ in range(30):
        m.update(1, 10, 9)   # 90% at b=1
        m.update(17, 10, 3)  # 30% at b=17
    assert abs(m.mean_p(1) - 0.9) < 0.06
    assert abs(m.mean_p(17) - 0.3) < 0.08
    assert m.mean_p(9) < m.mean_p(1)


def test_policy_prefers_cheap_rung_and_learns_batch_size(tmp_path):
    pol = AdaptivePolicy(LADDER, max_batch=24, explore=False)
    r, b = pol.choose(1, 0, 100)
    assert r == 0 and b >= 8  # easy tasks: cheapest effort, big batches
    # Evidence that big batches fail badly for difficulty 3 → policy shrinks them.
    for _ in range(20):
        pol.observe(3, 0, 16, 16, 2, 2000)
        pol.observe(3, 0, 2, 2, 2, 400)
    r, b = pol.choose(3, 0, 100)
    assert b <= 4
    pol.save(tmp_path / "p.json")
    pol2 = AdaptivePolicy(LADDER, explore=False).load(tmp_path / "p.json")
    assert pol2.choose(3, 0, 100) == (r, b)


def test_policy_respects_allowed_rungs():
    pol = AdaptivePolicy(LADDER, explore=False)
    assert pol.choose(1, 0, 10, allowed={2})[0] == 2


# ---------------------------------------------------------------- task
def test_task_checks():
    assert Task("a", "p", expected="Hello.").check(" hello ")
    assert not Task("a", "p", expected="1").check("2")
    assert Task("a", "p", pattern=r"^\d+$").check("42")
    assert Task("a", "p", validator=lambda s: s.startswith("y")).check("yes")
    assert not Task("a", "p").check(None)
    assert Task("a", "p").check("x", confidence=0.8, min_conf=0.6)
    assert not Task("a", "p").check("x", confidence=0.4, min_conf=0.6)


# ---------------------------------------------------------------- reliability primitives
def test_limiter_aimd():
    lim = AdaptiveLimiter(start=8)
    lim.on_throttle()
    assert lim.limit == 4
    for _ in range(100):
        lim.on_success()
    assert lim.limit == 8


def test_breaker_lifecycle():
    now = [0.0]
    br = CircuitBreaker(threshold=3, cooldown=1.0, clock=lambda: now[0])
    for _ in range(3):
        br.failure("a")
    assert not br.available("a") and br.trips == 1
    now[0] = 1.5
    assert br.available("a")
    br.acquire("a")  # claim the half-open probe
    assert not br.available("a")  # only one probe at a time
    br.failure("a")  # probe failed → open again
    assert not br.available("a")
    now[0] = 3.0
    br.acquire("a")
    br.success("a")
    assert br.available("a") and not br.is_open("a")


def test_hedger_and_budget():
    h = Hedger(warmup=3, max_fraction=0.1)
    assert h.delay() is None
    for x in (0.1, 0.1, 0.2):
        h.record(x)
    assert h.delay() >= 0.1
    h.calls, h.hedges = 10, 1
    assert not h.may_hedge()
    b = TokenBudget(100)
    assert b.try_reserve(60)
    assert not b.try_reserve(60)
    b.settle(60, 30)
    assert b.try_reserve(60)


def test_journal_roundtrip_and_torn_line(tmp_path):
    p = tmp_path / "j.jsonl"
    j = Journal(p)
    j.write({"task_id": "a", "fp": "f1", "answer": "1", "ok": True})
    j.close()
    with open(p, "a") as f:
        f.write('{"task_id": "b", "ans')  # simulated crash mid-write
    j2 = Journal(p)
    assert set(j2.done) == {"a"} and j2.by_fp == {"f1": "1"}


# ---------------------------------------------------------------- engine
def test_swarm_high_success_with_hostile_endpoint():
    tasks = mixed_workload(300, seed=3)
    be = MockBackend(seed=3, straggler_p=0.05, outage_at=0.05, outage_for=0.15)
    rep = Swarm(be, PlanConfig(breaker_cooldown=0.05)).run(tasks)
    assert set(rep.results) == {t.id for t in tasks}
    assert rep.success_rate >= 99
    assert rep.calls < len(tasks) / 3


def test_dedupe_runs_identical_prompts_once():
    tasks = [Task(f"t{i}", "same prompt", expected="ok") for i in range(20)]
    be = Echo()
    rep = Swarm(be).run(tasks)
    assert len(be.calls) == 1 and len(be.calls[0][2]) == 1
    assert rep.success_rate == 100


def test_resume_from_journal(tmp_path):
    j = str(tmp_path / "j.jsonl")
    tasks = [T(i) for i in range(30)]
    Swarm(Echo(), journal_path=j).run(tasks)
    be = Echo()
    rep = Swarm(be, journal_path=j).run(tasks)
    assert be.calls == [] and rep.counters["resumed"] == 30 and rep.success_rate == 100


class Flaky(Echo):
    def __init__(self, exc, n):
        super().__init__()
        self.exc, self.left = exc, n

    async def run_batch(self, tasks, effort, model):
        if self.left > 0:
            self.left -= 1
            await asyncio.sleep(0)
            raise self.exc("boom")
        return await super().run_batch(tasks, effort, model)


def test_rate_limits_and_transients_cost_no_attempts():
    for exc in (RateLimited, TransientError):
        tasks = [T(i) for i in range(30)]
        rep = Swarm(Flaky(exc, 4), PlanConfig(breaker_cooldown=0.01)).run(tasks)
        assert rep.success_rate == 100
        assert all(r.attempts == 1 for r in rep.results.values())


class Broken:
    bulk = False

    async def run_batch(self, tasks, effort, model):
        raise ValueError("400 bad request")


def test_hard_errors_terminate_and_report():
    rep = Swarm(Broken()).run([T(i) for i in range(10)])
    assert rep.success_rate == 0
    assert all("ValueError" in r.error for r in rep.results.values())


class WrongAtLow(Echo):
    async def run_batch(self, tasks, effort, model):
        resp = await super().run_batch(tasks, effort, model)
        if effort == "low":
            resp.answers = {k: "no" for k in resp.answers}
        return resp


def test_only_failures_escalate():
    rep = Swarm(WrongAtLow(), PlanConfig(ladder=LADDER, adaptive=False)).run([T(i) for i in range(25)])
    assert rep.success_rate == 100
    assert rep.per_rung_success == {"m:medium": 25}


class AlwaysWrong(Echo):
    async def run_batch(self, tasks, effort, model):
        resp = await super().run_batch(tasks, effort, model)
        resp.answers = {k: "no" for k in resp.answers}
        return resp


def test_top_rung_retries_then_fail():
    be = AlwaysWrong()
    cfg = PlanConfig(ladder=(Rung("m", "low"),), top_rung_retries=2, adaptive=False)
    rep = Swarm(be, cfg).run([T(i) for i in range(4)])
    assert rep.success_rate == 0
    assert all(r.attempts == 3 and r.error == "failed_validation" for r in rep.results.values())


def test_budget_stops_spending():
    be = Echo()
    cfg = PlanConfig(ladder=(Rung("m", "low"),), adaptive=False, static_batch=1, token_budget=600)
    rep = Swarm(be, cfg, agents=1).run([T(i) for i in range(40)])  # 20 billed tokens per call
    assert rep.counters["budget_stops"] > 0
    assert 15 < len(be.calls) < 40
    assert rep.billed_equiv_tokens <= 600
    assert any(r.error == "budget_exhausted" for r in rep.results.values())


class ModelADown(Echo):
    async def run_batch(self, tasks, effort, model):
        if model == "A":
            await asyncio.sleep(0)
            raise TransientError("503")
        return await super().run_batch(tasks, effort, model)


def test_breaker_reroutes_to_healthy_model():
    cfg = PlanConfig(ladder=(Rung("A", "low"), Rung("B", "low")), breaker_threshold=2,
                     breaker_cooldown=60, adaptive=False)
    be = ModelADown()
    rep = Swarm(be, cfg).run([T(i) for i in range(60)])
    assert rep.success_rate == 100
    assert rep.counters["breaker_trips"] == 1
    # Only the calls already in flight when it tripped hit A; then everything went to B.
    assert rep.counters["transient_errors"] <= 2 + rep.agents
    assert sum(1 for c in be.calls if c[0] == "B") >= 1


class OneStraggler(Echo):
    async def run_batch(self, tasks, effort, model):
        self.calls.append(0)
        await asyncio.sleep(1.0 if len(self.calls) == 15 else 0.005)
        return BatchResponse({t.id: t.expected for t in tasks}, CallUsage(input_tokens=5))


def test_hedging_cuts_straggler():
    cfg = PlanConfig(ladder=(Rung("m", "low"),), adaptive=False, static_batch=1)
    rep = Swarm(OneStraggler(), cfg, agents=2).run([T(i) for i in range(30)])
    assert rep.counters["hedge_wins"] >= 1
    assert rep.wall_time < 0.8


def test_confidence_gates_unchecked_tasks():
    class Conf(Echo):
        async def run_batch(self, tasks, effort, model):
            c = 0.9 if effort == "medium" else 0.3
            return BatchResponse({t.id: "ans" for t in tasks}, CallUsage(input_tokens=5),
                                 {t.id: c for t in tasks})

    tasks = [Task(f"u{i}", f"open question {i}") for i in range(6)]
    cfg = PlanConfig(ladder=LADDER, adaptive=False, unchecked_mode="confidence", min_confidence=0.6)
    rep = Swarm(Conf(), cfg).run(tasks)
    assert rep.success_rate == 100
    assert all(r.effort == "medium" and r.confidence == 0.9 for r in rep.results.values())


def test_priority_dispatches_first():
    be = Echo()
    tasks = [T(i) for i in range(10)] + [T(99, prio=-1)]
    Swarm(be, PlanConfig(adaptive=False, static_batch=1), agents=1).run(tasks)
    assert be.calls[0][2] == ["t99"]


# ---------------------------------------------------------------- real backends (no network)
def _msg(results, usage=None):
    return {"id": "msg", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "stop_reason": "end_turn", "stop_sequence": None,
            "content": [{"type": "text", "text": json.dumps({"results": results})}],
            "usage": usage or {"input_tokens": 50, "output_tokens": 9,
                               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}


def test_anthropic_backend_request_shape():
    import anthropic

    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["beta"] = req.headers.get("anthropic-beta")
        return httpx.Response(200, json=_msg([{"id": "a", "answer": "42", "confidence": 0.8}]))

    be = AnthropicBackend(shared_context="DOC " * 50)
    be.client = anthropic.AsyncAnthropic(api_key="test", max_retries=0,
                                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = asyncio.run(be.run_batch([Task("a", "What is 17+25?")], "low", "claude-opus-5-5"))
    assert r.answers == {"a": "42"} and r.confidence == {"a": 0.8}
    b = seen["body"]
    assert b["fallbacks"] == "default" and "server-side-fallback" in seen["beta"]
    assert b["output_config"]["effort"] == "low"
    assert b["output_config"]["format"]["type"] == "json_schema"
    assert "cache_control" in b["system"][-1] and "DOC" in b["system"][-1]["text"]


def test_batch_api_backend_coalesces_and_resolves():
    import anthropic

    created = {}

    def handler(req):
        url = str(req.url)
        batch = {"id": "b1", "type": "message_batch", "processing_status": "ended",
                 "request_counts": {"processing": 0, "succeeded": 2, "errored": 0,
                                    "canceled": 0, "expired": 0},
                 "created_at": "2026-11-20T00:00:00Z", "expires_at": "2026-11-21T00:00:00Z",
                 "ended_at": "2026-11-20T00:01:00Z", "archived_at": None, "cancel_initiated_at": None,
                 "results_url": "https://api.anthropic.com/v1/messages/batches/b1/results"}
        if req.method == "POST":
            created["reqs"] = json.loads(req.content)["requests"]
            return httpx.Response(200, json=batch)
        if url.endswith("/results"):
            lines = []
            for r in created["reqs"]:
                ids = [x["id"] for x in json.loads(r["params"]["messages"][0]["content"])]
                lines.append(json.dumps({"custom_id": r["custom_id"], "result": {
                    "type": "succeeded",
                    "message": _msg([{"id": i, "answer": "ok", "confidence": 1} for i in ids])}}))
            return httpx.Response(200, content="\n".join(lines).encode())
        return httpx.Response(200, json=batch)

    be = AnthropicBatchBackend(window_s=0.01, poll_s=0.01)
    be.client = anthropic.AsyncAnthropic(api_key="test", max_retries=0,
                                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def go():
        return await asyncio.gather(be.run_batch([Task("a", "p1")], "low", "claude-opus-5-5"),
                                    be.run_batch([Task("b", "p2")], "low", "claude-opus-5-5"))

    r1, r2 = asyncio.run(go())
    assert len(created["reqs"]) == 2  # two engine calls → one Batches API submission
    assert "fallbacks" not in created["reqs"][0]["params"]
    assert r1.answers == {"a": "ok"} and r2.answers == {"b": "ok"}
    assert r1.usage.price_mult == 0.5


def test_agreement_gates_unchecked_tasks():
    class Drift(Echo):
        """Answers change at low effort, then stabilise: only agreeing answers are accepted."""

        async def run_batch(self, tasks, effort, model):
            self.calls.append(effort)
            ans = {"low": "a1", "medium": "b", "high": "b"}[effort]
            return BatchResponse({t.id: ans for t in tasks}, CallUsage(input_tokens=5),
                                 {t.id: 0.99 for t in tasks})

    tasks = [Task(f"u{i}", f"open question {i}") for i in range(5)]
    rep = Swarm(Drift(), PlanConfig(ladder=LADDER, adaptive=False)).run(tasks)
    assert rep.success_rate == 100
    assert all(r.answer == "b" and r.effort == "high" for r in rep.results.values())

    class Never(Echo):
        async def run_batch(self, tasks, effort, model):
            self.calls.append(1)
            return BatchResponse({t.id: f"x{len(self.calls)}" for t in tasks}, CallUsage(input_tokens=5))

    rep = Swarm(Never(), PlanConfig(ladder=LADDER, adaptive=False)).run(tasks[:2])
    assert rep.success_rate == 0 and all(r.error == "no_agreement" for r in rep.results.values())


def test_classify_429():
    spend = {"type": "error", "error": {"type": "rate_limit_error", "message": "spend",
                                        "details": {"error_code": "enforced_spend_limit_reached"}}}
    assert isinstance(classify_429(spend, {}), SpendCapReached)
    ramp = {"error": {"type": "rate_limit_error", "details": {"error_code": "ramp_limit_exceeded"}}}
    e = classify_429(ramp, {"retry-after": "3"})
    assert isinstance(e, RateLimited) and e.kind == "ramp" and e.retry_after == 3.0
    e = classify_429({"error": {"type": "rate_limit_error"}}, {})
    assert e.kind == "rate" and e.retry_after is None


def test_spend_cap_stops_the_run_cleanly():
    class Capped(Echo):
        async def run_batch(self, tasks, effort, model):
            if len(self.calls) >= 2:  # Echo records successful calls; the 3rd hits the cap
                raise SpendCapReached("cap")
            return await Echo.run_batch(self, tasks, effort, model)

    be = Capped()
    cfg = PlanConfig(ladder=(Rung("m", "low"),), adaptive=False, static_batch=2)
    rep = Swarm(be, cfg, agents=1).run([T(i) for i in range(20)])
    assert rep.counters["spend_cap"] == 1
    assert len(be.calls) == 2  # no retry storm after the cap
    assert sum(r.ok for r in rep.results.values()) == 4
    assert sum(r.error == "spend_cap" for r in rep.results.values()) == 16


def test_ramp_limit_holds_rate():
    lim = AdaptiveLimiter(start=8)
    lim.on_ramp()
    assert lim.limit == 7  # hold, don't halve
