"""LLM backends — all expose  `async run_batch(tasks, effort, model) -> BatchResponse`.

* `AnthropicBackend`       real-time Messages API, one call answers a whole batch
* `AnthropicBatchBackend`  Message Batches API: coalesces calls, 50% price, minutes-hours latency
* `MockBackend`            seeded simulator with stragglers, outages, rate limits and
                           effort/batch-size-dependent accuracy, for free benchmarking
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .task import Task

SYSTEM_PROMPT = (
    "You are a worker in a task-processing swarm. You receive a JSON list of "
    "tasks, each with an `id` and a `prompt`. Solve every task independently. "
    "For each task give only the final answer — no explanation, no restating the "
    "question, no markdown; if a task asks for a single value, give just that value. "
    "Also give `confidence`: your probability (0 to 1) that the answer is correct. "
    "Be calibrated: use low values when unsure. Return one entry per task id."
)

CODE_SYSTEM_PROMPT = (
    "You are a worker in a task-processing swarm. You receive a JSON list of coding tasks, "
    "each with an `id` and a `prompt`. Solve every task independently. For each task the "
    "`answer` is only the complete, runnable source code (Python unless the task names another "
    "language): no markdown fences, no explanation, no example usage, no tests unless asked. "
    "Keep the exact function and class names the task asks for. Use only the standard library "
    "unless the task says otherwise. Also give `confidence`: your probability (0 to 1) that "
    "the code is correct. Return one entry per task id."
)

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "answer", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


class RateLimited(Exception):
    """Provider throttled us (429). Handled by the AIMD limiter; costs no attempt.

    `kind` is "rate" (plain limit: back off multiplicatively) or "ramp" (usage grew too
    fast: hold the current rate, then add one step at a time, as the provider advises).
    `retry_after` is the server's `retry-after` header in seconds, when present.
    """

    def __init__(self, msg: str = "", kind: str = "rate", retry_after: float | None = None):
        super().__init__(msg)
        self.kind = kind
        self.retry_after = retry_after


class SpendCapReached(Exception):
    """429 `enforced_spend_limit_reached`: retries can never succeed. The run stops cleanly."""


def classify_429(body: Any, headers: Any) -> Exception:
    """Map an Anthropic 429 to the right handling (see docs on rate-limit error codes)."""
    code = None
    if isinstance(body, dict):
        err = body.get("error", body)
        details = err.get("details") if isinstance(err, dict) else None
        if isinstance(details, dict):
            code = details.get("error_code")
    if code == "enforced_spend_limit_reached":
        return SpendCapReached("organization spend limit reached")
    ra = None
    try:
        ra = float(headers.get("retry-after")) if headers is not None and headers.get("retry-after") else None
    except (TypeError, ValueError):
        ra = None
    return RateLimited(str(code or "rate_limited"), kind="ramp" if code == "ramp_limit_exceeded" else "rate",
                       retry_after=ra)


class TransientError(Exception):
    """5xx / overloaded / connection error. Counted by the circuit breaker; costs no attempt."""


@dataclass
class CallUsage:
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    price_mult: float = 1.0  # 0.5 on the Batches API
    extra_billed: float = 0.0  # non-token fees (e.g. web searches) in input-token equivalents

    @property
    def billed_equiv(self) -> float:
        """Cost in input-token units: cache reads ~0.1x, cache writes ~1.25x.

        Output tokens cost 5x input on current Claude models, so they are weighted 5x.
        """
        return self.price_mult * (
            self.input_tokens + 0.1 * self.cache_read_tokens
            + 1.25 * self.cache_write_tokens + 5 * self.output_tokens
        ) + self.extra_billed

    @property
    def total(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens + self.output_tokens


@dataclass
class BatchResponse:
    answers: dict[str, str]
    usage: CallUsage
    confidence: dict[str, float] | None = None
    refused: bool = False
    truncated: bool = False


class Backend(Protocol):
    bulk: bool  # True → high-latency bulk API: engine disables hedging, raises concurrency

    async def run_batch(self, tasks: list[Task], effort: str, model: str) -> BatchResponse: ...


# ---------------------------------------------------------------- request building
def build_user_message(tasks: list[Task]) -> str:
    return json.dumps([{"id": t.id, "prompt": t.prompt} for t in tasks],
                      separators=(",", ":"), ensure_ascii=False)


def build_system(shared_context: str | None, system: str = SYSTEM_PROMPT) -> list[dict]:
    """Stable prefix first, cache breakpoint on the last stable block.

    A short system prompt alone is below the model's minimum cacheable prefix and
    silently won't cache; `shared_context` (a document, schema, rubric or few-shot
    examples common to every task) is what makes caching pay off at scale.
    """
    blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
    if shared_context:
        blocks.append({"type": "text", "text": "Shared context for all tasks:\n" + shared_context})
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def build_params(tasks: list[Task], effort: str, model: str, shared_context: str | None,
                 max_tokens_per_task: int, system: str = SYSTEM_PROMPT) -> dict:
    return dict(
        model=model,
        max_tokens=min(16000, 1024 + max_tokens_per_task * len(tasks)),
        system=build_system(shared_context, system),
        messages=[{"role": "user", "content": build_user_message(tasks)}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": RESULT_SCHEMA}},
    )


def parse_message(msg, price_mult: float = 1.0) -> BatchResponse:
    u = msg.usage
    usage = CallUsage(
        input_tokens=u.input_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        output_tokens=u.output_tokens or 0,
        price_mult=price_mult,
    )
    if msg.stop_reason == "refusal":
        return BatchResponse({}, usage, refused=True)
    if msg.stop_reason == "max_tokens":
        return BatchResponse({}, usage, truncated=True)  # policy learns this arm is too big
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        rows = json.loads(text).get("results", [])
        answers = {str(r["id"]): str(r["answer"]) for r in rows}
        conf = {str(r["id"]): float(r.get("confidence", 0.5)) for r in rows}
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
        answers, conf = {}, {}
    return BatchResponse(answers, usage, conf)


# ---------------------------------------------------------------- real-time API
class AnthropicBackend:
    bulk = False

    def __init__(self, shared_context: str | None = None, max_tokens_per_task: int = 400,
                 use_fallbacks: bool = True, timeout: float = 120.0, system: str = SYSTEM_PROMPT):
        import anthropic

        self._a = anthropic
        self.client = anthropic.AsyncAnthropic(max_retries=0, timeout=timeout)  # engine owns retries
        self.shared_context = shared_context
        self.max_tokens_per_task = max_tokens_per_task
        self.use_fallbacks = use_fallbacks
        self.system = system

    async def aclose(self) -> None:
        await self.client.close()

    async def run_batch(self, tasks: list[Task], effort: str, model: str) -> BatchResponse:
        a = self._a
        kw = build_params(tasks, effort, model, self.shared_context, self.max_tokens_per_task, self.system)
        try:
            if self.use_fallbacks:
                msg = await self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
            else:
                msg = await self.client.messages.create(**kw)
        except a.RateLimitError as e:
            raise classify_429(e.body, getattr(e.response, "headers", None)) from e
        except a.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientError(f"HTTP {e.status_code}") from e
            raise
        except (a.APIConnectionError, a.APITimeoutError) as e:
            raise TransientError(type(e).__name__) from e
        return parse_message(msg)


# ---------------------------------------------------------------- Batches API
class AnthropicBatchBackend:
    """Coalesces concurrent `run_batch` calls into Message Batches (50% price).

    Each engine call becomes one request inside a shared batch; a background poller
    resolves every caller's future when the batch ends. Use for offline workloads
    where hours of latency are fine. (`fallbacks` is not accepted by the Batches API.)
    """

    bulk = True

    def __init__(self, shared_context: str | None = None, max_tokens_per_task: int = 400,
                 window_s: float = 5.0, max_requests: int = 5000, poll_s: float = 30.0,
                 system: str = SYSTEM_PROMPT):
        import anthropic

        self._a = anthropic
        self.client = anthropic.AsyncAnthropic()
        self.system = system
        self.shared_context = shared_context
        self.max_tokens_per_task = max_tokens_per_task
        self.window_s = window_s
        self.max_requests = max_requests
        self.poll_s = poll_s
        self._pending: list[tuple[str, dict, asyncio.Future]] = []
        self._timer: asyncio.Task | None = None
        self._seq = 0
        self._bg: set[asyncio.Task] = set()

    async def run_batch(self, tasks: list[Task], effort: str, model: str) -> BatchResponse:
        fut = asyncio.get_running_loop().create_future()
        self._seq += 1
        params = build_params(tasks, effort, model, self.shared_context, self.max_tokens_per_task, self.system)
        self._pending.append((f"r{self._seq}", params, fut))
        if len(self._pending) >= self.max_requests:
            self._flush()
        elif self._timer is None:
            self._timer = asyncio.create_task(self._flush_later())
        return await fut

    async def _flush_later(self) -> None:
        await asyncio.sleep(self.window_s)
        self._timer = None
        self._flush()

    def _flush(self) -> None:
        reqs, self._pending = self._pending, []
        if reqs:
            t = asyncio.create_task(self._submit_and_poll(reqs))
            self._bg.add(t)
            t.add_done_callback(self._bg.discard)

    async def _submit_and_poll(self, reqs) -> None:
        futs = {cid: f for cid, _, f in reqs}
        try:
            batch = await self.client.messages.batches.create(
                requests=[{"custom_id": cid, "params": p} for cid, p, _ in reqs])
            while batch.processing_status != "ended":
                await asyncio.sleep(self.poll_s)
                batch = await self.client.messages.batches.retrieve(batch.id)
            async for item in await self.client.messages.batches.results(batch.id):
                f = futs.pop(item.custom_id, None)
                if f is None or f.done():
                    continue
                res: Any = item.result  # union of succeeded/errored/canceled/expired
                kind = res.type
                if kind == "succeeded":
                    f.set_result(parse_message(res.message, price_mult=0.5))
                elif kind == "errored" and res.error.error.type == "invalid_request_error":
                    f.set_exception(ValueError(f"invalid_request in batch {batch.id}"))
                else:  # server error, expired, canceled: safe to resubmit
                    f.set_exception(TransientError(f"batch result {kind}"))
        except Exception as e:
            err = TransientError(f"batch submit/poll failed: {e}")
            for f in futs.values():
                if not f.done():
                    f.set_exception(err)
        finally:
            for f in futs.values():
                if not f.done():
                    f.set_exception(TransientError("missing from batch results"))


# ---------------------------------------------------------------- simulator
EFFORT_OUT_MULT = {"low": 1.0, "medium": 2.2, "high": 4.0, "xhigh": 6.0, "max": 9.0}
EFFORT_SKILL = {"low": 0.0, "medium": 0.10, "high": 0.17, "xhigh": 0.20, "max": 0.22}


class MockBackend:
    """Seeded model of an LLM endpoint with the trade-offs that matter for scheduling.

    * bigger batches amortise the system prompt but dilute attention (more errors)
    * higher effort is more accurate but multiplies output (thinking) tokens
    * confidence is informative but imperfect (overlapping distributions)
    * heavy latency tail: `straggler_p` of calls run `straggler_x` times slower
    * outage window: from `outage_at` to `outage_at + outage_for` seconds after the
      first call, `outage_fail_p` of calls get 5xx (time-based, like a real incident)
    * `dilution`: accuracy lost per extra task in a batch (model/workload property)
    * concurrency above `rate_limit_concurrency` gets 429s
    """

    bulk = False
    SYSTEM_TOKENS = 1200

    def __init__(self, seed: int = 7, base_latency: float = 0.05, sec_per_out_token: float = 0.00015,
                 rate_limit_concurrency: int = 12, cache: bool = True, straggler_p: float = 0.04,
                 straggler_x: float = 10.0, outage_at: float | None = None,
                 outage_for: float = 0.0, outage_fail_p: float = 0.9, dilution: float = 0.004):
        self.rng = random.Random(seed)
        self.base_latency = base_latency
        self.sec_per_out_token = sec_per_out_token
        self.rate_limit_concurrency = rate_limit_concurrency
        self.cache = cache
        self.straggler_p = straggler_p
        self.straggler_x = straggler_x
        self.outage_at = outage_at
        self.outage_for = outage_for
        self.outage_fail_p = outage_fail_p
        self.dilution = dilution
        self._t0: float | None = None
        self._inflight = 0
        self._warm: set[str] = set()
        self.calls = 0  # successful calls
        self.attempts = 0  # every call that reached the "server"

    def p_correct(self, t: Task, effort: str, batch_size: int) -> float:
        base = {1: 0.95, 2: 0.82, 3: 0.62}.get(t.difficulty, 0.8)
        p = base + EFFORT_SKILL[effort] * (t.difficulty / 2)
        p -= self.dilution * max(0, batch_size - 1) * t.difficulty
        return max(0.0, min(0.995, p))

    async def run_batch(self, tasks: list[Task], effort: str, model: str) -> BatchResponse:
        self._inflight += 1
        try:
            if self._inflight > self.rate_limit_concurrency:
                await asyncio.sleep(0.005)
                raise RateLimited("simulated 429")
            self.attempts += 1
            now = time.monotonic()
            self._t0 = self._t0 or now
            if (self.outage_at is not None
                    and self.outage_at <= now - self._t0 < self.outage_at + self.outage_for
                    and self.rng.random() < self.outage_fail_p):
                await asyncio.sleep(0.01)
                raise TransientError("simulated 503")
            usage = CallUsage()
            if self.cache and model in self._warm:
                usage.cache_read_tokens = self.SYSTEM_TOKENS
            elif self.cache:
                usage.cache_write_tokens = self.SYSTEM_TOKENS
                self._warm.add(model)
            else:
                usage.input_tokens = self.SYSTEM_TOKENS
            usage.input_tokens += sum(t.est_tokens + 12 for t in tasks)

            answers, conf = {}, {}
            out = 20
            for t in tasks:
                out += 15 + int(40 * t.difficulty * (EFFORT_OUT_MULT[effort] - 1))
                if self.rng.random() < self.p_correct(t, effort, len(tasks)):
                    answers[t.id] = t.expected if t.expected is not None else "ok"
                    conf[t.id] = round(self.rng.uniform(0.55, 1.0), 2)
                else:
                    answers[t.id] = "WRONG"
                    conf[t.id] = round(self.rng.uniform(0.1, 0.75), 2)
            usage.output_tokens = out
            lat = (self.base_latency + out * self.sec_per_out_token) * (0.9 + 0.2 * self.rng.random())
            if self.rng.random() < self.straggler_p:
                lat *= self.straggler_x
            await asyncio.sleep(lat)
            self.calls += 1
            return BatchResponse(answers, usage, conf)
        finally:
            self._inflight -= 1
