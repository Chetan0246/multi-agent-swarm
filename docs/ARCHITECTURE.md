# Architecture and design decisions

This document explains *why* the swarm is built the way it is. Each decision links the
evidence behind it. The full literature review is in
[`research/Multi agent swarms state of the art.md`](research/Multi%20agent%20swarms%20state%20of%20the%20art.md).

## The problem

There are thousands of independent tasks (classification, extraction, Q&A, checks), and each
one is small. The goal is to finish all of them with the **highest success rate**, the
**fewest tokens per task**, the **shortest wall time** and the **fewest agents**.

This is the regime where multi-agent parallelism is known to help. Controlled studies find
large gains on decomposable work but losses on sequential tasks and when a single agent is
already strong (Kim et al., *Towards a Science of Scaling Agent Systems*, arXiv 2512.08296,
2025–26). The swarm therefore does not use agents that talk to each other. Its agents are
**independent workers on independent tasks**, and all coordination is done by code.

## Pipeline

```
tasks ─► resume (WAL) ─► dedupe/cache ─► pools[(min_rung, difficulty)]
                                               │  lazy batching at dispatch time
                                               ▼
                          scheduler ── policy picks (effort rung, batch size)
                                               │
          ┌──────────── agent pool (knee-sized, AIMD-limited) ────────────┐
          │  budget reserve → breaker route → call (+hedge if straggling) │
          └───────────────────────────────┬───────────────────────────────┘
                                          ▼
                 verify: deterministic checker, or agreement of 2 answers
                   ok → journal      fail → next rung (or retry at top)
```

## Decisions

### 1. Code orchestrates, and agents never coordinate with each other
Agent-to-agent conversation is where most measured failures come from. In the MAST taxonomy,
inter-agent misalignment accounts for roughly a third of failures across 1,642 traces from
7 frameworks (Cemri et al., arXiv 2503.13657). Debate rarely beats simple voting at equal
compute (Choi et al., NeurIPS 2025). Kim et al. measure 4.4× error amplification for
centralized coordination against 17.2× for independent agents, with the orchestrator acting
as a checkpoint. Practice in 2026 has moved the same way, towards **code-driven fan-out**
(e.g. Claude Code dynamic workflows, Managed Agents multiagent workflows) and away from
free-form swarms. Here the "orchestrator" is a deterministic scheduler, not an LLM, so it
costs zero tokens.

### 2. A pull-based shared queue instead of synchronous fan-out
Anthropic reports that synchronous subagent waves are a bottleneck, because the lead waits
for the slowest worker (*How we built our multi-agent research system*, Jun 2025). Here,
workers **pull** the next batch as soon as they are free. Batches are formed lazily, so each
decision uses the latest learned statistics, and a partial batch goes out only when nothing
full is ready.

### 3. Fewest agents: the makespan knee
Coordination overhead grows with agent count (58–515% in Kim et al.), and so do rate-limit
pressure and 429s. `plan_agents` chooses the *smallest* pool whose estimated makespan is
within 15% of the fastest possible. The benchmark includes a single-agent baseline with the
same pipeline. It has the same token cost but takes 3–5× longer, which shows that the extra
agents buy wall time and nothing else.

### 4. Many tasks per call (batch prompting)
Batch prompting cuts tokens and calls substantially: up to 5× at 6 tasks per call (Cheng et
al. 2023), and 18–31% of the tokens at batch size 32 (BatchPrompt). For reasoning models it
has also been reported to cut thinking tokens (arXiv 2511.04108). Accuracy drops as batches
grow, and how fast depends on the model and the task. That is why batch size is **learned**,
not fixed (§5).

### 5. Learn batch size and effort online (Thompson sampling + DP)
For each (difficulty, effort rung), a Bayesian logistic model learns P(success | batch size),
and an online linear model learns tokens per call. A dynamic programme over the ladder
minimises **expected cost per solved task**, *including* the cost of escalating failures:

```
Q(r,b) = c(r,b) + (1 − p(r,b))·V(r+1)      V(r) = min(min_b Q(r,b), V(r+1))
```

Prior art includes cascades and routers (FrugalGPT, RouteLLM, AutoMix) and joint
model-and-batch-size selection under a budget (RoBatch, arXiv 2605.28268, offline and
greedy). No prior work was found that applies **online Thompson sampling to batch size**.
Honest result: once learned (warm), it matches hand-tuned static rules within noise in the
simulator. Its value is not needing hand-tuning on a real endpoint. Its cost is an
exploration premium on the first run, which is why the learned state is persisted.

### 6. Escalate effort instead of debating
Upgrading the model or effort beat doubling the token budget in Anthropic's analysis. Most
tasks finish at `low`; only failures move up, then get two more tries at the top rung.
Rate limits, 5xx errors and stragglers **never** consume one of a task's attempts. The
ladder stays on one model by default, which keeps one cache namespace and avoids drift
between models.

### 7. Verify deterministically, otherwise require agreement
MAST lists verification failures as a whole failure category. Adding a verification step
lifted ChatDev by 15.6% (Cemri et al.). Checks here are code (`expected`, regex, callable), so
no "judge" LLM is needed. For tasks with no checker, self-reported confidence is a weak
signal (AUROC ≈ 0.52–0.61 in calibration studies), whereas agreement between samples is
better supported (self-consistency; Adaptive-Consistency needs up to 7.9× fewer samples by
stopping at agreement). By default an unchecked answer is accepted once **two independent
attempts agree**.

### 8. Provider-aware reliability
| Mechanism | Evidence | Behaviour |
|---|---|---|
| AIMD limiter | Anthropic ramp-limit guidance: "increase by at most one step per minute" | Halve on a plain 429; **hold** on `ramp_limit_exceeded`; honour `retry-after` |
| Spend-cap stop | Anthropic: the `enforced_spend_limit_reached` 429 "keeps failing until access resumes" | Stop the run cleanly, with no retry storm |
| Circuit breaker | Standard SRE practice; 529 means overload across all users | Open after N 5xx errors; reroute to another rung's model or wait; single half-open probe |
| Hedged requests | *The Tail at Scale* (Dean & Barroso 2013): large tail cut for ~2% extra requests | Duplicate a call past 1.5 × p95, at most 8% of calls, using reserved limiter headroom |
| Token budget | Cost-aware evaluation (Kapoor et al. 2024; HAL) | Reserve before each call, settle after; leftover tasks end as `budget_exhausted` |
| Write-ahead journal | Durable execution (Temporal, DBOS, Restate) | Append-only and fsynced, tolerates a torn last line; resume skips finished tasks |

### 9. Pay less per token
- **Batches API**: 50% off, for workloads that can wait up to 24 h. Engine calls are coalesced into one batch.
- **Prompt caching**: shared context sits behind a cache breakpoint; reads cost 0.05–0.1× input depending on the model. Cache reads also don't count toward input rate limits. A short system prompt alone falls below the minimum cacheable prefix, which is why `--shared-context` exists.
- **Structured output** removes parse retries and verbose prose.
- Known caveat: changing top-level `effort` between requests can cause a separate cache entry per rung.

## Evaluation methodology
This follows the cost-aware evaluation literature (Kapoor et al. 2024; HAL, ICLR 2026):
- **Several seeds**, reporting means with 95% confidence intervals.
- **Cost and time reported together** (Pareto view), not accuracy alone.
- A **compute-matched single-agent baseline** and the previous version as baselines.
- A **simulator** with adversarial conditions (stragglers, a 5xx incident, rate limits) and three endpoint profiles. It is clearly labelled as simulated: it measures the scheduler's trade-offs, not Claude's accuracy.

## Gaps from the research report: status

| Gap the report raised | Status |
|---|---|
| Branch on 429 error codes (spend cap, ramp limit, `retry-after`) | **Done** (`classify_429`, `SpendCapReached`, `AdaptiveLimiter.on_ramp`) |
| Compute-matched single-worker baseline, seeds with CIs, Pareto plot | **Done** (`v2-1agent`, `ci95`, `docs/assets/pareto.svg`) |
| Self-reported confidence is poorly calibrated | **Done**: agreement is the default for unchecked tasks |
| Batch position effects | **Done**: item order is shuffled within every batch |
| Bandit reward should be cost per *verified* item | Already the case: the DP minimises cost per solved (verified) task |
| Hedging only on the synchronous path | Already the case: hedging is off for the Batches backend |
| Discount old evidence (non-stationarity) | Open |
| Per-message effort to keep the cache across rungs | Open (each call is a fresh one-shot request today) |
| OpenTelemetry GenAI spans | Open (a JSONL event log exists) |
| Test the verifiers themselves | Partly: unit tests cover the checkers |

## Known gaps
- No real-endpoint benchmark yet (it costs money; see `swarm run`).
- Agreement mode doubles the minimum calls for unchecked tasks. Semantic (not exact) agreement would help free-form answers.
- No idempotency key on Messages requests, so a crash mid-call can duplicate spend (the journal prevents duplicated *results*).
- The policy assumes difficulty labels; unlabeled workloads pool into one class.
