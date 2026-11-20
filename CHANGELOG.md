# Changelog

## 2.3.0 — launch blockers: SDK 2.x, cost, claim checking, injection defences, evaluation
- **SDK 2.x:** moved to `anthropic>=2.12,<3`, which sends API version `2026-10-21` and the API
  key as `Authorization: Bearer`.
  - The code needed no request changes: it sends no sampling parameters, puts no documents
    in assistant turns, and only ever sends citations back on Claude's own turns.
  - The test mocks moved from `httpx` to `httpx2`, and `fallbacks` is now a typed parameter.
  - The SDK itself is now licensed Apache 2.0 (it was MIT); this project stays MIT.
  - Migration guide: https://platform.claude.com/docs/en/api/migration-2026-10-21
- **Cost:**
  - list-price module (`swarm/pricing.py`);
  - a dollar estimate before each run (`--estimate`) and the actual cost after it;
  - a hard dollar cap (`--max-usd`, default $2) that also counts web-search fees.
- **Claim check:** every cited sentence of the report is judged against the source excerpts
  it cites, batched through the engine. Unsupported or partly supported sentences are
  marked inline, and the report states its support rate.
- **Prompt-injection defences** (`swarm/safety.py`):
  - prompts treat web content as data, not instructions;
  - findings with injection markers are excluded from synthesis and named in the report;
  - the report is sanitised: no images, no HTML, and links only to cited sources.
- **Evaluation:** `swarm eval` compares the swarm with a single agent given the same search
  allowance, on known-answer questions (starter set in `evals/`). It is scored
  deterministically, with a total dollar cap.
- 13 new tests (47 in total).

## 2.2.0 — deep research
- New `swarm research "question"` command. A planner splits the question into independent
  sub-questions, and parallel Claude workers research them with server-side web search and
  fetch (resuming `pause_turn`).
- Each finding passes a deterministic citation gate (≥ 2 sources from ≥ 2 domains); weak
  findings are re-researched at higher effort.
- Synthesis writes a report from the findings, then a code check removes references to
  non-existent sources and counts uncited paragraphs.
- Runs on the existing engine (queue, AIMD, circuit breaker, budget), with hedging off and
  one worker per sub-question. `--mock` gives a full offline demo.
- 7 new tests, including the real client against a mocked HTTP transport.

## 2.1.0 — research-informed hardening
- Anthropic 429s are classified by `error.details.error_code`: a spend cap stops the run
  cleanly (no retry storm), a ramp limit holds the rate instead of halving it, and
  `retry-after` is honoured.
- Tasks without a deterministic checker are now accepted on **agreement** between two
  independent attempts (default), because self-reported confidence is a weak error signal.
  The old confidence gate is still available (`unchecked_mode="confidence"`).
- The benchmark adds a **single-agent baseline** with the same pipeline, 95% confidence
  intervals, JSON export, and a cost-vs-time (Pareto) chart.
- Research report in `docs/research/` and design rationale in `docs/ARCHITECTURE.md`.
- CI (ruff, mypy, pytest on Python 3.10–3.13, benchmark smoke test), MIT license, Makefile.

## 2.0.0 — adaptive scheduler and reliability layer
- Bayesian-logistic Thompson sampling with a dynamic programme over the escalation ladder
  picks batch size and effort level; the learned state persists across runs.
- Lazy, priority-aware scheduling; circuit breaker; request hedging; token budget;
  write-ahead journal with resume; Batches API backend; shared-context prompt caching;
  structured output with confidence; top-rung retries.

## 1.0.0 — static batching swarm
- Batched calls, effort escalation, deterministic validators, dedupe and disk cache,
  AIMD limiter, knee-based agent sizing, offline simulator.
