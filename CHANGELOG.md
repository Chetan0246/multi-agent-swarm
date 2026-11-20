# Changelog

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
