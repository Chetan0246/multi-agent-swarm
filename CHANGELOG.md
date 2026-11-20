# Changelog

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
