# Changelog

## 3.0.1 — bug fixes
Fixes from a code audit, each with a regression test (`tests/test_audit.py`).

- **Exhausted search quota or bad key now stops the run.** Before, the error went to the
  model as a tool message and the run kept paying for Claude while every search failed. Now
  the remaining sub-questions are cancelled. If nothing was found, the error is shown with its
  hint; otherwise the report says it stopped early (`stopped_early` in stats).
- **Cost accuracy:**
  - `--worker-model` runs are now priced, and capped, at the worker model's rates.
  - An unknown model fails before any spend, with a clear error instead of a traceback.
  - Tokens from worker calls that failed partway through are now counted.
- **Batches API:** `invalid_request` results were never recognised, so they were resubmitted
  as transient errors. They now fail at once.
- **Resume:** `swarm run` reuses a journalled answer only if the task's prompt is unchanged
  (before, a reused state dir could return another workload's answers under the same ids).
- **Follow-ups:**
  - In the web UI and API, follow-ups keep the parent's preset, search and source filters.
  - Finding ids no longer collide.
- **Reports:**
  - Source links with `&` in the URL work in HTML reports.
  - Page titles can no longer inject Markdown links or images into the Sources list.
- **Eval:** answers are matched as whole words ("Rust" no longer matches "trust").
- **Resources:** HTTP clients and the cache database are closed after each run and server job.
- **Validation:** the API checks `kind`, and the estimate endpoint uses the configured model.
- **Robustness:** non-JSON TinyFish responses and documents with no word tokens are handled.

## 3.0.0 — first public release
**Breaking:** the default web search provider is now TinyFish (free). Set `TINYFISH_API_KEY`,
or pass `--search anthropic` to keep Claude's server-side search.

- **Free search and fetch:** TinyFish Search and Fetch as Claude client-side tools, with a
  shared client-side rate limiter (30 searches and 150 page fetches per minute).
  - Retries on 429 and 5xx; clear errors for a bad key and for an exhausted daily allowance.
  - Search fees drop from $10 per 1,000 searches to $0.
- **Quote grounding:** workers return verbatim quotes, and code checks each one against the
  text the worker retrieved. Unmatched quotes are dropped and counted.
- **Your documents as sources:** `--docs` for txt, md, html and pdf files, searched locally
  with BM25. `--docs-only` researches your files without the web.
- **Presets:** `--preset quick|standard|deep`, each with its own default cost cap.
- **Source filters:** `--include-domains`, `--exclude-domains` (pinterest.com and quora.com are
  excluded by default), `--recency-days`, `--news`, `--papers` and `--language`.
- **Follow-ups:** `--follow-up report.json` keeps the earlier findings, researches only new
  sub-questions, and rewrites the report.
- **Exports:**
  - Markdown, JSON (round-trips for follow-ups), and self-contained HTML;
  - the HTML has hover-over source quotes, a strict Content Security Policy, and escapes
    everything.
- **Local cache:** search and fetch results are kept in SQLite, so repeats cost nothing and
  use no quota.
- **Web UI and REST API (`swarm serve`):**
  - jobs run in the background, with live progress over Server-Sent Events;
  - follow-ups, history, downloads, and finished jobs persisted to disk;
  - optional bearer token, and a server-side cost cap on every job;
  - a Docker image.
- **Config file:** `swarm.toml` or `~/.config/swarm/config.toml`, plus `SWARM_*` environment
  variables. Flags override the environment, and the environment overrides the file.
- **Errors and setup:** clear error messages with fixes, and `swarm doctor [--online]` to check
  your setup.
- **Launch packaging:**
  - PyPI metadata and extras (`[server]`, `[pdf]`, `[all]`);
  - release workflow (trusted publishing on a tag) and docs site (MkDocs to GitHub Pages);
  - Dependabot, pre-commit, issue and PR templates, CONTRIBUTING, CODE_OF_CONDUCT and SECURITY;
  - CI now reports coverage and installs the built wheel.
- 14 new tests (61 in total), all offline.
- Config files work on Python 3.10 through the `tomli` backport, which is installed only there.

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
