# Batch tasks: text, code and web

`swarm research` answers one big question. `swarm run` handles the other common job: **many
small, independent tasks**, each with an answer that can be checked. It batches tasks into
fewer calls, checks every answer, retries failures at a higher effort, resumes after a crash,
and stops at a dollar cap.

```bash
swarm run examples/tasks/text.jsonl --out results.jsonl
swarm run examples/tasks/code.jsonl --allow-code-execution --out results.jsonl
swarm run examples/tasks/web.jsonl --out results.jsonl
swarm run examples/tasks/mixed.jsonl --allow-code-execution --max-usd 1   # all three kinds at once
swarm run tasks.jsonl --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --price 0.12,0.48 \
    --base-url https://api.river.ai/v1 --api-key-env RIVER_API_KEY          # any OpenAI-compatible provider
swarm run examples/tasks/mixed.jsonl --mock                                 # offline, no cost
```

## Task file

One JSON object per line. Every task has an `id` (unique) and a `prompt`. Optional:
`difficulty` (1–3; harder tasks are batched less), `priority` (lower runs first), and
a way to check the answer:

| Kind | What it's for | How an answer is checked |
|---|---|---|
| `text` (default) | classification, routing, extraction, short answers, dates and units, translation, rewriting, summaries | `expected` (exact match, case- and space-insensitive) or `pattern` (regex). With neither, two independent attempts must agree |
| `code` | functions, classes, scripts, refactors, bug fixes; website code too | `tests`: Python asserts run against the answer in a sandbox. Needs `--allow-code-execution` |
| `web` | facts to look up online: dates, prices, specs, "who/what/when" across many entities | at least `min_sources` sources (default 1) whose quotes are verbatim on the fetched pages, plus `expected`/`pattern` if given |

```json
{"id": "route1", "prompt": "Route to billing, technical or account: 'charged twice'", "expected": "billing"}
{"id": "roman", "kind": "code", "prompt": "Write to_roman(n) for 1..3999", "tests": "assert to_roman(1994) == 'MCMXCIV'"}
{"id": "py3", "kind": "web", "prompt": "When was Python 3.0 released?", "expected": "2008", "min_sources": 2}
```

Each kind runs through its own engine at the same time as the others, with its own journal and
learned policy, so a slow web lookup never holds up a batch of text tasks. Results are written
in input order, one JSON line per task: `answer`, `ok`, `attempts`, the `effort` that
solved it, `error`, and `kind`. A web answer is itself JSON, `{"answer", "sources": [{url,
title, quote}]}`.

## How each kind works

- **Text:** up to 24 tasks share one call, which returns JSON answers. The batch size and starting
  effort are learned as the run goes.
- **Code:** at most 4 tasks per call. A solution that fails its tests is retried at the next
  effort level, then once more at the top. Without `tests`, a code task falls back to the
  "two attempts must agree" rule, which rarely holds for code, so give tests wherever you can.
- **Web:** one tool-using worker per task. It searches with free TinyFish, reads pages, and
  answers with verbatim quotes that are checked against the retrieved text (the same
  grounding that deep research uses).

## Running model-written code

`--allow-code-execution` runs each solution plus its tests with `python -I` in a subprocess:

- in a fresh temporary directory, deleted afterwards;
- with an empty environment, so **no API keys** and no `HOME`;
- with no stdin;
- with CPU, memory, file-size and open-file limits, and a wall-clock timeout (`--code-timeout`,
  default 10 s) that kills the whole process group.

This contains mistakes. It does **not** contain an attacker: the code can still read files
your user can read and open network connections. For prompts you didn't write, run inside a
container without network access:

```bash
docker run --rm --network none -v "$PWD:/w" -w /w -e ANTHROPIC_API_KEY python:3.12 \
    sh -c "pip install -q multi-agent-swarm && swarm run tasks.jsonl --allow-code-execution"
```

(With `--network none` only `--mock` runs work. Allow egress to the model API only, with a
proxy or firewall, for real runs.)

## Providers and costs

| | |
|---|---|
| Claude (default) | the Messages API. With `--batch-api`, text and code go through the Message Batches API at half price, with minutes-to-hours latency |
| OpenAI-compatible | DeepSeek, Together, OpenRouter, vLLM and others, via `--model ID --price IN,OUT --base-url URL --api-key-env NAME`, or `[run]` / `[research]` in `swarm.toml` |
| Web search | TinyFish (free within its allowance); `--search mock` runs offline |

`--max-usd` is a hard cap, converted to the engine's token budget at the model's input price.
That budget weights output at 5× input, which overstates the cost on providers whose output is
cheaper relative to input, so the cap errs on the safe side. The summary printed at the end
gives each kind's success rate, calls, agents and `usd_approx`.
