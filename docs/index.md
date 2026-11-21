# Multi-Agent Swarm

## AI agents that check their own work

Parallel agents split your work into parts and do them at the same time. Then code, not the
model's opinion, checks every result: quotes against the pages they came from, apps against
their own tests, numbers against your data. Free and open source; bring any model.

<video src="assets/demo-build.mp4" poster="assets/build-poster.png" autoplay muted loop playsinline
       style="width:100%;border-radius:12px;border:1px solid rgba(0,0,0,.08)"
       aria-label="swarm build writes a FastAPI app in six parallel waves, installs it and runs its tests: 14 files, tests pass, $0.0057"></video>

*A real run, replayed: `swarm build` writes a FastAPI + SQLite app in six parallel waves, then
installs it and runs its 22 tests. Total cost on DeepSeek: $0.0057.*

## Try it in a minute, free

```bash
pip install "multi-agent-swarm[all]"
swarm research "How do heat pumps compare to gas boilers in cold climates?" --mock --out demo.html
```

`--mock` runs the whole pipeline offline, with no keys and no cost, so you can see a cited report
before you spend anything. For real runs, add a model key and a free search key:

```bash
export ANTHROPIC_API_KEY=...     # or any OpenAI-compatible provider: see Configuration
export TINYFISH_API_KEY=...      # free web search: https://agent.tinyfish.ai/api-keys
swarm research "How do heat pumps compare to gas boilers in cold climates?" --out report.html
```

## What it does, and how each result is checked

| Job | Command | Checked by |
|---|---|---|
| Research a question, with citations | `swarm research` | every quote matched word for word to the page read; each cited claim checked against its quotes |
| Build an app (backend, frontend, CLI) | `swarm build "…" --out app --allow-execution` | the app's own tests, installed and run; failures go to a fixer |
| Analyze a CSV | `swarm analyze data.csv "…" --allow-code-execution` | every number must come from code that ran on your data |
| Review a codebase | `swarm review ./repo` | each finding must quote real code; a second check against the imported files refutes the ones that don't hold |
| Write a long document | `swarm write "…" --words 3000` | an outline every section follows; an edit pass that must keep every section |
| Translate, summarize or rewrite long documents | `swarm doc translate file.md --to German` | one glossary for every chunk; length checks |
| Thousands of small tasks | `swarm run tasks.jsonl` | exact answers, patterns, sandboxed tests, or grounded sources |

## Real results

Live runs on DeepSeek V4.1 Flash, with web search free through TinyFish:

| Run | Result | Cost |
|---|---|---|
| Build a FastAPI + SQLite todo API | 14 files, 22 passing tests, 0 fix rounds | $0.0057 |
| Research "heat pumps vs gas boilers in cold climates" | 3 of 3 sub-questions verified, 19 sources, 4 unmatched quotes dropped | $0.016 |
| Analyze a 144-row sales CSV | every figure matched the raw data | $0.0024 |
| Review an 11-file codebase | 6 findings kept; 4 refuted on a second check, listed with reasons | $0.0030 |

Every run has a hard cap (`--max-usd`), and the estimate comes before any spend.

## Where to go next

- [Getting started](getting-started.md): install, keys, your first report
- [Build, write, analyze, review](workflows.md): the everyday workflows
- [Batch tasks](tasks.md): thousands of small text, code or web tasks
- [Configuration](configuration.md): Claude or any OpenAI-compatible model, prices, presets
- [Deploying to production](deployment.md): a server for your team, with keys, budgets and HTTPS
