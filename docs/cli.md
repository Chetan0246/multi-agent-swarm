# Command line

## `swarm research QUESTION`

| Flag | Default | Meaning |
|---|---|---|
| `--preset quick\|standard\|deep` | standard | breadth, depth and default cost cap ($0.75 / $2 / $6) |
| `--search tinyfish\|anthropic` | tinyfish | free TinyFish tools, or Claude's server-side search ($10 per 1,000 searches) |
| `--docs PATH...` | | your files or folders as sources (txt, md, html, pdf) |
| `--docs-only` | | no web: research only `--docs` |
| `--include-domains` / `--exclude-domains` | excludes pinterest.com, quora.com | comma-separated domains |
| `--recency-days N` | | only recent results |
| `--news` / `--papers` | | search news or research papers |
| `--language CODE` | | language for searches and the report |
| `--out FILE` | stdout | `.md`, `.html` or `.json` (`.html`/`.md` also write a `.json` for follow-ups) |
| `--follow-up REPORT.json` | | extend an earlier report |
| `--max-usd X` | preset | hard dollar cap (0 = none) |
| `--estimate` | | print the estimate and exit |
| `--model` / `--worker-model` | claude-opus-5-5 | models for planning and writing, and for workers |
| `--max-subquestions`, `--max-agents`, `--max-searches` | preset | fine-grained limits |
| `--no-claim-check` | | skip the claim check (cheaper, less trustworthy) |
| `--no-cache` | | bypass the local search/fetch cache |
| `--mock` | | offline demo |
| `-v` | | print full run statistics |

## Other commands

| Command | What it does |
|---|---|
| `swarm serve [--host --port]` | web UI and REST API |
| `swarm doctor [--online]` | check install, keys and config |
| `swarm eval [questions.jsonl]` | known-answer evaluation against a single-agent baseline (`--estimate`, `--mock`, `--max-usd-total`) |
| `swarm bench` | offline benchmark of the scheduling engine |
| `swarm run tasks.jsonl` | the general batch engine, for classification or extraction workloads |
