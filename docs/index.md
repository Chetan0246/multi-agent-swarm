# Multi-Agent Swarm

**Ask a question; get a cited research report you can check.**

Parallel Claude workers each research one part of your question. They search the web for
free through TinyFish and read the sources. Every quote a worker cites is checked by code
against the page it actually read; quotes that don't match are thrown away. After the report
is written, each cited sentence is checked again against those quotes.

```bash
pip install "multi-agent-swarm[all]"
export ANTHROPIC_API_KEY=...       # Claude
export TINYFISH_API_KEY=...        # free search + fetch: https://agent.tinyfish.ai/api-keys
swarm research "How do heat pumps compare to gas boilers in cold climates?" --out report.html
```

| | |
|---|---|
| **Free web search** | TinyFish Search and Fetch: free within a daily allowance (12,000 searches and 1,000 pages per day) |
| **Checked citations** | quotes verified against retrieved text; claims checked against quotes; unsupported claims marked |
| **Predictable cost** | estimate before the run, actual cost after it, hard cap (default $2) |
| **Safe by default** | web text treated as data; injected instructions excluded; report sanitised |
| **Your documents** | add PDFs, Markdown, HTML or text files as sources with `--docs` |
| **Web UI and API** | `swarm serve`: live progress, follow-up questions, downloads |

Start with [Getting started](getting-started.md).
