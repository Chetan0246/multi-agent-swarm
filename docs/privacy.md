# Privacy and costs

## Where your data goes

| Data | Sent to |
|---|---|
| your question, sub-questions, retrieved page text, findings | Anthropic (Claude API) |
| search queries and URLs to fetch | TinyFish (or Anthropic, with `--search anthropic`) |
| files passed with `--docs` | read locally; passages you cite are sent to Anthropic as part of the research |
| search results and fetched pages | cached locally in `~/.cache/swarm` (delete it any time) |
| finished jobs (server only) | `SWARM_STATE_DIR` on your machine |

Nothing is sent to the project's authors, and there is no telemetry.

## What a run costs

- **Search and fetch:** free with TinyFish within its daily allowance (12,000 searches and
  1,000 fetched pages per day, resetting at 00:00 UTC). With `--search anthropic`, web search
  costs $10 per 1,000 searches.
- **Claude:** pay per token at Anthropic's list prices; for example `claude-opus-5-5` costs
  $4 per million input tokens and $20 per million output tokens. Use `--worker-model` to run
  the workers on a cheaper model.
- Every run prints an estimate first and the actual cost at the end, and stops at the cap.

Prices were checked on 2026-11-21. Confirm current prices at
<https://platform.claude.com/docs/en/about-claude/pricing> and <https://tinyfish.ai/pricing>.

## Respecting sources

Reports quote short excerpts (at most 300 characters each) with links to the original
sources. Check important facts against those sources, and follow each site's terms when you
reuse content.
