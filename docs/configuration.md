# Configuration

Settings are applied in this order, each overriding the one before:
preset defaults, then the config file, then environment variables, then command-line flags.

The config file is the first of these that exists: `./swarm.toml`, the file named by
`$SWARM_CONFIG`, or `~/.config/swarm/config.toml`. A commented example is in
[`examples/swarm.toml`](https://github.com/Chetan0246/multi-agent-swarm/blob/main/examples/swarm.toml).

| Environment variable | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | Claude credentials (or `ant auth login`) |
| `TINYFISH_API_KEY` | TinyFish search and fetch |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` (or `OPENAI_API_BASE`) | key and endpoint for an OpenAI-compatible provider |
| `SWARM_MODEL`, `SWARM_MAX_USD`, `SWARM_SEARCH`, `SWARM_PRESET`, `SWARM_PROVIDER`, `SWARM_PRICE` | override the config file |
| `SWARM_API_TOKEN` | require this bearer token on the web API |
| `SWARM_CACHE_DIR` | search/fetch cache (default `~/.cache/swarm`) |
| `SWARM_STATE_DIR` | where the server keeps finished jobs |

## Presets

| | quick | standard | deep |
|---|---|---|---|
| sub-questions / parallel workers | 3 | 5 | 8 |
| effort ladder | low → medium | low → medium → high | medium → high |
| searches per worker call | 3 | 5 | 8 |
| tool turns per worker | 4 | 6 | 10 |
| claim check | off | on | on |
| default cost cap | $0.75 | $2.00 | $6.00 |

## Other model providers

Besides Claude, any OpenAI-compatible chat-completions API works: DeepSeek, Together,
DeepInfra, Fireworks, OpenRouter, or a local vLLM or Ollama server.

```bash
export OPENAI_API_KEY=...                         # your provider's key
export OPENAI_BASE_URL=https://your-provider/v1
swarm research "question" --provider openai \
    --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --price 0.10,0.40 --estimate
```

Or in `swarm.toml`:

```toml
[research]
provider = "openai"
base_url = "https://your-provider/v1"
model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"

[prices]
"deepseek-ai/DeepSeek-V4.1-Flash-1017" = [0.10, 0.40]   # USD per million: [in, out] or [in, out, cached]
```

The pipeline is the same as with Claude: free TinyFish search, quote grounding, claim
checking, the cost estimate and the hard cap. A few things differ:

- **Prices:** the project can't know your provider's prices. Give them with `--price`, so the
  estimate and the cap mean something; without one, the run refuses to start. If you don't
  give a cached-token price, cached tokens are billed at the full input price, so cost is
  never understated.
- **Key variable:** the key is read from `OPENAI_API_KEY`; use `--api-key-env RIVER_API_KEY` (or
  `api_key_env = "RIVER_API_KEY"` in `[research]`) if your provider's key lives elsewhere.
- **Search:** must be client-side (`--search tinyfish` or `--docs`). Claude's server-side
  search is Anthropic-only.
- **Structured output:** JSON mode plus the schema in the prompt, parsed leniently. Models that
  don't follow JSON instructions well produce fewer verified findings.
- **Effort:** escalating a weak finding re-researches it with a fresh attempt, since effort
  levels aren't sent.
- **Not yet supported:** `swarm eval` and `swarm run` still use Claude only.
