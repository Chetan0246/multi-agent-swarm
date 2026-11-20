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
| `SWARM_MODEL`, `SWARM_MAX_USD`, `SWARM_SEARCH`, `SWARM_PRESET` | override the config file |
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
