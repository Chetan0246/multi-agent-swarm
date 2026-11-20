# Getting started

## 1. Install

```bash
pip install "multi-agent-swarm[all]"     # CLI + web UI + PDF support
```

You need Python 3.10 or later. `pip install multi-agent-swarm` installs the command line only.

## 2. Keys

| Key | What for | Cost |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude: planning, research workers, writing, checking | pay per token |
| `TINYFISH_API_KEY` | web search and page fetching | free within a daily allowance |

Get a TinyFish key at <https://agent.tinyfish.ai/api-keys> (no card needed). Instead of an
Anthropic key you can sign in with `ant auth login`. Then run:

```bash
swarm doctor            # checks install, keys and config
swarm doctor --online   # also makes one free TinyFish search
```

## 3. Try it offline first

```bash
swarm research "Do heat pumps work in cold climates?" --mock --out demo.html
```

`--mock` runs the whole pipeline with simulated models and sources: no keys, no cost.

## 4. Real research

```bash
swarm research "Do heat pumps work in cold climates?" --estimate        # cost estimate only
swarm research "Do heat pumps work in cold climates?" --out report.html
swarm research "What about running costs?" --follow-up report.json --out costs.html
```

Every run prints its cost, time, the share of findings verified and the claim support rate.
`--out report.html` also writes `report.json`, which `--follow-up` uses to build on the
earlier findings instead of starting over.

## 5. Web UI

```bash
swarm serve             # http://127.0.0.1:8000
```

See [Web UI and API](web-api.md).
