# Web UI and API

```bash
pip install "multi-agent-swarm[server]"
swarm serve                                   # http://127.0.0.1:8000
```

The UI shows an estimate before you start, then live progress per worker, and the finished
report with hover-over source quotes. From there you can download the report or ask a
follow-up question.

!!! warning "Exposing the server"
    By default the server listens only on `127.0.0.1`. Before using `--host 0.0.0.0`, set
    `SWARM_API_TOKEN`; otherwise anyone who can reach the port can spend your credits. Every
    job is capped at `max_usd_per_job` (default $2), whatever the request asks for.

## Docker

```bash
docker build -t swarm .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY -e TINYFISH_API_KEY -e SWARM_API_TOKEN=change-me -v swarm-data:/data swarm
```

## REST API

Interactive docs are served at `/api/docs`. Send `Authorization: Bearer $SWARM_API_TOKEN`
when a token is set.

| Method | Path | |
|---|---|---|
| GET | `/api/estimate?preset=&search=` | cost estimate |
| POST | `/api/research` | start a job: `{"question", "preset", "search", "include_domains", "exclude_domains", "recency_days", "kind", "language", "max_usd"}` → `{"id"}` |
| GET | `/api/jobs` | recent jobs |
| GET | `/api/jobs/{id}` | status, stats, recent events |
| GET | `/api/jobs/{id}/events` | live progress (Server-Sent Events) |
| GET | `/api/jobs/{id}/report?format=md\|html\|json` | the report |
| POST | `/api/jobs/{id}/follow-up` | `{"question"}`: research only what's new, rewrite the report |

```bash
curl -s -X POST localhost:8000/api/research -H 'content-type: application/json' \
  -d '{"question": "Do heat pumps work in cold climates?", "preset": "quick"}'
```
