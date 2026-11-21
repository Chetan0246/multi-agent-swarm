# Web UI and API

```bash
pip install "multi-agent-swarm[server]"
swarm serve                                   # http://127.0.0.1:8000
```

The UI shows an estimate before you start, then live progress per worker, and the finished
report with hover-over source quotes. From there you can download the report or ask a
follow-up question.

!!! warning "Exposing the server"
    By default the server listens only on `127.0.0.1` and needs no key. It refuses to listen
    on a public interface until a key exists (`swarm keys create`) or `SWARM_API_TOKEN` is
    set. Every job is capped at `SWARM_MAX_USD_PER_JOB` (default $2), whatever the request
    asks for. Requests addressed to any host name other than localhost and the names in
    `SWARM_ALLOWED_HOSTS` are rejected, which blocks DNS rebinding. For a real deployment
    (HTTPS, workers, keys, monitoring), see [Deploying to production](deployment.md).

## REST API

Interactive docs are served at `/api/docs`. Send `Authorization: Bearer <key>` once keys or
`SWARM_API_TOKEN` are configured. Each key sees only its own jobs. Every response carries an
`X-Request-ID` header.

| Method | Path | |
|---|---|---|
| GET | `/healthz`, `/readyz` | liveness; readiness (database reachable) |
| GET | `/metrics` | Prometheus metrics (admin key or `SWARM_METRICS_TOKEN`) |
| GET | `/api/me` | your key, limits and today's spend |
| GET | `/api/estimate?preset=&search=` | cost estimate for research |
| POST | `/api/jobs` | start a job (below) → `{"id"}` |
| POST | `/api/research` | start a research job: `{"question", "preset", "search", "include_domains", "exclude_domains", "recency_days", "kind", "language", "max_usd"}` |
| GET | `/api/jobs?limit=&offset=` | your recent jobs |
| GET | `/api/jobs/{id}` | status, stats, cost, attempts, recent events |
| DELETE | `/api/jobs/{id}` | cancel (queued: at once; running: at the worker's next heartbeat) |
| GET | `/api/jobs/{id}/events` | live progress as Server-Sent Events; send `Last-Event-ID` to resume |
| GET | `/api/jobs/{id}/report?format=md\|html\|json` | the result |
| POST | `/api/jobs/{id}/follow-up` | `{"question", "max_usd"}`: a follow-up research job with the same settings |

`POST /api/jobs` bodies, by `kind`:

```json
{"kind": "research", "question": "...", "preset": "standard", "source_kind": "news", "max_usd": 1}
{"kind": "write", "brief": "A guide to ...", "words": 2000, "sections": 6}
{"kind": "doc", "action": "translate", "text": "...", "to": "German"}
{"kind": "doc", "action": "summarize", "text": "...", "words": 300}
{"kind": "doc", "action": "rewrite", "text": "...", "style": "plain English for customers"}
```

Errors: `401` missing or bad key · `402` the key's daily budget would be exceeded · `404`
not found (or another user's job) · `409` result not ready · `413` request too large · `422`
invalid input · `429` rate limit (with `Retry-After`).

```bash
KEY=sws_...   # from `swarm keys create`
curl -s -X POST localhost:8000/api/research -H "Authorization: Bearer $KEY" \
  -H 'content-type: application/json' -d '{"question": "Do heat pumps work in cold climates?", "preset": "quick"}'
curl -sN localhost:8000/api/jobs/<id>/events -H "Authorization: Bearer $KEY"   # live progress
```

## Docker

```bash
docker build -t swarm .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY -e TINYFISH_API_KEY -e SWARM_API_TOKEN=change-me -v swarm-data:/data swarm
```

For HTTPS, separate workers, Kubernetes and systemd, see [Deploying to production](deployment.md).
