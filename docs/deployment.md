# Deploying to production

The server is the web UI and REST API for research, writing and long-document jobs, shared by
a team. This page covers running it reliably: architecture, three ways to deploy, settings,
API keys, monitoring, backups and upgrades.

## Architecture

```
            HTTPS                     SQLite job database (WAL), on one volume
users ──► reverse proxy ──► swarm serve ──enqueue──► jobs · events · api_keys
          (Caddy/nginx)     (API, UI)                   ▲        │
                                                        │ claim  │ progress
                                          swarm worker ─┴────────┘ (×N, heartbeats)
```

- **The API never runs jobs itself** (unless you use its embedded worker). It checks the key,
  rate limit and budget, then puts the job in the queue.
- **Workers claim jobs in a transaction**, so no two workers ever get the same job. They run
  it and stream progress into the database, where the browser follows it live (Server-Sent
  Events, which resume after a reconnect).
- **Jobs survive crashes.** A worker heartbeats every 10 seconds. If one dies, its jobs go
  back to the queue (up to 2 attempts), and after that they're marked failed with a reason.
- **Restarts don't lose work.** On a normal shutdown (SIGTERM) a worker hands its running jobs
  straight back to the queue, so upgrades lose nothing.
- **What runs on the server:** research, `write` and `doc` jobs. `build`, `analyze` and
  `review` execute code or read local files, so they're deliberately CLI-only.

## Option 1: Docker Compose with HTTPS (one server)

Needs Docker, a domain pointing at the server, and ports 80 and 443 open.

```bash
git clone https://github.com/Chetan0246/multi-agent-swarm && cd multi-agent-swarm
cp deploy/.env.example deploy/.env        # set SWARM_DOMAIN, model and search keys
docker compose -f deploy/compose.yaml up -d --build
docker compose -f deploy/compose.yaml exec api swarm keys create --name alice --daily-usd 5 --rpm 30
docker compose -f deploy/compose.yaml up -d --scale worker=3     # more parallel jobs
```

Caddy gets and renews a Let's Encrypt certificate automatically and streams progress events
without buffering. Every container:

- runs as a non-root user with a read-only filesystem;
- has no Linux capabilities and runs with `no-new-privileges`;
- has 45 seconds to stop gracefully.

Only Caddy publishes ports.

## Option 2: Kubernetes

```bash
kubectl create namespace swarm
kubectl -n swarm create secret generic swarm-secrets \
    --from-literal=ANTHROPIC_API_KEY=... --from-literal=TINYFISH_API_KEY=... \
    --from-literal=SWARM_METRICS_TOKEN=$(openssl rand -hex 24)
# edit the host names in deploy/k8s/swarm.yaml, then
kubectl -n swarm apply -f deploy/k8s/swarm.yaml
kubectl -n swarm exec deploy/swarm -c api -- swarm keys create --name alice --daily-usd 5
```

- **One pod, two containers:** the API and a worker share a persistent volume, and the pod
  has readiness and liveness probes, a restricted security context and resource limits.
- **Ingress:** an nginx Ingress with cert-manager, with buffering off so progress events stream.
- **Single replica:** the database is SQLite, so the Deployment is one replica with a `Recreate`
  strategy. Scale jobs with the worker's `--concurrency`, or by adding worker containers to the
  pod.

## Option 3: systemd on a VM

`deploy/systemd/` has a hardened `swarm-api.service` (with `ProtectSystem=strict`,
`NoNewPrivileges` and a write-only state directory) and a template `swarm-worker@.service`.
The install steps are at the top of `swarm-api.service`. Put TLS in your own reverse proxy
(Caddy, nginx), and list the host name in `SWARM_ALLOWED_HOSTS`.

## Settings

Everything is set through environment variables (or `[server]` in `swarm.toml`):

| Variable | Default | What it does |
|---|---|---|
| `SWARM_STATE_DIR` | `~/.local/share/swarm` (`/data/state` in the image) | where `jobs.sqlite` lives |
| `SWARM_ALLOWED_HOSTS` | localhost names | host names the server answers to (blocks DNS rebinding) |
| `SWARM_API_TOKEN` | | one admin key; prefer `swarm keys` per user |
| `SWARM_METRICS_TOKEN` | | lets a Prometheus scraper read `/metrics` |
| `SWARM_MAX_USD_PER_JOB` | 2 | hard cap per job, whatever the request asks for |
| `SWARM_DEFAULT_RPM` | 60 | requests per minute for keys without their own limit |
| `SWARM_JOB_TIMEOUT_S` | 1800 | a job running longer is stopped |
| `SWARM_RETENTION_DAYS` | 30 | finished jobs older than this are deleted |
| `SWARM_WORKER_CONCURRENCY` | 2 | jobs at once in the embedded worker |
| `SWARM_EMBEDDED_WORKER` | 1 | 0 = API only (same as `--no-worker`) |
| `SWARM_ALLOW_MOCK` | true | set to false in production |
| `SWARM_CORS_ORIGINS` | | browser origins allowed to call the API from other sites |
| `SWARM_MAX_BODY_BYTES` | 2000000 | larger requests get 413 |
| `SWARM_TRUSTED_PROXIES` | 127.0.0.1 | proxies whose `X-Forwarded-*` headers are trusted |
| `SWARM_LOG_FORMAT`, `SWARM_LOG_LEVEL` | text (json in the image), INFO | log format and level |

The model and search settings are the same as for the CLI: `ANTHROPIC_API_KEY`, or
`SWARM_PROVIDER=openai` with `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `SWARM_MODEL` and
`SWARM_PRICE`; plus `TINYFISH_API_KEY`.

## API keys and budgets

```bash
swarm keys create --name alice --daily-usd 5 --rpm 30   # prints the key once
swarm keys create --name ops --admin                     # sees every job and /metrics
swarm keys list
swarm keys revoke <id>
```

- **Keys are stored only as hashes**, so a leaked database doesn't leak usable keys.
- **Each key sees only its own jobs.** Admin keys see all of them.
- **Daily budgets can't be overshot.** A key's budget counts what its jobs spent today plus
  the caps of its jobs still queued or running, so a burst of submissions can't go over it.
  Going over returns `402`; going over the rate limit returns `429` with `Retry-After`.
- **No public server without a key.** `swarm serve` refuses to listen on a public interface
  until at least one key or `SWARM_API_TOKEN` exists.

## Monitoring

| Endpoint | For |
|---|---|
| `/healthz` | liveness: the process is up |
| `/readyz` | readiness: the database answers (503 if not) |
| `/metrics` | Prometheus, with an admin key or `SWARM_METRICS_TOKEN` |

The metrics are:
- `swarm_http_requests_total` and `swarm_http_request_duration_seconds`: requests by route and
  status, with latency histograms;
- `swarm_jobs{kind,status}`: job counts;
- `swarm_job_spend_usd{kind}`: spend by job type;
- `swarm_queue_oldest_age_seconds`: how long the oldest waiting job has waited;
- `swarm_process_uptime_seconds`.

Job, spend and queue figures come from the database, so they're correct however many workers
run.

Suggested alerts:

- `swarm_queue_oldest_age_seconds > 600`: jobs are waiting; add workers or check they run.
- A rising `swarm_jobs{status="error"}`: read the worker logs.
- `swarm_job_spend_usd` growing faster than expected: check the keys' budgets.

Logs are JSON lines with a `request_id`. The same id is returned in the `X-Request-ID`
response header, so a user's report can be traced to the exact log lines. Logs record key
ids, never keys or prompts.

## Backups and upgrades

- **Backup while running:**

  ```bash
  sqlite3 /data/state/jobs.sqlite ".backup /backup/jobs-$(date +%F).sqlite"
  ```

  (WAL mode makes this safe during writes.)
- **Upgrade:** pull or build the new image, then `docker compose up -d`. Workers hand running
  jobs back to the queue on SIGTERM, and the new workers pick them up. Released images are at
  `ghcr.io/chetan0246/multi-agent-swarm:<version>`. They're built for amd64 and arm64, with
  provenance and an SBOM attached.

## Security checklist

- [ ] HTTPS in front (Caddy in the Compose setup, or your own proxy or Ingress).
- [ ] `SWARM_ALLOWED_HOSTS` lists exactly your host names.
- [ ] One key per user, each with `--daily-usd`; no shared admin token.
- [ ] `SWARM_ALLOW_MOCK=false`, and `SWARM_MAX_USD_PER_JOB` set to what you'll accept.
- [ ] `/metrics` scraped with `SWARM_METRICS_TOKEN`, not left open.
- [ ] `deploy/.env` (or the Kubernetes Secret) readable only by the operator. It's gitignored
      and left out of image builds.
- [ ] Backups of `jobs.sqlite` on a schedule.

## Limits, plainly

- **One host for the job database.** It's SQLite, so the API and workers share one disk. That
  carries many workers on one machine. Spreading them over several machines needs a shared
  database server.
- **Rate limits are per API process.** With several API replicas, each enforces its own
  window. Daily dollar budgets are shared, because they're kept in the database.
