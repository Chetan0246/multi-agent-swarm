"""Web UI and REST API (optional extra: `pip install "multi-agent-swarm[server]"`).

    swarm serve                                   # http://127.0.0.1:8000, one embedded worker
    swarm serve --host 0.0.0.0 --no-worker        # API only; run `swarm worker` processes alongside
    swarm keys create --name alice --daily-usd 5  # per-user keys with budgets

Endpoints (JSON unless noted; all /api routes need `Authorization: Bearer <key>` once keys
or SWARM_API_TOKEN are configured):
    GET    /                          web UI
    GET    /healthz                   liveness (the process is up)
    GET    /readyz                    readiness (database reachable; a worker is running or expected)
    GET    /metrics                   Prometheus metrics (admin key or SWARM_METRICS_TOKEN)
    GET    /api/me                    who you are, your limits and today's spend
    GET    /api/estimate              cost estimate for a research preset
    POST   /api/jobs                  start a job: {"kind": "research" | "write" | "doc", ...} -> {"id"}
    POST   /api/research              start a research job (same as kind "research")
    GET    /api/jobs                  your recent jobs (?limit=&offset=)
    GET    /api/jobs/{id}             status, stats, error, recent events
    DELETE /api/jobs/{id}             cancel a queued or running job
    GET    /api/jobs/{id}/events      live progress (Server-Sent Events; resumes with Last-Event-ID)
    GET    /api/jobs/{id}/report      the result (?format=md|html|json)
    POST   /api/jobs/{id}/follow-up   a follow-up research job on a finished report

Jobs live in a SQLite queue (swarm/jobs.py) and run in workers (swarm/worker.py), so they
survive restarts and the API can scale apart from the workers.
"""
# No `from __future__ import annotations` here: FastAPI resolves the endpoint annotations at
# runtime, and the FastAPI types are imported inside create_app (the extra is optional).
import asyncio
import hmac
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from .app import ResearchOptions, aclose, register_prices, resolve
from .auth import Principal, RateLimiter, authenticate, has_keys
from .config import load_config
from .export import from_json, to_html
from .jobs import TERMINAL, BudgetExceeded, JobDB
from .observability import Metrics, request_id, setup_logging
from .pricing import estimate_research

log = logging.getLogger("swarm.server")
LOCAL_HOSTS = ["127.0.0.1", "localhost", "::1", "[::1]"]


def allowed_hosts_for(srv: dict[str, Any], bind_host: str | None = None) -> list[str]:
    """Host names the server answers to. Anything else is rejected with 400.

    This blocks DNS rebinding: a web page you visit can point its own domain at 127.0.0.1 and
    then call this API from your browser (spending your credits). Such requests carry the
    attacker's domain in the Host header, so they never match. Add the names you really use
    when serving on a network: SWARM_ALLOWED_HOSTS=swarm.example.lan,10.0.0.5 or
    `allowed_hosts = [...]` under [server].
    """
    hosts = list(LOCAL_HOSTS)
    env = os.environ.get("SWARM_ALLOWED_HOSTS")
    extra = [h.strip() for h in env.split(",")] if env else list(srv.get("allowed_hosts", []))
    if bind_host and bind_host not in ("0.0.0.0", "::"):
        extra.append(bind_host)
    return hosts + [h for h in extra if h and h not in hosts]


def state_dir() -> Path:
    return Path(os.environ.get("SWARM_STATE_DIR", Path.home() / ".local/share/swarm"))


SECURITY_HEADERS = {
    # The UI's script and styles are inline; reports render in a sandboxed srcdoc iframe.
    "content-security-policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' "
                               "'unsafe-inline'; img-src 'self' data:; frame-src 'self'; connect-src 'self'; "
                               "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
    "cross-origin-opener-policy": "same-origin",
}


class BodyLimit:
    """ASGI middleware: refuse request bodies over `limit` bytes, counting what actually arrives
    (a chunked upload has no Content-Length to check up front). A bad Content-Length is a 400."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                n = int(declared)
            except ValueError:
                return await _plain(send, 400, b'{"detail": "invalid Content-Length"}')
            if n > self.limit:
                return await _plain(send, 413, f'{{"detail": "request body over {self.limit} bytes"}}'.encode())
        seen = 0
        over = False
        replaced = False
        too_large = f'{{"detail": "request body over {self.limit} bytes"}}'.encode()

        async def counted():
            nonlocal seen, over
            if over:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.limit:  # stop reading; whatever the app answers becomes a 413
                    over = True
                    return {"type": "http.disconnect"}
            return msg

        async def guarded_send(msg):
            nonlocal replaced
            if not over:
                return await send(msg)
            if not replaced and msg["type"] == "http.response.start":
                replaced = True
                await _plain(send, 413, too_large)
            # drop the app's own response to the truncated body

        try:
            await self.app(scope, counted, guarded_send)
        except Exception:
            if not over:
                raise
            if not replaced:
                await _plain(send, 413, too_large)


async def _plain(send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


_RID = re.compile(r"[^A-Za-z0-9._:-]")


def create_app(config_path: str | None = None, state_dir: Path | None = None,  # noqa: C901
               allowed_hosts: list[str] | None = None, embedded_worker: bool | None = None):
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request, Response
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
        from pydantic import BaseModel, Field
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit('the web UI needs: pip install "multi-agent-swarm[server]"') from e

    cfg, _ = load_config(config_path)
    srv = cfg.get("server", {})
    max_usd_job = float(os.environ.get("SWARM_MAX_USD_PER_JOB") or srv.get("max_usd_per_job", 2.0))
    allow_mock = os.environ.get("SWARM_ALLOW_MOCK", str(srv.get("allow_mock", True))).lower() not in ("0", "false")
    admin_token = os.environ.get("SWARM_API_TOKEN") or srv.get("api_token")
    metrics_token = os.environ.get("SWARM_METRICS_TOKEN")
    max_body = int(os.environ.get("SWARM_MAX_BODY_BYTES") or srv.get("max_body_bytes", 2_000_000))
    default_rpm = int(os.environ.get("SWARM_DEFAULT_RPM") or srv.get("default_rpm", 60))
    if embedded_worker is None:
        embedded_worker = os.environ.get("SWARM_EMBEDDED_WORKER", "1") not in ("0", "false")
    sdir = state_dir or globals()["state_dir"]()
    db = JobDB(Path(sdir) / "jobs.sqlite")
    metrics = Metrics()
    limiter = RateLimiter()
    worker_state: dict[str, Any] = {"worker": None, "task": None}

    @asynccontextmanager
    async def lifespan(app):
        stop = asyncio.Event()
        if embedded_worker:
            from .worker import Worker

            w = Worker(db, concurrency=int(os.environ.get("SWARM_WORKER_CONCURRENCY")
                                           or srv.get("max_concurrent_jobs", 2)),
                       config_path=config_path, timeout=float(os.environ.get("SWARM_JOB_TIMEOUT_S", 1800)))
            worker_state["worker"], worker_state["task"] = w, asyncio.create_task(w.run(stop))
        log.info("server started", extra={"version": __version__, "embedded_worker": bool(embedded_worker),
                                           "auth": "keys" if has_keys(db) else "token" if admin_token else "open"})
        try:
            yield
        finally:
            stop.set()
            if worker_state["task"] is not None:
                await worker_state["task"]  # re-queues anything still running
            db.close()

    app = FastAPI(title="multi-agent-swarm", version=__version__, docs_url="/api/docs", lifespan=lifespan)
    app.state.db = db
    app.state.metrics = metrics
    origins = [o.strip() for o in (os.environ.get("SWARM_CORS_ORIGINS") or ",".join(srv.get("cors_origins", [])))
               .split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["authorization", "content-type", "last-event-id"])
    hosts = allowed_hosts or allowed_hosts_for(srv)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    app.add_middleware(BodyLimit, limit=max_body)
    # Without keys or a token the server is open, which is only safe when it can be reached by
    # local names alone. Behind a reverse proxy serving a real domain, open access would put
    # the API on the internet, so it is refused until a key exists.
    open_allowed = all(h in LOCAL_HOSTS or h == "testserver" for h in hosts)

    @app.middleware("http")
    async def edge(request: Request, call_next):
        """Request id, body-size limit, security headers, access log and metrics."""
        rid = _RID.sub("", request.headers.get("x-request-id", ""))[:64] or uuid.uuid4().hex[:16]
        token = request_id.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id.reset(token)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        dt = time.perf_counter() - t0
        metrics.observe(request.method, route, response.status_code, dt)
        response.headers["x-request-id"] = rid
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if route not in ("/healthz", "/readyz", "/metrics") and not route.endswith("/events"):
            log.info("request", extra={"request_id": rid, "method": request.method, "route": route,
                                       "status": response.status_code, "ms": round(dt * 1000, 1)})
        return response

    # ------------------------------------------------------------------ auth

    def auth(request: Request) -> Principal:
        p = authenticate(db, request.headers.get("authorization", ""), admin_token)
        if p is None:
            if admin_token or has_keys(db):
                raise HTTPException(401, "missing or invalid API key")
            if not open_allowed:
                raise HTTPException(401, "this server is reachable by a public name: create a key with "
                                         "`swarm keys create` (or set SWARM_API_TOKEN) to use it")
            p = Principal("local", "local", True)  # open server: only reachable via localhost names
        ok, retry = limiter.allow(p.id, p.rpm if p.rpm is not None else (None if p.admin else default_rpm))
        if not ok:
            raise HTTPException(429, f"rate limit: {p.rpm or default_rpm} requests per minute",
                                headers={"retry-after": str(int(retry) + 1)})
        return p

    def own(jid: str, p: Principal) -> dict:
        job = db.get(jid)
        if job is None or (not p.admin and job["owner"] != p.id):
            raise HTTPException(404, "no such job")  # never reveal other users' job ids
        return job

    def reserve(p: Principal, cap: float) -> None:
        """Fail fast before any setup work; the binding check is atomic, in `enqueue`."""
        if p.daily_usd is not None and db.spend_today(p.id) + cap > p.daily_usd + 1e-9:
            raise HTTPException(402, f"daily budget of ${p.daily_usd:.2f} would be exceeded "
                                     f"(${db.spend_today(p.id):.2f} spent or reserved today)")

    def enqueue(p: Principal, kind: str, params: dict, cap: float, parent: str | None = None) -> str:
        try:
            return db.enqueue(kind, params, owner=p.id, parent=parent, cap_usd=cap, daily_budget=p.daily_usd)
        except BudgetExceeded as e:
            raise HTTPException(402, str(e)) from e

    # ------------------------------------------------------------------ request models

    class ResearchIn(BaseModel):
        question: str = Field(min_length=3, max_length=2000)
        preset: str = "standard"
        search: str = "tinyfish"
        include_domains: list[str] = []
        exclude_domains: list[str] | None = None
        recency_days: int | None = Field(default=None, ge=1, le=3650)
        kind: str | None = None
        language: str | None = None
        max_usd: float | None = Field(default=None, gt=0)
        mock: bool = False

    class FollowUpIn(BaseModel):
        question: str = Field(min_length=3, max_length=2000)
        max_usd: float | None = Field(default=None, gt=0)

    class JobIn(BaseModel):
        kind: str = Field(pattern="^(research|write|doc)$")
        # research
        question: str | None = Field(default=None, max_length=2000)
        preset: str = "standard"
        search: str = "tinyfish"
        include_domains: list[str] = []
        exclude_domains: list[str] | None = None
        recency_days: int | None = Field(default=None, ge=1, le=3650)
        source_kind: str | None = None
        language: str | None = None
        mock: bool = False
        # write
        brief: str | None = Field(default=None, max_length=8000)
        words: int = Field(default=1500, ge=100, le=20000)
        sections: int = Field(default=8, ge=1, le=20)
        # doc
        action: str | None = Field(default=None, pattern="^(translate|summarize|rewrite)$")
        text: str | None = Field(default=None, max_length=1_500_000)
        to: str | None = Field(default=None, max_length=60)
        style: str | None = Field(default=None, max_length=300)
        # all
        model: str | None = Field(default=None, max_length=120)
        max_usd: float | None = Field(default=None, gt=0)

    def research_options(body: ResearchIn) -> ResearchOptions:
        if body.preset not in ("quick", "standard", "deep"):
            raise HTTPException(422, "preset must be quick, standard or deep")
        if body.search not in ("tinyfish", "anthropic"):
            raise HTTPException(422, "search must be tinyfish or anthropic")
        if body.kind not in (None, "web", "news", "research_paper"):
            raise HTTPException(422, "kind must be web, news or research_paper")
        if body.mock and not allow_mock:
            raise HTTPException(403, "mock runs are disabled on this server")
        return ResearchOptions(
            question=body.question, preset=body.preset, search=body.search, mock=body.mock,
            include_domains=body.include_domains or None, exclude_domains=body.exclude_domains,
            recency_days=body.recency_days, kind=body.kind, language=body.language, config_path=config_path)

    async def enqueue_research(body: ResearchIn, p: Principal, parent: str | None = None) -> str:
        opts = research_options(body)
        cap = min(body.max_usd or max_usd_job, max_usd_job)  # the server-side cap always applies
        reserve(p, cap)
        try:
            llm, _, _ = resolve(opts)  # fail fast on config/key errors
            await aclose(llm)
        except Exception as e:
            raise HTTPException(400, str(e)) from e
        params = {k: getattr(opts, k) for k in ("question", "preset", "search", "mock", "include_domains",
                                                "exclude_domains", "recency_days", "kind", "language")}
        jid = enqueue(p, "research", params, cap, parent)
        log.info("job queued", extra={"job": jid, "kind": "research", "owner": p.id, "cap_usd": cap})
        return jid

    def summary(job: dict) -> dict[str, Any]:
        params, result = job["params"], job.get("result") or {}
        title = params.get("question") or params.get("brief") or f"{params.get('action', '')} document"
        return {"id": job["id"], "kind": job["kind"], "question": title, "status": job["status"],
                "created": job["created"], "started": job["started"], "finished": job["finished"],
                "error": job["error"], "parent": job["parent"], "usd": job["usd"], "cap_usd": job["cap_usd"],
                "attempts": job["attempts"], "stats": result.get("stats") if isinstance(result, dict) else None}

    # ------------------------------------------------------------------ routes

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("swarm").joinpath("web/index.html").read_text()

    @app.get("/healthz")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/readyz")
    def ready(response: Response) -> dict:
        db_ok = db.ping()
        w = worker_state["worker"]
        out = {"ok": db_ok, "database": db_ok, "embedded_worker": w is not None,
               "running_jobs": len(w.running) if w else None}
        if not db_ok:
            response.status_code = 503
        return out

    @app.get("/metrics")
    def metrics_route(request: Request) -> PlainTextResponse:
        hdr = request.headers.get("authorization", "")
        allowed = (metrics_token and hmac.compare_digest(hdr, f"Bearer {metrics_token}")) or \
            ((p := authenticate(db, hdr, admin_token)) is not None and p.admin) or \
            (not admin_token and not has_keys(db) and not metrics_token)
        if not allowed:
            raise HTTPException(401, "metrics need an admin key or SWARM_METRICS_TOKEN")
        w = worker_state["worker"]
        return PlainTextResponse(metrics.render(db.stats(), workers=1 if w else 0),
                                 media_type="text/plain; version=0.0.4")

    @app.get("/api/me")
    def me(p: Principal = Depends(auth)) -> dict:
        return {"id": p.id, "name": p.name, "admin": p.admin, "daily_usd": p.daily_usd, "rpm": p.rpm,
                "spent_or_reserved_today_usd": round(db.spend_today(p.id), 4)}

    @app.get("/api/estimate")
    def estimate(preset: str = "standard", search: str = "tinyfish", p: Principal = Depends(auth)) -> dict:
        from .research import PRESETS

        pr = PRESETS.get(preset, PRESETS["standard"])
        rc = cfg.get("research", {})
        model = os.environ.get("SWARM_MODEL") or rc.get("model", "claude-opus-5-5")
        try:
            register_prices(cfg, [model], os.environ.get("SWARM_PRICE") or rc.get("price"))
            e = estimate_research(model, pr["max_subquestions"], pr["max_searches"], pr["efforts"],
                                  free_search=search != "anthropic")
        except Exception as err:
            raise HTTPException(400, str(err)) from err
        return {"low_usd": round(e.low_usd, 2), "high_usd": round(e.high_usd, 2), "text": str(e),
                "cap_usd": min(pr["max_usd"], max_usd_job)}

    @app.post("/api/research", status_code=202)
    async def research(body: ResearchIn, p: Principal = Depends(auth)) -> dict:
        return {"id": await enqueue_research(body, p)}

    @app.post("/api/jobs", status_code=202)
    async def create_job(body: JobIn, p: Principal = Depends(auth)) -> dict:
        if body.kind == "research":
            if not body.question:
                raise HTTPException(422, "research needs `question`")
            r = ResearchIn(question=body.question, preset=body.preset, search=body.search,
                           include_domains=body.include_domains, exclude_domains=body.exclude_domains,
                           recency_days=body.recency_days, kind=body.source_kind, language=body.language,
                           max_usd=body.max_usd, mock=body.mock)
            return {"id": await enqueue_research(r, p)}
        if body.mock:
            raise HTTPException(422, "mock is only available for research jobs")
        if body.kind == "write" and not body.brief:
            raise HTTPException(422, "write needs `brief`")
        if body.kind == "doc":
            if not body.action or not body.text:
                raise HTTPException(422, "doc needs `action` and `text`")
            if body.action == "translate" and not body.to:
                raise HTTPException(422, "translate needs `to`")
            if body.action == "rewrite" and not body.style:
                raise HTTPException(422, "rewrite needs `style`")
        cap = min(body.max_usd or max_usd_job, max_usd_job)
        reserve(p, cap)
        try:
            from .workflows.chat import make_chat

            chat = make_chat(body.model, max_usd=cap, config_path=config_path)  # fail fast on config/keys
            await chat.aclose()
        except Exception as e:
            raise HTTPException(400, str(e)) from e
        keep = {"write": ("brief", "words", "sections", "model"), "doc": ("action", "text", "to", "style", "words",
                                                                           "model")}[body.kind]
        jid = enqueue(p, body.kind, {k: getattr(body, k) for k in keep}, cap)
        log.info("job queued", extra={"job": jid, "kind": body.kind, "owner": p.id, "cap_usd": cap})
        return {"id": jid}

    @app.post("/api/jobs/{job_id}/follow-up", status_code=202)
    async def follow_up(job_id: str, body: FollowUpIn, p: Principal = Depends(auth)) -> dict:
        parent = own(job_id, p)
        if parent["kind"] != "research" or parent["status"] != "done":
            raise HTTPException(409, "follow-ups need a finished research job")
        inherited = {k: v for k, v in parent["params"].items() if k in ResearchIn.model_fields and v is not None}
        if body.max_usd is not None:
            inherited["max_usd"] = body.max_usd
        try:
            base = ResearchIn(**{**inherited, "question": body.question})
        except Exception as e:
            raise HTTPException(422, str(e)) from e
        return {"id": await enqueue_research(base, p, parent=job_id)}

    @app.get("/api/jobs")
    def jobs(limit: int = 50, offset: int = 0, p: Principal = Depends(auth)) -> list[dict]:
        rows = db.list_jobs(p.id, limit=max(1, min(limit, 200)), offset=max(0, offset), all_owners=p.admin)
        return [summary(j) for j in rows]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, p: Principal = Depends(auth)) -> dict:
        j = own(job_id, p)
        return {**summary(j), "params": {k: v for k, v in j["params"].items() if k != "text"},
                "events": [e for _, e in db.events(job_id)][-50:]}

    @app.delete("/api/jobs/{job_id}")
    def cancel(job_id: str, p: Principal = Depends(auth)) -> dict:
        own(job_id, p)
        status = db.cancel(job_id)
        return {"id": job_id, "status": status}

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, request: Request, p: Principal = Depends(auth)) -> StreamingResponse:
        own(job_id, p)
        try:
            after = int(request.headers.get("last-event-id") or 0)
        except ValueError:
            after = 0

        async def stream():
            seq, idle = after, 0.0
            while True:
                rows = db.events(job_id, after=seq)
                for s, e in rows:
                    seq = s
                    yield f"id: {s}\ndata: {json.dumps(e)}\n\n"
                job = db.get(job_id)
                if job is None or (job["status"] in TERMINAL and not rows):
                    yield "event: end\ndata: {}\n\n"
                    return
                if await request.is_disconnected():
                    return
                idle = 0.0 if rows else idle + 0.5
                if idle >= 15:
                    idle = 0.0
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/report")
    def report(job_id: str, format: str = "md", p: Principal = Depends(auth)):
        j = own(job_id, p)
        if j["status"] != "done" or not j.get("result"):
            raise HTTPException(409, f"job is {j['status']}")
        res = j["result"]
        if j["kind"] == "research":
            rep = from_json(res["report"])
            if format == "html":
                return HTMLResponse(to_html(rep))
            if format == "json":
                return PlainTextResponse(res["report"], media_type="application/json")
            return PlainTextResponse(rep.markdown, media_type="text/markdown")
        body = res.get("markdown") or res.get("text") or ""
        if format == "json":
            return JSONResponse(res)
        if format == "html":
            from .export import markdown_to_html

            return HTMLResponse(f"<!doctype html><meta charset='utf-8'><body>{markdown_to_html(body)}</body>")
        return PlainTextResponse(body, media_type="text/markdown")

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, config_path: str | None = None,
          embedded_worker: bool = True, insecure: bool = False, workers: int = 1) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit('the web UI needs: pip install "multi-agent-swarm[server]"') from e
    setup_logging()
    srv = load_config(config_path)[0].get("server", {})
    public = host not in ("127.0.0.1", "localhost", "::1")
    db = JobDB(state_dir() / "jobs.sqlite")
    secured = bool(os.environ.get("SWARM_API_TOKEN") or srv.get("api_token") or has_keys(db))
    db.close()
    if public and not secured and not insecure:
        raise SystemExit("refusing to serve on a public interface without authentication: create a key with "
                         "`swarm keys create --name <you>` or set SWARM_API_TOKEN (or pass --insecure)")
    hosts = allowed_hosts_for(srv, host)
    if host in ("0.0.0.0", "::") and hosts == LOCAL_HOSTS:
        log.warning("only localhost names are accepted; set SWARM_ALLOWED_HOSTS to the names you serve on")
    os.environ.setdefault("SWARM_EMBEDDED_WORKER", "1" if embedded_worker else "0")
    uvicorn.run(create_app(config_path, allowed_hosts=hosts, embedded_worker=embedded_worker), host=host,
                port=port, proxy_headers=True, forwarded_allow_ips=os.environ.get("SWARM_TRUSTED_PROXIES", "127.0.0.1"),
                log_config=None, timeout_graceful_shutdown=30)
