"""Web UI and REST API (optional extra: `pip install "multi-agent-swarm[server]"`).

    swarm serve                      # http://127.0.0.1:8000
    SWARM_API_TOKEN=... swarm serve --host 0.0.0.0   # require a bearer token when exposed

Endpoints (JSON unless noted):
    GET  /                          web UI
    GET  /healthz                   liveness
    GET  /api/estimate              cost estimate for a preset/search
    POST /api/research              start a job -> {"id": ...}
    GET  /api/jobs                  recent jobs
    GET  /api/jobs/{id}             status, stats, error
    GET  /api/jobs/{id}/events      live progress (Server-Sent Events)
    GET  /api/jobs/{id}/report      ?format=md|html|json
    POST /api/jobs/{id}/follow-up   start a follow-up job on a finished report
"""
# No `from __future__ import annotations` here: FastAPI resolves the endpoint annotations at
# runtime, and the FastAPI types are imported inside create_app (the extra is optional).
import asyncio
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .app import ResearchOptions, resolve, run_research
from .config import load_config
from .export import from_json, to_html, to_json
from .pricing import estimate_research
from .research import ResearchReport


@dataclass
class Job:
    id: str
    question: str
    status: str = "queued"  # queued | running | done | error
    created: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    report: ResearchReport | None = None
    error: str | None = None
    parent: str | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def emit(self, kind: str, data: dict) -> None:
        self.events.append({"t": round(time.time() - self.created, 2), "event": kind, **data})
        self.changed.set()

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "question": self.question, "status": self.status, "created": self.created,
                "error": self.error, "parent": self.parent,
                "stats": self.report.stats if self.report else None}


class JobStore:
    """In-memory jobs, persisted as JSON on completion so finished reports survive restarts."""

    def __init__(self, state_dir: Path, max_concurrent: int = 2):
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.slots = asyncio.Semaphore(max_concurrent)
        for p in sorted(self.dir.glob("*.json"))[-200:]:
            try:
                d = json.loads(p.read_text())
                job = Job(d["id"], d["question"], d["status"], d["created"], d.get("events", []),
                          from_json(d["report"]) if d.get("report") else None, d.get("error"), d.get("parent"))
                self.jobs[job.id] = job
            except (ValueError, KeyError, TypeError):
                continue

    def save(self, job: Job) -> None:
        d = {"id": job.id, "question": job.question, "status": job.status, "created": job.created,
             "events": job.events[-200:], "error": job.error, "parent": job.parent,
             "report": to_json(job.report) if job.report else None}
        (self.dir / f"{job.id}.json").write_text(json.dumps(d))


def create_app(config_path: str | None = None, state_dir: Path | None = None):
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
        from pydantic import BaseModel, Field
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit('the web UI needs: pip install "multi-agent-swarm[server]"') from e

    cfg, _ = load_config(config_path)
    srv = cfg.get("server", {})
    max_usd_job = float(srv.get("max_usd_per_job", 2.0))
    allow_mock = bool(srv.get("allow_mock", True))
    token = os.environ.get("SWARM_API_TOKEN") or srv.get("api_token")
    store = JobStore(state_dir or Path(os.environ.get("SWARM_STATE_DIR", Path.home() / ".local/share/swarm/jobs")),
                     int(srv.get("max_concurrent_jobs", 2)))

    def auth(request: Request) -> None:
        if token:
            got = request.headers.get("authorization", "")
            if not secrets.compare_digest(got, f"Bearer {token}"):
                raise HTTPException(401, "missing or invalid bearer token")

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

    app = FastAPI(title="multi-agent-swarm", version="3.0.0", docs_url="/api/docs")

    def options(body: ResearchIn, question: str | None = None) -> ResearchOptions:
        if body.preset not in ("quick", "standard", "deep"):
            raise HTTPException(422, "preset must be quick, standard or deep")
        if body.search not in ("tinyfish", "anthropic"):
            raise HTTPException(422, "search must be tinyfish or anthropic")
        if body.mock and not allow_mock:
            raise HTTPException(403, "mock runs are disabled on this server")
        return ResearchOptions(
            question=question or body.question, preset=body.preset, search=body.search, mock=body.mock,
            include_domains=body.include_domains or None, exclude_domains=body.exclude_domains,
            recency_days=body.recency_days, kind=body.kind, language=body.language,
            max_usd=min(body.max_usd or max_usd_job, max_usd_job),  # server-side cap always applies
            config_path=config_path)

    async def run_job(job: Job, opts: ResearchOptions, prior: ResearchReport | None) -> None:
        async with store.slots:
            job.status = "running"
            job.emit("status", {"status": "running"})
            try:
                job.report = await run_research(opts, on_event=job.emit, prior=prior)
                job.status = "done"
                job.emit("done", {"stats": job.report.stats})
            except Exception as e:
                job.status, job.error = "error", str(e)
                job.emit("error", {"error": str(e)})
            store.save(job)

    def start(opts: ResearchOptions, prior: ResearchReport | None = None, parent: str | None = None) -> Job:
        resolve(opts)  # fail fast on config/key errors (raised as 400 below)
        job = Job(uuid.uuid4().hex[:12], opts.question, parent=parent)
        store.jobs[job.id] = job
        asyncio.get_running_loop().create_task(run_job(job, opts, prior))
        return job

    def get(job_id: str) -> Job:
        if job_id not in store.jobs:
            raise HTTPException(404, "no such job")
        return store.jobs[job_id]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("swarm").joinpath("web/index.html").read_text()

    @app.get("/healthz")
    def health() -> dict:
        return {"ok": True, "jobs": len(store.jobs)}

    @app.get("/api/estimate", dependencies=[Depends(auth)])
    def estimate(preset: str = "standard", search: str = "tinyfish") -> dict:
        from .research import PRESETS

        p = PRESETS.get(preset, PRESETS["standard"])
        e = estimate_research("claude-opus-5-5", p["max_subquestions"], p["max_searches"], p["efforts"],
                              free_search=search != "anthropic")
        return {"low_usd": round(e.low_usd, 2), "high_usd": round(e.high_usd, 2), "text": str(e),
                "cap_usd": min(p["max_usd"], max_usd_job)}

    @app.post("/api/research", dependencies=[Depends(auth)], status_code=202)
    async def research(body: ResearchIn) -> dict:
        try:
            job = start(options(body))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, str(e)) from e
        return {"id": job.id}

    @app.post("/api/jobs/{job_id}/follow-up", dependencies=[Depends(auth)], status_code=202)
    async def follow_up(job_id: str, body: FollowUpIn) -> dict:
        parent = get(job_id)
        if parent.report is None:
            raise HTTPException(409, "the job has no finished report yet")
        mock = bool(parent.report.stats.get("run_search") == "mock")
        base = ResearchIn(question=body.question, max_usd=body.max_usd, mock=mock)
        try:
            job = start(options(base), prior=parent.report, parent=job_id)
        except Exception as e:
            raise HTTPException(400, str(e)) from e
        return {"id": job.id}

    @app.get("/api/jobs", dependencies=[Depends(auth)])
    def jobs() -> list[dict]:
        return [j.summary() for j in sorted(store.jobs.values(), key=lambda j: -j.created)[:50]]

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(auth)])
    def job(job_id: str) -> dict:
        j = get(job_id)
        return {**j.summary(), "events": j.events[-50:]}

    @app.get("/api/jobs/{job_id}/events", dependencies=[Depends(auth)])
    async def events(job_id: str) -> StreamingResponse:
        j = get(job_id)

        async def stream():
            sent = 0
            while True:
                while sent < len(j.events):
                    yield f"data: {json.dumps(j.events[sent])}\n\n"
                    sent += 1
                if j.status in ("done", "error"):
                    yield "event: end\ndata: {}\n\n"
                    return
                j.changed.clear()
                try:
                    await asyncio.wait_for(j.changed.wait(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/report", dependencies=[Depends(auth)])
    def report(job_id: str, format: str = "md"):
        j = get(job_id)
        if j.report is None:
            raise HTTPException(409, f"job is {j.status}")
        if format == "html":
            return HTMLResponse(to_html(j.report))
        if format == "json":
            return PlainTextResponse(to_json(j.report), media_type="application/json")
        return PlainTextResponse(j.report.markdown, media_type="text/markdown")

    app.state.store = store
    return app


def serve(host: str = "127.0.0.1", port: int = 8000, config_path: str | None = None) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit('the web UI needs: pip install "multi-agent-swarm[server]"') from e
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("SWARM_API_TOKEN"):
        print("warning: serving on a public interface without SWARM_API_TOKEN; anyone who can reach "
              "this port can spend your API credits")
    uvicorn.run(create_app(config_path), host=host, port=port)
