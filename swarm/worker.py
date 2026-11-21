"""Job worker: runs queued research, write and doc jobs from the JobDB.

    swarm worker --concurrency 4          # standalone process (scale these out)
    swarm serve                           # also runs one embedded worker unless --no-worker

Each claimed job runs as an asyncio task. Every `heartbeat` seconds the worker refreshes the
job's heartbeat and checks for cancellation or a timeout. On SIGTERM/SIGINT it stops claiming,
cancels its running jobs and puts them back in the queue, so a deploy or a scale-down never
loses work (another worker picks them up and restarts them).

Only text-in, text-out jobs run on the server. `build`, `analyze` and `review` execute code or
read local files, which a shared server must not do on a caller's behalf; they stay CLI-only.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections.abc import Callable, Coroutine
from typing import Any

from .jobs import JobDB

log = logging.getLogger("swarm.worker")
Emit = Callable[[str, dict], None]
JOB_KINDS = ("research", "write", "doc")


# ----------------------------------------------------------------------------- runners


def _tag_spend(e: BaseException, usd: float) -> None:
    """A failed or cancelled job still spent money; carry it to the worker for the budget."""
    with contextlib.suppress(AttributeError):  # exceptions with __slots__ can't carry it
        e.swarm_usd = usd  # type: ignore[attr-defined]


async def run_research(job: dict, emit: Emit, db: JobDB, config_path: str | None) -> tuple[Any, float | None]:
    from .app import ResearchOptions
    from .app import run_research as _run
    from .export import from_json, to_json

    p = dict(job["params"])
    prior = None
    if job.get("parent"):
        parent = db.get(job["parent"])
        if parent and parent.get("result"):
            prior = from_json(parent["result"]["report"])
    opts = ResearchOptions(**{k: v for k, v in p.items() if k in ResearchOptions.__dataclass_fields__},
                           max_usd=job["cap_usd"] or None, config_path=config_path)
    report = await _run(opts, on_event=emit, prior=prior)
    return {"report": to_json(report), "stats": report.stats}, float(report.stats.get("usd") or 0.0)


async def run_write(job: dict, emit: Emit, db: JobDB, config_path: str | None) -> tuple[Any, float | None]:
    from .workflows.chat import make_chat
    from .workflows.write import write

    p = job["params"]
    chat = make_chat(p.get("model"), max_usd=job["cap_usd"] or None, config_path=config_path)
    try:
        res = await write(p["brief"], chat, words=int(p.get("words", 1500)), max_sections=int(p.get("sections", 8)),
                          log=lambda m: emit("log", {"message": m}))
    except BaseException as e:
        _tag_spend(e, chat.usd)
        raise
    finally:
        await chat.aclose()
    return {"markdown": res.markdown, "title": res.title, "words": res.words, "edited": res.edited}, res.usd


async def run_doc(job: dict, emit: Emit, db: JobDB, config_path: str | None) -> tuple[Any, float | None]:
    from .workflows.chat import make_chat
    from .workflows.doc import summarize, transform

    p = job["params"]
    chat = make_chat(p.get("model"), max_usd=job["cap_usd"] or None, config_path=config_path)
    say = lambda m: emit("log", {"message": m})  # noqa: E731
    try:
        if p["action"] == "summarize":
            res = await summarize(p["text"], chat, words=int(p.get("words", 400)), log=say)
        elif p["action"] == "translate":
            res = await transform(p["text"], chat, f"translate it into {p['to']}", f"their {p['to']} translation",
                                  log=say)
        else:
            res = await transform(p["text"], chat, f"rewrite it as: {p['style']}",
                                  "how each is written in the new style", log=say)
    except BaseException as e:
        _tag_spend(e, chat.usd)
        raise
    finally:
        await chat.aclose()
    return {"text": res.text, "chunks": res.chunks, "failed_chunks": res.failed_chunks}, res.usd


RUNNERS: dict[str, Callable[..., Coroutine[Any, Any, tuple[Any, float | None]]]] = {
    "research": run_research, "write": run_write, "doc": run_doc}


# ----------------------------------------------------------------------------- worker


class Worker:
    def __init__(self, db: JobDB, *, concurrency: int = 2, config_path: str | None = None, poll: float = 1.0,
                 heartbeat: float = 10.0, timeout: float = 1800.0, kinds: tuple[str, ...] = JOB_KINDS,
                 name: str | None = None, runners: dict[str, Any] | None = None):
        self.db = db
        self.concurrency = concurrency
        self.config_path = config_path
        self.poll = poll
        self.heartbeat = heartbeat
        self.timeout = timeout
        self.kinds = kinds
        self.name = name or f"{socket.gethostname()}:{os.getpid()}"
        self.runners = runners or RUNNERS
        self.running: dict[str, asyncio.Task] = {}
        self.completed = 0

    async def run(self, stop: asyncio.Event) -> None:
        log.info("worker started", extra={"worker": self.name, "concurrency": self.concurrency})
        last_purge = 0.0
        try:
            while not stop.is_set():
                if len(self.running) < self.concurrency:
                    job = self.db.claim(self.name, self.kinds)
                    if job is not None:
                        self.running[job["id"]] = asyncio.create_task(self._supervise(job))
                        continue  # try to fill the next slot at once
                if time.time() - last_purge > 3600:
                    days = float(os.environ.get("SWARM_RETENTION_DAYS", "30"))
                    n = self.db.purge(days)
                    if n:
                        log.info("purged old jobs", extra={"count": n, "retention_days": days})
                    last_purge = time.time()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self.poll)
        finally:
            await self._drain()

    async def _drain(self) -> None:
        """Shutdown: cancel running jobs and put them back in the queue for another worker."""
        if not self.running:
            return
        log.info("worker stopping: re-queueing running jobs", extra={"count": len(self.running)})
        for task in list(self.running.values()):
            task.cancel()
        await asyncio.gather(*self.running.values(), return_exceptions=True)

    async def _supervise(self, job: dict) -> None:
        jid = job["id"]
        t0 = time.monotonic()
        emit = lambda kind, data: self.db.event(jid, kind, data)  # noqa: E731
        runner = self.runners.get(job["kind"])
        task: asyncio.Task | None = None

        async def stop_task() -> float | None:
            """Cancel the running job; return what it spent, if the runner reported it."""
            assert task is not None
            task.cancel()
            outcome: Any = (await asyncio.gather(task, return_exceptions=True))[0]
            return getattr(outcome, "swarm_usd", None)

        try:
            if runner is None:
                self.db.finish(jid, "error", error=f"unknown job kind {job['kind']!r}", worker=self.name)
                return
            task = asyncio.create_task(runner(job, emit, self.db, self.config_path))
            while True:
                done, _ = await asyncio.wait({task}, timeout=self.heartbeat)
                if done:
                    break
                status = self.db.heartbeat(jid, self.name)
                if status == "lost":  # recovered and re-claimed elsewhere while we stalled: stop, don't finish
                    await stop_task()
                    log.warning("job lost to another worker; stopped this copy", extra={"job": jid})
                    return
                if status == "cancelling":
                    usd = await stop_task()
                    self.db.finish(jid, "cancelled", error="cancelled by request", usd=usd, worker=self.name)
                    log.info("job cancelled", extra={"job": jid})
                    return
                if time.monotonic() - t0 > self.timeout:
                    usd = await stop_task()
                    self.db.finish(jid, "error", error=f"timed out after {self.timeout:.0f}s", usd=usd,
                                   worker=self.name)
                    log.warning("job timed out", extra={"job": jid, "timeout_s": self.timeout})
                    return
            try:
                result, usd = task.result()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # the job failed; the worker carries on
                self.db.finish(jid, "error", error=str(e)[:2000], usd=getattr(e, "swarm_usd", None),
                               worker=self.name)
                log.warning("job failed", extra={"job": jid, "kind": job["kind"], "error": type(e).__name__})
                return
            if self.db.finish(jid, "done", result=result, usd=round(usd, 6) if usd is not None else None,
                              worker=self.name):
                self.completed += 1
                log.info("job done", extra={"job": jid, "kind": job["kind"], "usd": usd,
                                            "seconds": round(time.monotonic() - t0, 1)})
            else:
                log.warning("job finished after another worker took it over; result discarded", extra={"job": jid})
        except asyncio.CancelledError:  # worker shutdown
            if task is not None:
                await stop_task()
            self.db.requeue(jid, "worker shutting down", worker=self.name)
            raise
        finally:
            self.running.pop(jid, None)


def run_worker(db_path: str, concurrency: int, config_path: str | None, timeout: float) -> None:
    """Entry point for `swarm worker`: runs until SIGTERM/SIGINT."""
    import signal

    from .observability import setup_logging

    setup_logging()
    db = JobDB(db_path)

    async def main() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        await Worker(db, concurrency=concurrency, config_path=config_path, timeout=timeout).run(stop)

    asyncio.run(main())
