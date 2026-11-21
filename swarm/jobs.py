"""Durable job queue for the server: SQLite in WAL mode, shared by API and worker processes.

    API process ──enqueue──► jobs table ◄──claim (one transaction)── worker process(es)
                            events table ◄──progress events────────── (heartbeat every 10 s)

* A job is a row: kind, params, owner (API key id), status, cost cap, result, error.
* `claim()` takes the oldest queued job inside BEGIN IMMEDIATE, so two workers never get the
  same job.
* Workers heartbeat. A running job whose heartbeat is older than `stale_after` (its worker
  crashed or was killed) goes back to the queue, up to `max_attempts`, then fails as
  "interrupted". Graceful shutdown re-queues at once.
* Cancellation is a flag the worker polls; timeouts are enforced by the worker.
* Old finished jobs are deleted after `retention_days`.

SQLite is the right store for one host (the API and workers share a volume). For several hosts,
put the same schema behind Postgres; the interface here is deliberately small.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

TERMINAL = ("done", "error", "cancelled")


class BudgetExceeded(Exception):
    def __init__(self, spent: float, budget: float):
        super().__init__(f"daily budget of ${budget:.2f} would be exceeded (${spent:.2f} spent or reserved today)")
        self.spent, self.budget = spent, budget


# What a job counts against its owner's daily budget: its cap while it may still spend; its
# actual cost once known; its cap again when it failed or was cancelled after starting without
# a known cost (it may have spent up to the cap, so the budget errs on the safe side).
SPEND_SQL = ("SELECT COALESCE(SUM(CASE WHEN status IN ('queued', 'running', 'cancelling') THEN cap_usd "
             "WHEN usd IS NOT NULL THEN usd WHEN started IS NOT NULL THEN cap_usd ELSE 0 END), 0) AS s "
             "FROM jobs WHERE owner IS ? AND created >= ?")
ACTIVE = ("queued", "running", "cancelling")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    params TEXT NOT NULL,
    owner TEXT,
    parent TEXT,
    status TEXT NOT NULL,
    cap_usd REAL NOT NULL DEFAULT 0,
    usd REAL,
    result TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    worker TEXT,
    created REAL NOT NULL,
    started REAL,
    finished REAL,
    heartbeat REAL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created);
CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, created);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    t REAL NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id, seq);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    created REAL NOT NULL,
    revoked REAL,
    daily_usd REAL,
    rpm INTEGER,
    admin INTEGER NOT NULL DEFAULT 0
);
"""


class JobDB:
    def __init__(self, path: str | Path, stale_after: float = 90.0, max_attempts: int = 2):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stale_after = stale_after
        self.max_attempts = max_attempts
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []  # every thread's connection, closed together
        self._lock = threading.Lock()
        self._conn().executescript(SCHEMA)  # executescript manages its own transaction

    # A connection per thread: FastAPI runs sync endpoints in a thread pool.
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
            self._local.conn = c
            with self._lock:
                self._all.append(c)
        return c

    @contextmanager
    def tx(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        c = self._conn()
        c.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def close(self) -> None:
        with self._lock:
            conns, self._all = self._all, []
        for c in conns:
            c.close()
        self._local = threading.local()

    def ping(self) -> bool:
        try:
            self._conn().execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    # ------------------------------------------------------------------ jobs

    def enqueue(self, kind: str, params: dict[str, Any], *, owner: str | None = None, parent: str | None = None,
                cap_usd: float = 0.0, daily_budget: float | None = None) -> str:
        """Queue a job. With `daily_budget`, the budget check and the insert happen in one write
        transaction, so concurrent submissions can't both squeeze under the limit."""
        jid = uuid.uuid4().hex[:16]
        with self.tx(immediate=True) as c:
            if daily_budget is not None:
                spent = float(c.execute(SPEND_SQL, (owner, _day_start())).fetchone()["s"])
                if spent + cap_usd > daily_budget + 1e-9:
                    raise BudgetExceeded(spent, daily_budget)
            c.execute("INSERT INTO jobs (id, kind, params, owner, parent, status, cap_usd, created) "
                      "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
                      (jid, kind, json.dumps(params), owner, parent, cap_usd, time.time()))
            self.event(jid, "status", {"status": "queued"}, c)
        return jid

    def get(self, jid: str) -> dict[str, Any] | None:
        row = self._conn().execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
        return _row(row) if row else None

    def list_jobs(self, owner: str | None = None, limit: int = 50, offset: int = 0, all_owners: bool = False
             ) -> list[dict[str, Any]]:
        if all_owners:
            rows = self._conn().execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ? OFFSET ?",
                                        (limit, offset)).fetchall()
        else:
            rows = self._conn().execute("SELECT * FROM jobs WHERE owner IS ? ORDER BY created DESC LIMIT ? OFFSET ?",
                                        (owner, limit, offset)).fetchall()
        return [_row(r) for r in rows]

    def claim(self, worker: str, kinds: tuple[str, ...] | None = None) -> dict[str, Any] | None:
        """Atomically take the oldest queued job (of the given kinds)."""
        self.recover()
        with self.tx(immediate=True) as c:
            if kinds:
                marks = ",".join("?" * len(kinds))
                row = c.execute(f"SELECT id FROM jobs WHERE status = 'queued' AND kind IN ({marks}) "
                                "ORDER BY created LIMIT 1", kinds).fetchone()
            else:
                row = c.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            c.execute("UPDATE jobs SET status = 'running', worker = ?, started = ?, heartbeat = ?, "
                      "attempts = attempts + 1 WHERE id = ?", (worker, now, now, row["id"]))
        self.event(row["id"], "status", {"status": "running", "worker": worker})
        return self.get(row["id"])

    def heartbeat(self, jid: str, worker: str | None = None) -> str | None:
        """Refresh a running job's heartbeat; returns its status (e.g. 'cancelling').

        With `worker`, only that worker's claim is refreshed, and "lost" is returned when the job
        no longer belongs to it (it stalled, was recovered and re-claimed by another worker): the
        caller must stop, or the job would run twice and be paid for twice."""
        with self.tx() as c:
            row = c.execute("SELECT status, worker FROM jobs WHERE id = ?", (jid,)).fetchone()
            if row is None:
                return None
            if worker is not None and (row["worker"] != worker or row["status"] not in ("running", "cancelling")):
                return "lost"
            c.execute("UPDATE jobs SET heartbeat = ? WHERE id = ?", (time.time(), jid))
        return row["status"]

    def finish(self, jid: str, status: str, *, result: Any = None, error: str | None = None,
               usd: float | None = None, worker: str | None = None) -> bool:
        """Record the outcome. With `worker`, only if that worker still holds the job (returns
        False otherwise, and changes nothing). The final event goes in the same transaction, so
        a progress stream never sees a finished job without its final event."""
        assert status in TERMINAL
        data: dict[str, Any] = {"status": status}
        if error:
            data["error"] = error
        if usd is not None:
            data["usd"] = usd
        with self.tx() as c:
            if worker is None:
                n = c.execute("UPDATE jobs SET status = ?, result = ?, error = ?, usd = ?, finished = ? WHERE id = ?",
                              (status, json.dumps(result) if result is not None else None, error, usd, time.time(),
                               jid)).rowcount
            else:
                n = c.execute("UPDATE jobs SET status = ?, result = ?, error = ?, usd = ?, finished = ? WHERE id = ? "
                              "AND worker = ? AND status IN ('running', 'cancelling')",
                              (status, json.dumps(result) if result is not None else None, error, usd, time.time(),
                               jid, worker)).rowcount
            if n:
                self.event(jid, "done" if status == "done" else status, data, c)
        return bool(n)

    def requeue(self, jid: str, reason: str, worker: str | None = None) -> None:
        with self.tx() as c:
            n = c.execute("UPDATE jobs SET status = 'queued', worker = NULL, heartbeat = NULL WHERE id = ? "
                          "AND status IN ('running', 'cancelling') AND (? IS NULL OR worker = ?)",
                          (jid, worker, worker)).rowcount
            if n:
                self.event(jid, "status", {"status": "queued", "reason": reason}, c)

    def cancel(self, jid: str) -> str | None:
        """Queued: cancelled at once. Running: flagged; the worker stops it at its next heartbeat."""
        with self.tx() as c:
            row = c.execute("SELECT status FROM jobs WHERE id = ?", (jid,)).fetchone()
            if row is None:
                return None
            if row["status"] == "queued":
                c.execute("UPDATE jobs SET status = 'cancelled', finished = ? WHERE id = ?", (time.time(), jid))
            elif row["status"] == "running":
                c.execute("UPDATE jobs SET status = 'cancelling' WHERE id = ?", (jid,))
            else:
                return row["status"]
        new = "cancelled" if row["status"] == "queued" else "cancelling"
        self.event(jid, "status", {"status": new})
        return new

    def recover(self) -> int:
        """Jobs whose worker stopped heartbeating: back to the queue, or failed after max_attempts."""
        cutoff = time.time() - self.stale_after
        with self.tx(immediate=True) as c:
            rows = c.execute("SELECT id, attempts, status FROM jobs WHERE status IN ('running', 'cancelling') "
                             "AND heartbeat < ?", (cutoff,)).fetchall()
            for r in rows:
                if r["status"] == "cancelling":
                    c.execute("UPDATE jobs SET status = 'cancelled', finished = ? WHERE id = ?", (time.time(), r["id"]))
                elif r["attempts"] >= self.max_attempts:
                    c.execute("UPDATE jobs SET status = 'error', error = ?, finished = ? WHERE id = ?",
                              ("interrupted: its worker stopped responding (tried "
                               f"{r['attempts']} times)", time.time(), r["id"]))
                else:
                    c.execute("UPDATE jobs SET status = 'queued', worker = NULL, heartbeat = NULL WHERE id = ?",
                              (r["id"],))
        for r in rows:
            self.event(r["id"], "status", {"status": "recovered", "from": r["status"]})
        return len(rows)

    def purge(self, retention_days: float) -> int:
        cutoff = time.time() - retention_days * 86400
        with self.tx() as c:
            ids = [r["id"] for r in c.execute("SELECT id FROM jobs WHERE status IN ('done', 'error', 'cancelled') "
                                              "AND finished < ?", (cutoff,)).fetchall()]
            for jid in ids:
                c.execute("DELETE FROM events WHERE job_id = ?", (jid,))
                c.execute("DELETE FROM jobs WHERE id = ?", (jid,))
        return len(ids)

    # ------------------------------------------------------------------ events

    def event(self, jid: str, kind: str, data: dict[str, Any], conn: sqlite3.Connection | None = None) -> None:
        (conn or self._conn()).execute("INSERT INTO events (job_id, t, data) VALUES (?, ?, ?)",
                                       (jid, time.time(), json.dumps({"event": kind, **data}, default=str)))

    def events(self, jid: str, after: int = 0, limit: int = 500) -> list[tuple[int, dict[str, Any]]]:
        rows = self._conn().execute("SELECT seq, t, data FROM events WHERE job_id = ? AND seq > ? "
                                    "ORDER BY seq LIMIT ?", (jid, after, limit)).fetchall()
        return [(r["seq"], {"t": round(r["t"], 2), **json.loads(r["data"])}) for r in rows]

    # ------------------------------------------------------------------ spend and stats

    def spend_today(self, owner: str | None) -> float:
        """Spent today plus the caps of jobs still queued or running (reserved)."""
        return float(self._conn().execute(SPEND_SQL, (owner, _day_start())).fetchone()["s"])

    def stats(self) -> dict[str, Any]:
        c = self._conn()
        by = [dict(r) for r in c.execute("SELECT kind, status, COUNT(*) AS n, COALESCE(SUM(usd), 0) AS usd, "
                                         "COALESCE(SUM(finished - started), 0) AS seconds FROM jobs "
                                         "GROUP BY kind, status").fetchall()]
        oldest = c.execute("SELECT MIN(created) AS t FROM jobs WHERE status = 'queued'").fetchone()["t"]
        return {"by_kind_status": by, "oldest_queued_age_s": round(time.time() - oldest, 1) if oldest else 0.0}


def _day_start() -> float:
    now = time.time()
    return now - (now % 86400)  # UTC midnight


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["params"] = json.loads(d["params"]) if d.get("params") else {}
    d["result"] = json.loads(d["result"]) if d.get("result") else None
    return d
