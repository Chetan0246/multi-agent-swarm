"""Production features: durable queue, workers, API keys, budgets, limits, metrics (offline)."""
import asyncio
import time

import pytest

from swarm.auth import RateLimiter, authenticate, create_key, list_keys, revoke_key
from swarm.jobs import JobDB
from swarm.worker import Worker

# ------------------------------------------------------------------ queue


def test_claim_is_exclusive_and_ordered(tmp_path):
    db = JobDB(tmp_path / "j.sqlite")
    a = db.enqueue("research", {"question": "a"})
    b = db.enqueue("write", {"brief": "b"})
    other = JobDB(tmp_path / "j.sqlite")  # a second process on the same file
    first, second = db.claim("w1"), other.claim("w2")
    assert first["id"] == a and second["id"] == b and db.claim("w3") is None
    assert db.get(a)["status"] == "running" and db.get(a)["worker"] == "w1" and db.get(a)["attempts"] == 1
    assert db.claim("w4", kinds=("doc",)) is None
    db.close()
    other.close()


def test_stale_jobs_are_recovered_then_failed(tmp_path):
    db = JobDB(tmp_path / "j.sqlite", stale_after=0.01, max_attempts=2)
    jid = db.enqueue("research", {})
    db.claim("w1")
    time.sleep(0.02)  # the worker died: no heartbeat
    assert db.recover() == 1 and db.get(jid)["status"] == "queued"
    db.claim("w2")
    time.sleep(0.02)
    db.recover()
    job = db.get(jid)
    assert job["status"] == "error" and "interrupted" in job["error"]
    db.close()


def test_cancel_purge_and_spend(tmp_path):
    db = JobDB(tmp_path / "j.sqlite")
    q = db.enqueue("write", {}, owner="k1", cap_usd=1.0)
    assert db.cancel(q) == "cancelled"
    r = db.enqueue("write", {}, owner="k1", cap_usd=0.5)
    db.claim("w")
    assert db.cancel(r) == "cancelling" and db.heartbeat(r) == "cancelling"
    done = db.enqueue("doc", {}, owner="k1", cap_usd=2.0)
    db.finish(done, "done", result={"text": "x"}, usd=0.25)
    assert db.spend_today("k1") == pytest.approx(0.5 + 0.25)  # reserved cap + actual spend; cancelled: 0
    assert db.purge(0) == 2 and db.get(done) is None  # the finished and cancelled jobs
    db.close()


# ------------------------------------------------------------------ worker


def _run_worker(db, runners, **kw):
    async def go():
        stop = asyncio.Event()
        w = Worker(db, poll=0.02, heartbeat=0.05, runners=runners, **kw)
        task = asyncio.create_task(w.run(stop))
        return w, stop, task

    return go


def test_worker_runs_cancels_times_out_and_requeues_on_shutdown(tmp_path):
    db = JobDB(tmp_path / "j.sqlite")

    async def quick(job, emit, db_, cfg):
        emit("log", {"message": "hi"})
        return {"text": "ok"}, 0.01

    async def slow(job, emit, db_, cfg):
        await asyncio.sleep(30)
        return {}, 0.0

    async def broken(job, emit, db_, cfg):
        raise RuntimeError("model said no")

    ok = db.enqueue("write", {})
    bad = db.enqueue("doc", {})
    to_cancel = db.enqueue("research", {})

    async def main():
        stop = asyncio.Event()
        w = Worker(db, poll=0.02, heartbeat=0.05, concurrency=3,
                   runners={"write": quick, "doc": broken, "research": slow})
        task = asyncio.create_task(w.run(stop))
        for _ in range(100):
            if db.get(ok)["status"] == "done" and db.get(bad)["status"] == "error":
                break
            await asyncio.sleep(0.02)
        db.cancel(to_cancel)
        for _ in range(100):
            if db.get(to_cancel)["status"] == "cancelled":
                break
            await asyncio.sleep(0.02)
        # a timeout, then a graceful shutdown with a job still running
        w.timeout = 0.1
        late = db.enqueue("research", {})
        for _ in range(100):
            if db.get(late)["status"] == "error":
                break
            await asyncio.sleep(0.02)
        w.timeout = 60
        survivor = db.enqueue("research", {})
        for _ in range(100):
            if db.get(survivor)["status"] == "running":
                break
            await asyncio.sleep(0.02)
        stop.set()
        await task
        return late, survivor

    late, survivor = asyncio.run(main())
    assert db.get(ok)["result"] == {"text": "ok"} and db.get(ok)["usd"] == 0.01
    assert "model said no" in db.get(bad)["error"]
    assert db.get(to_cancel)["status"] == "cancelled"
    assert "timed out" in db.get(late)["error"]
    assert db.get(survivor)["status"] == "queued"  # handed back, not lost
    assert any(e["event"] == "log" for _, e in db.events(ok))
    db.close()


# ------------------------------------------------------------------ keys


def test_keys_are_hashed_and_revocable(tmp_path):
    db = JobDB(tmp_path / "j.sqlite")
    kid, key = create_key(db, "alice", daily_usd=5, rpm=10)
    assert key.startswith(f"sws_{kid}_") and key.encode() not in (tmp_path / "j.sqlite").read_bytes()
    p = authenticate(db, f"Bearer {key}", None)
    assert p is not None and p.id == kid and p.daily_usd == 5 and not p.admin
    assert authenticate(db, f"Bearer {key}x", None) is None and authenticate(db, "Basic abc", None) is None
    assert authenticate(db, "Bearer tok", "tok").admin
    assert revoke_key(db, kid) and authenticate(db, f"Bearer {key}", None) is None
    assert list_keys(db)[0]["revoked"]
    db.close()


def test_rate_limiter_window():
    rl = RateLimiter()
    assert all(rl.allow("a", 3)[0] for _ in range(3))
    ok, retry = rl.allow("a", 3)
    assert not ok and 0 < retry <= 60
    assert rl.allow("b", 3)[0] and rl.allow("a", None)[0]


# ------------------------------------------------------------------ server


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from swarm.server import create_app

    for v in ("SWARM_API_TOKEN", "SWARM_METRICS_TOKEN", "SWARM_ALLOWED_HOSTS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SWARM_DEFAULT_RPM", "1000")
    app = create_app(state_dir=tmp_path, allowed_hosts=["testserver"])
    with TestClient(app) as c:
        yield c, app.state.db


def _wait(c, jid, headers, status=("done", "error", "cancelled")):
    for _ in range(200):
        j = c.get(f"/api/jobs/{jid}", headers=headers).json()
        if j["status"] in status:
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {jid} stuck in {j['status']}")


def test_keys_isolate_users_and_enforce_budgets(client):
    c, db = client
    _, alice = create_key(db, "alice", daily_usd=1.0)
    _, bob = create_key(db, "bob", rpm=2)
    A, B = {"Authorization": f"Bearer {alice}"}, {"Authorization": f"Bearer {bob}"}
    assert c.get("/api/jobs").status_code == 401  # keys exist: anonymous access is closed
    r = c.post("/api/research", json={"question": "Heat pumps?", "preset": "quick", "mock": True, "max_usd": 0.75},
               headers=A)
    assert r.status_code == 202
    jid = r.json()["id"]
    assert c.get(f"/api/jobs/{jid}", headers=B).status_code == 404  # bob can't see alice's job
    over = c.post("/api/research", json={"question": "More?", "mock": True, "max_usd": 0.5}, headers=A)
    assert over.status_code == 402 and "daily budget" in over.json()["detail"]  # 0.75 reserved + 0.5 > 1.0
    assert _wait(c, jid, A)["status"] == "done"
    me = c.get("/api/me", headers=A).json()
    assert me["name"] == "alice" and 0 < me["spent_or_reserved_today_usd"] < 0.75
    assert [j["id"] for j in c.get("/api/jobs", headers=A).json()] == [jid]
    assert c.get("/api/jobs", headers=B).json() == []  # bob's 2nd request this minute (rpm=2)
    limited = c.get("/api/me", headers=B)  # his 3rd
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1


def test_job_kinds_validation_cancel_and_events_resume(client):
    c, db = client
    assert c.post("/api/jobs", json={"kind": "build", "brief": "x"}).status_code == 422  # not a server job
    assert c.post("/api/jobs", json={"kind": "write"}).status_code == 422
    assert c.post("/api/jobs", json={"kind": "doc", "action": "translate", "text": "hi"}).status_code == 422
    r = c.post("/api/jobs", json={"kind": "research", "question": "Heat pumps?", "preset": "quick", "mock": True})
    jid = r.json()["id"]
    _wait(c, jid, {})
    body = c.get(f"/api/jobs/{jid}/events").text
    assert "event: end" in body and "id: " in body
    last = max(int(line[4:]) for line in body.splitlines() if line.startswith("id: "))
    resumed = c.get(f"/api/jobs/{jid}/events", headers={"Last-Event-ID": str(last)}).text
    assert "id: " not in resumed and "event: end" in resumed  # nothing replayed after a reconnect
    queued = db.enqueue("write", {"brief": "b"}, owner="local")
    db.claim("someone-else")  # running elsewhere
    assert c.delete(f"/api/jobs/{queued}").json()["status"] == "cancelling"


def test_metrics_health_headers_and_limits(client, monkeypatch):
    c, db = client
    r = c.get("/healthz")
    assert r.status_code == 200 and r.headers["x-request-id"] and "frame-ancestors 'none'" in \
        r.headers["content-security-policy"] and r.headers["x-content-type-options"] == "nosniff"
    assert c.get("/healthz", headers={"x-request-id": "abc123"}).headers["x-request-id"] == "abc123"
    assert c.get("/readyz").json()["database"] is True
    db.enqueue("write", {}, owner="local")
    m = c.get("/metrics").text
    assert 'swarm_jobs{kind="write",status="queued"} 1' in m or 'kind="write",status="running"' in m
    assert "swarm_http_requests_total" in m and "swarm_http_request_duration_seconds_bucket" in m
    big = c.post("/api/jobs", content=b"x" * 10, headers={"content-length": str(5_000_000),
                                                         "content-type": "application/json"})
    assert big.status_code == 413
    _, admin = create_key(db, "ops", admin=True)
    _, user = create_key(db, "u")
    assert c.get("/metrics").status_code == 401
    assert c.get("/metrics", headers={"Authorization": f"Bearer {user}"}).status_code == 401
    assert c.get("/metrics", headers={"Authorization": f"Bearer {admin}"}).status_code == 200


def test_serve_refuses_public_interface_without_auth(tmp_path, monkeypatch):
    pytest.importorskip("uvicorn")
    from swarm.server import serve

    monkeypatch.setenv("SWARM_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("SWARM_API_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="without authentication"):
        serve(host="0.0.0.0")
