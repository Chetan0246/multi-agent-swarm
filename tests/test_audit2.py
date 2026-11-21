"""Regression tests for bugs found in the 3.3.0 audit (all offline)."""
import asyncio
import json
import threading
import time

import pytest

from swarm.jobs import BudgetExceeded, JobDB
from swarm.worker import Worker

# ------------------------------------------------------------------ queue and worker


def test_a_stale_worker_cannot_finish_a_job_another_worker_took_over(tmp_path):
    db = JobDB(tmp_path / "j.sqlite", stale_after=0.01)
    jid = db.enqueue("write", {})
    db.claim("w1")
    time.sleep(0.02)
    db.recover()  # w1 looked dead
    db.claim("w2")
    assert db.heartbeat(jid, "w1") == "lost" and db.heartbeat(jid, "w2") == "running"
    assert db.finish(jid, "done", result={"from": "w1"}, worker="w1") is False  # discarded
    assert db.get(jid)["status"] == "running" and db.get(jid)["worker"] == "w2"
    assert db.finish(jid, "done", result={"from": "w2"}, worker="w2") is True
    assert db.get(jid)["result"] == {"from": "w2"}
    db.close()


def test_worker_stops_a_job_it_has_lost(tmp_path):
    db = JobDB(tmp_path / "j.sqlite")
    calls = []

    async def slow(job, emit, db_, cfg):
        calls.append(job["id"])
        await asyncio.sleep(30)
        return {}, 0.0

    jid = db.enqueue("write", {})

    async def main():
        stop = asyncio.Event()
        w = Worker(db, poll=0.02, heartbeat=0.05, runners={"write": slow}, name="w1")
        task = asyncio.create_task(w.run(stop))
        for _ in range(100):
            if calls:
                break
            await asyncio.sleep(0.02)
        with db.tx() as c:  # another worker re-claimed it while w1 stalled
            c.execute("UPDATE jobs SET worker = 'w2' WHERE id = ?", (jid,))
        for _ in range(100):
            if not w.running:
                break
            await asyncio.sleep(0.02)
        assert not w.running  # w1 stopped its copy
        stop.set()
        await task

    asyncio.run(main())
    assert db.get(jid)["worker"] == "w2" and db.get(jid)["status"] == "running"  # w2's claim untouched
    db.close()


def test_failed_and_cancelled_jobs_still_count_against_the_budget(tmp_path):
    db = JobDB(tmp_path / "j.sqlite")
    failed = db.enqueue("write", {}, owner="k", cap_usd=1.0)
    db.claim("w")
    db.finish(failed, "error", error="boom", worker="w")  # cost unknown: counts the cap
    assert db.spend_today("k") == pytest.approx(1.0)
    known = db.enqueue("write", {}, owner="k", cap_usd=1.0)
    db.claim("w")
    db.finish(known, "cancelled", usd=0.2, worker="w")  # cost known: counts that
    never_started = db.enqueue("write", {}, owner="k", cap_usd=5.0)
    db.cancel(never_started)  # cancelled before running: costs nothing
    assert db.spend_today("k") == pytest.approx(1.2)
    db.close()


def test_write_runner_reports_spend_when_it_fails(tmp_path, monkeypatch):
    from swarm import worker as wmod
    from swarm.workflows import chat as chatmod

    class Boom(Exception):
        pass

    async def failing_write(brief, chat, **kw):
        await chat.complete("s", "u" * 4000)
        raise Boom("model refused")

    monkeypatch.setattr("swarm.workflows.write.write", failing_write)
    monkeypatch.setattr(chatmod, "make_chat",
                        lambda *a, **k: chatmod.MockChat(lambda *x: "y" * 400, model="claude-haiku-5-5"))
    db = JobDB(tmp_path / "j.sqlite")
    job = {"id": "x", "params": {"brief": "b"}, "cap_usd": 1.0}
    with pytest.raises(Boom) as e:
        asyncio.run(wmod.run_write(job, lambda *a: None, db, None))
    assert getattr(e.value, "swarm_usd", 0) > 0
    db.close()


def test_budget_check_is_atomic_under_concurrent_submissions(tmp_path):
    path = tmp_path / "j.sqlite"
    JobDB(path).close()
    results: list[str] = []

    def submit():
        db = JobDB(path)  # its own connection, like a separate API worker thread
        try:
            db.enqueue("write", {}, owner="k", cap_usd=0.6, daily_budget=1.0)
            results.append("ok")
        except BudgetExceeded:
            results.append("refused")
        finally:
            db.close()

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1 and results.count("refused") == 7  # 0.6 fits once in 1.0


def test_final_event_is_written_with_the_status(tmp_path):
    db = JobDB(tmp_path / "j.sqlite")
    jid = db.enqueue("write", {})
    db.claim("w")
    db.finish(jid, "done", result={}, usd=0.1, worker="w")
    last = db.events(jid)[-1][1]
    assert last["event"] == "done" and last["usd"] == 0.1
    assert db.finish(jid, "error", error="late", worker="w") is False  # no second outcome, no stray event
    assert db.events(jid)[-1][1]["event"] == "done"
    db.close()


# ------------------------------------------------------------------ server


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from swarm.server import create_app

    for v in ("SWARM_API_TOKEN", "SWARM_METRICS_TOKEN", "SWARM_ALLOWED_HOSTS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SWARM_MAX_BODY_BYTES", "1000")
    clients = []

    def make(hosts):
        c = TestClient(create_app(state_dir=tmp_path, allowed_hosts=hosts, embedded_worker=False))
        clients.append(c)
        return c.__enter__()

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def test_chunked_and_malformed_bodies_are_limited(make_client):
    c = make_client(["testserver"])

    def chunks():
        for _ in range(10):
            yield b"x" * 500  # 5,000 bytes, no Content-Length

    assert c.post("/api/jobs", content=chunks(), headers={"content-type": "application/json"}).status_code == 413
    bad = c.post("/api/jobs", content=b"{}", headers={"content-length": "abc", "content-type": "application/json"})
    assert bad.status_code == 400
    assert c.post("/api/jobs", json={"kind": "write"}).status_code == 422  # small bodies still work


def test_open_mode_is_refused_when_a_public_name_is_allowed(make_client):
    c = make_client(["testserver", "swarm.example.com"])
    r = c.get("/api/jobs")
    assert r.status_code == 401 and "public name" in r.json()["detail"]
    assert c.get("/healthz").status_code == 200  # liveness stays open for probes


def test_request_ids_are_sanitised(make_client):
    c = make_client(["testserver"])
    rid = c.get("/healthz", headers={"x-request-id": 'abc"<script>\\x'}).headers["x-request-id"]
    assert rid == "abcscriptx"


# ------------------------------------------------------------------ workflows


def test_review_tolerates_odd_line_numbers():
    from swarm.workflows.review import ground

    text = "a = 1\nb = a / 0\n"
    kept, dropped = ground([{"line": "12-14", "severity": "high", "quote": "b = a / 0", "title": "t"},
                            {"line": "L2", "severity": "low", "quote": "a = 1", "title": "u"},
                            {"line": None, "severity": "low", "quote": "b = a / 0", "title": "v"}], "f.py", text)
    assert dropped == 0 and [k.line for k in kept] == [2, 1, 2]


def test_summarize_merge_rounds_are_bounded():
    from swarm.workflows.chat import MockChat
    from swarm.workflows.doc import summarize

    calls = {"summary": 0}

    def respond(role, system, user, schema):
        if role == "doc.summary":
            calls["summary"] += 1
            return "word " * 12_000  # a model that never gets shorter (60k chars)
        return "final"

    text = "\n\n".join(f"## {i}\n\n" + "text " * 2000 for i in range(6))
    res = asyncio.run(summarize(text, MockChat(respond), words=100, max_chars=12_000))
    assert res.text.strip() == "final" and calls["summary"] < 20  # finished, no endless loop


def _build_model(tests, fix_path="calc.py"):
    spec = {"name": "c", "summary": "s", "stack": "py", "start": "python calc.py",
            "files": [{"path": "calc.py", "purpose": "p", "interface": "i", "depends_on": []}],
            "setup": [{"run": "python -c \"import calc\"", "cwd": "."}], "test": tests}

    def respond(role, system, user, schema):
        if role == "build.plan":
            return json.dumps(spec)
        if role == "build.file":
            return "VALUE = 1\n"
        return json.dumps({"analysis": "a", "edits": [{"path": fix_path, "content": "VALUE = 2\n"},
                                                      {"path": "calc.py", "content": "VALUE = 3\n"}],
                           "setup": [], "test": [{"run": "python -c \"import calc; assert calc.VALUE == 3\"",
                                                  "cwd": "."}]})

    return respond


def test_build_without_tests_does_not_claim_a_pass(tmp_path):
    from swarm.workflows.build import build
    from swarm.workflows.chat import MockChat

    res = asyncio.run(build("x", MockChat(_build_model([])), tmp_path / "a", allow_execution=True))
    assert res.tests_passed is None and any("no test command" in n for n in res.notes)


def test_build_skips_an_unsafe_fixer_edit_and_keeps_going(tmp_path):
    from swarm.workflows.build import build
    from swarm.workflows.chat import MockChat

    failing = [{"run": "python -c \"import calc; assert calc.VALUE == 3\"", "cwd": "."}]
    res = asyncio.run(build("x", MockChat(_build_model(failing, fix_path="../outside.py")), tmp_path / "b",
                            allow_execution=True))
    assert res.tests_passed is True and res.fix_rounds == 1
    assert not (tmp_path / "outside.py").exists() and any("unsafe edit" in n for n in res.notes)


def test_sandbox_runs_from_threads_with_limits():
    from swarm.sandbox import run_python

    async def main():
        return await asyncio.gather(*(asyncio.to_thread(run_python, "import resource\n"
                                                         "print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])", "")
                                      for _ in range(4)))

    results = asyncio.run(main())
    assert all(r.ok and r.output.strip() == "64" for r in results)  # limits applied, no preexec_fn
