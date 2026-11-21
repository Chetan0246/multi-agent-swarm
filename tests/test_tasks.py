"""Task kinds for `swarm run`: text, code (sandboxed tests) and web lookups (offline)."""
import asyncio
import json

import httpx2
import pytest

from swarm.errors import ConfigError
from swarm.openai_compat import OpenAICompatBackend, OpenAICompatLLM
from swarm.planner import PlanConfig, Rung
from swarm.research import MockResearchLLM
from swarm.sandbox import run_python, strip_fences
from swarm.tasks import WebTaskBackend, load_tasks, run_kinds, web_validator

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
CFG = PlanConfig(ladder=(Rung(MODEL, "low"), Rung(MODEL, "medium")), adaptive=False, hedging=False)


def _jsonl(tmp_path, rows):
    p = tmp_path / "tasks.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_sandbox_runs_tests_without_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    assert run_python("def add(a, b):\n    return a + b", "assert add(2, 3) == 5").ok
    assert not run_python("def add(a, b):\n    return a - b", "assert add(2, 3) == 5").ok
    env = run_python("import os\nprint(dict(os.environ))", "").output
    assert "sk-must-not-leak" not in env and "OPENAI_API_KEY" not in env
    slow = run_python("while True:\n    pass", "", timeout=1.0)
    assert slow.timed_out and not slow.ok
    assert strip_fences("```python\nx = 1\n```") == "x = 1"


def test_code_tasks_need_explicit_opt_in(tmp_path):
    p = _jsonl(tmp_path, [{"id": "f", "kind": "code", "prompt": "write f", "tests": "assert f() == 1"}])
    with pytest.raises(ConfigError, match="allow-code-execution"):
        load_tasks(p)
    assert load_tasks(p, allow_code=True)[0].validator is not None
    with pytest.raises(ConfigError, match="unknown kind"):
        load_tasks(_jsonl(tmp_path, [{"id": "x", "kind": "video", "prompt": "p"}]))
    with pytest.raises(ConfigError, match="unique"):
        load_tasks(_jsonl(tmp_path, [{"id": "x", "prompt": "a"}, {"id": "x", "prompt": "b"}]))


def _fake_provider(seen):
    """Answers text tasks and writes code; the first attempt at `inc` is wrong (forces a retry)."""
    attempts = {"inc": 0}

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        tasks = json.loads(body["messages"][1]["content"])
        rows = []
        for t in tasks:
            if t["id"] == "inc":
                attempts["inc"] += 1
                wrong, right = "def inc(x):\n    return x", "```python\ndef inc(x):\n    return x + 1\n```"
                code = wrong if attempts["inc"] == 1 else right
                rows.append({"id": "inc", "answer": code, "confidence": 0.9})
            elif t["id"] == "sq":
                rows.append({"id": "sq", "answer": "def sq(x):\n    return x * x", "confidence": 0.9})
            else:
                rows.append({"id": t["id"], "answer": "positive", "confidence": 0.9})
        return httpx2.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps(
            {"results": rows})}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 300, "completion_tokens": 60}})

    return handler


def test_mixed_text_code_and_web_workload(tmp_path):
    p = _jsonl(tmp_path, [
        {"id": "s1", "prompt": "Sentiment of 'great!'", "expected": "positive"},
        {"id": "inc", "kind": "code", "prompt": "Write inc(x) returning x + 1", "tests": "assert inc(1) == 2"},
        {"id": "sq", "kind": "code", "prompt": "Write sq(x)", "tests": "assert sq(3) == 9"},
        {"id": "w1", "kind": "web", "prompt": "How do heat pumps do in the cold?", "min_sources": 2},
    ])
    tasks = load_tasks(p, allow_code=True)
    seen: list[dict] = []

    def client():
        return OpenAICompatLLM(base_url="https://p.test/v1", api_key="k", require_search=False,
                               http=httpx2.AsyncClient(transport=httpx2.MockTransport(_fake_provider(seen))))

    from swarm.backends import CODE_SYSTEM_PROMPT

    backends = {"text": OpenAICompatBackend(client()), "code": OpenAICompatBackend(client(), CODE_SYSTEM_PROMPT),
                "web": WebTaskBackend(MockResearchLLM(fail_low=False))}
    runs = asyncio.run(run_kinds(tasks, backends, CFG, state_dir=tmp_path))
    res = {tid: r for run in runs for tid, r in run.report.results.items()}
    assert all(r.ok for r in res.values()), {k: (r.ok, r.error) for k, r in res.items()}
    assert res["inc"].attempts == 2  # failed its tests once, then escalated and passed
    web = json.loads(res["w1"].answer)
    assert web["answer"] and len(web["sources"]) >= 2 and web["sources"][0]["quote"]
    code_calls = [b for b in seen if "coding tasks" in b["messages"][0]["content"]]
    assert code_calls and code_calls[0]["response_format"] == {"type": "json_object"}
    assert any(r.kind == "web" and r.extra["web_searches"] > 0 for r in runs)


def test_web_validator():
    good = json.dumps({"answer": "Released in 2008.", "sources": [{"url": "https://a.org"}, {"url": "https://b.org"}]})
    assert web_validator("2008", None, 2)(good)
    assert not web_validator("2009", None, 1)(good)
    assert not web_validator(None, None, 3)(good)  # too few sources
    assert not web_validator(None, None, 1)(json.dumps({"answer": "x", "sources": []}))
    assert not web_validator(None, None, 1)("not json")


def test_run_cli_mixed_mock(tmp_path, monkeypatch, capsys):
    import sys

    from swarm.__main__ import main

    p = _jsonl(tmp_path, [{"id": "a", "prompt": "What is 2 + 2?", "expected": "4"},
                          {"id": "w", "kind": "web", "prompt": "Heat pump COP at -20C?"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["swarm", "run", str(p), "--mock", "--out", str(out),
                                      "--state-dir", str(tmp_path / "st")])
    main()
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["text", "web"] and all("ok" in r for r in rows)
    assert "2 tasks (1 text, 1 web)" in capsys.readouterr().err
