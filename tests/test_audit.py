"""Regression tests for bugs found in the v3.0.0 code audit (all offline)."""
import asyncio
from types import SimpleNamespace

import httpx2
import pytest

from swarm.backends import AnthropicBatchBackend, BatchResponse, CallUsage, TransientError
from swarm.errors import SearchQuotaExhausted, UnknownModel
from swarm.evaluate import is_correct
from swarm.export import markdown_to_html
from swarm.orchestrator import Swarm
from swarm.planner import PlanConfig, Rung
from swarm.research import (
    AnthropicResearchLLM,
    Citation,
    MockResearchLLM,
    _parse_json_obj,
    config_from,
    deep_research,
)
from swarm.search import MockSearch
from swarm.task import Task


def _msg(content, stop):
    return {"id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "stop_reason": stop, "stop_sequence": None, "content": content,
            "usage": {"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}}


def _llm(handler, **kw) -> AnthropicResearchLLM:
    import anthropic

    llm = AnthropicResearchLLM(**kw)
    llm.client = anthropic.AsyncAnthropic(
        api_key="t", max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    return llm


class _QuotaGone(MockSearch):
    async def search(self, query, filters=None, purpose=None):
        raise SearchQuotaExhausted("TinyFish daily free allowance exhausted")


# ------------------------------------------------------------------ fatal search errors stop the run


def test_quota_exhaustion_reaches_the_engine_instead_of_the_model():
    llm = _llm(lambda req: httpx2.Response(500), search=_QuotaGone())
    with pytest.raises(SearchQuotaExhausted):
        asyncio.run(llm._run_tool("web_search", {"query": "x"}, {}))


def test_fatal_error_drains_the_queue_without_more_calls():
    calls = []

    class Backend:
        bulk = False

        async def run_batch(self, tasks, effort, model):
            calls.append(1)
            raise SearchQuotaExhausted("quota gone")

    cfg = PlanConfig(ladder=(Rung("claude-opus-5-5", "low"),), adaptive=False, static_batch=1,
                     max_batch=1, hedging=False)
    engine = Swarm(Backend(), cfg, agents=1)
    run = asyncio.run(engine.arun([Task(f"t{i}", f"p{i}", validator=lambda a: True) for i in range(20)]))
    assert len(calls) == 1  # the other 19 tasks were cancelled, not each tried and failed
    assert isinstance(engine.fatal, SearchQuotaExhausted)
    assert all(not r.ok and r.error.startswith("fatal") for r in run.results.values())


def test_deep_research_raises_fatal_error_when_nothing_was_found():
    class Dead(MockResearchLLM):
        async def investigate(self, question, sub_question, model, effort):
            raise SearchQuotaExhausted("TinyFish daily free allowance exhausted")

    with pytest.raises(SearchQuotaExhausted):
        asyncio.run(deep_research("Q?", Dead(), config_from("quick")))


def test_deep_research_reports_a_partial_run():
    class HalfDead(MockResearchLLM):
        n = 0

        async def investigate(self, question, sub_question, model, effort):
            HalfDead.n += 1
            if HalfDead.n > 1:
                raise SearchQuotaExhausted("TinyFish daily free allowance exhausted")
            return await super().investigate(question, sub_question, model, "high")

    rep = asyncio.run(deep_research("Q?", HalfDead(fail_low=False), config_from("quick", max_agents=1)))
    assert rep.stats["stopped_early"] == "SearchQuotaExhausted"
    assert "Research stopped early" in rep.markdown


# ------------------------------------------------------------------ cost accounting


def test_usage_of_a_worker_call_that_fails_midway_is_kept():
    n = []

    def handler(req):
        n.append(1)
        if len(n) == 1:
            return httpx2.Response(200, json=_msg([{"type": "tool_use", "id": "t1", "name": "web_search",
                                                    "input": {"query": "q"}}], "tool_use"))
        return httpx2.Response(503, json={"type": "error", "error": {"type": "overloaded_error",
                                                                      "message": "busy"}})

    llm = _llm(handler, search=MockSearch())
    with pytest.raises(TransientError):
        asyncio.run(llm.investigate("Q", "sub", "claude-opus-5-5", "low"))
    lost = llm.take_lost_usage()
    assert lost.input_tokens == 1000 and lost.output_tokens == 100  # the billed first turn
    assert llm.take_lost_usage().input_tokens == 0  # taken once


def test_workers_are_priced_by_the_worker_model():
    cfg = config_from("quick")
    opus = asyncio.run(deep_research("Q?", MockResearchLLM(seed=3), cfg))
    cheap = MockResearchLLM(seed=3)
    cheap.worker_model = "claude-haiku-5-5"  # type: ignore[attr-defined]
    haiku = asyncio.run(deep_research("Q?", cheap, cfg))
    assert haiku.stats["usd"] < opus.stats["usd"]


def test_unknown_worker_model_fails_before_any_spend():
    llm = MockResearchLLM()
    llm.worker_model = "claude-unknown-9"  # type: ignore[attr-defined]
    planned = []
    orig = llm.plan

    async def plan(*a):
        planned.append(1)
        return await orig(*a)

    llm.plan = plan  # type: ignore[method-assign]
    with pytest.raises(UnknownModel, match="claude-unknown-9"):
        asyncio.run(deep_research("Q?", llm, config_from("quick")))
    assert not planned


# ------------------------------------------------------------------ engine, batches, eval, export


def test_journal_resume_requires_the_same_prompt(tmp_path):
    class Echo:
        bulk = False

        async def run_batch(self, tasks, effort, model):
            return BatchResponse({t.id: t.prompt.upper() for t in tasks}, CallUsage(input_tokens=1))

    cfg = PlanConfig(ladder=(Rung("claude-opus-5-5", "low"),), adaptive=False, hedging=False)
    j = str(tmp_path / "journal.jsonl")
    Swarm(Echo(), cfg, journal_path=j).run([Task("t1", "alpha", validator=lambda a: True)])
    run = Swarm(Echo(), cfg, journal_path=j).run([Task("t1", "beta", validator=lambda a: True)])
    assert run.results["t1"].answer == "BETA" and run.counters["resumed"] == 0


def test_batch_invalid_request_is_not_retried():
    be = object.__new__(AnthropicBatchBackend)
    err = SimpleNamespace(error=SimpleNamespace(type="invalid_request_error"), type="error")
    item = SimpleNamespace(custom_id="r1", result=SimpleNamespace(type="errored", error=err))

    async def create(**kw):
        return SimpleNamespace(id="b", processing_status="ended")

    async def results(_):
        async def gen():
            yield item
        return gen()

    be.client = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(create=create,
                                                                                 results=results)))

    async def go():
        fut = asyncio.get_running_loop().create_future()
        await be._submit_and_poll([("r1", {}, fut)])
        return fut.exception()

    assert isinstance(asyncio.run(go()), ValueError)  # not TransientError, which would resubmit forever


def test_eval_matches_whole_words_only():
    assert not is_correct("# Q\n\nWe can trust this result.", ["Rust"])
    assert is_correct("# Q\n\nThe language is Rust.", ["Rust"])
    assert not is_correct("# Q\n\nAbout 57000 years.", ["5700"])
    assert is_correct("# Q\n\nAbout 5,730 years.", ["5730"])


def test_html_links_keep_query_strings():
    html = markdown_to_html("See [the data](https://a.org/x?a=1&b=2).")
    assert 'href="https://a.org/x?a=1&amp;b=2"' in html and "&amp;amp;" not in html


def test_source_titles_cannot_inject_markdown():
    class Hostile(MockResearchLLM):
        async def investigate(self, question, sub_question, model, effort):
            text, cites, u, s = await super().investigate(question, sub_question, model, "high")
            cites[0] = Citation(cites[0].url, "Evil](https://evil.test) ![x](https://evil.test/p.png", "q")
            return text, cites, u, s

    rep = asyncio.run(deep_research("Q?", Hostile(fail_low=False), config_from("quick")))
    sources = rep.markdown.split("## Sources", 1)[1]
    assert "](https://evil.test" not in sources and "![" not in sources


def test_follow_up_ids_never_collide():
    first = asyncio.run(deep_research("Q?", MockResearchLLM(seed=1), config_from("quick")))
    first.findings = [f for f in first.findings if f.sub_id != "q2"]  # a gap, as when a finding is empty
    second = asyncio.run(deep_research("More?", MockResearchLLM(seed=2), config_from("quick"), prior=first))
    ids = [f.sub_id for f in second.findings]
    assert len(ids) == len(set(ids))


def test_parse_json_obj_ignores_non_objects():
    assert _parse_json_obj("[1, 2]") == {}
    assert _parse_json_obj('note: {"answer": "x"}') == {"answer": "x"}


# ------------------------------------------------------------------ server


def test_server_follow_up_inherits_options_and_validates_kind(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from swarm.server import create_app

    monkeypatch.delenv("SWARM_API_TOKEN", raising=False)
    app = create_app(state_dir=tmp_path, allowed_hosts=["testserver"])
    with TestClient(app) as c:
        bad = c.post("/api/research", json={"question": "Q?", "kind": "videos", "mock": True})
        assert bad.status_code == 422
        r = c.post("/api/research", json={"question": "Heat pumps?", "preset": "quick", "mock": True,
                                          "include_domains": ["nrel.gov"]})
        job_id = r.json()["id"]
        c.get(f"/api/jobs/{job_id}/events")  # wait for the end of the stream
        f = c.post(f"/api/jobs/{job_id}/follow-up", json={"question": "And costs?"})
        assert f.status_code == 202, f.text
        child = app.state.store.jobs[f.json()["id"]]
        assert child.options["preset"] == "quick" and child.options["include_domains"] == ["nrel.gov"]


def test_eval_counts_spend_of_a_failed_run():
    from swarm.evaluate import EvalItem, run_eval

    class FailsAtSynthesis(MockResearchLLM):
        async def synthesize(self, *a):
            raise RuntimeError("400 at synthesis")

    rs = asyncio.run(run_eval([EvalItem("1", "Q?", ["x"])], FailsAtSynthesis(), None, config_from("quick")))
    assert rs[0].error and rs[0].usd > 0  # plan + workers were billed before the failure


def test_server_rejects_foreign_host_headers(tmp_path, monkeypatch):
    """DNS rebinding: a page on evil.test resolved to 127.0.0.1 sends Host: evil.test."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from swarm.server import allowed_hosts_for, create_app

    monkeypatch.delenv("SWARM_API_TOKEN", raising=False)
    monkeypatch.delenv("SWARM_ALLOWED_HOSTS", raising=False)
    app = create_app(state_dir=tmp_path)  # default: localhost names only
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/healthz", headers={"host": "localhost:8000"}).status_code == 200
        assert c.get("/healthz", headers={"host": "evil.test"}).status_code == 400
        r = c.post("/api/research", headers={"host": "evil.test:8000"}, json={"question": "Q?", "mock": True})
        assert r.status_code == 400
    monkeypatch.setenv("SWARM_ALLOWED_HOSTS", "swarm.lan")
    assert "swarm.lan" in allowed_hosts_for({}) and "evil.test" not in allowed_hosts_for({}, "0.0.0.0")
