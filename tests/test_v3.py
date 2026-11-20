"""v3: free client-side search (TinyFish), quote grounding, documents, cache, export, server."""
import asyncio
import json

import httpx2
import pytest

from swarm.app import ResearchOptions, resolve, run_research
from swarm.errors import SearchAuthError, SearchQuotaExhausted
from swarm.export import from_json, markdown_to_html, to_html, to_json
from swarm.research import PRESETS, AnthropicResearchLLM, MockResearchLLM, config_from, deep_research
from swarm.search import CachedProvider, LocalDocs, MockSearch, SearchFilters, quote_is_grounded
from swarm.search.tinyfish import TinyFish

# ------------------------------------------------------------------ grounding


def test_quote_grounding_tolerates_markup_and_ellipses():
    page = "## Results\n\nThe **heat pump** kept a COP above 1.5 at −20 °C, said the report."
    from swarm.search import normalize

    hay = normalize(page)
    assert quote_is_grounded("The heat pump kept a COP above 1.5", hay)
    assert quote_is_grounded("The heat pump kept a COP ... said the report", hay)
    assert not quote_is_grounded("The heat pump failed below zero degrees entirely", hay)
    assert not quote_is_grounded("short", hay)  # too short to verify


# ------------------------------------------------------------------ TinyFish client (mock HTTP)


def _tf(handler) -> TinyFish:
    return TinyFish(api_key="k", http=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))


def test_tinyfish_search_and_fetch_shapes():
    seen = []

    def handler(req):
        seen.append(req)
        if req.method == "GET":
            return httpx2.Response(200, json={"query": "q", "page": 0, "total_results": 1, "results": [
                {"position": 1, "site_name": "a.org", "title": "A", "snippet": "S", "url": "https://a.org/x"}]})
        body = json.loads(req.content)
        return httpx2.Response(200, json={
            "results": [{"url": body["urls"][0], "final_url": body["urls"][0], "title": "T", "text": "# Page"}],
            "errors": [{"url": u, "error": "timeout"} for u in body["urls"][1:]]})

    tf = _tf(handler)
    f = SearchFilters(exclude_domains=["pinterest.com"], recency_days=7, domain_type="news")
    hits = asyncio.run(tf.search("heat pumps", f))
    assert hits[0].url == "https://a.org/x" and hits[0].snippet == "S"
    q = dict(seen[0].url.params)
    assert seen[0].headers["x-api-key"] == "k" and q["query"] == "heat pumps"
    assert q["exclude_domains"] == "pinterest.com" and q["recency_minutes"] == str(7 * 1440)
    assert q["domain_type"] == "news"
    pages = asyncio.run(tf.fetch(["https://a.org/x", "https://b.org/y"]))
    assert pages[0].text == "# Page" and pages[1].error == "timeout"
    assert json.loads(seen[1].content)["format"] == "markdown"
    assert tf.stats == {"searches": 1, "fetched_urls": 1, "fetch_errors": 1}


def test_tinyfish_errors_are_actionable():
    tf = _tf(lambda req: httpx2.Response(401, json={"error": {"code": "UNAUTHORIZED", "message": "bad"}}))
    with pytest.raises(SearchAuthError, match="401"):
        asyncio.run(tf.search("x"))
    body = {"error": {"code": "RATE_LIMIT_EXCEEDED", "details": {"reason": "free_allowance_exhausted"}}}
    tf = _tf(lambda req: httpx2.Response(429, json=body))
    with pytest.raises(SearchQuotaExhausted) as e:
        asyncio.run(tf.search("x"))
    assert "00:00 UTC" in str(e.value)


def test_tinyfish_retries_transient_errors():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx2.Response(503, headers={"retry-after": "0"})
        return httpx2.Response(200, json={"results": []})

    assert asyncio.run(_tf(handler).search("x")) == [] and len(calls) == 3


def test_missing_tinyfish_key(monkeypatch):
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    with pytest.raises(Exception, match="TINYFISH_API_KEY"):
        TinyFish()


# ------------------------------------------------------------------ cache and documents


def test_cache_serves_repeats_without_calling_provider(tmp_path):
    inner = MockSearch()
    c = CachedProvider(inner, tmp_path / "c.sqlite")
    a = asyncio.run(c.search("heat pumps"))
    b = asyncio.run(c.search("heat pumps"))
    assert a == b and inner.stats["searches"] == 1 and c.hits == 1
    asyncio.run(c.fetch([a[0].url]))
    asyncio.run(c.fetch([a[0].url]))
    assert inner.stats["fetched_urls"] == 1


def test_local_docs_search_and_read(tmp_path):
    (tmp_path / "a.md").write_text("# Heat pumps\n\nCold-climate heat pumps keep working at minus 20 degrees.\n\n"
                                   "Unrelated paragraph about gardening and tomatoes.")
    (tmp_path / "b.txt").write_text("Gas boilers have an efficiency near 90 percent.")
    docs = LocalDocs([tmp_path])
    hits = asyncio.run(docs.search("heat pumps cold climate"))
    assert hits and hits[0].url.startswith("file://") and "a.md" in hits[0].url
    page = asyncio.run(docs.fetch([hits[0].url]))[0]
    assert "minus 20 degrees" in page.text
    assert docs.stats["documents"] == 2


# ------------------------------------------------------------------ worker tool loop (mock Claude + mock search)


def _msg(content, stop):
    return {"id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "stop_reason": stop, "stop_sequence": None, "content": content,
            "usage": {"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 500,
                      "cache_creation_input_tokens": 0}}


def test_tool_loop_searches_fetches_and_drops_ungrounded_quotes():
    search = MockSearch()
    hits = asyncio.run(search.search("cold climate heat pumps"))
    url1, url2 = hits[0].url, hits[1].url
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        n = len(bodies)
        if n == 1:
            return httpx2.Response(200, json=_msg([{"type": "tool_use", "id": "t1", "name": "web_search",
                                                    "input": {"query": "cold climate heat pumps"}}], "tool_use"))
        if n == 2:
            return httpx2.Response(200, json=_msg([{"type": "tool_use", "id": "t2", "name": "web_fetch",
                                                    "input": {"urls": [url1, url2]}}], "tool_use"))
        final = {"answer": "Cold-climate heat pumps keep a COP above 1.5 at -20C.", "sources": [
            {"url": url1, "quote": "cold-climate heat pumps kept a coefficient of performance above 1.5"},
            {"url": url2, "quote": "Installation costs varied by region"},
            {"url": url2, "quote": "heat pumps never work below freezing point at all"}]}  # fabricated
        return httpx2.Response(200, json=_msg([{"type": "text", "text": json.dumps(final)}], "end_turn"))

    import anthropic

    llm = AnthropicResearchLLM(search=search, filters=SearchFilters(exclude_domains=["x.com"]))
    llm.client = anthropic.AsyncAnthropic(api_key="t", max_retries=0,
                                          http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    text, cites, usage, paid = asyncio.run(llm.investigate("Q", "sub", "claude-opus-5-5", "low"))
    assert "COP above 1.5" in text and paid == 0
    assert [c.url for c in cites] == [url1, url2]  # the fabricated quote was dropped
    assert llm.stats["ungrounded_quotes_dropped"] == 1
    assert {t["name"] for t in bodies[0]["tools"]} == {"web_search", "web_fetch"}
    assert bodies[0]["cache_control"] == {"type": "ephemeral"}
    assert bodies[0]["output_config"]["format"]["type"] == "json_schema"
    # Tool results go back as one user message, wrapped as untrusted content.
    tr = bodies[2]["messages"][-1]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "t2"
    assert tr["content"].startswith("<untrusted_content>") and url1 in tr["content"]
    assert usage.cache_read_tokens == 1500


def test_last_tool_turn_forbids_more_tools():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        if len(bodies) <= 2:
            return httpx2.Response(200, json=_msg([{"type": "tool_use", "id": f"t{len(bodies)}", "name": "web_search",
                                                    "input": {"query": "q"}}], "tool_use"))
        return httpx2.Response(200, json=_msg([{"type": "text", "text": '{"answer": "a", "sources": []}'}],
                                              "end_turn"))

    import anthropic

    llm = AnthropicResearchLLM(search=MockSearch(), max_tool_turns=2)
    llm.client = anthropic.AsyncAnthropic(api_key="t", max_retries=0,
                                          http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    asyncio.run(llm.investigate("Q", "sub", "claude-opus-5-5", "low"))
    assert [b["tool_choice"]["type"] for b in bodies] == ["auto", "auto", "none"]


# ------------------------------------------------------------------ presets, follow-ups, export


def test_presets_scale_cost_and_breadth():
    q, s, d = (config_from(p) for p in ("quick", "standard", "deep"))
    assert q.max_subquestions < s.max_subquestions < d.max_subquestions
    assert q.max_usd < s.max_usd < d.max_usd and not q.check_claims
    assert config_from("deep", max_usd=1.0).max_usd == 1.0
    with pytest.raises(ValueError):
        config_from("huge")
    assert set(PRESETS) == {"quick", "standard", "deep"}


def test_follow_up_keeps_prior_findings():
    first = asyncio.run(deep_research("Q?", MockResearchLLM(seed=1), config_from("quick")))
    second = asyncio.run(deep_research("What about costs?", MockResearchLLM(seed=2), config_from("quick"),
                                       prior=first))
    assert len(second.findings) == len(first.findings) + second.stats["subquestions"]
    assert second.stats["followup_of"] == "Q?" and "Follow-up request" in second.question
    assert second.stats["verified_%"] <= 100
    assert {f.sub_id for f in first.findings} <= {f.sub_id for f in second.findings}


def test_json_round_trip_and_safe_html():
    rep = asyncio.run(deep_research("Q <script>alert(1)</script>?", MockResearchLLM(seed=1), config_from("quick")))
    again = from_json(to_json(rep))
    assert again.markdown == rep.markdown and len(again.findings) == len(rep.findings)
    page = to_html(rep)
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "Content-Security-Policy" in page and 'href="#src-1"' in page
    assert markdown_to_html("[x](javascript:alert(1))") == "<p>[x](javascript:alert(1))</p>"  # not linkified


def test_app_resolve_mock_and_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "swarm.toml"
    cfg.write_text('[research]\npreset = "quick"\nmax_usd = 0.5\n')
    llm, rc, info = resolve(ResearchOptions("Q?", mock=True, config_path=str(cfg)))
    assert isinstance(llm, MockResearchLLM) and info["preset"] == "quick" and rc.max_usd == 0.5
    monkeypatch.setenv("SWARM_MAX_USD", "0.9")
    _, rc, _ = resolve(ResearchOptions("Q?", mock=True, config_path=str(cfg)))
    assert rc.max_usd == 0.9  # environment beats config file
    _, rc, _ = resolve(ResearchOptions("Q?", mock=True, max_usd=1.1, config_path=str(cfg)))
    assert rc.max_usd == 1.1  # explicit option beats both
    rep = asyncio.run(run_research(ResearchOptions("Q?", mock=True, config_path=str(cfg))))
    assert rep.stats["run_preset"] == "quick"


# ------------------------------------------------------------------ server


def test_server_end_to_end_with_mock(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from swarm.server import create_app

    monkeypatch.setenv("SWARM_API_TOKEN", "secret")
    app = create_app(state_dir=tmp_path)
    auth = {"Authorization": "Bearer secret"}
    with TestClient(app) as c:
        assert c.get("/healthz").json()["ok"]
        assert "Swarm Research" in c.get("/").text
        assert c.get("/api/jobs").status_code == 401  # token required
        est = c.get("/api/estimate?preset=quick", headers=auth).json()
        assert est["low_usd"] < est["high_usd"]
        r = c.post("/api/research", json={"question": "Do heat pumps work?", "preset": "quick", "mock": True},
                   headers=auth)
        assert r.status_code == 202
        job_id = r.json()["id"]
        events = c.get(f"/api/jobs/{job_id}/events", headers=auth).text
        assert '"event": "plan"' in events and "event: end" in events
        j = c.get(f"/api/jobs/{job_id}", headers=auth).json()
        assert j["status"] == "done" and j["stats"]["verified_%"] > 0
        assert "## Sources" in c.get(f"/api/jobs/{job_id}/report", headers=auth).text
        assert "<!doctype html>" in c.get(f"/api/jobs/{job_id}/report?format=html", headers=auth).text
        f = c.post(f"/api/jobs/{job_id}/follow-up", json={"question": "And costs?"}, headers=auth)
        assert f.status_code == 202
        c.get(f"/api/jobs/{f.json()['id']}/events", headers=auth)
        assert c.get(f"/api/jobs/{f.json()['id']}", headers=auth).json()["parent"] == job_id
        bad = c.post("/api/research", json={"question": "Q?", "preset": "huge"}, headers=auth)
        assert bad.status_code == 422
    assert list(tmp_path.glob("*.json"))  # finished jobs persisted
