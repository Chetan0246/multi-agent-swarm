import asyncio
import json

import httpx2

from swarm.research import (
    AnthropicResearchLLM,
    Citation,
    MockResearchLLM,
    ResearchConfig,
    check_references,
    citation_validator,
    deep_research,
)


def _ans(text_words=60, urls=("https://a.org/1", "https://b.org/2")):
    return json.dumps({"text": "w " * text_words,
                       "citations": [{"url": u, "title": "t", "quote": ""} for u in urls]})


def test_citation_validator():
    ok = citation_validator(min_citations=2, min_domains=2, min_words=40)
    assert ok(_ans())
    assert not ok(_ans(urls=("https://a.org/1",)))  # one source
    assert not ok(_ans(urls=("https://a.org/1", "https://www.a.org/2")))  # same domain
    assert not ok(_ans(text_words=10))  # too thin
    assert not ok("not json")


def test_check_references_removes_invalid_and_measures():
    md = ("# T\n\nA claim backed by evidence with many words to count as a factual paragraph "
          "for the checker here [1][2] and also [7].\n\nAnother long paragraph that states facts "
          "but forgets to cite anything at all which the checker should notice clearly, even though "
          "it is long enough to count.")
    clean, st = check_references(md, n_sources=2)
    assert "[7]" not in clean and "[1][2]" in clean
    assert st["invalid_refs_removed"] == 1 and st["sources_cited"] == 2
    assert st["uncited_paragraphs"] == 1


def test_deep_research_mock_end_to_end():
    rep = asyncio.run(deep_research("Q?", MockResearchLLM(seed=1), ResearchConfig(max_subquestions=4)))
    s = rep.stats
    assert s["subquestions"] == 4 and s["verified_%"] == 100
    assert s["agents"] == 4  # one worker per sub-question
    assert s["invalid_refs_removed"] == 1  # the mock cites a non-existent [99]
    assert "[99]" not in rep.markdown and "## Sources" in rep.markdown
    assert len({c.url for c in rep.sources}) == len(rep.sources) == s["sources"]


def test_weak_findings_escalate_and_unverified_are_flagged():
    rep = asyncio.run(deep_research("Q?", MockResearchLLM(seed=3), ResearchConfig(max_subquestions=4)))
    assert rep.stats["escalated"] >= 1
    # Strict gate that no mock finding can pass: findings are kept but marked unverified.
    rep = asyncio.run(deep_research("Q?", MockResearchLLM(seed=3),
                                    ResearchConfig(max_subquestions=2, min_citations=5)))
    assert rep.stats["verified_findings"] == 0
    assert all(not f.verified for f in rep.findings)


def test_research_survives_rate_limits():
    rep = asyncio.run(deep_research("Q?", MockResearchLLM(seed=1, throttle_first=3),
                                    ResearchConfig(max_subquestions=3)))
    assert rep.stats["verified_%"] == 100


# ------------------------------------------------------------- real client, mocked HTTP


def _msg(content, stop="end_turn", searches=0):
    return {"id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "stop_reason": stop, "stop_sequence": None, "content": content,
            "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0,
                      "server_tool_use": {"web_search_requests": searches}}}


def _client(handler):
    import anthropic

    llm = AnthropicResearchLLM()
    llm.client = anthropic.AsyncAnthropic(
        api_key="test", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    return llm


def test_investigate_resumes_pause_turn_and_collects_citations():
    bodies = []
    paused = [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
               "input": {"query": "heat pumps"}}]
    final = [{"type": "text", "text": "Heat pumps work to -25C.", "citations": [
        {"type": "web_search_result_location", "url": "https://energy.gov/hp", "title": "DOE",
         "cited_text": "cold-climate heat pumps", "encrypted_index": "x"}]}]

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx2.Response(200, json=_msg(paused, "pause_turn", 1) if len(bodies) == 1
                              else _msg(final, "end_turn", 2))

    text, cites, usage, searches = asyncio.run(
        _client(handler).investigate("Q", "sub", "claude-opus-5-5", "low"))
    assert text == "Heat pumps work to -25C."
    assert cites == [Citation("https://energy.gov/hp", "DOE", "cold-climate heat pumps")]
    assert searches == 3 and usage.input_tokens == 200
    b0, b1 = bodies
    assert {t["type"] for t in b0["tools"]} == {"web_search_20260209", "web_fetch_20260209"}
    assert b0["output_config"] == {"effort": "low"} and b0["fallbacks"] == "default"
    # Resume = original user turn + the paused assistant turn, and no extra "continue" message.
    assert [m["role"] for m in b1["messages"]] == ["user", "assistant"]
    assert b1["messages"][1]["content"][0]["type"] == "server_tool_use"


def test_plan_uses_structured_output():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        out = {"subquestions": [{"question": "a?", "difficulty": 1}, {"question": "b?", "difficulty": 3}]}
        return httpx2.Response(200, json=_msg([{"type": "text", "text": json.dumps(out)}]))

    subs, _ = asyncio.run(_client(handler).plan("Q", 4, "claude-opus-5-5", "medium"))
    assert [s["question"] for s in subs] == ["a?", "b?"]
    assert seen["output_config"]["format"]["type"] == "json_schema"
    assert "Never return more than 4" in seen["system"]
