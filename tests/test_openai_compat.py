"""The OpenAI-compatible provider, against a fake chat-completions server (offline)."""
import asyncio
import json

import httpx2
import pytest

from swarm.app import ResearchOptions, register_prices, resolve
from swarm.backends import RateLimited, SpendCapReached, TransientError
from swarm.errors import ConfigError, MissingKey
from swarm.openai_compat import OpenAICompatLLM
from swarm.pricing import PRICES
from swarm.research import config_from, deep_research
from swarm.search import MockSearch

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
QUERY = "cold climate heat pumps"


def _reply(content=None, tool_calls=None, usage=(1000, 200, 0)):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "x", "object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                      "prompt_tokens_details": {"cached_tokens": usage[2]}}}


def _call(i, name, args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _fake_provider(urls: list[str], seen: list[dict]):
    def handler(req):
        body = json.loads(req.content)
        seen.append({"auth": req.headers.get("authorization"), "url": str(req.url), **body})
        system = body["messages"][0]["content"]
        if "You plan web research" in system:
            return httpx2.Response(200, json=_reply(json.dumps({"subquestions": [
                {"question": "How efficient are heat pumps at -20C?", "difficulty": 2},
                {"question": "What do heat pumps cost to install?", "difficulty": 1}]})))
        if "You write a research report" in system:
            return httpx2.Response(200, json=_reply(
                "# Heat pumps\n\nHeat pumps keep a COP above 1.5 at -20C [1]. Costs vary by region [2].\n"))
        if "one research worker" in system:
            n_tool = sum(1 for m in body["messages"] if m["role"] == "tool")
            if n_tool == 0:
                return httpx2.Response(200, json=_reply(tool_calls=[_call(1, "web_search", {"query": QUERY})]))
            if n_tool == 1:
                return httpx2.Response(200, json=_reply(tool_calls=[_call(2, "web_fetch", {"urls": urls[:2]})]))
            final = {"answer": "Cold-climate heat pumps keep a coefficient of performance above 1.5 at "
                               "minus 20 degrees Celsius in field studies, and installation costs vary by "
                               "region, so the economics depend on local prices and the existing system. " * 2,
                     "sources": [{"url": urls[0], "quote": "cold-climate heat pumps kept a coefficient of "
                                                           "performance above 1.5"},
                                 {"url": urls[1], "quote": "Installation costs varied by region"},
                                 {"url": urls[1], "quote": "a sentence that is not on the page at all"}]}
            return httpx2.Response(200, json=_reply("```json\n" + json.dumps(final) + "\n```"))
        if "You check claims" in system:
            ids = [p.split('"')[0] for p in body["messages"][1]["content"].split('<claim id="')[1:]]
            return httpx2.Response(200, json=_reply(json.dumps(
                {"results": [{"id": i, "verdict": "supported"} for i in ids]})))
        return httpx2.Response(400, json={"error": {"message": "unexpected request"}})

    return handler


def _llm(handler, **kw) -> OpenAICompatLLM:
    return OpenAICompatLLM(base_url="https://provider.test/v1/", api_key="sk-test",
                           http=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)), **kw)


def test_full_research_run_on_an_openai_compatible_provider():
    search = MockSearch()
    urls = [h.url for h in asyncio.run(search.search(QUERY))]
    seen: list[dict] = []
    register_prices({}, [MODEL], "0.10,0.40")
    llm = _llm(_fake_provider(urls, seen), search=search)
    rep = asyncio.run(deep_research("How do heat pumps compare to gas boilers?", llm,
                                    config_from("standard", model=MODEL, free_search=True)))
    assert rep.stats["verified_%"] == 100.0 and rep.stats["sources"] == 2
    assert rep.stats["search_ungrounded_quotes_dropped"] == 2  # the made-up quote, once per worker
    assert rep.stats["claim_support_%"] == 100.0 and 0 < rep.stats["usd"] < 0.05
    assert "## Sources" in rep.markdown and urls[0] in rep.markdown
    first = seen[0]
    assert first["url"] == "https://provider.test/v1/chat/completions" and first["auth"] == "Bearer sk-test"
    assert first["model"] == MODEL and first["response_format"] == {"type": "json_object"}
    worker = [s for s in seen if "one research worker" in s["messages"][0]["content"]]
    assert worker[0]["tools"][0]["type"] == "function"
    assert worker[0]["tools"][0]["function"]["name"] == "web_search"
    assert worker[0]["tool_choice"] == "auto"
    tool_msg = next(m for s in worker for m in s["messages"] if m["role"] == "tool")  # workers interleave
    assert tool_msg["tool_call_id"] == "c1" and tool_msg["content"].startswith("<untrusted_content>")
    assert "reasoning_effort" not in first  # not sent unless enabled: providers differ


def test_last_tool_turn_disables_tools():
    seen: list[dict] = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx2.Response(200, json=_reply(tool_calls=[_call(len(seen), "web_search", {"query": "q"})]))

    llm = _llm(handler, search=MockSearch(), max_tool_turns=2)
    text, cites, usage, _ = asyncio.run(llm.investigate("Q", "sub", MODEL, "low"))
    assert [s.get("tool_choice") for s in seen] == ["auto", "auto", "none", None]  # + recovery call
    assert text == "" and cites == [] and usage.input_tokens == 4000


@pytest.mark.parametrize("status,exc", [(401, MissingKey), (402, SpendCapReached), (429, RateLimited),
                                        (503, TransientError)])
def test_provider_errors_map_to_engine_handling(status, exc):
    llm = _llm(lambda req: httpx2.Response(status, headers={"retry-after": "3"},
                                           json={"error": {"message": "nope"}}), search=MockSearch())
    with pytest.raises(exc) as e:
        asyncio.run(llm.plan("Q", 3, MODEL, "low"))
    if exc is RateLimited:
        assert e.value.retry_after == 3.0
    if exc is MissingKey:
        assert e.value.fatal


def test_cached_tokens_billed_as_input_unless_priced():
    data = _reply("x", usage=(1000, 10, 800))
    plain = _llm(lambda r: None, search=MockSearch())._usage(data)
    priced = _llm(lambda r: None, search=MockSearch(), cached_price_known=True)._usage(data)
    assert (plain.input_tokens, plain.cache_read_tokens) == (1000, 0)
    assert (priced.input_tokens, priced.cache_read_tokens) == (200, 800)


def test_resolve_openai_provider(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-test")
    monkeypatch.setenv("SWARM_CACHE_DIR", str(tmp_path))
    llm, cfg, info = resolve(ResearchOptions("Q?", base_url="https://p.test/v1", model=MODEL,
                                             price="0.1,0.4,0.01"))
    assert isinstance(llm, OpenAICompatLLM) and info["provider"] == "openai" and cfg.model == MODEL
    assert llm.cached_price_known and PRICES[MODEL].cache_read_mult == pytest.approx(0.1)
    asyncio.run(llm.aclose())
    with pytest.raises(ConfigError, match="model name"):
        resolve(ResearchOptions("Q?", provider="openai", base_url="https://p.test/v1"))
    with pytest.raises(ConfigError, match="only with Claude"):
        resolve(ResearchOptions("Q?", provider="openai", base_url="https://p.test/v1", model=MODEL,
                                search="anthropic"))
    with pytest.raises(ConfigError, match="--price"):
        resolve(ResearchOptions("Q?", base_url="https://p.test/v1", model=MODEL, price="cheap"))
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(MissingKey, match="OPENAI_API_KEY"):
        resolve(ResearchOptions("Q?", base_url="https://p.test/v1", model=MODEL, price="0.1,0.4"))


def test_non_claude_model_selects_openai_provider(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_BASE", "https://p.test/v1")
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-test")
    monkeypatch.setenv("SWARM_CACHE_DIR", str(tmp_path))
    llm, _, info = resolve(ResearchOptions("Q?", model=MODEL, price="0.1,0.4"))
    assert info["provider"] == "openai" and llm.base_url == "https://p.test/v1"
    asyncio.run(llm.aclose())
    llm, _, info = resolve(ResearchOptions("Q?"))  # env endpoint alone: still Claude
    assert info["provider"] == "anthropic"
    asyncio.run(llm.aclose())


def test_final_answer_is_recovered_when_last_turn_still_wants_tools():
    seen: list[dict] = []
    search = MockSearch()
    url = asyncio.run(search.search("q"))[0].url

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        if "tools" in body:  # the model ignores tool_choice "none" and keeps calling tools
            return httpx2.Response(200, json=_reply(tool_calls=[_call(len(seen), "web_search", {"query": "q"})]))
        final = {"answer": "Heat pumps kept a COP above 1.5 at minus 20 degrees.",
                 "sources": [{"url": url, "quote": "cold-climate heat pumps kept a coefficient of performance"}]}
        return httpx2.Response(200, json=_reply(json.dumps(final)))

    llm = _llm(handler, search=search, max_tool_turns=2)
    text, cites, usage, _ = asyncio.run(llm.investigate("Q", "sub", MODEL, "low"))
    assert "COP above 1.5" in text and [c.url for c in cites] == [url]
    last = seen[-1]
    assert "tools" not in last and last["response_format"] == {"type": "json_object"}
    # every tool call the model made got a tool result, so the history is valid
    asked = {c["id"] for m in last["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])}
    answered = {m["tool_call_id"] for m in last["messages"] if m["role"] == "tool"}
    assert asked == answered and llm.stats["final_answer_retries"] == 1
    assert usage.input_tokens == 4000  # 3 tool turns + the recovery call


def test_prose_answer_is_kept_and_unanswered_subquestions_are_reported():
    llm = _llm(lambda r: None, search=MockSearch())
    text, cites = llm._ground("Heat pumps work well in the cold, studies say.", {})
    assert text.startswith("Heat pumps") and cites == [] and llm.stats["answers_not_json"] == 1

    from swarm.research import MockResearchLLM

    class OneEmpty(MockResearchLLM):
        async def investigate(self, question, sub_question, model, effort):
            text, cites, u, s = await super().investigate(question, sub_question, model, "high")
            return ("" if sub_question == self._subs[0] else text), cites, u, s

    rep = asyncio.run(deep_research("Q?", OneEmpty(fail_low=False), config_from("quick")))
    assert rep.stats["unanswered_subquestions"] == 1 and rep.stats["verified_%"] < 100
    assert "Not answered" in rep.markdown and rep.stats["escalated"] >= 1


def test_key_can_come_from_a_named_variable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("RIVER_API_KEY", "rk-test")
    monkeypatch.setenv("TINYFISH_API_KEY", "tf-test")
    monkeypatch.setenv("SWARM_CACHE_DIR", str(tmp_path))
    opts = dict(base_url="https://api.river.test/v1", model=MODEL, price="0.12,0.48,0.0024")
    llm, _, _ = resolve(ResearchOptions("Q?", api_key_env="RIVER_API_KEY", **opts))
    assert llm._key == "rk-test" and llm.cached_price_known
    asyncio.run(llm.aclose())
    with pytest.raises(MissingKey, match="OPENAI_API_KEY is not set"):
        resolve(ResearchOptions("Q?", **opts))
