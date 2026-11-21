"""Any OpenAI-compatible chat-completions provider (DeepSeek, Together, DeepInfra, Fireworks,
OpenRouter, vLLM, Ollama ...), as an alternative to Claude.

    export OPENAI_API_KEY=...   OPENAI_BASE_URL=https://your-provider/v1
    swarm research "question" --provider openai --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \\
        --price 0.10,0.40

The pipeline is the same as with Claude: free TinyFish search (or your documents) as
function tools, verbatim quotes checked against the retrieved text, claim checking, cost cap.
Differences:
* Search must be client-side (`--search tinyfish` or `--docs`); Claude's server-side search
  is Anthropic-only.
* Structured output uses JSON mode (`response_format: json_object`) plus the schema in the
  prompt, since strict JSON-schema output isn't universal. Answers are parsed leniently.
* Effort levels aren't sent (providers differ); escalating a weak finding re-researches it
  with a fresh attempt instead. Set `reasoning_effort=True` for providers that accept it.
* Prices aren't known in advance: pass them with `--price IN,OUT[,CACHED]` (USD per million
  tokens) or in swarm.toml, so the estimate and the hard cap stay meaningful.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx2

from .backends import CallUsage, RateLimited, SpendCapReached, TransientError
from .errors import ConfigError, MissingKey
from .research import (
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    PLAN_SCHEMA,
    PLAN_SYSTEM,
    SYNTH_SYSTEM,
    TOOL_WORKER_SCHEMA,
    TOOL_WORKER_SYSTEM,
    ClientToolsLLM,
    _add,
    _parse_json_obj,
    _tool_defs,
)
from .search.base import SearchFilters, SearchProvider


class ProviderError(Exception):
    """A non-retryable error from the provider (e.g. 400 for an unsupported parameter)."""


def _json_rule(schema: dict) -> str:
    return ("\n\nRespond with only a JSON object (no prose, no code fences) matching this JSON "
            f"schema:\n{json.dumps(schema, separators=(',', ':'))}")


def _functions(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


class OpenAICompatLLM(ClientToolsLLM):
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 search: SearchProvider | None = None, docs: SearchProvider | None = None,
                 filters: SearchFilters | None = None, max_tool_turns: int = 6, page_chars: int = 8000,
                 worker_model: str | None = None, language: str | None = None, timeout: float = 300.0,
                 max_tokens: int = 8000, json_mode: bool = True, reasoning_effort: bool = False,
                 cached_price_known: bool = False, http: httpx2.AsyncClient | None = None,
                 api_key_env: str = "OPENAI_API_KEY"):
        url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        if not url:
            raise ConfigError("no base URL for the OpenAI-compatible provider",
                              "pass --base-url https://your-provider/v1 or set OPENAI_BASE_URL")
        # `api_key_env` names the variable holding the key, e.g. RIVER_API_KEY.
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise MissingKey(f"{api_key_env} is not set",
                             f"export {api_key_env}=<your provider's key> (it is sent only to the base URL), "
                             "or name the variable that holds it with --api-key-env")
        self.api_key_env = api_key_env
        if search is None and docs is None:
            raise ConfigError("the OpenAI-compatible provider needs client-side search",
                              "use --search tinyfish (free) and/or --docs; Claude's server-side "
                              "search is Anthropic-only")
        self.base_url = url.rstrip("/")
        self._key = key
        self.http = http or httpx2.AsyncClient(timeout=timeout)
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.reasoning_effort = reasoning_effort
        # Only split out cached prompt tokens when their (discounted) price is known; otherwise
        # bill them as ordinary input so the cost and the cap are never understated.
        self.cached_price_known = cached_price_known
        self._init_tools(search, docs, filters, max_tool_turns, page_chars, worker_model, language)

    async def aclose(self) -> None:
        await self._close_providers()
        await self.http.aclose()

    # ------------------------------------------------------------------ transport

    async def _chat(self, model: str, messages: list[dict], effort: str | None = None,
                    json_out: bool = False, **extra: Any) -> dict:
        body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": self.max_tokens, **extra}
        if json_out and self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if effort and self.reasoning_effort:
            body["reasoning_effort"] = effort
        try:
            r = await self.http.post(f"{self.base_url}/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {self._key}"})
        except httpx2.TransportError as e:  # connect/read timeouts, resets
            raise TransientError(type(e).__name__) from e
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError as e:
                raise TransientError("provider returned a response that is not JSON") from e
            if not isinstance(data, dict) or not data.get("choices"):
                err = data.get("error") if isinstance(data, dict) else None
                raise TransientError(f"provider returned no choices: {str(err)[:200]}")
            return data
        msg = ""
        try:
            e_body = r.json()
            err = e_body.get("error", e_body) if isinstance(e_body, dict) else {}
            msg = str(err.get("message", "") if isinstance(err, dict) else err)[:300]
        except ValueError:
            msg = r.text[:300]
        if r.status_code in (401, 403):
            raise MissingKey(f"the provider rejected the API key ({r.status_code}): {msg}",
                             f"check {self.api_key_env} (or pick another variable with --api-key-env) "
                             f"and the base URL {self.base_url}")
        if r.status_code == 402:
            raise SpendCapReached(f"provider: payment required (402): {msg}")
        if r.status_code == 429:
            try:
                ra: float | None = float(r.headers.get("retry-after", ""))
            except ValueError:
                ra = None
            raise RateLimited(msg or "rate limited", retry_after=ra)
        if r.status_code >= 500 or r.status_code in (408, 409):
            raise TransientError(f"HTTP {r.status_code}")
        raise ProviderError(f"provider returned HTTP {r.status_code}: {msg}")

    def _usage(self, data: dict) -> CallUsage:
        u = data.get("usage") or {}
        prompt = int(u.get("prompt_tokens") or 0)
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens")
                     or u.get("prompt_cache_hit_tokens") or 0) if self.cached_price_known else 0
        return CallUsage(input_tokens=prompt - cached, cache_read_tokens=cached,
                         output_tokens=int(u.get("completion_tokens") or 0))

    @staticmethod
    def _message(data: dict) -> tuple[dict, str]:
        choice = data["choices"][0]
        return choice.get("message") or {}, str(choice.get("finish_reason") or "")

    # ------------------------------------------------------------------ ResearchLLM

    async def plan(self, question, max_subquestions, model, effort):
        data = await self._chat(model, [
            {"role": "system", "content": PLAN_SYSTEM + f" Never return more than {max_subquestions} "
                                          "sub-questions." + _json_rule(PLAN_SCHEMA)},
            {"role": "user", "content": question}], effort, json_out=True)
        msg, _ = self._message(data)
        subs = _parse_json_obj(msg.get("content") or "").get("subquestions", [])
        return [s for s in subs if isinstance(s, dict)] if isinstance(subs, list) else [], self._usage(data)

    async def investigate(self, question, sub_question, model, effort):
        model = self.worker_model or model
        tools = _functions(_tool_defs(web=self.search is not None, docs=self.docs is not None))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": TOOL_WORKER_SYSTEM + _json_rule(TOOL_WORKER_SCHEMA)},
            {"role": "user", "content": self._worker_prompt(question, sub_question)}]
        usage = CallUsage()
        corpus: dict[str, tuple[str, str]] = {}
        content = ""
        for turn in range(self.max_tool_turns + 1):
            last = turn == self.max_tool_turns
            try:
                data = await self._chat(model, messages, effort, tools=tools,
                                        tool_choice="none" if last else "auto")
            except BaseException:
                _add(self._lost, usage)  # earlier turns were billed even though this call failed
                raise
            _add(usage, self._usage(data))
            msg, finish = self._message(data)
            content = msg.get("content") or ""
            calls = msg.get("tool_calls") or []
            if not calls or last:
                break
            assistant: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": calls}
            if msg.get("reasoning_content"):  # some reasoning models need it echoed back mid-turn
                assistant["reasoning_content"] = msg["reasoning_content"]
            messages.append(assistant)

            async def run(call: dict) -> str:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    return "Tool error: arguments were not valid JSON"
                return await self._run_tool(str(fn.get("name", "")), args if isinstance(args, dict) else {},
                                            corpus)

            try:
                outputs = await asyncio.gather(*(run(c) for c in calls))
            except BaseException:
                _add(self._lost, usage)
                raise
            messages += [{"role": "tool", "tool_call_id": c.get("id", ""), "content": out}
                         for c, out in zip(calls, outputs)]
        if not _parse_json_obj(content).get("answer"):
            # The last turn produced no usable answer: some providers ignore tool_choice "none"
            # and ask for more tools, or the JSON was cut off. Without this, all the research
            # this worker did is lost. Close any open tool calls and ask once for the answer.
            self.stats["final_answer_retries"] = self.stats.get("final_answer_retries", 0) + 1
            if calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": calls})
                messages += [{"role": "tool", "tool_call_id": c.get("id", ""),
                              "content": "Not run: the tool budget for this task is used up."} for c in calls]
            messages.append({"role": "user", "content": "Stop searching. Using only what the tools "
                             "returned above, write your final answer now as the JSON object described."})
            try:
                data = await self._chat(model, messages, effort, json_out=True)
            except ProviderError:
                data = None  # e.g. the provider insists on `tools` alongside tool history
            except BaseException:
                _add(self._lost, usage)
                raise
            if data is not None:
                _add(usage, self._usage(data))
                content = self._message(data)[0].get("content") or content
        answer, cites = self._ground(content, corpus)
        return answer, cites, usage, 0  # client-side search: no per-search fee

    async def synthesize(self, question, findings_block, n_sources, model, effort):
        data = await self._chat(model, [
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\n{findings_block}\n\n"
                                        f"Valid citation numbers: 1-{n_sources}."}], effort)
        msg, _ = self._message(data)
        return (msg.get("content") or "").strip() or "_The synthesis step returned no text._", self._usage(data)

    async def judge(self, claims, model, effort):
        body = "\n\n".join(
            f"<claim id=\"{c['id']}\">{c['claim']}</claim>\n<excerpts>\n"
            + ("\n".join(c["evidence"]) or "(no excerpt available)") + "\n</excerpts>" for c in claims)
        data = await self._chat(model, [{"role": "system", "content": JUDGE_SYSTEM + _json_rule(JUDGE_SCHEMA)},
                                        {"role": "user", "content": body}], effort, json_out=True)
        msg, _ = self._message(data)
        rows = _parse_json_obj(msg.get("content") or "").get("results", [])
        out = {str(r["id"]): str(r["verdict"]).lower() for r in rows if isinstance(r, dict)
               and "id" in r and "verdict" in r} if isinstance(rows, list) else {}
        return out, self._usage(data)
