"""Long documents in parallel chunks.

    swarm doc translate manual.md --to German --out manual.de.md
    swarm doc summarize report.pdf --words 400 --out summary.md
    swarm doc rewrite policy.md --style "plain English for customers" --out policy-plain.md

    document ─► split at headings and paragraph breaks (formatting and order preserved)
             ─► translate / rewrite: a glossary of key terms is fixed first, so every chunk
                renders names and terms the same way; chunks then run in parallel, each with
                the glossary and the end of the previous chunk for context (checks: no empty
                or truncated chunk, length within a sane ratio of the source)
             ─► summarize: chunk summaries in parallel, then merged (in rounds, for very
                long documents) into one summary of the requested length

Text, Markdown and HTML work as is; PDFs with `pip install pypdf`.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from .chat import Chat, dumps, parallel

GLOSSARY_SCHEMA = {
    "type": "object",
    "properties": {"terms": {"type": "array", "items": {"type": "object", "properties": {
        "source": {"type": "string"}, "target": {"type": "string"}},
        "required": ["source", "target"], "additionalProperties": False}}},
    "required": ["terms"], "additionalProperties": False,
}
GLOSSARY_SYSTEM = (
    "List the key terms of this document (product names, technical terms, recurring phrases, up "
    "to 40) with how each must be rendered: {how}. Names that must not change map to themselves."
)
CHUNK_SYSTEM = (
    "You are processing one part of a long document; other parts are done in parallel. Task: {task}. "
    "Use the glossary for every listed term. Keep all Markdown formatting, headings, lists, "
    "tables, code blocks, links and numbers exactly. Do not add or drop content, and do not add "
    "comments or notes. Output only the processed text of THIS part."
)
SUMMARY_SYSTEM = (
    "Summarize this part of a long document faithfully: the main points, decisions, numbers and "
    "names, in about {words} words. No commentary and nothing that isn't in the text."
)
MERGE_SYSTEM = (
    "Merge these partial summaries of one document, in order, into a single coherent summary of "
    "about {words} words: one voice, no repetition, every important point, decision and number "
    "kept, nothing added. Markdown."
)

CONDENSE_SYSTEM = (
    "Shorten this summary to about {words} words. Keep every key decision, number and name; drop "
    "repetition, examples and detail first. Same language and Markdown style. Output only the summary."
)


@dataclass
class DocResult:
    text: str
    chunks: int
    failed_chunks: int
    usd: float
    calls: int


def read_document(path: Path) -> str:
    if not path.is_file():
        raise ConfigError(f"document not found: {path}")
    from ..search.local import _read

    text = _read(path)
    if not text.strip():
        raise ConfigError(f"{path.name} has no text")
    return text


def chunk(text: str, max_chars: int = 6000) -> list[str]:
    """Split before headings and at paragraph breaks; never inside a fenced code block."""
    blocks, cur, in_code = [], [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        cur.append(line)
        if not in_code and line.strip() == "":
            blocks.append("".join(cur))
            cur = []
    if cur:
        blocks.append("".join(cur))
    chunks: list[str] = []
    buf = ""
    for b in blocks:
        starts_section = bool(re.match(r"\s*#{1,3}\s", b))
        if buf and (len(buf) + len(b) > max_chars or (starts_section and len(buf) > max_chars // 2)):
            chunks.append(buf)
            buf = ""
        buf += b
        while len(buf) > max_chars * 1.5:  # one huge paragraph: cut at a sentence end
            cut = buf.rfind(". ", 0, max_chars) + 1 or max_chars
            chunks.append(buf[:cut])
            buf = buf[cut:]
    if buf.strip():
        chunks.append(buf)
    return chunks


async def transform(text: str, chat: Chat, task: str, glossary_how: str, *, max_chars: int = 6000,
                    agents: int = 8, log: Callable[[str], None] = lambda s: None) -> DocResult:
    parts = chunk(text, max_chars)
    gl = await chat.json(GLOSSARY_SYSTEM.format(how=glossary_how), text[:15_000], GLOSSARY_SCHEMA,
                         effort="medium", role="doc.glossary")
    terms = [t for t in gl.get("terms", []) if isinstance(t, dict) and t.get("source")][:40]
    glossary = dumps({t["source"]: t.get("target", t["source"]) for t in terms})
    log(f"{len(parts)} chunks, glossary of {len(terms)} terms")

    def job(i: int) -> Callable[[str], Any]:
        async def run(effort: str) -> str:
            prev = parts[i - 1][-600:] if i else ""
            ctx = f"End of the previous part (context only, do not output it):\n{prev}\n\n" if prev else ""
            return await chat.complete(CHUNK_SYSTEM.format(task=task),
                                       f"Glossary:\n{glossary}\n\n{ctx}Part {i + 1} of {len(parts)}:\n{parts[i]}",
                                       effort=effort, max_tokens=12000, role="doc.chunk")
        return run

    def ok(cid: str, out: str) -> bool:
        src = parts[int(cid[1:])]
        return bool(out.strip()) and 0.3 <= len(out) / max(1, len(src)) <= 3.5

    got = await parallel(chat, {f"c{i}": job(i) for i in range(len(parts))}, check=ok, agents=agents)
    out, failed = [], 0
    for i in range(len(parts)):
        piece = got.get(f"c{i}")
        if piece is None:
            failed += 1
            piece = f"\n> **[part {i + 1} could not be processed; original text follows]**\n\n{parts[i]}"
        out.append(piece if piece.endswith("\n") else piece + "\n")
    return DocResult("".join(out), len(parts), failed, round(chat.usd, 4), chat.calls)


async def summarize(text: str, chat: Chat, *, words: int = 400, max_chars: int = 12_000, agents: int = 8,
                    log: Callable[[str], None] = lambda s: None) -> DocResult:
    parts = chunk(text, max_chars)
    per = max(60, min(300, 3 * words // max(1, len(parts))))
    log(f"{len(parts)} chunks, about {per} words each before merging")

    async def summarize_all(pieces: list[str], target: int, role: str) -> list[str]:
        def job(i: int) -> Callable[[str], Any]:
            async def run(effort: str) -> str:
                return await chat.complete(SUMMARY_SYSTEM.format(words=target), pieces[i], effort=effort,
                                           max_tokens=4000, role=role)
            return run
        got = await parallel(chat, {f"c{i}": job(i) for i in range(len(pieces))}, agents=agents)
        return [g for g in (got.get(f"c{i}") for i in range(len(pieces))) if g]

    summaries = await summarize_all(parts, per, "doc.summary")
    failed = len(parts) - len(summaries)
    rounds = 0
    # Very long documents: merge in rounds. Stop when one summary is left (it can't shrink by
    # grouping) or after a few rounds, so a model that never shortens can't loop until the cap.
    while sum(len(s) for s in summaries) > 40_000 and len(summaries) > 1 and rounds < 4:
        groups = ["\n\n".join(summaries[i:i + 8]) for i in range(0, len(summaries), 8)]
        summaries = await summarize_all(groups, per * 2, "doc.summary") or summaries[:1]
        rounds += 1
    final = await chat.complete(MERGE_SYSTEM.format(words=words),
                                "\n\n".join(f"Part {i + 1}:\n{s}" for i, s in enumerate(summaries)),
                                effort="medium", max_tokens=8000, role="doc.merge")
    # Models overshoot length targets; one condensing pass when the summary is well over it.
    if len(final.split()) > 1.3 * words:
        shorter = await chat.complete(CONDENSE_SYSTEM.format(words=words), final, effort="low", max_tokens=4000,
                                      role="doc.condense")
        if shorter.strip() and len(shorter.split()) < len(final.split()):
            final = shorter
    return DocResult(final.strip() + "\n", len(parts), failed, round(chat.usd, 4), chat.calls)
