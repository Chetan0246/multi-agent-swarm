"""`swarm write "a beginner's guide to home solar" --words 3000 --out guide.md`

    brief (+ optional --context files: notes, data, a research report)
        ─► editor: outline = title, audience, style guide, sections with briefs and word targets
        ─► writers draft every section in parallel; each sees the whole outline and style guide,
           so sections don't overlap and read alike (checks: length near target, no stray
           headings); a short draft is rewritten at higher effort
        ─► editor pass over the whole draft for one voice and smooth transitions (documents
           up to ~8,000 words; longer ones keep the section drafts as written)

Facts come only from the brief and the context you give; writers are told not to invent
statistics or citations. For sourced writing, run `swarm research` first and pass the report
with --context.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigError
from .chat import Chat, dumps, parallel

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"}, "audience": {"type": "string"}, "style_guide": {"type": "string"},
        "sections": {"type": "array", "items": {"type": "object", "properties": {
            "heading": {"type": "string"}, "brief": {"type": "string"}, "words": {"type": "integer"}},
            "required": ["heading", "brief", "words"], "additionalProperties": False}},
    },
    "required": ["title", "audience", "style_guide", "sections"],
    "additionalProperties": False,
}

OUTLINE_SYSTEM = (
    "You are the editor planning a document that several writers will draft in parallel, one "
    "section each. Return a title, the audience, a style guide (tone, person, terminology, "
    "formatting rules every writer must follow) and {n} sections at most, each with a heading, a "
    "brief that says exactly what it covers and what it must NOT cover (so sections don't "
    "overlap), and a word target. Targets should add up to about {words} words."
)
SECTION_SYSTEM = (
    "You are one of several writers drafting a document in parallel. Write only your assigned "
    "section, in Markdown, starting directly with the prose (no heading: it is added for you; "
    "use ### subheadings only if the section is long). Follow the style guide exactly and stay "
    "inside your brief: other writers cover the other sections. Use only facts from the brief "
    "and the provided context; never invent statistics, quotes or citations. Aim for the word target."
)
EDIT_SYSTEM = (
    "You are the final editor. Revise this draft into one consistent document: a single voice, "
    "smooth transitions between sections, no repetition, consistent terminology, following the "
    "style guide. Keep every section, heading and fact; do not add facts or shorten it "
    "substantially. Return the complete revised document in Markdown, nothing else."
)


@dataclass
class WriteResult:
    markdown: str
    title: str
    sections: int
    words: int
    edited: bool
    usd: float
    calls: int


def _words(s: str) -> int:
    return len(re.findall(r"\w+", s))


async def write(brief: str, chat: Chat, *, words: int = 1500, max_sections: int = 8,
                context: list[Path] | None = None, agents: int = 6,
                log: Callable[[str], None] = lambda s: None) -> WriteResult:
    ctx = ""
    for p in context or []:
        if not p.is_file():
            raise ConfigError(f"context file not found: {p}")
        ctx += f"\n\n<context file=\"{p.name}\">\n{p.read_text(errors='replace')[:40_000]}\n</context>"
    outline = await chat.json(OUTLINE_SYSTEM.format(n=max_sections, words=words),
                              f"Brief:\n{brief}{ctx}", OUTLINE_SCHEMA, effort="high", role="write.outline")
    sections = [s for s in outline.get("sections", []) if isinstance(s, dict) and s.get("heading")][:max_sections]
    if not sections:
        raise ConfigError("the editor returned no outline", "make the brief more specific")
    log(f"outline: {outline.get('title', '')!r}, {len(sections)} sections")
    plan = dumps({k: outline.get(k) for k in ("title", "audience", "style_guide")}
                 | {"sections": [{"heading": s["heading"], "brief": s["brief"]} for s in sections]})

    def job(i: int, s: dict) -> Callable[[str], object]:
        async def run(effort: str) -> str:
            return (await chat.complete(
                SECTION_SYSTEM, f"Document plan:\n{plan}\n\nYour section: {s['heading']}\nBrief: {s['brief']}\n"
                                f"Word target: {s.get('words', 300)}{ctx}", effort=effort, max_tokens=8000,
                role="write.section")).strip()
        return run

    targets = {f"s{i}": int(s.get("words") or 300) for i, s in enumerate(sections)}
    drafts = await parallel(chat, {f"s{i}": job(i, s) for i, s in enumerate(sections)},  # type: ignore[misc]
                            check=lambda sid, text: _words(text) >= 0.5 * targets[sid] and not text.startswith("# "),
                            agents=agents)
    parts = [f"# {outline.get('title') or brief[:80]}"]
    for i, s in enumerate(sections):
        body = drafts.get(f"s{i}")
        parts.append(f"## {s['heading']}\n\n{body or '_This section could not be written._'}")
    draft = "\n\n".join(parts) + "\n"
    log(f"drafted {sum(1 for v in drafts.values() if v)}/{len(sections)} sections, {_words(draft)} words")

    edited = False
    if _words(draft) <= 8000:
        revised = (await chat.complete(EDIT_SYSTEM, f"Style guide: {outline.get('style_guide', '')}\n\n{draft}",
                                       effort="medium", max_tokens=16000, role="write.edit")).strip()
        # Keep the edit only if it kept the structure and most of the text.
        if revised.count("\n## ") >= draft.count("\n## ") and _words(revised) >= 0.8 * _words(draft):
            draft, edited = revised + "\n", True
        else:
            log("editor pass dropped content; keeping the section drafts")
    return WriteResult(draft, str(outline.get("title", "")), len(sections), _words(draft), edited,
                       round(chat.usd, 4), chat.calls)
