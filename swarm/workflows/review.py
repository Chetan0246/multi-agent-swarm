"""`swarm review ./my-repo --out review.md`

    repository ─► files to review (git-tracked, or a walk that skips vendored/build folders)
               ─► reviewers in parallel, one file each (long files in overlapping windows), shown
                  with line numbers and the repo's file list for context
               ─► every finding must quote the code it is about; a finding whose quote is not
                  in the file is dropped (the reviewer's version of quote grounding), and its
                  line number is corrected to where the quote really is
               ─► duplicates merged, ranked by severity, written as Markdown + JSON

Findings are about correctness, security and reliability, not style, unless --focus says so.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from .chat import Chat, parallel

CODE_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".rb", ".php", ".cs", ".c",
            ".h", ".cpp", ".hpp", ".swift", ".scala", ".sh", ".sql", ".vue", ".svelte", ".html", ".css"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "dist", "build", "vendor", "__pycache__", ".next",
             "target", ".swarm", "site", ".mypy_cache", ".pytest_cache", "coverage"}
SEVERITY = {"high": 0, "medium": 1, "low": 2}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"findings": {"type": "array", "items": {"type": "object", "properties": {
        "line": {"type": "integer"}, "severity": {"type": "string", "enum": ["high", "medium", "low"]},
        "category": {"type": "string"}, "title": {"type": "string"}, "explanation": {"type": "string"},
        "suggestion": {"type": "string"}, "quote": {"type": "string"}},
        "required": ["line", "severity", "category", "title", "explanation", "suggestion", "quote"],
        "additionalProperties": False}}},
    "required": ["findings"], "additionalProperties": False,
}
REVIEW_SYSTEM = (
    "You are a senior engineer reviewing one file of a codebase, one reviewer per file in parallel. "
    "Report only real problems: bugs, security holes, data loss, race conditions, resource leaks, "
    "wrong error handling, broken edge cases{focus}. No style nits, no praise, no speculative "
    "'consider adding' notes. For each finding give the line, severity (high: wrong results, "
    "crashes, security; medium: likely bug in an edge case; low: minor), a category, a title, the "
    "explanation of the failure scenario, a concrete fix, and `quote`: the exact code (copied "
    "verbatim, without the line-number prefix) that the finding is about. Quotes are checked "
    "against the file and findings with a wrong quote are discarded. An empty list is a fine answer."
)


@dataclass
class Finding:
    file: str
    line: int
    severity: str
    category: str
    title: str
    explanation: str
    suggestion: str
    quote: str


@dataclass
class ReviewResult:
    markdown: str
    findings: list[Finding]
    files_reviewed: int
    dropped_ungrounded: int
    usd: float
    calls: int


def collect(root: Path, max_files: int = 80, max_bytes: int = 200_000) -> list[Path]:
    if root.is_file():
        return [root]
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, timeout=30)
        paths = [root / p for p in out.stdout.splitlines()] if out.returncode == 0 and out.stdout else []
    except (OSError, subprocess.TimeoutExpired):
        paths = []
    if not paths:
        paths = [p for p in root.rglob("*") if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS)]
    files = [p for p in sorted(paths) if p.suffix.lower() in CODE_EXT and p.is_file()
             and p.stat().st_size <= max_bytes and not (set(p.relative_to(root).parts) & SKIP_DIRS)]
    return files[:max_files]


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def ground(findings: list[dict], file: str, text: str) -> tuple[list[Finding], int]:
    """Keep findings whose quote is really in the file; fix their line numbers."""
    lines = text.splitlines()
    kept, dropped = [], 0
    for f in findings:
        if not isinstance(f, dict):
            continue
        quote = str(f.get("quote", "")).strip()
        first = _ws(quote.splitlines()[0]) if quote else ""
        if len(first) < 4 or _ws(quote) not in _ws(text):
            dropped += 1
            continue
        m = re.match(r"\s*(\d+)", str(f.get("line") or ""))  # models sometimes say "12-14" or "L12"
        line = int(m.group(1)) if m else 0
        near = [i + 1 for i, ln in enumerate(lines) if first in _ws(ln)]
        if near and line not in near:
            line = min(near, key=lambda n: abs(n - line))
        sev = str(f.get("severity", "low")).lower()
        kept.append(Finding(file, line, sev if sev in SEVERITY else "low", str(f.get("category", "")),
                            str(f.get("title", "")).strip(), str(f.get("explanation", "")).strip(),
                            str(f.get("suggestion", "")).strip(), quote))
    return kept, dropped


async def review(root: Path, chat: Chat, *, focus: str = "", max_files: int = 80, window: int = 600,
                 agents: int = 8, log: Callable[[str], None] = lambda s: None) -> ReviewResult:
    if not root.exists():
        raise ConfigError(f"not found: {root}")
    files = collect(root, max_files)
    if not files:
        raise ConfigError(f"no source files to review under {root}")
    base = root if root.is_dir() else root.parent
    listing = "\n".join(str(p.relative_to(base)) for p in files)
    texts = {str(p.relative_to(base)): p.read_text(errors="replace") for p in files}
    jobs: dict[str, Callable[[str], Any]] = {}
    for rel, text in texts.items():
        lines = text.splitlines()
        for start in range(0, max(1, len(lines)), window - 50):  # overlapping windows for long files
            chunk = "\n".join(f"{n:5d}  {ln}" for n, ln in enumerate(lines[start:start + window], start + 1))

            def mk(rel: str = rel, chunk: str = chunk) -> Callable[[str], Any]:
                async def run(effort: str) -> str:
                    return await chat.complete(
                        REVIEW_SYSTEM.format(focus=f", and especially: {focus}" if focus else ""),
                        f"Repository files:\n{listing}\n\nFile: {rel}\n```\n{chunk}\n```",
                        schema=REVIEW_SCHEMA, effort=effort, max_tokens=6000, role="review.file")
                return run

            jobs[f"{rel}@{start}"] = mk()
            if start + window >= len(lines):
                break
    log(f"reviewing {len(files)} files in {len(jobs)} parts")

    from ..research import _parse_json_obj

    def valid(_: str, answer: str) -> bool:
        return isinstance(_parse_json_obj(answer).get("findings"), list)

    got = await parallel(chat, jobs, check=valid, efforts=("medium", "high"), agents=agents)
    all_findings: list[Finding] = []
    dropped = 0
    for jid, answer in got.items():
        if not answer:
            continue
        rel = jid.rsplit("@", 1)[0]
        kept, d = ground(_parse_json_obj(answer).get("findings", []), rel, texts[rel])
        all_findings += kept
        dropped += d
    seen: set[tuple[str, int, str]] = set()
    unique = []
    for f in sorted(all_findings, key=lambda f: (SEVERITY[f.severity], f.file, f.line)):
        key = (f.file, f.line, f.title.lower())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    log(f"{len(unique)} findings kept, {dropped} dropped (quote not in the file)")
    return ReviewResult(_markdown(root, unique, len(files), dropped), unique, len(files), dropped,
                        round(chat.usd, 4), chat.calls)


def _markdown(root: Path, findings: list[Finding], n_files: int, dropped: int) -> str:
    counts = {s: sum(f.severity == s for f in findings) for s in SEVERITY}
    out = [f"# Code review: {root.name}", "",
           f"{n_files} files reviewed; {len(findings)} findings ({counts['high']} high, {counts['medium']} medium, "
           f"{counts['low']} low). {dropped} findings were discarded because their quoted code "
           "was not in the file.", ""]
    for f in findings:
        quote = "\n".join(f.quote.splitlines()[:6])
        out += [f"## [{f.severity}] {f.title}", f"`{f.file}:{f.line}` · {f.category}", "",
                f.explanation, "", "```", quote, "```", "", f"**Fix:** {f.suggestion}", ""]
    return "\n".join(out)


def to_json(result: ReviewResult) -> list[dict[str, Any]]:
    return [asdict(f) for f in result.findings]
