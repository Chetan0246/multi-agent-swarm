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
from dataclasses import asdict, dataclass, field
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
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["confirmed", "refuted", "unsure"]},
                   "reason": {"type": "string"}},
    "required": ["verdict", "reason"], "additionalProperties": False,
}
VERIFY_SYSTEM = (
    "You are a skeptical senior engineer double-checking one code-review finding written by a "
    "colleague who saw only one file. You see that file and the project files it imports. "
    "Decide: 'confirmed' (the code really fails as described), 'refuted' (the code shown "
    "disproves it, e.g. the related module behaves differently than the finding assumes), or "
    "'unsure' (it depends on code or context you can't see). Judge only from the code given, "
    "and give a one-sentence reason."
)

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
    status: str = "unverified"  # confirmed | unconfirmed | refuted (after the second check)
    verifier_reason: str = ""


@dataclass
class ReviewResult:
    markdown: str
    findings: list[Finding]
    files_reviewed: int
    dropped_ungrounded: int
    usd: float
    calls: int
    refuted: int = 0  # findings a verifier showed to be wrong, given the related code
    refuted_findings: list[Finding] = field(default_factory=list)  # kept for audit, never hidden


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
                 agents: int = 8, verify_findings: bool = True,
                 log: Callable[[str], None] = lambda s: None) -> ReviewResult:
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
    refuted: list[Finding] = []
    if verify_findings and unique:
        unique, refuted = await verify(unique, texts, chat, agents=agents)
    log(f"{len(unique)} findings kept, {dropped} dropped (quote not in the file), "
        f"{len(refuted)} refuted on a second check")
    return ReviewResult(_markdown(root, unique, len(files), dropped, refuted), unique, len(files), dropped,
                        round(chat.usd, 4), chat.calls, len(refuted), refuted)


_IMPORT = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))|require\(['\"]\.{1,2}/([\w./-]+)['\"]\)"
                     r"|from\s+['\"]\.{1,2}/([\w./-]+)['\"]", re.M)


def related_files(rel: str, texts: dict[str, str], limit_chars: int = 30_000, depth: int = 2) -> dict[str, str]:
    """Project files the reviewed file imports (Python modules, relative JS/TS imports), and
    what those import in turn (`depth` levels), up to `limit_chars` in total."""
    by_name: dict[str, list[str]] = {}
    for path in texts:
        p = Path(path)
        by_name.setdefault(p.with_suffix("").as_posix().replace("/", "."), []).append(path)
        by_name.setdefault(p.stem, []).append(path)

    def imports(path: str) -> list[str]:
        found = []
        for m in _IMPORT.finditer(texts[path]):
            name = next(g for g in m.groups() if g)
            for key in (name.lstrip("."), name.split(".")[-1], Path(name).stem):
                found += [c for c in by_name.get(key, []) if c not in found]
        return found

    out: dict[str, str] = {}
    frontier = [rel]
    for _ in range(depth):
        nxt = []
        for path in frontier:
            for c in imports(path):
                if c != rel and c not in out and sum(map(len, out.values())) + len(texts[c]) <= limit_chars:
                    out[c] = texts[c]
                    nxt.append(c)
        frontier = nxt
    return out


async def verify(findings: list[Finding], texts: dict[str, str], chat: Chat, agents: int = 8
                 ) -> tuple[list[Finding], list[Finding]]:
    """A second, skeptical pass with the related code: drop findings it refutes."""
    from ..research import _parse_json_obj

    def job(i: int, f: Finding) -> Callable[[str], Any]:
        async def run(effort: str) -> str:
            ctx = "".join(f"\n--- {p} ---\n{t}\n" for p, t in related_files(f.file, texts).items())
            return await chat.complete(
                VERIFY_SYSTEM, f"Finding in {f.file}:{f.line} [{f.severity}] {f.title}\n{f.explanation}\n"
                               f"Quoted code: {f.quote}\n\n--- {f.file} ---\n{texts[f.file]}\n{ctx}",
                schema=VERIFY_SCHEMA, effort=effort, max_tokens=1500, role="review.verify")
        return run

    got = await parallel(chat, {f"v{i}": job(i, f) for i, f in enumerate(findings)}, agents=agents,
                         check=lambda _, a: _parse_json_obj(a).get("verdict") in ("confirmed", "refuted", "unsure"))
    kept: list[Finding] = []
    refuted: list[Finding] = []
    for i, f in enumerate(findings):
        v = _parse_json_obj(got.get(f"v{i}") or "")
        verdict = v.get("verdict", "unsure")
        f.verifier_reason = str(v.get("reason", ""))[:500]
        if verdict == "refuted":
            f.status = "refuted"
            refuted.append(f)
            continue
        f.status = "confirmed" if verdict == "confirmed" else "unconfirmed"
        if verdict != "confirmed":
            f.category = f"{f.category} (unconfirmed)"
        kept.append(f)
    return kept, refuted


def _markdown(root: Path, findings: list[Finding], n_files: int, dropped: int,
              refuted: list[Finding] | None = None) -> str:
    refuted = refuted or []
    counts = {s: sum(f.severity == s for f in findings) for s in SEVERITY}
    refuted_note = (f", and {len(refuted)} because a second check of the related code refuted them (listed at "
                    "the end)") if refuted else ""
    out = [f"# Code review: {root.name}", "",
           f"{n_files} files reviewed; {len(findings)} findings ({counts['high']} high, {counts['medium']} medium, "
           f"{counts['low']} low). {dropped} findings were discarded because their quoted code "
           f"was not in the file{refuted_note}.", ""]
    for f in findings:
        quote = "\n".join(f.quote.splitlines()[:6])
        out += [f"## [{f.severity}] {f.title}", f"`{f.file}:{f.line}` · {f.category}", "",
                f.explanation, "", "```", quote, "```", "", f"**Fix:** {f.suggestion}", ""]
        if f.status == "unconfirmed" and f.verifier_reason:
            out += [f"*Second check:* {f.verifier_reason}", ""]
    if refuted:
        out += ["## Refuted on a second check", "",
                "The first reviewer raised these, but the related code shows they don't hold. They are listed "
                "so you can disagree.", ""]
        out += [f"- `{f.file}:{f.line}` [{f.severity}] {f.title}: {f.verifier_reason or 'no reason given'}"
                for f in refuted]
        out.append("")
    return "\n".join(out)


def to_json(result: ReviewResult) -> list[dict[str, Any]]:
    """Every finding, refuted ones included, each with its `status` and the verifier's reason."""
    return [asdict(f) for f in result.findings + result.refuted_findings]
