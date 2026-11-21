"""`swarm build "a todo app with a React frontend and FastAPI backend" --out todo/`

Multi-agent software building that stays coherent:

    description ─► lead engineer: spec = files, each with its purpose, the interface other
                   files rely on (signatures, endpoints, props, data shapes) and dependencies,
                   plus setup and test commands
                ─► engineers write files in parallel, in dependency waves: each sees the
                   whole spec and the finished files it depends on, and codes to the contract
                   (deterministic checks: Python must compile, JSON must parse)
                ─► (with --allow-execution) install, build and run the tests for real
                ─► one fixer reads the failing output and the files involved and edits them;
                   repeat up to --fix-rounds
                ─► the project folder, plus .swarm/build.json (spec, command results, cost)

Parallel writers never talk to each other; the shared spec is the only coordination, which is
what keeps independently written files compatible. Fixing is a single agent with the full
error and code, because fixes cut across files.

Commands come from the model, so they are checked before running (sandbox.check_command): no
shell, an allowlist of build tools, inside the project only, and a private HOME with no API
keys in the environment. Python projects get their own .venv.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from ..sandbox import ALLOWED_PROGRAMS, CommandRejected, CommandResult, run_command, strip_fences
from .chat import Chat, dumps, parallel

CMD = {"type": "object", "properties": {"run": {"type": "string"}, "cwd": {"type": "string"}},
       "required": ["run", "cwd"], "additionalProperties": False}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "summary": {"type": "string"}, "stack": {"type": "string"},
        "files": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "purpose": {"type": "string"}, "interface": {"type": "string"},
            "depends_on": {"type": "array", "items": {"type": "string"}}},
            "required": ["path", "purpose", "interface", "depends_on"], "additionalProperties": False}},
        "setup": {"type": "array", "items": CMD}, "test": {"type": "array", "items": CMD},
        "start": {"type": "string"},
    },
    "required": ["name", "summary", "stack", "files", "setup", "test", "start"],
    "additionalProperties": False,
}
FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"], "additionalProperties": False}},
        "setup": {"type": "array", "items": CMD}, "test": {"type": "array", "items": CMD},
    },
    "required": ["analysis", "edits", "setup", "test"],
    "additionalProperties": False,
}

PLAN_SYSTEM = (
    "You are the lead engineer planning a software project that a team will write in parallel, "
    "each engineer writing one file without seeing the others' work. Your spec is their only "
    "coordination, so make every file's `interface` precise: the exact exported names and "
    "signatures, HTTP endpoints with methods, request and response JSON shapes, component props, "
    "CLI arguments, or config keys that other files rely on. `depends_on` lists the files whose "
    "interface this file uses.\n\nRules:\n"
    "- A complete, minimal, working project: at most {max_files} files. Include README.md (what it "
    "is, how to install, run and test) and automated tests that run offline with no external "
    "services (e.g. pytest with FastAPI's TestClient; vitest or node --test for JavaScript).\n"
    "- Few dependencies, all declared (requirements.txt, package.json). Python: a virtualenv is "
    "already active, so use `pip install -r requirements.txt`; include pytest if you use it.\n"
    "- `setup` installs dependencies and `test` runs the tests (non-zero exit on failure). Each "
    "command is ONE program with arguments, run with no shell: no &&, pipes, redirects or $VARS; "
    "use `cwd` (relative to the project root, \".\" for the root) for subfolders. Allowed "
    "programs: {programs}. No global installs, no sudo, nothing that starts a long-running server "
    "in `setup`/`test`. `start` is the command a user runs to start the app.\n"
    "- Relative paths with forward slashes; no binary files."
)

FILE_SYSTEM = (
    "You are an engineer on a team writing one project in parallel. Write exactly one file. "
    "Follow the spec and every interface exactly: the other files are being written at the same "
    "time against the same contract. Use only dependencies the spec declares. Output only the "
    "file's complete contents: no markdown fences, no commentary."
)

FIX_SYSTEM = (
    "You fix a software project whose setup or tests failed. You get the spec, the failing "
    "command with its output, and the relevant files. Find the root cause and return `edits`: "
    "the complete new content of each file you change (you may add files). Change as little as "
    "needed. Do not delete or weaken tests to make them pass, unless a test is itself wrong; say "
    "so in `analysis` if so. Return `setup`/`test` unchanged unless a command itself is wrong "
    "(same rules: one program per command, no shell syntax)."
)


# ----------------------------------------------------------------------------- spec handling


def safe_path(project: Path, rel: str) -> Path:
    """A model-chosen path, refused unless it stays inside the project (and out of .git etc.)."""
    p = Path(rel.strip().removeprefix("./"))
    if not rel.strip() or p.is_absolute() or ".." in p.parts or p.parts[0] in (".git", ".swarm", ".venv"):
        raise ConfigError(f"unsafe file path in the plan: {rel!r}")
    full = (project / p).resolve()
    if project.resolve() not in full.parents:
        raise ConfigError(f"file path escapes the project: {rel!r}")
    return full


def waves(files: list[dict]) -> list[list[dict]]:
    """Dependency order: a file is written after the files whose interface it uses.
    (Interfaces are in the spec, so a cycle is broken by writing the rest together.)"""
    by_path = {f["path"]: f for f in files}
    done: set[str] = set()
    out: list[list[dict]] = []
    while len(done) < len(files):
        ready = [f for f in files if f["path"] not in done
                 and all(d in done or d not in by_path for d in f.get("depends_on", []))]
        if not ready:  # cycle
            ready = [f for f in files if f["path"] not in done]
        out.append(ready)
        done |= {f["path"] for f in ready}
    return out


def check_file(path: str, content: str) -> bool:
    """Cheap deterministic checks before a file is accepted (else it is rewritten)."""
    if not content.strip():
        return False
    if path.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError:
            return False
    if path.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError:
            return False
    return True


# ----------------------------------------------------------------------------- the workflow


@dataclass
class BuildResult:
    project: str
    spec: dict[str, Any]
    files_written: list[str]
    commands: list[dict[str, Any]] = field(default_factory=list)  # every run, in order
    fix_rounds: int = 0
    tests_passed: bool | None = None  # None: not run (no --allow-execution)
    usd: float = 0.0
    calls: int = 0
    notes: list[str] = field(default_factory=list)


def _cmds(spec: dict, key: str) -> list[dict[str, str]]:
    return [{"run": str(c.get("run", "")), "cwd": str(c.get("cwd") or ".")}
            for c in spec.get(key, []) if isinstance(c, dict) and c.get("run")]


async def build(description: str, chat: Chat, out: Path, *, allow_execution: bool = False,
                max_files: int = 25, fix_rounds: int = 3, agents: int = 6, force: bool = False,
                log: Callable[[str], None] = lambda s: None) -> BuildResult:
    out = out.resolve()
    if out.exists() and any(out.iterdir()) and not force:
        raise ConfigError(f"{out} is not empty", "choose a new --out folder, or pass --force to write into it")
    out.mkdir(parents=True, exist_ok=True)

    # 1. spec -----------------------------------------------------------------------------
    spec = await chat.json(PLAN_SYSTEM.format(max_files=max_files, programs=", ".join(sorted(ALLOWED_PROGRAMS))),
                           f"Build this:\n{description}", PLAN_SCHEMA, effort="high", max_tokens=12000,
                           role="build.plan")
    files = [f for f in spec.get("files", []) if isinstance(f, dict) and f.get("path")][:max_files]
    if not files:
        raise ConfigError("the planner returned no files", "rephrase the description, or try a stronger model")
    for f in files:
        safe_path(out, f["path"])  # refuse the whole plan if any path is unsafe
    spec["files"] = files
    log(f"planned {len(files)} files ({spec.get('stack', '')}); writing in {len(waves(files))} waves")

    # 2. files, in dependency waves -----------------------------------------------------------
    contract = dumps({k: spec.get(k) for k in ("name", "summary", "stack", "setup", "test", "start")}
                     | {"files": [{k: f.get(k) for k in ("path", "purpose", "interface", "depends_on")}
                                  for f in files]})
    written: dict[str, str] = {}
    for n, wave in enumerate(waves(files), 1):
        def job(f: dict) -> Callable[[str], Any]:
            async def run(effort: str) -> str:
                deps = "".join(f"\n--- {d} (already written) ---\n{written[d]}\n"
                               for d in f.get("depends_on", []) if d in written)
                text = await chat.complete(
                    FILE_SYSTEM, f"Project spec:\n{contract}\n{deps}\nWrite the file `{f['path']}` "
                                 f"(purpose: {f['purpose']}).", effort=effort, max_tokens=16000, role="build.file")
                return strip_fences(text)
            return run

        got = await parallel(chat, {f["path"]: job(f) for f in wave}, check=check_file, agents=agents)
        for path, content in got.items():
            if content is None:
                raise ConfigError(f"could not write a valid {path}", "try again or use a stronger model")
            target = safe_path(out, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content if content.endswith("\n") else content + "\n")
            written[path] = content
        log(f"wave {n}: wrote {', '.join(sorted(got))}")

    result = BuildResult(str(out), spec, sorted(written))
    if not allow_execution:
        result.notes.append("not run: pass --allow-execution to install, build and test it")
        return _finish(result, chat, out)

    # 3. run setup + tests; 4. fix and re-run --------------------------------------------------
    for attempt in range(fix_rounds + 1):
        failed = _run_all(out, spec, result, log, setup=attempt == 0 or result.notes[-1:] == ["setup changed"])
        if failed is None:
            result.tests_passed = True
            log("all setup and test commands passed")
            break
        if "not installed on this machine" in failed.output or "not an allowed build tool" in failed.output:
            result.notes.append(f"cannot run {failed.command!r}: {failed.output.strip()[:200]}")
            result.tests_passed = False
            break
        if attempt == fix_rounds:
            result.tests_passed = False
            break
        result.fix_rounds += 1
        log(f"fix round {result.fix_rounds}: {failed.command!r} failed; asking the fixer")
        fix = await chat.json(FIX_SYSTEM, _fix_prompt(spec, failed, out, written), FIX_SCHEMA, effort="high",
                              max_tokens=16000, role="build.fix")
        changed_setup = False
        for e in fix.get("edits", []) if isinstance(fix.get("edits"), list) else []:
            if not isinstance(e, dict) or not e.get("path"):
                continue
            target = safe_path(out, str(e["path"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            content = strip_fences(str(e.get("content", "")))
            target.write_text(content if content.endswith("\n") else content + "\n")
            written[str(e["path"])] = content
            changed_setup |= Path(str(e["path"])).name in ("requirements.txt", "package.json", "pyproject.toml")
        for key in ("setup", "test"):
            if _cmds(fix, key) and _cmds(fix, key) != _cmds(spec, key):
                spec[key] = _cmds(fix, key)
                changed_setup |= key == "setup"
        result.notes.append("setup changed" if changed_setup else f"fixed: {str(fix.get('analysis', ''))[:300]}")
    result.files_written = sorted(written)
    return _finish(result, chat, out)


def _run_all(out: Path, spec: dict, result: BuildResult, log: Callable[[str], None],
             setup: bool) -> CommandResult | None:
    """Run setup (when needed) then tests; return the first failure, or None if all pass."""
    if setup and not (out / ".venv").exists() and (any(out.rglob("*.py")) or (out / "requirements.txt").exists()):
        subprocess.run([sys.executable, "-m", "venv", str(out / ".venv")], check=False, capture_output=True)
    for key in (("setup", "test") if setup else ("test",)):
        for c in _cmds(spec, key):
            try:
                r = run_command(c["run"], out, c["cwd"], timeout=900 if key == "setup" else 300)
            except CommandRejected as e:
                r = CommandResult(c["run"], False, f"{e} (rejected before running)")
            result.commands.append({"stage": key, **asdict(r)})
            log(f"  {key}: {c['run']} ({c['cwd']}) -> {'ok' if r.ok else 'FAILED'} in {r.seconds}s")
            if not r.ok:
                return r
    if not _cmds(spec, "test"):
        result.notes.append("the plan has no test command")
    return None


def _fix_prompt(spec: dict, failed: CommandResult, out: Path, written: dict[str, str]) -> str:
    total = sum(len(v) for v in written.values())
    if total <= 80_000:
        chosen = list(written)
    else:  # big project: the files the error mentions, and what they depend on
        named = [p for p in written if re.search(re.escape(Path(p).name), failed.output)]
        deps = {d for f in spec["files"] if f["path"] in named for d in f.get("depends_on", [])}
        chosen = [p for p in written if p in named or p in deps] or list(written)[:10]
    files = "".join(f"\n--- {p} ---\n{written[p]}\n" for p in chosen)
    return (f"Spec:\n{dumps({k: spec.get(k) for k in ('summary', 'stack', 'setup', 'test')})}\n\n"
            f"Failing command: {failed.command}\nOutput (last part):\n{failed.output[-5000:]}\n\n"
            f"Project files: {', '.join(sorted(written))}\n{files}")


def _finish(result: BuildResult, chat: Chat, out: Path) -> BuildResult:
    result.usd, result.calls = round(chat.usd, 4), chat.calls
    meta = out / ".swarm"
    meta.mkdir(exist_ok=True)
    (meta / "build.json").write_text(json.dumps(asdict(result), indent=1, ensure_ascii=False))
    gi = out / ".gitignore"
    if not gi.exists():
        gi.write_text(".swarm/\n.venv/\nnode_modules/\n__pycache__/\n")
    return result
