"""Run model-written Python against a task's tests, as the deterministic check for code tasks.

This executes code an LLM wrote, so it is opt-in (`swarm run --allow-code-execution`) and
contained as far as the standard library allows:

* a fresh temporary directory as the working directory, deleted afterwards;
* an empty environment: no API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, TINYFISH_API_KEY ...)
  and no HOME, so the code can't read your credentials from the environment;
* `python -I` (isolated mode: no user site-packages, no PYTHON* variables, script dir not on
  the path);
* hard limits on Linux/macOS: CPU seconds, address space, file size and open files, plus a
  wall-clock timeout that kills the whole process group;
* no stdin.

It is NOT a security boundary against hostile code: the process can still read files your
user can read and open network connections. Run untrusted workloads in a container or VM
(e.g. `docker run --network none`), as the README explains.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_FENCE = re.compile(r"^\s*```[\w+-]*\s*\n(.*?)\n\s*```\s*$", re.S)


def strip_fences(code: str) -> str:
    """Models sometimes wrap code in ``` fences despite being asked not to."""
    m = _FENCE.match(code)
    return m.group(1) if m else code


@dataclass
class RunResult:
    ok: bool
    output: str  # stdout + stderr, truncated
    timed_out: bool = False


def _limits(cpu_s: int, mem_mb: int):  # pragma: no cover - runs in the child process
    def apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            mem = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        except (ImportError, ValueError, OSError):
            pass  # not available on this platform: the wall-clock timeout still applies

    return apply


def run_python(code: str, tests: str = "", timeout: float = 10.0, mem_mb: int = 512,
               workdir: Path | None = None, python: str | None = None, isolated: bool = True) -> RunResult:
    """Run `code` followed by `tests` (plain asserts) in a contained subprocess.

    `workdir`: run there and keep its files (e.g. charts) instead of a throwaway temp dir.
    `python`/`isolated=False`: an interpreter whose user site-packages (pandas, matplotlib) are
    visible, for data analysis; the environment is still emptied of API keys."""
    tmp_ctx = tempfile.TemporaryDirectory(prefix="swarm-code-") if workdir is None else None
    tmp = Path(tmp_ctx.name) if tmp_ctx is not None else workdir
    assert tmp is not None
    try:
        script = tmp / "solution.py"
        script.write_text(strip_fences(code) + "\n\n# --- tests ---\n" + tests + "\n")
        env = {"PATH": os.defpath, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
               "MPLCONFIGDIR": str(tmp / ".mpl"), "MPLBACKEND": "Agg"}
        posix = os.name == "posix"
        try:
            proc = subprocess.Popen(
                [python or sys.executable, *(["-I"] if isolated else ["-E"]), str(script)], cwd=tmp, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=posix,
                preexec_fn=_limits(int(timeout) + 1, mem_mb) if posix else None)
        except OSError as e:
            return RunResult(False, f"could not start python: {e}")
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if posix:
                os.killpg(proc.pid, signal.SIGKILL)  # the whole group, including anything it spawned
            else:  # pragma: no cover
                proc.kill()
            out, _ = proc.communicate()
            return RunResult(False, (out or b"").decode(errors="replace")[-4000:], timed_out=True)
        return RunResult(proc.returncode == 0, (out or b"").decode(errors="replace")[-4000:])
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def code_validator(tests: str, timeout: float = 10.0, mem_mb: int = 512):
    """Engine validator: the answer passes when the task's tests run without error."""

    def check(answer: str) -> bool:
        return run_python(answer, tests, timeout, mem_mb).ok

    return check


# ----------------------------------------------------------------------------- build commands

ALLOWED_PROGRAMS = {
    "python", "python3", "pip", "pip3", "uv", "pytest", "ruff", "mypy", "black",
    "node", "npm", "npx", "pnpm", "yarn", "bun", "deno", "tsc", "eslint", "prettier", "jest", "vitest",
    "go", "cargo", "rustc", "make", "cmake", "mvn", "gradle", "java", "javac", "ruby", "bundle", "rake",
    "php", "composer", "dotnet",
}
_SHELL_OPS = {"|", "||", "&&", "&", ";", ">", ">>", "<", "<<", "2>", "2>&1", "&>"}


class CommandRejected(Exception):
    pass


@dataclass
class CommandResult:
    command: str
    ok: bool
    output: str
    timed_out: bool = False
    seconds: float = 0.0


def check_command(command: str) -> list[str]:
    """Parse a model-proposed command, refusing anything outside the build-tool allowlist.

    No shell is ever used: pipes, redirects, `&&`, `$(...)` and backticks are rejected rather
    than interpreted, and the program must be a known build/test tool."""
    import shlex

    try:
        argv = shlex.split(command)
    except ValueError as e:
        raise CommandRejected(f"cannot parse {command!r}: {e}") from e
    if not argv:
        raise CommandRejected("empty command")
    # Arguments are passed as-is (no shell), so a `;` inside a quoted `python -c` argument is
    # harmless. Bare shell operators and expansions mean the model expected a shell: refuse
    # them instead of running something different from what it meant.
    for tok in argv:
        if tok in _SHELL_OPS or tok.startswith(("$", "|", ">", "<", "&", ";")) or "$(" in tok or "`" in tok:
            raise CommandRejected(f"shell syntax is not allowed: {command!r} (one command per entry, no pipes, "
                                  "no &&, no redirects, no $VARS)")
    if any(t in ("sudo", "su", "doas") for t in argv):
        raise CommandRejected(f"privilege escalation is not allowed: {command!r}")
    prog = Path(argv[0]).name
    if "/" in argv[0] or prog not in ALLOWED_PROGRAMS:
        raise CommandRejected(f"{argv[0]!r} is not an allowed build tool ({', '.join(sorted(ALLOWED_PROGRAMS))})")
    return argv


def build_env(project: Path) -> dict[str, str]:
    """Environment for build commands: your PATH (to find node/npm/python), a private HOME
    inside the project (so ~/.npmrc, ~/.pypirc and other credentials aren't used), no API keys."""
    home = project / ".swarm" / "home"
    home.mkdir(parents=True, exist_ok=True)
    path = os.environ.get("PATH", os.defpath)
    venv = project / ".venv"
    env = {"PATH": path, "HOME": str(home), "LANG": "C.UTF-8", "CI": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PIP_DISABLE_PIP_VERSION_CHECK": "1", "npm_config_fund": "false", "npm_config_audit": "false",
           "npm_config_update_notifier": "false", "XDG_CACHE_HOME": str(home / ".cache")}
    if venv.exists():  # Python projects get their own virtualenv, first on PATH
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{path}"
        env["VIRTUAL_ENV"] = str(venv)
    return env


def run_command(command: str, project: Path, cwd: str = ".", timeout: float = 600.0) -> CommandResult:
    """Run one allowlisted command inside `project` (never outside it), with a timeout that
    kills the whole process group."""
    import shutil
    import time

    project = project.resolve()
    workdir = (project / cwd).resolve()
    if project != workdir and project not in workdir.parents:
        raise CommandRejected(f"cwd {cwd!r} is outside the project")
    argv = check_command(command)
    env = build_env(project)
    exe = shutil.which(argv[0], path=env["PATH"])
    if exe is None:
        return CommandResult(command, False, f"{argv[0]}: not installed on this machine")
    posix = os.name == "posix"
    t0 = time.monotonic()
    proc = subprocess.Popen([exe, *argv[1:]], cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=posix)
    try:
        out, _ = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        if posix:
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
        out, _ = proc.communicate()
        timed_out = True
    text = (out or b"").decode(errors="replace")
    return CommandResult(command, proc.returncode == 0 and not timed_out, text[-6000:], timed_out,
                         round(time.monotonic() - t0, 1))
