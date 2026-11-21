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


def run_python(code: str, tests: str = "", timeout: float = 10.0, mem_mb: int = 512) -> RunResult:
    """Run `code` followed by `tests` (plain asserts) in a contained subprocess."""
    with tempfile.TemporaryDirectory(prefix="swarm-code-") as tmp:
        script = Path(tmp) / "solution.py"
        script.write_text(strip_fences(code) + "\n\n# --- tests ---\n" + tests + "\n")
        env = {"PATH": os.defpath, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
        posix = os.name == "posix"
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(script)], cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=posix,
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
            return RunResult(False, (out or b"").decode(errors="replace")[-2000:], timed_out=True)
        return RunResult(proc.returncode == 0, (out or b"").decode(errors="replace")[-2000:])


def code_validator(tests: str, timeout: float = 10.0, mem_mb: int = 512):
    """Engine validator: the answer passes when the task's tests run without error."""

    def check(answer: str) -> bool:
        return run_python(answer, tests, timeout, mem_mb).ok

    return check
