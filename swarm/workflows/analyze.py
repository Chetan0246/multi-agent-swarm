"""`swarm analyze sales.csv "What drives revenue, and what changed this year?" --allow-code-execution`

    data file ─► profile (columns, types, row count, samples, numeric ranges) in code, not by a model
              ─► lead analyst: up to N independent analysis questions
              ─► analysts in parallel: each writes Python that reads the data, prints its
                 result as JSON and may save a chart; the code RUNS in the sandbox and an
                 error is sent back for a fix (up to 3 tries)
              ─► report written only from the results the code printed; every number in it
                 is checked against those results, and numbers that aren't there are counted
                 (`unverified_numbers`)

Charts and each analyst's code are kept next to the report (analysis/), so every figure can
be reproduced. Uses pandas and matplotlib when your Python has them, else the standard library.
"""
from __future__ import annotations

import asyncio
import contextlib
import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from ..sandbox import run_python
from .chat import Chat, dumps, parallel

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "question": {"type": "string"}, "method": {"type": "string"}},
        "required": ["id", "question", "method"], "additionalProperties": False}}},
    "required": ["questions"], "additionalProperties": False,
}
PLAN_SYSTEM = (
    "You are the lead data analyst. From the user's question and the data profile, choose at most "
    "{n} independent, concrete analysis questions that together answer it (each answerable by "
    "one short script: an aggregate, a comparison, a trend, a correlation, a distribution). Give "
    "each a short id (letters/digits) and the method."
)
CODE_SYSTEM = (
    "You are a data analyst. Write one Python script that answers your question from the file "
    "`{file}` in the current directory. Libraries available: {libs}. Print exactly one final "
    "line: RESULT: followed by a JSON object with `finding` (one or two sentences with the key "
    "numbers) and `numbers` (an object of the named figures you computed). If a chart helps, save "
    "ONE PNG named chart.png (matplotlib, Agg backend; no plt.show()). No network, no other files. "
    "Output only the code."
)
REPORT_SYSTEM = (
    "You are the lead data analyst writing the report. Use ONLY the results below, which came from "
    "code run on the data; never invent or estimate numbers. Structure: a short direct answer, "
    "then one section per finding (embed its chart where given, with the exact Markdown image "
    "line shown), then 'Limitations' (data quality, failed analyses, what the data can't show). "
    "Markdown only."
)


@dataclass
class AnalysisResult:
    markdown: str
    findings: dict[str, Any]
    failed: list[str]
    unverified_numbers: int
    usd: float
    calls: int
    files: list[str] = field(default_factory=list)


def find_python() -> tuple[str, list[str]]:
    """An interpreter with pandas/matplotlib if one exists (this one, or python3 on PATH)."""
    candidates = [sys.executable] + [p for p in (shutil.which("python3"), shutil.which("python")) if p]
    for py in candidates:
        try:
            r = subprocess.run([py, "-E", "-c", "import pandas, matplotlib"], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            return py, ["pandas", "numpy", "matplotlib", "the standard library"]
    return sys.executable, ["the standard library only (csv, statistics, json); no pandas, no charts"]


def profile(path: Path, max_rows: int = 200_000) -> dict[str, Any]:
    """Describe a CSV/TSV file with the standard library (no model involved)."""
    if path.suffix.lower() not in (".csv", ".tsv", ".txt"):
        raise ConfigError(f"{path.name}: only CSV/TSV files are supported", "export the sheet as CSV")
    with path.open(newline="", errors="replace") as fh:
        sample = fh.read(64_000)
        fh.seek(0)
        try:
            dialect: Any = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(fh, dialect))[: max_rows + 1]
    if len(rows) < 2:
        raise ConfigError(f"{path.name}: needs a header row and at least one data row")
    header, data = rows[0], rows[1:]
    cols: list[dict[str, Any]] = []
    for j, name in enumerate(header):
        vals = [r[j] for r in data if j < len(r) and r[j].strip()]
        nums = []
        for v in vals:
            with contextlib.suppress(ValueError):
                nums.append(float(v.replace(",", "")))
        col: dict[str, Any] = {"name": name, "non_empty": len(vals), "distinct": len(set(vals))}
        if vals and len(nums) >= 0.9 * len(vals):
            col |= {"type": "number", "min": min(nums), "max": max(nums), "mean": round(statistics.fmean(nums), 4)}
        else:
            col |= {"type": "text", "examples": sorted(set(vals))[:5]}
        cols.append(col)
    return {"file": path.name, "rows": len(data), "columns": cols, "first_rows": data[:5],
            "delimiter": getattr(dialect, "delimiter", ",")}


_NUM = re.compile(r"(?<![\w.])[-−]?\d[\d,]*\.?\d*%?")


def _numbers(text: str) -> set[str]:
    """Numbers as written, without sign: prose says "fell 31.19%" for a computed -31.19."""
    out = set()
    for m in _NUM.findall(text):
        n = m.rstrip("%").replace(",", "").rstrip(".").lstrip("-−")
        if n and len(n.replace(".", "")) >= 2:  # single digits are list markers etc.
            out.add(n)
    return out


def _matches(printed: str, value: float) -> bool:
    """True when `value`, rounded to the precision `printed` shows, is `printed` (sign ignored)."""
    decimals = len(printed.split(".", 1)[1]) if "." in printed else 0
    return f"{round(abs(value), decimals):.{decimals}f}" == f"{float(printed):.{decimals}f}"


def unsupported_numbers(report: str, computed: list[float]) -> list[str]:
    """Numbers in the report that no computed figure supports.

    A number is supported when it is a computed figure, a percentage of one (shares stored as
    0.305), or one simple step from two of them: a difference, a sum, a ratio, or a percentage
    or percent change. Each must match at exactly the precision the report prints, so a
    derived figure passes only if the arithmetic is right."""
    vals = sorted({abs(v) for v in computed if v == v})[:400]  # NaN-safe; bound the pair search
    out = []
    for n in sorted(_numbers(report)):
        direct = any(_matches(n, v) or _matches(n, 100 * v) for v in vals)
        if direct:
            continue
        derived = any(_matches(n, a - b) or _matches(n, a + b)
                      or (b != 0 and (_matches(n, a / b) or _matches(n, 100 * a / b) or _matches(n, 100 * (a / b - 1))))
                      for a in vals for b in vals if a != b)
        if not derived:
            out.append(n)
    return out


def _computed(obj: object) -> list[float]:
    """Every number in the analyses' JSON results (and the data profile)."""
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", json.dumps(obj))]


async def analyze(data: Path, question: str, chat: Chat, out: Path, *, max_questions: int = 5,
                  allow_code: bool = False, timeout: float = 60.0, agents: int = 5,
                  log: Callable[[str], None] = lambda s: None) -> AnalysisResult:
    if not allow_code:
        raise ConfigError("analysis runs model-written code on your data",
                          "add --allow-code-execution (it runs in the sandbox: no API keys, time and memory limits)")
    prof = profile(data)
    python, libs = find_python()
    log(f"profiled {prof['rows']} rows x {len(prof['columns'])} columns; using {libs[0]}")
    plan = await chat.json(PLAN_SYSTEM.format(n=max_questions), f"Question: {question}\n\nData profile:\n{dumps(prof)}",
                           QUESTIONS_SCHEMA, effort="high", role="analyze.plan")
    qs: list[dict[str, Any]] = []
    for i, q in enumerate(plan.get("questions", [])[:max_questions]):
        if isinstance(q, dict) and q.get("question"):
            qid = re.sub(r"[^A-Za-z0-9_-]", "", str(q.get("id") or "")) or f"q{i + 1}"
            qs.append({**q, "id": qid if qid not in {x["id"] for x in qs} else f"q{i + 1}"})
    if not qs:
        raise ConfigError("the analyst returned no questions", "make the question more specific")
    work = out / "analysis"
    work.mkdir(parents=True, exist_ok=True)

    def job(q: dict) -> Callable[[str], Any]:
        async def run(effort: str) -> str:
            qdir = work / q["id"]
            qdir.mkdir(exist_ok=True)
            shutil.copy(data, qdir / data.name)
            feedback = ""
            for _ in range(3):  # write, run, and fix on error
                code = await chat.complete(CODE_SYSTEM.format(file=data.name, libs=", ".join(libs)),
                                           f"Question: {q['question']}\nMethod: {q['method']}\nData profile:\n"
                                           f"{dumps(prof)}{feedback}", effort=effort, max_tokens=6000,
                                           role="analyze.code")
                # In a thread: a slow script must not stall the other analysts' model calls.
                r = await asyncio.to_thread(run_python, code, workdir=qdir, timeout=timeout, mem_mb=2048,
                                            python=python, isolated=False)
                (qdir / "analysis.py").write_text((qdir / "solution.py").read_text())
                line = next((ln for ln in reversed(r.output.splitlines()) if ln.startswith("RESULT:")), None)
                if r.ok and line:
                    try:
                        res = json.loads(line[len("RESULT:"):].strip())
                    except json.JSONDecodeError:
                        res = None
                    if isinstance(res, dict) and res.get("finding"):
                        res["chart"] = (qdir / "chart.png").exists()
                        return json.dumps(res)
                feedback = f"\n\nYour previous script failed. Fix it. Output:\n{r.output[-3000:]}"
            return ""
        return run

    got = await parallel(chat, {q["id"]: job(q) for q in qs}, agents=agents,
                         check=lambda qid, a: bool(a), efforts=("medium", "high"))
    findings: dict[str, Any] = {}
    files: list[str] = []
    for q in qs:
        if got.get(q["id"]):
            res = json.loads(got[q["id"]] or "{}")
            if res.get("chart"):
                rel = f"analysis/{q['id']}/chart.png"
                res["chart_markdown"] = f"![{q['question']}]({rel})"
                files.append(rel)
            findings[q["id"]] = {"question": q["question"], **res}
    failed = [q["question"] for q in qs if q["id"] not in findings]
    log(f"{len(findings)}/{len(qs)} analyses ran; {len(failed)} failed")
    if not findings:
        raise ConfigError("no analysis script ran successfully", "check the data file, or try a stronger model")
    md = await chat.complete(REPORT_SYSTEM, f"Question: {question}\nData: {prof['file']}, {prof['rows']} rows\n\n"
                                            f"Results:\n{dumps(findings)}\n\nFailed analyses: {failed or 'none'}",
                             effort="medium", max_tokens=8000, role="analyze.report")
    unverified = unsupported_numbers(md, _computed([findings, prof]))
    # Keep what the code computed next to the report, so every figure can be audited later.
    (work / "results.json").write_text(json.dumps({"question": question, "profile": prof, "findings": findings,
                                                   "failed": failed, "unverified_numbers": unverified}, indent=1))
    if unverified:
        md = md.rstrip() + (f"\n\n> **Number check:** {len(unverified)} number(s) in this report do not appear in "
                            f"the computed results: {', '.join(unverified[:12])}.\n")
    return AnalysisResult(md, findings, failed, len(unverified), round(chat.usd, 4), chat.calls, files)
