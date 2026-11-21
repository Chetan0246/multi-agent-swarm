"""Report export: Markdown, JSON (round-trips for follow-ups) and self-contained HTML."""
from __future__ import annotations

import html
import json
import re
from dataclasses import asdict
from pathlib import Path

from .research import Citation, Finding, ResearchReport

FORMATS = ("md", "html", "json")


def to_json(report: ResearchReport) -> str:
    return json.dumps({"version": 1, "question": report.question, "markdown": report.markdown,
                       "findings": [asdict(f) for f in report.findings],
                       "sources": [asdict(c) for c in report.sources], "stats": report.stats},
                      indent=2, ensure_ascii=False)


def from_json(text: str) -> ResearchReport:
    d = json.loads(text)
    findings = [Finding(**{**f, "citations": [Citation(**c) for c in f["citations"]]})
                for f in d["findings"]]
    return ResearchReport(d["question"], d["markdown"], findings,
                          [Citation(**c) for c in d["sources"]], d.get("stats", {}))


def load_report(path: str | Path) -> ResearchReport:
    return from_json(Path(path).read_text())


# ----------------------------------------------------------------------------- markdown -> HTML

_INLINE = [
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\w)"), r"<em>\1</em>"),
]
_LINK = re.compile(r"\[([^\]]+)\]\(((?:https?|file)://[^)\s]+)\)")
_REF = re.compile(r"\[(\d{1,3})\]")


def _inline(text: str, tips: dict[int, str]) -> str:
    out = html.escape(text, quote=False)
    for rx, rep in _INLINE:
        out = rx.sub(rep, out)
    out = _LINK.sub(lambda m: f'<a href="{html.escape(m.group(2))}" target="_blank" '
                              f'rel="noopener noreferrer">{m.group(1)}</a>', out)

    def ref(m: re.Match) -> str:
        k = int(m.group(1))
        tip = html.escape(tips.get(k, ""), quote=True)
        return f'<a class="ref" href="#src-{k}" title="{tip}">[{k}]</a>'

    return _REF.sub(ref, out)


def markdown_to_html(md: str, tips: dict[int, str] | None = None) -> str:
    """Small, safe Markdown subset: headings, paragraphs, lists, blockquotes, emphasis, links.

    Everything is HTML-escaped first, so report text can never inject markup.
    """
    tips = tips or {}
    blocks: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if m := re.match(r"^(#{1,4})\s+(.*)", line):
            n = len(m.group(1))
            blocks.append(f"<h{n}>{_inline(m.group(2), tips)}</h{n}>")
            i += 1
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.", line))
            items = []
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i]))
                i += 1
            tag = "ol" if ordered else "ul"
            blocks.append(f"<{tag}>" + "".join(f"<li>{_inline(x, tips)}</li>" for x in items) + f"</{tag}>")
            continue
        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> "))
                i += 1
            blocks.append(f"<blockquote>{_inline(' '.join(quote), tips)}</blockquote>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|>|\s*([-*]|\d+\.)\s)", lines[i]):
            para.append(lines[i])
            i += 1
        blocks.append(f"<p>{_inline(' '.join(para), tips)}</p>")
    return "\n".join(blocks)


_CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--accent:#2a6fdb;--card:#f5f6f8;--warn:#b54708}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--muted:#a0a0a8;
--accent:#7aa7ff;--card:#1f2024;--warn:#f0a050}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:46rem;margin:0 auto;padding:2.5rem 1rem 4rem}
h1{font-size:1.7rem;line-height:1.25}h2{margin-top:2rem;font-size:1.25rem}
a{color:var(--accent)}a.ref{text-decoration:none;font-size:.8em;vertical-align:super;padding:0 1px}
blockquote{margin:1.2rem 0;padding:.7rem 1rem;background:var(--card);
border-left:3px solid var(--warn);border-radius:6px}
.meta{color:var(--muted);font-size:.9rem;border-top:1px solid var(--card);margin-top:2.5rem;padding-top:1rem}
li{margin:.25rem 0}code{background:var(--card);padding:0 .3em;border-radius:4px}
.sources li{word-break:break-word}.sources .q{display:block;color:var(--muted);font-size:.9em}
"""


def cost_label(stats: dict) -> str:
    """'$0.45', or '$0.45 simulated (mock run, no real spend)' for offline demo runs."""
    usd = f"${stats['usd']:.2f}"
    return f"{usd} simulated (mock run, no real spend)" if stats.get("run_search") == "mock" else usd


def to_html(report: ResearchReport) -> str:
    tips = {i: f"{c.title}: “{c.quote}”" if c.quote else c.title
            for i, c in enumerate(report.sources, 1)}
    body_md = report.markdown.split("\n## Sources\n", 1)[0]
    body = markdown_to_html(body_md, tips)
    srcs = "".join(
        f'<li id="src-{i}"><a href="{html.escape(c.url)}" target="_blank" rel="noopener noreferrer">'
        f'{html.escape(c.title or c.url)}</a>'
        + (f'<span class="q">“{html.escape(c.quote)}”</span>' if c.quote else "") + "</li>"
        for i, c in enumerate(report.sources, 1))
    s = report.stats
    meta = " · ".join(x for x in [
        f"{s.get('verified_%', '?')}% of findings verified" if "verified_%" in s else "",
        f"claim support {s['claim_support_%']}%" if s.get("claim_support_%") is not None else "",
        f"{len(report.sources)} sources",
        f"cost {cost_label(s)}" if isinstance(s.get("usd"), (int, float)) else "",
        f"{s['wall_time_s']}s" if "wall_time_s" in s else ""] if x)
    title = html.escape(report.question.splitlines()[0][:120])
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<meta http-equiv=\"Content-Security-Policy\" "
            "content=\"default-src 'none'; style-src 'unsafe-inline'\">"
            f"<title>{title}</title><style>{_CSS}</style></head><body><main>{body}"
            f"<h2>Sources</h2><ol class=\"sources\">{srcs}</ol>"
            f"<p class=\"meta\">{html.escape(meta)}<br>Generated by multi-agent-swarm deep research. "
            f"Claims are checked against quoted source text; check important facts yourself.</p>"
            f"</main></body></html>")


def write_report(report: ResearchReport, path: str | Path, fmt: str | None = None) -> str:
    p = Path(path)
    fmt = fmt or (p.suffix.lstrip(".").lower() or "md")
    fmt = {"markdown": "md", "htm": "html"}.get(fmt, fmt)
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; use one of {', '.join(FORMATS)}")
    text = {"md": lambda: report.markdown, "html": lambda: to_html(report),
            "json": lambda: to_json(report)}[fmt]()
    p.write_text(text)
    return fmt
