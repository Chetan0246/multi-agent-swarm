"""Your own documents as a source: .txt, .md, .html and (with `pip install pypdf`) .pdf.

Documents are split into passages and searched with BM25 (no external service, no cost).
Passages are addressed as `file://<path>#p<n>`, so they are cited and quote-checked exactly
like web pages.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ..errors import ConfigError
from .base import Page, SearchFilters, SearchHit

_TOK = re.compile(r"[a-z0-9]+")
_TAG = re.compile(r"<[^>]+>")
SUPPORTED = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


def _read(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ModuleNotFoundError as e:
            raise ConfigError(f"{path}: reading PDFs needs `pip install pypdf`") from e
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    text = path.read_text(errors="replace")
    if suffix in (".html", ".htm"):
        text = _TAG.sub(" ", re.sub(r"(?is)<(script|style).*?</\1>", " ", text))
    return text


class LocalDocs:
    name = "documents"
    free = True

    def __init__(self, paths: Sequence[str | Path], passage_chars: int = 1500):
        self.passages: list[tuple[str, str, str]] = []  # (url, title, text)
        files: list[Path] = []
        for p in map(Path, paths):
            if p.is_dir():
                files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
            elif p.is_file():
                files.append(p)
            else:
                raise ConfigError(f"document not found: {p}")
        for f in files:
            text = _read(f)
            paras = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
            buf, n = "", 0
            for para in paras + [""]:
                if para and len(buf) + len(para) < passage_chars:
                    buf = f"{buf}\n\n{para}" if buf else para
                    continue
                if buf:
                    n += 1
                    self.passages.append((f"file://{f.resolve()}#p{n}", f"{f.name} (part {n})", buf))
                buf = para
        self._tf = [Counter(_TOK.findall(t.lower())) for _, _, t in self.passages]
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n_docs = max(1, len(self.passages))
        self._idf = {w: math.log(1 + (n_docs - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        # max(1, ...): passages with no word tokens (e.g. only symbols) must not divide by zero
        self._avg = max(1.0, sum(sum(tf.values()) for tf in self._tf) / n_docs)
        self.stats = {"documents": len(files), "passages": len(self.passages), "searches": 0}

    def _score(self, q: list[str], i: int, k1: float = 1.5, b: float = 0.75) -> float:
        tf, length = self._tf[i], sum(self._tf[i].values())
        return sum(self._idf.get(w, 0) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * length / self._avg))
                   for w in q if w in tf)

    async def search(self, query: str, filters: SearchFilters | None = None,
                     purpose: str | None = None) -> list[SearchHit]:
        self.stats["searches"] += 1
        q = _TOK.findall(query.lower())
        ranked = sorted(range(len(self.passages)), key=lambda i: -self._score(q, i))
        out = []
        for i in ranked[:8]:
            if self._score(q, i) <= 0:
                break
            url, title, text = self.passages[i]
            out.append(SearchHit(url, title, text[:300], "documents"))
        return out

    async def fetch(self, urls: list[str], purpose: str | None = None) -> list[Page]:
        by_url = {u: (t, x) for u, t, x in self.passages}
        return [Page(u, by_url[u][0], by_url[u][1]) if u in by_url
                else Page(u, "", "", error="not_a_document") for u in urls]
