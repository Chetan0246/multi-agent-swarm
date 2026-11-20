"""Core data types: tasks, results, and validators."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

Validator = Callable[[str], bool]


@dataclass
class Task:
    """One unit of work.

    `difficulty` (1=easy .. 3=hard) keys the adaptive policy's statistics.
    `priority`   lower runs sooner (0 = normal).
    `expected` / `pattern` / `validator` give a cheap deterministic check. Tasks
    with no checker fall back to the model's self-reported confidence.
    """

    id: str
    prompt: str
    difficulty: int = 1
    expected: str | None = None
    pattern: str | None = None
    validator: Validator | None = None
    priority: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def est_tokens(self) -> int:
        return max(1, len(self.prompt) // 4)  # ~4 chars/token, no API call

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.prompt.strip().encode()).hexdigest()[:24]

    @property
    def has_checker(self) -> bool:
        return self.validator is not None or self.expected is not None or self.pattern is not None

    def check(self, answer: str | None, confidence: float = 1.0, min_conf: float = 0.0) -> bool:
        if answer is None or not str(answer).strip():
            return False
        a = str(answer).strip()
        if self.validator is not None:
            try:
                return bool(self.validator(a))
            except Exception:
                return False
        if self.expected is not None:
            return _norm(a) == _norm(self.expected)
        if self.pattern is not None:
            return re.search(self.pattern, a, re.S) is not None
        return confidence >= min_conf  # unchecked task: trust calibrated confidence

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(
            id=str(d["id"]),
            prompt=d["prompt"],
            difficulty=int(d.get("difficulty", 1)),
            expected=d.get("expected"),
            pattern=d.get("pattern"),
            priority=int(d.get("priority", 0)),
            meta=d.get("meta", {}),
        )


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().rstrip("."))


@dataclass
class Result:
    task_id: str
    answer: str | None
    ok: bool
    attempts: int
    effort: str
    tokens: int  # billed-equivalent tokens attributed to this task
    cached: bool = False
    confidence: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
