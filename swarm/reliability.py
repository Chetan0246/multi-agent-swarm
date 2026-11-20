"""Reliability primitives: circuit breaker, hedging, token budget, write-ahead journal."""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import IO


class CircuitBreaker:
    """Per-model breaker: opens after N consecutive transient failures.

    open      → requests to that model are rerouted to another rung or wait
    half-open → after `cooldown` seconds one probe goes through
    closed    → on a success
    """

    def __init__(self, threshold: int = 5, cooldown: float = 2.0, clock=time.monotonic):
        self.threshold = threshold
        self.cooldown = cooldown
        self.clock = clock
        self._fails: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._probing: set[str] = set()
        self.trips = 0

    def available(self, model: str) -> bool:
        """Pure check: may a call to `model` be dispatched right now?"""
        t = self._opened_at.get(model)
        if t is None:
            return True
        return self.clock() - t >= self.cooldown and model not in self._probing

    def acquire(self, model: str) -> None:
        """Call when dispatching. If the breaker is half-open this claims the single probe."""
        if model in self._opened_at:
            self._probing.add(model)

    def release(self, model: str) -> None:
        """A dispatched probe ended without a verdict (e.g. 429): free the probe slot."""
        self._probing.discard(model)

    def retry_after(self, model: str) -> float:
        t = self._opened_at.get(model)
        return 0.0 if t is None else max(0.01, self.cooldown - (self.clock() - t))

    def success(self, model: str) -> None:
        self._fails[model] = 0
        self._opened_at.pop(model, None)
        self._probing.discard(model)

    def failure(self, model: str) -> None:
        if model in self._probing:  # failed probe: re-open for another cooldown
            self._probing.discard(model)
            self._opened_at[model] = self.clock()
            return
        self._fails[model] = self._fails.get(model, 0) + 1
        if self._fails[model] >= self.threshold and model not in self._opened_at:
            self._opened_at[model] = self.clock()
            self.trips += 1

    def is_open(self, model: str) -> bool:
        return model in self._opened_at


class Hedger:
    """Straggler mitigation: duplicate a call that runs past the observed p95.

    Capped at `max_fraction` of calls, so it buys tail latency with a few % of tokens.
    """

    def __init__(self, quantile: float = 0.95, factor: float = 1.5, min_delay: float = 0.05,
                 max_fraction: float = 0.08, window: int = 200, warmup: int = 10):
        self.q = quantile
        self.factor = factor
        self.min_delay = min_delay
        self.max_fraction = max_fraction
        self.lat: deque[float] = deque(maxlen=window)
        self.warmup = warmup
        self.calls = 0
        self.hedges = 0
        self.wins = 0

    def record(self, seconds: float) -> None:
        self.lat.append(seconds)

    def delay(self) -> float | None:
        if len(self.lat) < self.warmup:
            return None
        s = sorted(self.lat)
        return max(self.min_delay, self.factor * s[int(self.q * (len(s) - 1))])

    def may_hedge(self) -> bool:
        return self.hedges < self.max_fraction * max(1, self.calls)


class TokenBudget:
    """Hard ceiling on billed-equivalent tokens. Reserve before a call, settle after."""

    def __init__(self, limit: float | None):
        self.limit = limit
        self.spent = 0.0
        self.reserved = 0.0
        self.denied = 0

    def try_reserve(self, est: float) -> bool:
        if self.limit is None:
            return True
        if self.spent + self.reserved + est > self.limit:
            self.denied += 1
            return False
        self.reserved += est
        return True

    def settle(self, est: float, actual: float) -> None:
        if self.limit is not None:
            self.reserved = max(0.0, self.reserved - est)
        self.spent += actual

    @property
    def remaining(self) -> float:
        return float("inf") if self.limit is None else self.limit - self.spent - self.reserved


class Journal:
    """Append-only write-ahead log of finished tasks.

    A crashed or interrupted run resumes from it: finished task ids are skipped,
    and their answers also seed the content cache (by prompt fingerprint).
    """

    def __init__(self, path: str | Path | None, fsync_every: int = 50):
        self.path = Path(path) if path else None
        self.fsync_every = fsync_every
        self.done: dict[str, dict] = {}
        self.by_fp: dict[str, str] = {}
        self._fh: IO[str] | None = None
        self._n = 0
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line after a crash: ignore it
                self.done[rec["task_id"]] = rec
                if rec.get("ok") and rec.get("fp"):
                    self.by_fp[rec["fp"]] = rec["answer"]

    def write(self, rec: dict) -> None:
        if not self.path:
            return
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        fh = self._fh
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n % self.fsync_every == 0:
            fh.flush()
            os.fsync(fh.fileno())

    def close(self) -> None:
        fh = self._fh
        if fh is not None:
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
            self._fh = None
