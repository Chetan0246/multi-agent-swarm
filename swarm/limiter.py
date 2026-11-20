"""AIMD concurrency limiter: back off hard on 429s, creep back up on success."""
from __future__ import annotations

import asyncio


class AdaptiveLimiter:
    def __init__(self, start: int, floor: int = 1, ceiling: int | None = None):
        self.limit = float(max(floor, start))
        self.floor = floor
        self.ceiling = ceiling or start
        self.inflight = 0
        self._cond = asyncio.Condition()
        self.throttle_events = 0

    async def __aenter__(self):
        async with self._cond:
            await self._cond.wait_for(lambda: self.inflight < int(self.limit))
            self.inflight += 1
        return self

    async def __aexit__(self, *exc):
        async with self._cond:
            self.inflight -= 1
            self._cond.notify_all()

    def has_free_slot(self) -> bool:
        return self.inflight < int(self.limit)

    def on_success(self) -> None:
        self.limit = min(self.ceiling, self.limit + 1.0 / max(1.0, self.limit))

    def on_ramp(self) -> None:
        """Ramp limit: hold just below the current rate instead of halving."""
        self.throttle_events += 1
        self.limit = max(self.floor, self.limit - 1)

    def on_throttle(self) -> None:
        self.throttle_events += 1
        self.limit = max(self.floor, self.limit * 0.5)
