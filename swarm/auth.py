"""API keys for the server: per-user identity, daily dollar budgets and request rate limits.

    swarm keys create --name alice --daily-usd 5 --rpm 30    # prints the key ONCE
    swarm keys list
    swarm keys revoke <id>

* Keys look like `sws_<id>_<secret>`; only a SHA-256 hash is stored, so a leaked database
  doesn't leak usable keys. Lookups compare hashes in constant time.
* A key's daily budget covers what its jobs spent today plus the caps of its jobs still
  queued or running, so a burst of submissions can't overshoot it.
* `SWARM_API_TOKEN` still works as one admin key (handy for a single-user deployment).
* With no keys and no token the server is open, which is only allowed on localhost.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .jobs import JobDB


@dataclass
class Principal:
    id: str  # key id ("admin" for SWARM_API_TOKEN, "local" for an open localhost server)
    name: str
    admin: bool
    daily_usd: float | None = None
    rpm: int | None = None


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(db: JobDB, name: str, daily_usd: float | None = None, rpm: int | None = None,
               admin: bool = False) -> tuple[str, str]:
    """Returns (key id, the full key). The full key is never stored."""
    kid = secrets.token_hex(4)
    key = f"sws_{kid}_{secrets.token_urlsafe(24)}"
    with db.tx() as c:
        c.execute("INSERT INTO api_keys (id, name, hash, created, daily_usd, rpm, admin) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (kid, name, _hash(key), time.time(), daily_usd, rpm, int(admin)))
    return kid, key


def list_keys(db: JobDB) -> list[dict[str, Any]]:
    rows = db._conn().execute("SELECT id, name, created, revoked, daily_usd, rpm, admin FROM api_keys "
                              "ORDER BY created").fetchall()
    return [dict(r) for r in rows]


def revoke_key(db: JobDB, kid: str) -> bool:
    with db.tx() as c:
        n = c.execute("UPDATE api_keys SET revoked = ? WHERE id = ? AND revoked IS NULL", (time.time(), kid)).rowcount
    return n > 0


def has_keys(db: JobDB) -> bool:
    return db._conn().execute("SELECT 1 FROM api_keys WHERE revoked IS NULL LIMIT 1").fetchone() is not None


def authenticate(db: JobDB, header: str, admin_token: str | None) -> Principal | None:
    """Resolve an `Authorization: Bearer ...` header to a principal, or None."""
    if not header.startswith("Bearer "):
        return None
    key = header[len("Bearer "):].strip()
    if admin_token and hmac.compare_digest(key, admin_token):
        return Principal("admin", "admin", True)
    if not key.startswith("sws_"):
        return None
    row = db._conn().execute("SELECT * FROM api_keys WHERE hash = ? AND revoked IS NULL", (_hash(key),)).fetchone()
    if row is None or not hmac.compare_digest(row["hash"], _hash(key)):
        return None
    return Principal(row["id"], row["name"], bool(row["admin"]), row["daily_usd"], row["rpm"])


class RateLimiter:
    """Sliding one-minute window per principal (per API process)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, who: str, rpm: int | None) -> tuple[bool, float]:
        if not rpm:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            q = self._hits[who]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= rpm:
                return False, round(60 - (now - q[0]), 1)
            q.append(now)
            return True, 0.0
