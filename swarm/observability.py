"""Logs and metrics for production.

* `setup_logging()`: JSON lines (SWARM_LOG_FORMAT=json, the default in containers) or plain
  text, at SWARM_LOG_LEVEL. Every line carries the request id when there is one, and any
  `extra={...}` fields. Secrets are never logged: only key ids, never keys or prompts.
* `Metrics`: Prometheus text exposition without extra dependencies. HTTP counters and latency
  histograms are per API process; job counts, spend and queue depth come from the job
  database, so they are correct however many workers you run.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from typing import Any

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {"ts": round(record.created, 3), "level": record.levelname.lower(),
                               "logger": record.name, "msg": record.getMessage()}
        if rid := request_id.get():
            out["request_id"] = rid
        for k, v in record.__dict__.items():
            if k not in _STD and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extra = " ".join(f"{k}={v}" for k, v in record.__dict__.items() if k not in _STD and not k.startswith("_"))
        rid = request_id.get()
        return (f"{time.strftime('%H:%M:%S', time.localtime(record.created))} {record.levelname.lower():7s} "
                f"{record.name}: {record.getMessage()}{' ' + extra if extra else ''}{f' rid={rid}' if rid else ''}")


def setup_logging(fmt: str | None = None, level: str | None = None) -> None:
    fmt = fmt or os.environ.get("SWARM_LOG_FORMAT", "text")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger("swarm")
    root.handlers[:] = [handler]
    root.setLevel((level or os.environ.get("SWARM_LOG_LEVEL", "INFO")).upper())
    root.propagate = False


BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self.latency: dict[tuple[str, str], list[float]] = {}  # bucket counts + sum + count
        self.started = time.time()

    def observe(self, method: str, route: str, status: int, seconds: float) -> None:
        with self.lock:
            self.requests[(method, route, status)] += 1
            h = self.latency.setdefault((method, route), [0.0] * (len(BUCKETS) + 2))
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    h[i] += 1
            h[-2] += seconds
            h[-1] += 1

    def render(self, db_stats: dict[str, Any] | None = None, workers: int = 0) -> str:
        lines = ["# HELP swarm_http_requests_total HTTP requests by method, route and status.",
                 "# TYPE swarm_http_requests_total counter"]
        with self.lock:
            for (m, r, s), n in sorted(self.requests.items()):
                lines.append(f'swarm_http_requests_total{{method="{m}",route="{r}",status="{s}"}} {n}')
            lines += ["# HELP swarm_http_request_duration_seconds Request latency.",
                      "# TYPE swarm_http_request_duration_seconds histogram"]
            for (m, r), h in sorted(self.latency.items()):
                lab = f'method="{m}",route="{r}"'
                for i, b in enumerate(BUCKETS):
                    lines.append(f'swarm_http_request_duration_seconds_bucket{{{lab},le="{b}"}} {int(h[i])}')
                lines.append(f'swarm_http_request_duration_seconds_bucket{{{lab},le="+Inf"}} {int(h[-1])}')
                lines.append(f"swarm_http_request_duration_seconds_sum{{{lab}}} {h[-2]:.6f}")
                lines.append(f"swarm_http_request_duration_seconds_count{{{lab}}} {int(h[-1])}")
        if db_stats is not None:
            lines += ["# HELP swarm_jobs Jobs in the database by kind and status.", "# TYPE swarm_jobs gauge"]
            for row in db_stats["by_kind_status"]:
                lines.append(f'swarm_jobs{{kind="{row["kind"]}",status="{row["status"]}"}} {row["n"]}')
            lines += ["# HELP swarm_job_spend_usd Dollars spent by finished jobs, by kind.",
                      "# TYPE swarm_job_spend_usd gauge"]
            spend: dict[str, float] = defaultdict(float)
            for row in db_stats["by_kind_status"]:
                spend[row["kind"]] += row["usd"]
            for k, v in sorted(spend.items()):
                lines.append(f'swarm_job_spend_usd{{kind="{k}"}} {v:.6f}')
            lines += ["# HELP swarm_queue_oldest_age_seconds Age of the oldest queued job.",
                      "# TYPE swarm_queue_oldest_age_seconds gauge",
                      f"swarm_queue_oldest_age_seconds {db_stats['oldest_queued_age_s']}"]
        lines += ["# HELP swarm_embedded_workers Workers running inside this API process.",
                  "# TYPE swarm_embedded_workers gauge", f"swarm_embedded_workers {workers}",
                  "# HELP swarm_process_uptime_seconds Seconds since this process started.",
                  "# TYPE swarm_process_uptime_seconds gauge",
                  f"swarm_process_uptime_seconds {time.time() - self.started:.0f}"]
        return "\n".join(lines) + "\n"
