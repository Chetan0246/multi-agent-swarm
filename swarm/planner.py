"""Static configuration and agent-count sizing (pure functions, no I/O)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rung:
    """One step of the escalation ladder."""

    model: str
    effort: str


DEFAULT_LADDER = (
    # Same model, rising effort: one prompt-cache namespace and no cross-model drift.
    Rung("claude-opus-5-5", "low"),
    Rung("claude-opus-5-5", "medium"),
    Rung("claude-opus-5-5", "high"),
)


@dataclass
class PlanConfig:
    ladder: tuple[Rung, ...] = field(default=DEFAULT_LADDER)
    # batching
    max_batch: int = 24
    max_batch_input_tokens: int = 6000
    adaptive: bool = True  # learn batch size / start rung online (else static v1 rules)
    static_batch: int = 12  # batch size for difficulty 1 when adaptive=False
    # pool sizing
    max_agents: int = 8
    knee_slack: float = 0.15
    bulk_max_agents: int = 256  # for the Batches API backend
    # verification
    # Tasks with no deterministic checker: "agreement" accepts an answer once two
    # independent attempts agree (self-reported confidence is a weak error signal);
    # "confidence" accepts when the model's confidence >= min_confidence.
    unchecked_mode: str = "agreement"
    min_confidence: float = 0.6
    top_rung_retries: int = 2  # extra attempts at the last rung before giving up
    # reliability
    hedging: bool = True
    breaker_threshold: int = 5
    breaker_cooldown: float = 0.5
    max_throttle_retries: int = 8
    max_transient_retries: int = 6
    token_budget: float | None = None  # billed-equivalent tokens for the whole run
    seed: int = 0


def plan_agents(n_batches: int, max_agents: int, knee_slack: float = 0.15,
                expected_retry_frac: float = 0.1) -> int:
    """Smallest agent count whose makespan is within `knee_slack` of the best possible.

    Makespan with k agents ≈ ceil(B / k) rounds. Past the knee, extra agents add
    rate-limit pressure and overhead but barely cut wall time.
    """
    if n_batches <= 0:
        return 1
    b = math.ceil(n_batches * (1 + expected_retry_frac))
    bound = math.ceil(b / max_agents) * (1 + knee_slack)
    for k in range(1, max_agents + 1):
        if math.ceil(b / k) <= bound:
            return k
    return max_agents
