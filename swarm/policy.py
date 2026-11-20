"""Adaptive policy: learn, online, which (effort rung, batch size) is cheapest per success.

For every difficulty class d the policy keeps:

* a Bayesian logistic model  P(success | d, rung, b) = sigmoid(w0 + w1*(b-1)/8)
  with a Laplace (Gaussian) posterior over w, updated online; one posterior per
  (d, rung) shares evidence across all batch sizes, so it learns from few calls
* an online linear cost model  cost(call) = overhead + per_task * n   per (d, rung)

and picks the arm that minimises sampled  expected tokens per solved task:

        cost_per_success(r, b) = (overhead_r/b + per_task_r) / p(r, b)

So the swarm discovers, for example, that easy tasks are cheapest in batches of 16
at `low`, while hard tasks are cheaper going straight to `medium` in batches of 4.
The state can be saved and reused across runs, so later runs start already tuned.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .planner import Rung

ARMS = (1, 2, 4, 8, 12, 16, 24)
_EFFORT_TOKENS = {"low": 1.0, "medium": 2.2, "high": 4.0, "xhigh": 6.0, "max": 9.0}


@dataclass
class _LinFit:
    """Running least squares for y = a + c*x, seeded with prior pseudo-points."""

    n: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    sxy: float = 0.0

    def add(self, x: float, y: float, w: float = 1.0) -> None:
        self.n += w
        self.sx += w * x
        self.sy += w * y
        self.sxx += w * x * x
        self.sxy += w * x * y

    def coef(self) -> tuple[float, float]:
        den = self.n * self.sxx - self.sx**2
        if abs(den) < 1e-9:  # all observations at one batch size: no slope information
            c = self.sy / max(1e-9, self.sx)
            return 0.0, c
        c = (self.n * self.sxy - self.sx * self.sy) / den
        a = (self.sy - c * self.sx) / self.n
        return max(0.0, a), max(1.0, c)


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


@dataclass
class _BayesLogit:
    """2-feature Bayesian logistic regression with a full 2x2 Laplace posterior.

    Features x = [1, (b-1)/8]. Observations are binomial (n tasks, k successes)
    at one batch size. Update = MAP Newton steps, then precision += Fisher info.
    """

    m0: float
    m1: float
    p00: float = 1.0  # precision matrix [[p00, p01], [p01, p11]]
    p01: float = 0.0
    p11: float = 1.0

    @staticmethod
    def feat(b: int) -> float:
        return (b - 1) / 8.0

    def mean_p(self, b: int) -> float:
        return _sig(self.m0 + self.m1 * self.feat(b))

    def sample(self, rng: random.Random, temperature: float = 1.0) -> tuple[float, float]:
        # covariance = inverse precision; sample via Cholesky of the 2x2 covariance
        det = self.p00 * self.p11 - self.p01 ** 2
        c00, c01, c11 = self.p11 / det, -self.p01 / det, self.p00 / det
        l00 = math.sqrt(max(c00, 1e-12))
        l10 = c01 / l00
        l11 = math.sqrt(max(c11 - l10 ** 2, 1e-12))
        z0, z1 = rng.gauss(0, temperature), rng.gauss(0, temperature)
        return self.m0 + l00 * z0, self.m1 + l10 * z0 + l11 * z1

    def update(self, b: int, n: int, k: int) -> None:
        if n <= 0:
            return
        x1 = self.feat(b)
        w0, w1 = self.m0, self.m1
        for _ in range(8):  # Newton on: 0.5 (w-m)'P(w-m) - loglik(w)
            p = _sig(w0 + w1 * x1)
            r = n * p - k  # d(-loglik)/dz
            g0 = self.p00 * (w0 - self.m0) + self.p01 * (w1 - self.m1) + r
            g1 = self.p01 * (w0 - self.m0) + self.p11 * (w1 - self.m1) + r * x1
            h = n * p * (1 - p)
            h00, h01, h11 = self.p00 + h, self.p01 + h * x1, self.p11 + h * x1 * x1
            det = h00 * h11 - h01 ** 2
            d0 = (h11 * g0 - h01 * g1) / det
            d1 = (h00 * g1 - h01 * g0) / det
            w0, w1 = w0 - d0, w1 - d1
            if abs(d0) + abs(d1) < 1e-6:
                break
        p = _sig(w0 + w1 * x1)
        h = n * p * (1 - p)
        self.m0, self.m1 = w0, w1
        self.p00 += h
        self.p01 += h * x1
        self.p11 += h * x1 * x1


class AdaptivePolicy:
    def __init__(
        self,
        ladder: tuple[Rung, ...],
        max_batch: int = 24,
        seed: int = 0,
        explore: bool = True,
        prior_overhead: float = 250.0,
        prior_per_task: float = 100.0,
        prior_precision: float = 1.5,
        temperature: float = 0.5,
    ):
        self.ladder = ladder
        self.arms = tuple(b for b in ARMS if b <= max_batch) or (1,)
        self.rng = random.Random(seed)
        self.explore = explore
        self.prior_overhead = prior_overhead
        self.prior_per_task = prior_per_task
        self.prior_precision = prior_precision
        self.temperature = temperature  # <1 narrows Thompson draws: cheaper exploration
        self.model: dict[str, _BayesLogit] = {}
        self.cost: dict[str, _LinFit] = {}

    # ------------------------------------------------------------ priors
    def _prior_p(self, d: int, r: int, b: int) -> float:
        # Weakly informative: harder → lower, more effort → higher, bigger batch → lower.
        p = 0.92 - 0.12 * (d - 1) + 0.07 * r * d / 2 - 0.004 * (b - 1) * d
        return min(0.97, max(0.2, p))

    def _model(self, d: int, r: int) -> _BayesLogit:
        k = f"{d}|{r}"
        if k not in self.model:
            w0 = _logit(self._prior_p(d, r, 1))
            w1 = (_logit(self._prior_p(d, r, 9)) - w0)  # feature is 1.0 at b = 9
            pp = self.prior_precision
            self.model[k] = _BayesLogit(w0, w1, pp, 0.0, pp)
        return self.model[k]

    def _fit(self, d: int, r: int) -> _LinFit:
        k = f"{d}|{r}"
        if k not in self.cost:
            f = _LinFit()
            # Output (thinking) tokens grow roughly with these per-effort multipliers.
            mult = _EFFORT_TOKENS.get(self.ladder[r].effort, 3.0)
            per = self.prior_per_task + 2 * self.prior_per_task * d * (mult - 1)
            for x in (1, 8):  # two prior pseudo-points pin intercept and slope
                f.add(x, self.prior_overhead + per * x, w=1.0)
            self.cost[k] = f
        return self.cost[k]

    # ------------------------------------------------------------ API
    def per_task_cost(self, d: int, r: int, b: int) -> float:
        a, c = self._fit(d, r).coef()
        return a / b + c

    def choose(
        self, d: int, min_rung: int, available: int, allowed: set[int] | None = None
    ) -> tuple[int, int]:
        """Return (rung, batch_size) for the next batch of difficulty-d tasks.

        Dynamic programme over the ladder, from the top rung down:

            Q(r, b) = c(r, b) + (1 - p(r, b)) * V(r + 1)      (a failure escalates)
            V(r)    = min( min_b Q(r, b),  V(r + 1) )         (or skip this rung)
            V(top)  = min_b c(top, b) / p(top, b)              (retry in place)

        One posterior draw per (d, rung) gives a coherent success-vs-batch curve (Thompson).
        """
        L = len(self.ladder)
        usable = [r for r in range(min_rung, L) if allowed is None or r in allowed]
        if not usable:
            return min_rung, 1
        q: dict[tuple[int, int], float] = {}
        v_next = None
        for r in reversed(range(min_rung, L)):
            mdl = self._model(d, r)
            w0, w1 = mdl.sample(self.rng, self.temperature) if self.explore else (mdl.m0, mdl.m1)
            for b in self.arms:
                p = max(_sig(w0 + w1 * mdl.feat(b)), 0.02)
                c = self.per_task_cost(d, r, min(b, max(1, available)))  # partial batch: less discount
                q[(r, b)] = c / p if v_next is None else c + (1 - p) * v_next
            here = min(q[(r, b)] for b in self.arms)
            v_next = here if v_next is None else min(here, v_next)
        return min(((r, b) for r in usable for b in self.arms), key=lambda k: q[k])

    def observe(self, d: int, r: int, b: int, n: int, successes: int, billed: float) -> None:
        # Learn from the batch size actually sent (a partial batch is smaller than the arm).
        self._model(d, r).update(n, n, successes)
        if n > 0 and billed > 0:
            self._fit(d, r).add(n, billed)

    def expected_success(self, d: int, r: int, b: int) -> float:
        return self._model(d, r).mean_p(b)

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "model": {k: [m.m0, m.m1, m.p00, m.p01, m.p11] for k, m in self.model.items()},
            "cost": {k: [f.n, f.sx, f.sy, f.sxx, f.sxy] for k, f in self.cost.items()},
        }))

    def load(self, path: str | Path) -> AdaptivePolicy:
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            self.model = {k: _BayesLogit(*v) for k, v in data.get("model", {}).items()}
            self.cost = {k: _LinFit(*v) for k, v in data.get("cost", {}).items()}
        return self


class StaticPolicy:
    """Fixed batching (v1 behaviour): start at rung 0, halve batch on each retry."""

    def __init__(self, ladder: tuple[Rung, ...], max_batch: int = 12):
        self.ladder = ladder
        self.max_batch = max_batch

    def choose(self, d, min_rung, available, allowed=None):
        r = min_rung
        if allowed is not None:
            r = next((x for x in range(min_rung, len(self.ladder)) if x in allowed), min_rung)
        return r, max(1, (self.max_batch // max(1, d)) >> min_rung)

    def observe(self, *a, **k):
        pass

    def per_task_cost(self, d, r, b):
        mult = _EFFORT_TOKENS.get(self.ladder[r].effort, 3.0)
        return 250 / b + 100 + 200 * d * (mult - 1)

    def save(self, path):
        pass
