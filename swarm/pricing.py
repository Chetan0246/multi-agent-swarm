"""Dollar costs: prices, actual cost of a run, and a pre-run estimate for deep research.

Prices are Anthropic first-party API list prices, read from
https://platform.claude.com/docs/en/about-claude/pricing on 2026-11-21. Prices change, so
check that page before relying on a figure; override with `register_price`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .backends import CallUsage


@dataclass(frozen=True)
class Price:
    input: float  # USD per million input tokens
    output: float  # USD per million output tokens
    cache_read_mult: float = 0.1  # multiple of input price
    cache_write_mult: float = 1.25  # 5-minute TTL write


PRICES: dict[str, Price] = {
    "claude-fable-5-2": Price(8.00, 40.00, 0.025),
    "claude-opus-5-5": Price(4.00, 20.00, 0.05),
    "claude-opus-5": Price(5.00, 25.00),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.05),
    "claude-sonnet-5": Price(2.00, 10.00),
    "claude-haiku-5-5": Price(0.10, 0.50),  # prompts up to 100k tokens
}
WEB_SEARCH_USD_PER_1K = 10.0  # failed searches are not billed; web fetch has no extra fee


def register_price(model: str, price: Price) -> None:
    PRICES[model] = price


def price_of(model: str) -> Price:
    if model not in PRICES:
        raise KeyError(f"no price for {model!r}; add one with swarm.pricing.register_price")
    return PRICES[model]


def usage_usd(model: str, u: CallUsage, searches: int = 0) -> float:
    p = price_of(model)
    tokens = (u.input_tokens * p.input
              + u.cache_read_tokens * p.input * p.cache_read_mult
              + u.cache_write_tokens * p.input * p.cache_write_mult
              + u.output_tokens * p.output) / 1e6
    return u.price_mult * tokens + searches * WEB_SEARCH_USD_PER_1K / 1000


def usd_to_billed_tokens(model: str, usd: float) -> float:
    """Convert dollars to the engine's budget unit (input-token equivalents).

    The engine's billed-equivalent weights output 5x and cache reads 0.1x. For models whose
    output costs 5x input that matches list price, except cache reads, which it
    over-counts on the newest models, so the budget errs on the safe side.
    """
    return usd / price_of(model).input * 1e6


@dataclass
class Estimate:
    low_usd: float
    high_usd: float
    low_s: float
    high_s: float

    def __str__(self) -> str:
        return (f"~${self.low_usd:.2f}-${self.high_usd:.2f}, "
                f"~{self.low_s / 60:.0f}-{max(1, round(self.high_s / 60))} min")


# Per worker call, by effort: (input tokens incl. search results, output tokens). Calibrated
# as a range; replace with your own run stats (`swarm research` prints actuals).
_WORKER_TOKENS = {"low": ((8_000, 20_000), (600, 1_500)),
                  "medium": ((10_000, 28_000), (1_200, 3_000)),
                  "high": ((12_000, 35_000), (2_000, 5_000))}


def estimate_research(model: str, n_subquestions: int, max_searches: int,
                      efforts: tuple[str, ...] = ("low", "medium", "high"),
                      escalation_rate: tuple[float, float] = (0.1, 0.5),
                      free_search: bool = False) -> Estimate:
    """Cost and time range before any call is made (planner and synthesis included).

    `free_search`: client-side tools on a free provider (TinyFish): no per-search fee, but the
    tool loop re-reads its growing context each turn (mostly as cheap cache reads).
    """
    p = price_of(model)
    fee = 0.0 if free_search else WEB_SEARCH_USD_PER_1K / 1000
    loop_factor = 1.5 if free_search else 1.0

    def call(effort: str, hi: bool) -> float:
        (ilo, ihi), (olo, ohi) = _WORKER_TOKENS.get(effort, _WORKER_TOKENS["medium"])
        i, o = (ihi, ohi) if hi else (ilo, olo)
        s = max_searches if hi else max(1, max_searches // 2)
        return (i * loop_factor * p.input + o * p.output) / 1e6 + s * fee

    first = efforts[0]
    second = efforts[1] if len(efforts) > 1 else first
    plan_synth = ((3_000 * p.input + 600 * p.output) + (12_000 * p.input + 2_500 * p.output)) / 1e6
    lo = n_subquestions * (call(first, False) + escalation_rate[0] * call(second, False)) + plan_synth
    hi = n_subquestions * (call(first, True) + escalation_rate[1] * call(second, True)) + 2 * plan_synth
    # Workers run in parallel: wall time is roughly plan + one or two worker rounds + synthesis.
    return Estimate(lo, hi, 60.0, 360.0)
