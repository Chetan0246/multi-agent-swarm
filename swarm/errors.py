"""User-facing errors: what went wrong and what to do about it."""
from __future__ import annotations


class SwarmError(Exception):
    """Base class. `hint` tells the user how to fix it."""

    hint: str = ""
    # Fatal: retrying any task can't help (bad key, quota gone). The engine stops the run.
    fatal: bool = False

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        if hint is not None:
            self.hint = hint

    def __str__(self) -> str:
        msg = super().__str__()
        return f"{msg}\n  hint: {self.hint}" if self.hint else msg


class ConfigError(SwarmError):
    hint = "check swarm.toml or your environment variables (swarm doctor shows the resolved config)"


class UnknownModel(ConfigError, KeyError):
    hint = "use a model listed in swarm/pricing.py, or add a price with swarm.pricing.register_price"


class MissingKey(SwarmError):
    fatal = True


class SearchAuthError(SwarmError):
    fatal = True
    hint = "check TINYFISH_API_KEY; create a free key at https://agent.tinyfish.ai/api-keys"


class SearchQuotaExhausted(SwarmError):
    fatal = True
    hint = ("TinyFish's free daily allowance is used up (it resets at 00:00 UTC). Wait, use "
            "--search anthropic (paid), or rely on the local cache for repeated questions")


class SearchUnavailable(SwarmError):
    hint = "the search service is temporarily unavailable; try again in a minute"


def missing_anthropic_key() -> MissingKey:
    return MissingKey(
        "no Anthropic credentials found",
        "set ANTHROPIC_API_KEY (https://platform.claude.com/settings/keys) or run `ant auth login`; "
        "add --mock to try the pipeline offline",
    )


def missing_tinyfish_key() -> MissingKey:
    return MissingKey(
        "TINYFISH_API_KEY is not set",
        "create a free key at https://agent.tinyfish.ai/api-keys (search and fetch are free within "
        "a daily allowance), or use --search anthropic",
    )
