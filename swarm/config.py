"""Configuration: defaults < config file < environment < command-line flags.

Config file (TOML), first found of: ./swarm.toml, $SWARM_CONFIG, ~/.config/swarm/config.toml

    [research]
    preset = "standard"          # quick | standard | deep
    model = "claude-opus-5-5"
    max_usd = 2.0
    search = "tinyfish"          # tinyfish (free) | anthropic (server-side, $10/1k searches)
    exclude_domains = ["pinterest.com", "quora.com"]
    language = "en"
    # provider = "openai"        # any OpenAI-compatible API (key in OPENAI_API_KEY)
    # base_url = "https://your-provider/v1"
    # model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"

    [prices]                     # USD per million tokens, for models not built in
    # "deepseek-ai/DeepSeek-V4.1-Flash-1017" = [0.10, 0.40]   # [in, out] or [in, out, cached]

    [server]
    host = "127.0.0.1"
    port = 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .errors import ConfigError

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 uses the tomli backport
    import tomli as _toml


def config_paths() -> list[Path]:
    paths = [Path("swarm.toml")]
    if os.environ.get("SWARM_CONFIG"):
        paths.append(Path(os.environ["SWARM_CONFIG"]))
    paths.append(Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "swarm" / "config.toml")
    return paths


def load_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path | None]:
    """Return (config, path it came from). Empty config when no file exists."""
    candidates = [Path(path)] if path else config_paths()
    for p in candidates:
        if p.is_file():
            try:
                return _toml.loads(p.read_text()), p
            except Exception as e:
                raise ConfigError(f"{p}: invalid TOML: {e}") from e
    if path:
        raise ConfigError(f"config file not found: {path}")
    return {}, None


def cache_dir() -> Path:
    if os.environ.get("SWARM_CACHE_DIR"):
        d = Path(os.environ["SWARM_CACHE_DIR"])
    else:
        d = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "swarm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, env, cast in [("model", "SWARM_MODEL", str), ("max_usd", "SWARM_MAX_USD", float),
                           ("search", "SWARM_SEARCH", str), ("preset", "SWARM_PRESET", str),
                           ("provider", "SWARM_PROVIDER", str), ("price", "SWARM_PRICE", str)]:
        if os.environ.get(env):
            try:
                out[key] = cast(os.environ[env])
            except ValueError as e:
                raise ConfigError(f"{env}={os.environ[env]!r} is not a valid {cast.__name__}") from e
    return out
