# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Inference engine registry.

An instance's engine is `config["engine"]`; a missing or empty value means
llama.cpp, so every instance, preset and container label written before
engines existed keeps meaning exactly what it did. llama.cpp configs are
deliberately NOT stamped with "engine": "llamacpp" - the key would land in the
container's llamaman.config label and change the llama.cpp container spec.
"""

from core.engines.base import Engine
from core.engines.llamacpp import LlamaCppEngine

DEFAULT_ENGINE = "llamacpp"

ENGINES: dict[str, Engine] = {
    LlamaCppEngine.name: LlamaCppEngine(),
}


def engine_name(config: dict | None, model_path: str | None = None) -> str:
    """The engine an instance config (or preset) runs on."""
    name = ((config or {}).get("engine") or "").strip().lower()
    return name if name in ENGINES else DEFAULT_ENGINE


def get_engine(config_or_name: dict | str | None = None, model_path: str | None = None) -> Engine:
    if isinstance(config_or_name, str):
        return ENGINES.get(config_or_name.strip().lower(), ENGINES[DEFAULT_ENGINE])
    return ENGINES[engine_name(config_or_name, model_path)]


def describe_engines(vendor: str | None) -> list[dict]:
    return [e.describe(vendor) for e in ENGINES.values()]


def parse_engine(body: dict | None) -> tuple[str, str | None]:
    """Validate an API body's `engine` field. Returns (engine_name, error).

    Missing / empty / null means llama.cpp (every client written before
    engines existed). An unknown name is an error rather than a silent
    fallback, so a typo can't launch the wrong backend."""
    raw = (body or {}).get("engine")
    if raw in (None, ""):
        return DEFAULT_ENGINE, None
    if not isinstance(raw, str):
        return DEFAULT_ENGINE, "engine must be a string"
    name = raw.strip().lower()
    if not name:
        return DEFAULT_ENGINE, None
    if name not in ENGINES:
        return DEFAULT_ENGINE, f"unknown engine '{raw}' (available: {', '.join(sorted(ENGINES))})"
    return name, None
