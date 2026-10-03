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
from core.engines.strata import StrataEngine

DEFAULT_ENGINE = "llamacpp"

ENGINES: dict[str, Engine] = {
    LlamaCppEngine.name: LlamaCppEngine(),
    StrataEngine.name: StrataEngine(),
}


def engine_for_path(model_path: str | None) -> str | None:
    """The engine whose virtual model this path is, or None for a file."""
    if not model_path:
        return None
    for name, eng in ENGINES.items():
        if eng.owns_model_path(model_path):
            return name
    return None


def engine_name(config: dict | None, model_path: str | None = None) -> str:
    """The engine an instance config (or preset) runs on: the recorded one,
    else the engine owning the model path's virtual model, else llama.cpp."""
    name = ((config or {}).get("engine") or "").strip().lower()
    if name in ENGINES:
        return name
    return engine_for_path(model_path) or DEFAULT_ENGINE


def get_engine(config_or_name: dict | str | None = None, model_path: str | None = None) -> Engine:
    if isinstance(config_or_name, str):
        return ENGINES.get(config_or_name.strip().lower(), ENGINES[DEFAULT_ENGINE])
    return ENGINES[engine_name(config_or_name, model_path)]


def describe_engines(vendor: str | None) -> list[dict]:
    return [e.describe(vendor) for e in ENGINES.values()]


def available_virtual_models(vendor: str | None) -> list[dict]:
    """Virtual model entries of every engine this node can launch."""
    out = []
    for eng in ENGINES.values():
        if eng.capabilities.get("virtual_models") and eng.availability(vendor)[0]:
            out.extend(eng.virtual_models())
    return out


def parse_engine(body: dict | None, model_path: str | None = None) -> tuple[str, str | None]:
    """Validate an API body's `engine` field. Returns (engine_name, error).

    Missing / empty / null means the engine owning model_path's virtual model,
    else llama.cpp (every client written before engines existed). An unknown
    name is an error rather than a silent fallback, so a typo can't launch the
    wrong backend; so is an engine that contradicts the model path."""
    owner = engine_for_path(model_path)
    raw = (body or {}).get("engine")
    if raw in (None, ""):
        return owner or DEFAULT_ENGINE, None
    if not isinstance(raw, str):
        return DEFAULT_ENGINE, "engine must be a string"
    name = raw.strip().lower()
    if not name:
        return owner or DEFAULT_ENGINE, None
    if name not in ENGINES:
        return DEFAULT_ENGINE, f"unknown engine '{raw}' (available: {', '.join(sorted(ENGINES))})"
    if model_path and (owner or DEFAULT_ENGINE) != name:
        if owner:
            return name, f"'{model_path}' is a {ENGINES[owner].label} model, not {ENGINES[name].label}"
        if ENGINES[name].capabilities.get("virtual_models"):
            return name, f"{ENGINES[name].label} cannot run '{model_path}'; pick one of its models"
    return name, None


def validate_launch(body: dict, model_path: str, vendor: str | None) -> tuple[str, dict, str | None]:
    """Everything engine-related an API launch / preset body must pass:
    a known engine that matches the model, available on this node, no fields
    the engine would ignore, valid engine options.
    Returns (engine_name, engine_options, error)."""
    name, err = parse_engine(body, model_path)
    if err:
        return name, {}, err
    eng = ENGINES[name]
    ok, reason = eng.availability(vendor)
    if not ok:
        return name, {}, reason
    err = eng.reject_unsupported_fields(body)
    if err:
        return name, {}, err
    options, err = eng.parse_options(body, model_path)
    return name, options, err


def load_timeout_for(inst_or_config: dict | None, model_path: str | None = None) -> int:
    """Seconds to wait for an instance (or a config) to become ready."""
    d = inst_or_config or {}
    config = d.get("config", d)
    return get_engine(config, model_path or d.get("model_path")).load_timeout()
