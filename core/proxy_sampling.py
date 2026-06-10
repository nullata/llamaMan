# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

DEFAULT_PROXY_SAMPLING_TEMPERATURE = 0.8
MAX_PROXY_SAMPLING_TEMPERATURE = 2.0
DEFAULT_PROXY_SAMPLING_TOP_K = 40
DEFAULT_PROXY_SAMPLING_TOP_P = 0.95
DEFAULT_PROXY_SAMPLING_PRESENCE_PENALTY = 0.0
MIN_PROXY_SAMPLING_PRESENCE_PENALTY = -2.0
MAX_PROXY_SAMPLING_PRESENCE_PENALTY = 2.0
DEFAULT_PROXY_SAMPLING_REPEAT_PENALTY = 0.0
MIN_PROXY_SAMPLING_REPEAT_PENALTY = 0.0
MAX_PROXY_SAMPLING_REPEAT_PENALTY = 2.0

PROXY_SAMPLING_OVERRIDE_KEYS = (
    "proxy_sampling_override_enabled",
    "proxy_sampling_temperature",
    "proxy_sampling_top_k",
    "proxy_sampling_top_p",
    "proxy_sampling_presence_penalty",
    "proxy_sampling_repeat_penalty",
)

PROXY_SAMPLING_PATHS = frozenset({
    "/v1/chat/completions",
    "/v1/completions",
    "/completion",
    "/chat/completions",
})


def parse_proxy_sampling_config(body: dict) -> tuple[dict, str | None]:
    enabled = bool(body.get("proxy_sampling_override_enabled", False))

    try:
        temperature = float(body.get("proxy_sampling_temperature", DEFAULT_PROXY_SAMPLING_TEMPERATURE))
    except (TypeError, ValueError):
        return {}, "proxy_sampling_temperature must be a number"
    if temperature < 0 or temperature > MAX_PROXY_SAMPLING_TEMPERATURE:
        return {}, f"proxy_sampling_temperature must be >= 0 and <= {MAX_PROXY_SAMPLING_TEMPERATURE:g}"

    try:
        top_k = int(body.get("proxy_sampling_top_k", DEFAULT_PROXY_SAMPLING_TOP_K))
    except (TypeError, ValueError):
        return {}, "proxy_sampling_top_k must be an integer"
    if top_k < 0:
        return {}, "proxy_sampling_top_k must be >= 0"

    try:
        top_p = float(body.get("proxy_sampling_top_p", DEFAULT_PROXY_SAMPLING_TOP_P))
    except (TypeError, ValueError):
        return {}, "proxy_sampling_top_p must be a number"
    if top_p <= 0 or top_p > 1:
        return {}, "proxy_sampling_top_p must be > 0 and <= 1"

    try:
        presence_penalty = float(
            body.get(
                "proxy_sampling_presence_penalty",
                DEFAULT_PROXY_SAMPLING_PRESENCE_PENALTY,
            )
        )
    except (TypeError, ValueError):
        return {}, "proxy_sampling_presence_penalty must be a number"
    if (
        presence_penalty < MIN_PROXY_SAMPLING_PRESENCE_PENALTY
        or presence_penalty > MAX_PROXY_SAMPLING_PRESENCE_PENALTY
    ):
        return (
            {},
            "proxy_sampling_presence_penalty must be >= "
            f"{MIN_PROXY_SAMPLING_PRESENCE_PENALTY:g} and <= {MAX_PROXY_SAMPLING_PRESENCE_PENALTY:g}",
        )

    try:
        repeat_penalty = float(
            body.get(
                "proxy_sampling_repeat_penalty",
                DEFAULT_PROXY_SAMPLING_REPEAT_PENALTY,
            )
        )
    except (TypeError, ValueError):
        return {}, "proxy_sampling_repeat_penalty must be a number"
    if (
        repeat_penalty < MIN_PROXY_SAMPLING_REPEAT_PENALTY
        or repeat_penalty > MAX_PROXY_SAMPLING_REPEAT_PENALTY
    ):
        return (
            {},
            "proxy_sampling_repeat_penalty must be >= "
            f"{MIN_PROXY_SAMPLING_REPEAT_PENALTY:g} and <= {MAX_PROXY_SAMPLING_REPEAT_PENALTY:g}",
        )

    return {
        "proxy_sampling_override_enabled": enabled,
        "proxy_sampling_temperature": temperature,
        "proxy_sampling_top_k": top_k,
        "proxy_sampling_top_p": top_p,
        "proxy_sampling_presence_penalty": presence_penalty,
        "proxy_sampling_repeat_penalty": repeat_penalty,
    }, None


def proxy_sampling_overrides_enabled(config: dict | None) -> bool:
    return bool((config or {}).get("proxy_sampling_override_enabled", False))


def apply_proxy_sampling_overrides(body: dict, config: dict | None) -> dict:
    if not proxy_sampling_overrides_enabled(config):
        return body

    updated = dict(body)
    updated["temperature"] = float(config.get("proxy_sampling_temperature", DEFAULT_PROXY_SAMPLING_TEMPERATURE))
    updated["top_k"] = int(config.get("proxy_sampling_top_k", DEFAULT_PROXY_SAMPLING_TOP_K))
    updated["top_p"] = float(config.get("proxy_sampling_top_p", DEFAULT_PROXY_SAMPLING_TOP_P))
    updated["presence_penalty"] = float(
        config.get(
            "proxy_sampling_presence_penalty",
            DEFAULT_PROXY_SAMPLING_PRESENCE_PENALTY,
        )
    )
    repeat_penalty = float(
        config.get(
            "proxy_sampling_repeat_penalty",
            DEFAULT_PROXY_SAMPLING_REPEAT_PENALTY,
        )
    )
    if repeat_penalty > 0:
        updated["repeat_penalty"] = repeat_penalty
    return updated
