# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""DRY (Don't Repeat Yourself) sampler config parsing and emission helpers.

llama.cpp's DRY sampler penalizes tokens that would extend a recently-seen
n-gram, so it's a soft, sampling-time anti-repeat that complements the
post-hoc loop detector in core/loop_detect.py. Both are configured per model
preset; both are off by default (dry_multiplier=0 disables DRY entirely at
llama.cpp's level, matching its own default).

Flag reference from common/arg.cpp:
  --dry-multiplier  N   (default 0.0, 0.0 = disabled)
  --dry-base        N   (default 1.75; llama.cpp silently ignores values < 1.0)
  --dry-allowed-length N (default 2)
  --dry-penalty-last-n N (default -1 internally; the CLI THROWS on negative
                          user input, so we only accept 0 or positive here)
"""

DEFAULT_DRY_MULTIPLIER = 0.0
DEFAULT_DRY_BASE = 1.75
DEFAULT_DRY_ALLOWED_LENGTH = 2
# llama.cpp's internal default is -1 ("use ctx size") but the CLI rejects
# negative input. Blank in our UI = flag omitted = llama.cpp's own default.
DEFAULT_DRY_PENALTY_LAST_N = None

MIN_DRY_BASE = 1.0  # llama.cpp silently clamps below this
MAX_DRY_MULTIPLIER = 10.0  # sanity cap - any real value is well below this

DRY_SAMPLER_KEYS = (
    "dry_enabled",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
)


def parse_dry_config(body: dict) -> tuple[dict, str | None]:
    """Normalize DRY sampler fields out of a launch/preset body.

    Returns (config_dict, error_message_or_None). Follows the same
    boundary-validation contract as parse_proxy_sampling_config /
    parse_spec_config: reject invalid values with a message the API caller
    surfaces as a 400, never silently drop or coerce out-of-range values.
    """
    enabled = bool(body.get("dry_enabled", False))

    try:
        multiplier = float(body.get("dry_multiplier", DEFAULT_DRY_MULTIPLIER))
    except (TypeError, ValueError):
        return {}, "dry_multiplier must be a number"
    if multiplier < 0 or multiplier > MAX_DRY_MULTIPLIER:
        return {}, f"dry_multiplier must be >= 0 and <= {MAX_DRY_MULTIPLIER:g}"

    try:
        base = float(body.get("dry_base", DEFAULT_DRY_BASE))
    except (TypeError, ValueError):
        return {}, "dry_base must be a number"
    if base < MIN_DRY_BASE:
        # llama.cpp silently ignores < 1.0. Reject at the boundary so the
        # user sees "this doesn't do anything" instead of a silent no-op.
        return {}, f"dry_base must be >= {MIN_DRY_BASE:g}"

    try:
        allowed_length = int(body.get("dry_allowed_length", DEFAULT_DRY_ALLOWED_LENGTH))
    except (TypeError, ValueError):
        return {}, "dry_allowed_length must be an integer"
    if allowed_length < 0:
        return {}, "dry_allowed_length must be >= 0"

    # penalty_last_n: None / omitted means "leave llama.cpp's own default in
    # place" (build_llama_cmd omits the flag entirely). 0 explicitly disables
    # the penalty history per llama.cpp docs ("0 = disable"). Negative values
    # are rejected because llama.cpp's arg parser throws on them.
    raw = body.get("dry_penalty_last_n", DEFAULT_DRY_PENALTY_LAST_N)
    if raw in (None, ""):
        penalty_last_n = None
    else:
        try:
            penalty_last_n = int(raw)
        except (TypeError, ValueError):
            return {}, "dry_penalty_last_n must be an integer"
        if penalty_last_n < 0:
            return {}, "dry_penalty_last_n must be >= 0 (llama.cpp rejects negative values)"

    return {
        "dry_enabled": enabled,
        "dry_multiplier": multiplier,
        "dry_base": base,
        "dry_allowed_length": allowed_length,
        "dry_penalty_last_n": penalty_last_n,
    }, None


def dry_enabled(config: dict | None) -> bool:
    """True when DRY should actually be emitted to llama-server.

    Uses BOTH the explicit toggle and the multiplier > 0 check: the toggle is
    the user's intent, but llama.cpp treats multiplier=0 as disabled anyway,
    so emitting a bunch of DRY flags with multiplier=0 would be noise.
    """
    if not config:
        return False
    if not config.get("dry_enabled", False):
        return False
    try:
        return float(config.get("dry_multiplier", 0.0)) > 0
    except (TypeError, ValueError):
        return False
