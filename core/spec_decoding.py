# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

DEFAULT_SPEC_TYPE = "draft-mtp"

# Values accepted by llama-server's --spec-type. All of these are the
# "draft-model" family - a small model drafts tokens for the target to
# verify. They all take a drafter via -md; they differ in whether the
# drafter is mandatory and in what kind of checkpoint the drafter has to
# be. draft-mtp is unique in accepting no drafter and falling back to
# heads built into the main model (e.g. Gemma 4's built-in MTP heads).
# The n-gram family (ngram-simple / ngram-map-k / ngram-map-k4v /
# ngram-mod / ngram-cache) is intentionally NOT exposed here - those
# don't need a drafter model at all, so surfacing them alongside these
# would confuse the Draft Model field's meaning; that's a separate UI
# shape and lives outside this module.
SPEC_TYPES = (
    "draft-simple",
    DEFAULT_SPEC_TYPE,      # draft-mtp
    "draft-dflash",
    "draft-dspark",
    "draft-eagle3",
)
# Every draft type except MTP requires a drafter model of the appropriate
# format (a plain smaller model for draft-simple, DFlash checkpoint for
# draft-dflash, DSpark for draft-dspark, EAGLE-3 for draft-eagle3).
SPEC_TYPES_NEEDING_DRAFT_MODEL = frozenset({
    "draft-simple",
    "draft-dflash",
    "draft-dspark",
    "draft-eagle3",
})

SPEC_CONFIG_KEYS = (
    "spec_enabled",
    "spec_type",
    "spec_draft_model",
    "spec_draft_n_max",
    "spec_draft_n_min",
    "spec_draft_p_split",
    "spec_draft_p_min",
)


def spec_type_needs_draft_model(spec_type: str | None) -> bool:
    """Whether a drafter is mandatory. All types *accept* one; only some require it."""
    return (spec_type or DEFAULT_SPEC_TYPE) in SPEC_TYPES_NEEDING_DRAFT_MODEL


def _parse_optional_int(body: dict, key: str, min_value: int = 0) -> tuple[int | None, str | None]:
    """Parse an optional non-negative integer. Empty/None -> None (omit flag)."""
    raw = body.get(key)
    if raw in (None, ""):
        return None, None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, f"{key} must be an integer"
    if value < min_value:
        return None, f"{key} must be >= {min_value}"
    return value, None


def _parse_optional_probability(body: dict, key: str) -> tuple[float | None, str | None]:
    """Parse an optional float in [0, 1]. Empty/None -> None (omit flag)."""
    raw = body.get(key)
    if raw in (None, ""):
        return None, None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"{key} must be a number"
    if value < 0 or value > 1:
        return None, f"{key} must be between 0 and 1"
    return value, None


def parse_spec_config(body: dict) -> tuple[dict, str | None]:
    """Validate the speculative-decoding fields of a launch/preset payload."""
    enabled = bool(body.get("spec_enabled", False))

    spec_type = (body.get("spec_type") or DEFAULT_SPEC_TYPE).strip() or DEFAULT_SPEC_TYPE
    if spec_type not in SPEC_TYPES:
        return {}, f"spec_type must be one of: {', '.join(SPEC_TYPES)}"

    draft_model = (body.get("spec_draft_model") or "").strip()
    if enabled and spec_type_needs_draft_model(spec_type) and not draft_model:
        return {}, f"spec_draft_model is required when spec_type is {spec_type}"

    # spec_draft_n_max is the pre-existing "how many tokens to draft per step"
    # knob and was previously the only advanced field. It stays in the visible
    # part of the section; the three below are collapsed under Advanced.
    draft_n_max, err = _parse_optional_int(body, "spec_draft_n_max", min_value=0)
    if err:
        return {}, err

    # The three "Advanced" knobs. Empty means "don't pass the flag at all" so
    # llama-server uses its own default (which we deliberately don't hard-code
    # here - it drifts across llama.cpp versions). Same shape for all three:
    # accept a value in the natural range, otherwise omit the flag.
    draft_n_min, err = _parse_optional_int(body, "spec_draft_n_min", min_value=0)
    if err:
        return {}, err

    draft_p_split, err = _parse_optional_probability(body, "spec_draft_p_split")
    if err:
        return {}, err

    draft_p_min, err = _parse_optional_probability(body, "spec_draft_p_min")
    if err:
        return {}, err

    return {
        "spec_enabled": enabled,
        "spec_type": spec_type,
        "spec_draft_model": draft_model,
        "spec_draft_n_max": draft_n_max,
        "spec_draft_n_min": draft_n_min,
        "spec_draft_p_split": draft_p_split,
        "spec_draft_p_min": draft_p_min,
    }, None
