# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

DEFAULT_SPEC_TYPE = "draft-mtp"

# Values accepted by llama-server's --spec-type. draft-mtp uses the main model's
# MTP heads; draft-dflash needs a separate DFlash drafter passed via -md.
SPEC_TYPES = (DEFAULT_SPEC_TYPE, "draft-dflash")
SPEC_TYPES_NEEDING_DRAFT_MODEL = frozenset({"draft-dflash"})

SPEC_CONFIG_KEYS = (
    "spec_enabled",
    "spec_type",
    "spec_draft_model",
    "spec_draft_n_max",
)


def spec_type_needs_draft_model(spec_type: str | None) -> bool:
    return (spec_type or DEFAULT_SPEC_TYPE) in SPEC_TYPES_NEEDING_DRAFT_MODEL


def parse_spec_config(body: dict) -> tuple[dict, str | None]:
    """Validate the speculative-decoding fields of a launch/preset payload."""
    enabled = bool(body.get("spec_enabled", False))

    spec_type = (body.get("spec_type") or DEFAULT_SPEC_TYPE).strip() or DEFAULT_SPEC_TYPE
    if spec_type not in SPEC_TYPES:
        return {}, f"spec_type must be one of: {', '.join(SPEC_TYPES)}"

    draft_model = (body.get("spec_draft_model") or "").strip()
    if enabled and spec_type_needs_draft_model(spec_type) and not draft_model:
        return {}, f"spec_draft_model is required when spec_type is {spec_type}"

    raw_n_max = body.get("spec_draft_n_max")
    if raw_n_max in (None, ""):
        draft_n_max = None
    else:
        try:
            draft_n_max = int(raw_n_max)
        except (TypeError, ValueError):
            return {}, "spec_draft_n_max must be an integer"
        if draft_n_max < 0:
            return {}, "spec_draft_n_max must be >= 0"

    return {
        "spec_enabled": enabled,
        "spec_type": spec_type,
        "spec_draft_model": draft_model,
        "spec_draft_n_max": draft_n_max,
    }, None
