# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

# Image input (vision) support: llama-server loads a separate multimodal
# projector file via --mmproj so a vision model can accept image inputs. The
# projector ships as its own GGUF alongside the main model.

MMPROJ_CONFIG_KEYS = (
    "mmproj_enabled",
    "mmproj_path",
)


def parse_mmproj_config(body: dict) -> tuple[dict, str | None]:
    """Validate the image-input (mmproj) fields of a launch/preset payload."""
    enabled = bool(body.get("mmproj_enabled", False))
    path = (body.get("mmproj_path") or "").strip()
    if enabled and not path:
        return {}, "mmproj_path is required when image input is enabled"
    return {
        "mmproj_enabled": enabled,
        "mmproj_path": path,
    }, None
