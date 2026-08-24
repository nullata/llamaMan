# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

# Image and PDF input support. Vision itself is a llama.cpp feature: llama-server
# loads a separate multimodal projector via --mmproj so a vision model can accept
# image inputs. That projector ships as its own GGUF alongside the main model.
#
# PDF input is llamaman-side plumbing on top of that: inbound PDFs are rewritten
# to image blocks (or, optionally, the embedded text layer) before forwarding.
# Because the vision model has to actually consume the rasterized pages, PDF
# input is gated on mmproj_enabled - turning it on without a projector loaded
# would just produce blocks the model can't see.

MMPROJ_CONFIG_KEYS = (
    "mmproj_enabled",
    "mmproj_path",
    "pdf_input_enabled",
    "pdf_extract_text_first",
    "pdf_dpi",
    "pdf_max_pages",
)


def parse_mmproj_config(body: dict) -> tuple[dict, str | None]:
    """Validate the image+PDF-input fields of a launch/preset payload."""
    enabled = bool(body.get("mmproj_enabled", False))
    path = (body.get("mmproj_path") or "").strip()
    if enabled and not path:
        return {}, "mmproj_path is required when image input is enabled"

    pdf_enabled = bool(body.get("pdf_input_enabled", False))
    if pdf_enabled and not enabled:
        # No vision model = the rasterized PDF pages have nowhere to go.
        return {}, "pdf_input_enabled requires mmproj_enabled"

    # dpi and max_pages are always parsed (even when the feature is off) so
    # a saved preset always has consistent shape. Bounds are conservative:
    # 72 DPI is barely readable; 600 DPI is huge and pointless past vision
    # models' internal resize; 1..200 pages caps a single request's raster
    # cost without preventing legitimate multi-page docs.
    #
    # We can't use `body.get(k) or default` here: an explicit 0 would silently
    # map to the default and slip past the bounds check. Only a missing key
    # (None) should fall back.
    dpi_raw = body.get("pdf_dpi")
    try:
        dpi = int(dpi_raw) if dpi_raw is not None else 200
    except (TypeError, ValueError):
        return {}, "pdf_dpi must be an integer"
    if not 72 <= dpi <= 600:
        return {}, "pdf_dpi must be between 72 and 600"

    max_pages_raw = body.get("pdf_max_pages")
    try:
        max_pages = int(max_pages_raw) if max_pages_raw is not None else 20
    except (TypeError, ValueError):
        return {}, "pdf_max_pages must be an integer"
    if not 1 <= max_pages <= 200:
        return {}, "pdf_max_pages must be between 1 and 200"

    return {
        "mmproj_enabled": enabled,
        "mmproj_path": path,
        "pdf_input_enabled": pdf_enabled,
        "pdf_extract_text_first": bool(body.get("pdf_extract_text_first", False)),
        "pdf_dpi": dpi,
        "pdf_max_pages": max_pages,
    }, None
