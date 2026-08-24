# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

# PDF input: llama-server has no PDF support, so we detect PDF payloads inside
# inbound chat messages here and rewrite the offending content blocks in place
# before forwarding. Two output shapes:
#
#   pdf_extract_text_first = False (default)  -> N image_url blocks, one PNG
#                                                per rasterized page. Needs a
#                                                vision model + mmproj.
#   pdf_extract_text_first = True             -> one text block with the PDF's
#                                                embedded text layer, IF that
#                                                text is substantive (see the
#                                                50-chars/page heuristic in
#                                                extract_text). Sparse output
#                                                falls through to rasterization
#                                                so scanned PDFs still work.
#
# Everything runs in the Flask worker thread (gunicorn gthread). Rasterization
# is CPU/RAM-heavy and independent of any per-instance RequestGate, so a
# process-wide semaphore caps concurrent conversions. Anything above the cap
# blocks until a slot frees up rather than piling on load.

import base64
import io
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_bytes
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError as e:  # keep the import block itself importable
    PDF_SUPPORT = False
    logger.warning("PDF input disabled: %s", e)

# %PDF- is the only mandatory prefix; the version digits vary (1.3..2.0).
_PDF_MAGIC = b"%PDF-"

# Cap concurrent rasterizations across the whole process. Tunable via the
# LLAMAMAN_PDF_MAX_CONCURRENT env var (default 4); raise on beefy boxes with
# PDF-heavy traffic, lower on tight ones. Independent of per-instance
# RequestGate, which caps inference concurrency downstream.
from config import LLAMAMAN_PDF_MAX_CONCURRENT
_raster_sem = threading.BoundedSemaphore(LLAMAMAN_PDF_MAX_CONCURRENT)


class PDFError(Exception):
    """A PDF payload could not be processed. Callers should surface as 4xx."""


# ---------- detection ----------

def is_pdf_payload(b64_or_data_url: str) -> bool:
    """Cheap %PDF- header sniff. Handles both raw base64 and data: URLs.

    Kept intentionally tolerant: a data URL that claims application/pdf is
    trusted without decoding (fast path); anything else is header-sniffed
    from the first few base64 chars."""
    if not b64_or_data_url or not isinstance(b64_or_data_url, str):
        return False
    if b64_or_data_url.startswith("data:application/pdf"):
        return True
    if b64_or_data_url.startswith("data:"):
        try:
            _, b64 = b64_or_data_url.split(",", 1)
        except ValueError:
            return False
    else:
        b64 = b64_or_data_url
    try:
        head = base64.b64decode(b64[:12], validate=False)
    except Exception:
        return False
    return head.startswith(_PDF_MAGIC)


def _decode(b64_or_data_url: str) -> bytes:
    if b64_or_data_url.startswith("data:"):
        try:
            _, b64 = b64_or_data_url.split(",", 1)
        except ValueError:
            raise PDFError("malformed data URL")
    else:
        b64 = b64_or_data_url
    try:
        return base64.b64decode(b64, validate=False)
    except Exception as e:
        raise PDFError(f"invalid base64: {e}")


# ---------- extraction / rasterization ----------

def extract_text(pdf_bytes: bytes, max_pages: int) -> Optional[str]:
    """Return the PDF's embedded text if it's substantive, else None.

    A born-digital PDF (generated from Word/LaTeX/browser print) carries real
    text objects that pypdf can pull out in milliseconds - free, exact, no
    OCR. A scanned PDF has no text layer and returns near-empty strings; the
    50-chars/page average is a conservative threshold to distinguish the two
    so the caller falls through to rasterization on scans."""
    if not PDF_SUPPORT:
        return None
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        raise PDFError(f"unreadable PDF: {e}")
    n = min(len(reader.pages), max_pages)
    if n == 0:
        return None
    parts = []
    for i in range(n):
        try:
            parts.append((reader.pages[i].extract_text() or "").strip())
        except Exception:
            parts.append("")
    text = "\n\n".join(p for p in parts if p)
    if not text or len(text) < 50 * n:
        return None
    return text


def rasterize(pdf_bytes: bytes, dpi: int, max_pages: int) -> list:
    """Render each page to PNG bytes. Enforces max_pages up front so a huge
    PDF can't monopolize the raster semaphore for minutes."""
    if not PDF_SUPPORT:
        raise PDFError("PDF support unavailable (poppler/pdf2image not installed)")
    try:
        page_count = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception as e:
        raise PDFError(f"cannot read PDF: {e}")
    if page_count == 0:
        raise PDFError("PDF has no pages")
    if page_count > max_pages:
        raise PDFError(f"PDF has {page_count} pages, exceeds max_pages={max_pages}")

    with _raster_sem:
        try:
            imgs = convert_from_bytes(
                pdf_bytes,
                dpi=dpi,
                first_page=1,
                last_page=page_count,
                fmt="png",
            )
        except Exception as e:
            raise PDFError(f"rasterization failed: {e}")

    out = []
    for img in imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        out.append(buf.getvalue())
    return out


def _png_data_url(png: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"


# ---------- public entry points ----------

def _pdf_payload_of(block) -> Optional[str]:
    """Return the base64-or-data-url of a PDF content block, or None if the
    block isn't a PDF-carrying shape. Handles both the OpenAI vision
    `image_url` block (when the URL is a PDF data URL) and the OpenAI file
    input block (`type: "file"` with inline `file_data`)."""
    if not isinstance(block, dict):
        return None
    if block.get("type") == "image_url":
        url = (block.get("image_url") or {}).get("url") or ""
        if is_pdf_payload(url):
            return url
    if block.get("type") == "file":
        data = (block.get("file") or {}).get("file_data") or ""
        if is_pdf_payload(data):
            return data
    return None


def expand_pdf_blocks(content, config: dict):
    """Walk an OpenAI content array in place; replace PDF blocks with either
    a text block (text-layer shortcut) or N image_url blocks (rasterized
    pages). Non-PDF blocks pass through untouched. Non-list content passes
    through unchanged.

    Config keys (all optional, safe defaults):
      pdf_input_enabled       - if False, pass through entirely
      pdf_extract_text_first  - default False; try the text layer first
      pdf_dpi                 - default 200
      pdf_max_pages           - default 20
    """
    if not isinstance(content, list) or not content:
        return content
    if not config.get("pdf_input_enabled", False):
        return content

    max_pages = int(config.get("pdf_max_pages") or 20)
    dpi = int(config.get("pdf_dpi") or 200)
    try_text = bool(config.get("pdf_extract_text_first", False))

    out = []
    for block in content:
        payload = _pdf_payload_of(block)
        if payload is None:
            out.append(block)
            continue
        pdf_bytes = _decode(payload)
        if try_text:
            text = extract_text(pdf_bytes, max_pages)
            if text:
                out.append({"type": "text", "text": text})
                continue
        for png in rasterize(pdf_bytes, dpi, max_pages):
            out.append({
                "type": "image_url",
                "image_url": {"url": _png_data_url(png)},
            })
    return out


def expand_ollama_images(images, config: dict):
    """Split an Ollama-style images[] array. PDFs are pulled out and returned
    as ready-made content blocks; real images stay in the returned list and
    get lifted to image_url blocks by the existing translator.

    Returns (remaining_image_b64s, extra_content_blocks). Both empty when
    input is empty or non-list."""
    if not isinstance(images, list) or not images:
        return images, []

    max_pages = int(config.get("pdf_max_pages") or 20)
    dpi = int(config.get("pdf_dpi") or 200)
    try_text = bool(config.get("pdf_extract_text_first", False))
    feature_on = bool(config.get("pdf_input_enabled", False))

    keep, extra = [], []
    for img in images:
        if not isinstance(img, str):
            continue
        if feature_on and is_pdf_payload(img):
            pdf_bytes = _decode(img)
            if try_text:
                text = extract_text(pdf_bytes, max_pages)
                if text:
                    extra.append({"type": "text", "text": text})
                    continue
            for png in rasterize(pdf_bytes, dpi, max_pages):
                extra.append({
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(png)},
                })
        else:
            # Non-PDF, or feature off - preserve as-is so the caller's
            # existing image_url lifting sees it. If the feature is off
            # and the bytes happen to be a PDF, llama-server will complain;
            # that's the correct "you didn't enable PDF input" behavior.
            keep.append(img)
    return keep, extra
