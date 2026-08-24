# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import base64
import io
import os
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core import pdf_input
from core.multimodal import parse_mmproj_config


# A minimal well-formed PDF (5 pages, empty). Built once in setUpModule so
# tests don't depend on external files.
_TINY_PDF_B64 = None


def _make_tiny_pdf(pages: int = 1) -> bytes:
    """Build a tiny multi-page PDF in memory. Uses pypdf if available so we
    don't need a fixture file; if pypdf is missing the test module skips."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        raise unittest.SkipTest("pypdf not installed")
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=612, height=792)  # US Letter, 72 DPI
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


class IsPDFPayloadTests(unittest.TestCase):
    """The header sniff has to catch real PDFs and reject everything else
    without false-positiving on random base64. It's the gate to the rest of
    the pipeline, so a miss on either side is a real bug."""

    def test_data_url_with_pdf_mime_is_pdf(self):
        self.assertTrue(pdf_input.is_pdf_payload("data:application/pdf;base64,JVBERi0="))

    def test_raw_base64_starting_with_pdf_header_is_pdf(self):
        # %PDF-1.4 -> base64
        b64 = base64.b64encode(b"%PDF-1.4\n").decode()
        self.assertTrue(pdf_input.is_pdf_payload(b64))

    def test_png_is_not_pdf(self):
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        self.assertFalse(pdf_input.is_pdf_payload(b64))

    def test_jpeg_data_url_is_not_pdf(self):
        self.assertFalse(pdf_input.is_pdf_payload("data:image/jpeg;base64,/9j/4A=="))

    def test_empty_and_junk_do_not_crash(self):
        self.assertFalse(pdf_input.is_pdf_payload(""))
        self.assertFalse(pdf_input.is_pdf_payload("not base64!"))
        self.assertFalse(pdf_input.is_pdf_payload(None))  # type: ignore[arg-type]


class ExtractTextTests(unittest.TestCase):
    """The <50 chars/page heuristic distinguishes born-digital PDFs (worth
    inlining) from scanned PDFs (which need rasterization)."""

    def test_blank_pdf_returns_none(self):
        pdf = _make_tiny_pdf(pages=3)
        # A blank PDF's pages have no text objects -> extraction returns
        # empty strings -> None, so the caller falls through to rasterize.
        self.assertIsNone(pdf_input.extract_text(pdf, max_pages=20))

    def test_unreadable_pdf_raises(self):
        with self.assertRaises(pdf_input.PDFError):
            pdf_input.extract_text(b"not a pdf", max_pages=5)


class RasterizeGuardTests(unittest.TestCase):
    """Rasterization guards - page cap and empty-input handling - must fire
    BEFORE convert_from_bytes is invoked (which is expensive and requires
    poppler). Patching convert_from_bytes lets these tests run in CI without
    poppler installed."""

    def test_over_page_cap_raises(self):
        pdf = _make_tiny_pdf(pages=5)
        with self.assertRaises(pdf_input.PDFError) as ctx:
            pdf_input.rasterize(pdf, dpi=200, max_pages=3)
        self.assertIn("exceeds max_pages", str(ctx.exception))

    def test_unreadable_pdf_raises_before_convert(self):
        with self.assertRaises(pdf_input.PDFError):
            pdf_input.rasterize(b"not a pdf", dpi=200, max_pages=5)

    @patch("core.pdf_input.convert_from_bytes")
    def test_within_cap_calls_converter(self, mock_convert):
        # Fake PIL image with a save() that writes something recognizable.
        class _FakeImg:
            def save(self, buf, format=None, optimize=False):
                buf.write(b"\x89PNG\r\n\x1a\nfake")
        mock_convert.return_value = [_FakeImg(), _FakeImg()]
        pdf = _make_tiny_pdf(pages=2)
        pages = pdf_input.rasterize(pdf, dpi=150, max_pages=10)
        self.assertEqual(len(pages), 2)
        self.assertTrue(all(p.startswith(b"\x89PNG") for p in pages))
        # DPI must be forwarded verbatim; last_page must equal actual page count.
        _, kwargs = mock_convert.call_args
        self.assertEqual(kwargs["dpi"], 150)
        self.assertEqual(kwargs["last_page"], 2)


class ExpandPDFBlocksTests(unittest.TestCase):
    """The content-block rewriter is the public entry point - its shape
    contract is what everything else in the pipeline depends on."""

    def _pdf_url(self, pages: int = 1) -> str:
        pdf = _make_tiny_pdf(pages=pages)
        return "data:application/pdf;base64," + base64.b64encode(pdf).decode()

    def test_feature_off_passes_through_unchanged(self):
        content = [{"type": "image_url", "image_url": {"url": self._pdf_url()}}]
        out = pdf_input.expand_pdf_blocks(content, {"pdf_input_enabled": False})
        self.assertIs(out, content)  # exact same list, no work done

    def test_non_list_passes_through(self):
        self.assertEqual(pdf_input.expand_pdf_blocks("hello", {"pdf_input_enabled": True}), "hello")
        self.assertEqual(pdf_input.expand_pdf_blocks(None, {"pdf_input_enabled": True}), None)

    def test_non_pdf_blocks_pass_through(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        ]
        out = pdf_input.expand_pdf_blocks(content, {"pdf_input_enabled": True})
        self.assertEqual(out, content)

    @patch("core.pdf_input.convert_from_bytes")
    def test_pdf_image_url_block_expands_to_image_urls(self, mock_convert):
        class _FakeImg:
            def save(self, buf, format=None, optimize=False):
                buf.write(b"\x89PNGpng-bytes")
        mock_convert.return_value = [_FakeImg(), _FakeImg()]
        content = [{"type": "image_url", "image_url": {"url": self._pdf_url(pages=2)}}]
        out = pdf_input.expand_pdf_blocks(content, {
            "pdf_input_enabled": True, "pdf_max_pages": 20,
        })
        self.assertEqual(len(out), 2)
        for block in out:
            self.assertEqual(block["type"], "image_url")
            self.assertTrue(block["image_url"]["url"].startswith("data:image/png;base64,"))

    @patch("core.pdf_input.convert_from_bytes")
    def test_pdf_file_block_expands_to_image_urls(self, mock_convert):
        # OpenAI's newer file-input shape: {"type":"file","file":{"file_data":"data:application/pdf;..."}}
        class _FakeImg:
            def save(self, buf, format=None, optimize=False):
                buf.write(b"\x89PNG")
        mock_convert.return_value = [_FakeImg()]
        content = [{"type": "file", "file": {"file_data": self._pdf_url(pages=1)}}]
        out = pdf_input.expand_pdf_blocks(content, {
            "pdf_input_enabled": True, "pdf_max_pages": 20,
        })
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "image_url")

    def test_text_first_falls_through_when_layer_is_sparse(self):
        # Blank PDF has no text -> extract_text returns None -> falls through
        # to rasterize. We patch rasterize so we don't need poppler.
        with patch("core.pdf_input.rasterize", return_value=[b"\x89PNGfake"]):
            content = [{"type": "image_url", "image_url": {"url": self._pdf_url()}}]
            out = pdf_input.expand_pdf_blocks(content, {
                "pdf_input_enabled": True,
                "pdf_extract_text_first": True,
                "pdf_max_pages": 20,
            })
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "image_url")


class ExpandOllamaImagesTests(unittest.TestCase):
    """Ollama's images[] is a flat list of base64 strings - PDFs and JPEGs
    get mixed in the same list, so the splitter must return non-PDF entries
    unchanged (they get lifted by the existing translator loop)."""

    def test_mixed_list_splits_correctly(self):
        pdf_b64 = base64.b64encode(_make_tiny_pdf(pages=1)).decode()
        jpeg_b64 = base64.b64encode(b"\xff\xd8\xff\xe0fake").decode()
        with patch("core.pdf_input.rasterize", return_value=[b"\x89PNGraster"]):
            keep, extra = pdf_input.expand_ollama_images(
                [jpeg_b64, pdf_b64],
                {"pdf_input_enabled": True, "pdf_max_pages": 20},
            )
        self.assertEqual(keep, [jpeg_b64])
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0]["type"], "image_url")

    def test_feature_off_keeps_pdf_in_place(self):
        # With the feature off we don't touch PDFs - llama-server will error,
        # which is the correct "you didn't enable this" surfacing.
        pdf_b64 = base64.b64encode(_make_tiny_pdf(pages=1)).decode()
        keep, extra = pdf_input.expand_ollama_images(
            [pdf_b64], {"pdf_input_enabled": False},
        )
        self.assertEqual(keep, [pdf_b64])
        self.assertEqual(extra, [])


class MMProjConfigTests(unittest.TestCase):
    """The preset validator has to gate PDF input on mmproj (no vision model
    means the rasterized pages have nowhere to go) and enforce the numeric
    bounds on DPI and page count."""

    def test_defaults(self):
        cfg, err = parse_mmproj_config({})
        self.assertIsNone(err)
        self.assertEqual(cfg, {
            "mmproj_enabled": False,
            "mmproj_path": "",
            "pdf_input_enabled": False,
            "pdf_extract_text_first": False,
            "pdf_dpi": 200,
            "pdf_max_pages": 20,
        })

    def test_pdf_without_mmproj_rejected(self):
        _, err = parse_mmproj_config({"pdf_input_enabled": True})
        self.assertIsNotNone(err)
        self.assertIn("mmproj_enabled", err)

    def test_dpi_bounds(self):
        _, err = parse_mmproj_config({
            "mmproj_enabled": True, "mmproj_path": "/x",
            "pdf_input_enabled": True, "pdf_dpi": 50,
        })
        self.assertIn("pdf_dpi", err)
        _, err = parse_mmproj_config({
            "mmproj_enabled": True, "mmproj_path": "/x",
            "pdf_input_enabled": True, "pdf_dpi": 900,
        })
        self.assertIn("pdf_dpi", err)

    def test_max_pages_bounds(self):
        _, err = parse_mmproj_config({
            "mmproj_enabled": True, "mmproj_path": "/x",
            "pdf_input_enabled": True, "pdf_max_pages": 0,
        })
        self.assertIn("pdf_max_pages", err)
        _, err = parse_mmproj_config({
            "mmproj_enabled": True, "mmproj_path": "/x",
            "pdf_input_enabled": True, "pdf_max_pages": 500,
        })
        self.assertIn("pdf_max_pages", err)

    def test_valid_full_config(self):
        cfg, err = parse_mmproj_config({
            "mmproj_enabled": True, "mmproj_path": "/models/vp.gguf",
            "pdf_input_enabled": True, "pdf_extract_text_first": True,
            "pdf_dpi": 300, "pdf_max_pages": 50,
        })
        self.assertIsNone(err)
        self.assertEqual(cfg["pdf_dpi"], 300)
        self.assertEqual(cfg["pdf_max_pages"], 50)
        self.assertTrue(cfg["pdf_extract_text_first"])


if __name__ == "__main__":
    unittest.main()
