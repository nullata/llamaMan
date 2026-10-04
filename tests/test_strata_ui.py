# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Server-side pieces of the Strata UI: the Docker Images tab's
"built locally" entry, and the launch-form markup that static/js/engines.js
toggles (llama.cpp-only fields tagged, Strata section present and hidden)."""

import os
import re
import unittest
from unittest.mock import patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.images as images_api

TEMPLATE = os.path.join(REPO_ROOT, "templates", "index.html")


class EngineImagesTests(unittest.TestCase):

    def _get(self, enabled, present):
        app = Flask(__name__)
        app.register_blueprint(images_api.bp)
        with patch("config.STRATA_ENABLED", enabled), \
             patch("config.STRATA_IMAGE", "strata:latest"), \
             patch("api.images._read_docker_images", return_value={}), \
             patch("api.images._get_image_local_info", return_value={"present": present}):
            return app.test_client().get("/api/images").get_json()

    def test_hidden_when_disabled(self):
        self.assertEqual(self._get(False, True)["engine_images"], [])

    def test_strata_image_listed_with_build_command(self):
        data = self._get(True, False)
        (img,) = data["engine_images"]
        self.assertEqual(img["name"], "strata:latest")
        self.assertFalse(img["present"])
        self.assertFalse(img["pullable"])
        self.assertEqual(img["build_command"], "docker build -t strata:latest .")
        # Never offered for pulling / auto-update.
        self.assertNotIn("strata:latest", [i["name"] for i in data["images"]])


class StrataPdfInputTests(unittest.TestCase):
    """PDF input on Strata is gated on its own Image Input, not on mmproj."""

    def _launch_check(self, body):
        from core.engines import get_engine
        from core.multimodal import parse_mmproj_config
        path = "/strata/qwen-IQ2_XS"
        full = {"engine": "strata", **body}
        return parse_mmproj_config(full, get_engine(full, path).image_input_enabled(full))

    def test_pdf_needs_strata_image_input(self):
        cfg, err = self._launch_check({"pdf_input_enabled": True, "strata_vision": "no"})
        self.assertIn("image input", err)
        for vision in ("yes", "cpu"):
            cfg, err = self._launch_check({"pdf_input_enabled": True, "strata_vision": vision})
            self.assertIsNone(err, vision)
            self.assertTrue(cfg["pdf_input_enabled"])
            self.assertFalse(cfg["mmproj_enabled"])

    def test_llamacpp_rule_unchanged(self):
        from core.multimodal import parse_mmproj_config
        self.assertEqual(parse_mmproj_config({"pdf_input_enabled": True})[1],
                         "pdf_input_enabled requires mmproj_enabled")
        self.assertIsNone(parse_mmproj_config({"pdf_input_enabled": True, "mmproj_enabled": True,
                                               "mmproj_path": "/models/m.gguf"})[1])

    def test_strata_launch_accepts_pdf_fields(self):
        from core.engines import validate_launch
        with patch("config.STRATA_ENABLED", True):
            err = validate_launch({"ctx_size": 32768, "strata_vision": "yes", "pdf_input_enabled": True,
                                   "pdf_extract_text_first": True, "pdf_dpi": 150, "pdf_max_pages": 10},
                                  "/strata/qwen-IQ2_XS", "cuda")[2]
        self.assertIsNone(err)


class PresetLoadOrderTests(unittest.TestCase):

    def test_strata_fields_applied_before_mmproj_state(self):
        # updateMmprojState clears the PDF toggles when Strata's Image Input
        # is off; run before the preset set Image Input, it wiped a saved
        # "Accept PDF uploads".
        with open(os.path.join(REPO_ROOT, "static", "js", "models.js"), encoding="utf-8") as f:
            js = f.read()
        body = js[js.index("function applyPresetToLaunchForm("):]
        body = body[:body.index("\n}\n")]
        self.assertLess(body.index("applyStrataPresetToLaunchForm(p)"),
                        body.index("updateMmprojState()"))


class LaunchFormMarkupTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(TEMPLATE, encoding="utf-8") as f:
            cls.html = f.read()

    def _group_tag(self, field_id):
        m = re.search(r'<div class="form-group"(?: id="[^"]+")?( data-engine-only="\w+"(?: hidden)?)?>\s*'
                      r'<span class="label-row">\s*<label for="' + re.escape(field_id) + '">', self.html)
        self.assertIsNotNone(m, field_id)
        return m.group(1)

    def _section_of(self, marker):
        """The id of the launch-settings section containing `marker`."""
        i = self.html.index(marker)
        starts = [m for m in re.finditer(r'<div class="form-group full launch-settings-section"[^>]*>', self.html)
                  if m.start() < i]
        m = re.search(r'id="([^"]+)"', starts[-1].group(0))
        return m.group(1) if m else None

    def test_llamacpp_only_fields_tagged(self):
        for fid in ("f-gpu-layers", "f-n-cpu-moe", "f-threads", "f-parallel", "f-flash-attn",
                    "f-cache-type-k", "f-split-mode", "f-tensor-split"):
            self.assertEqual(self._group_tag(fid), ' data-engine-only="llamacpp"', fid)

    def test_shared_fields_not_tagged(self):
        for fid in ("f-ctx-size", "f-port", "f-memory-limit", "f-idle-timeout",
                    "f-max-concurrent", "f-gpu-devices"):
            self.assertIsNone(self._group_tag(fid), fid)

    def test_strata_options_sit_beside_their_llamacpp_counterparts(self):
        # No separate Strata section any more.
        self.assertNotIn('id="strata-settings-section"', self.html)
        for fid, section in (("f-strata-kv", "model-settings-section"),
                             ("f-strata-low-ram", "model-settings-section"),
                             ("f-strata-vision", "image-pdf-section"),
                             ("f-strata-layer-split", "gpu-settings-section")):
            self.assertEqual(self._group_tag(fid), ' data-engine-only="strata" hidden', fid)
            self.assertEqual(self._section_of(f'id="{fid}"'), section, fid)
        for marker in ('id="strata-model-summary"', 'id="btn-strata-download"',
                       'id="strata-warnings"', 'id="strata-context-options"'):
            self.assertEqual(self._section_of(marker), "model-settings-section", marker)
        self.assertIn("js/engines.js", self.html)

    def test_dry_explanations_llamacpp_only(self):
        self.assertIn('<p class="hint-text" data-engine-only="llamacpp">Two independent controls.', self.html)
        self.assertIn('<span class="text-meta" data-engine-only="llamacpp">Two-tier defense', self.html)

    def test_image_pdf_section_shared_mmproj_parts_llamacpp_only(self):
        sec = re.search(r'<div class="form-group full launch-settings-section" id="image-pdf-section">', self.html)
        self.assertIsNotNone(sec)                                   # not engine-gated
        self.assertRegex(self.html, r'<label class="toggle-switch" for="f-mmproj-enabled" data-engine-only="llamacpp"')
        self.assertEqual(self._group_tag("f-mmproj-path"), ' data-engine-only="llamacpp"')
        self.assertRegex(self.html, r'mmproj-toggle-table" data-engine-only="llamacpp">\s*'
                                    r'<div class="launch-toggle-pair">\s*<label for="f-mmproj-offload"')
        for fid in ("f-pdf-input-enabled", "f-pdf-extract-text-first", "f-pdf-dpi", "f-pdf-max-pages"):
            self.assertEqual(self._section_of(f'id="{fid}"'), "image-pdf-section", fid)


if __name__ == "__main__":
    unittest.main()
