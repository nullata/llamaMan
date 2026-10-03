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


class LaunchFormMarkupTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(TEMPLATE, encoding="utf-8") as f:
            cls.html = f.read()

    def _group_tag(self, field_id):
        m = re.search(r'<div class="form-group"(?: id="[^"]+")?( data-engine-only="\w+")?>\s*'
                      r'<span class="label-row">\s*<label for="' + re.escape(field_id) + '">', self.html)
        self.assertIsNotNone(m, field_id)
        return m.group(1)

    def test_llamacpp_only_fields_tagged(self):
        for fid in ("f-gpu-layers", "f-n-cpu-moe", "f-threads", "f-parallel", "f-flash-attn",
                    "f-cache-type-k", "f-split-mode", "f-tensor-split"):
            self.assertEqual(self._group_tag(fid), ' data-engine-only="llamacpp"', fid)

    def test_shared_fields_not_tagged(self):
        for fid in ("f-ctx-size", "f-port", "f-memory-limit", "f-idle-timeout",
                    "f-max-concurrent", "f-gpu-devices"):
            self.assertIsNone(self._group_tag(fid), fid)

    def test_strata_section_present_and_hidden(self):
        self.assertRegex(self.html, r'id="strata-settings-section" data-engine-only="strata" hidden')
        for fid in ("f-strata-vision", "f-strata-kv", "f-strata-low-ram", "btn-strata-download"):
            self.assertIn(f'id="{fid}"', self.html)
        self.assertIn("js/engines.js", self.html)


if __name__ == "__main__":
    unittest.main()
