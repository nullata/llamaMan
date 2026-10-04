# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""llama.cpp's built-in web UI: the Web UI launch toggle (webui_enabled,
default on -> --no-webui when off) and the instance card's link to it
(_public_instance web_ui): hidden for embedding models and with the UI off,
always the port the server's own container publishes."""

import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.instances as instances_api
import api.presets as presets_api
from core.engines import validate_launch
from core.helpers import build_llama_cmd

GGUF = "/models/chat-Q4_K_M.gguf"


class CommandTests(unittest.TestCase):

    def _cmd(self, **cfg):
        return build_llama_cmd(GGUF, 8080, {"ctx_size": 4096, **cfg})

    def test_on_by_default(self):
        self.assertNotIn("--no-webui", self._cmd())                 # configs before the toggle
        self.assertNotIn("--no-webui", self._cmd(webui_enabled=True))

    def test_off_passes_no_webui_once(self):
        self.assertEqual(self._cmd(webui_enabled=False).count("--no-webui"), 1)
        self.assertEqual(self._cmd(webui_enabled=False, extra_args="--no-webui").count("--no-webui"), 1)


class LinkTests(unittest.TestCase):

    def _public(self, **inst):
        base = {"id": "i1", "model_path": GGUF, "status": "healthy", "port": 8001, "config": {}}
        base.update(inst)
        return instances_api._public_instance(base)

    def test_direct_instance_uses_its_port(self):
        self.assertEqual(self._public()["web_ui"], {"port": 8001, "path": "/"})

    def test_proxied_instance_uses_the_servers_own_published_port(self):
        # Not llamaman's proxy port: no auth, no host port mapping in between.
        self.assertEqual(self._public(_internal_port=12001)["web_ui"], {"port": 12001, "path": "/"})

    def test_no_link_for_embeddings_or_ui_off(self):
        self.assertNotIn("web_ui", self._public(config={"embedding_model": True}))
        self.assertNotIn("web_ui", self._public(config={"webui_enabled": False}))
        self.assertNotIn("web_ui", self._public(config={"extra_args": "--jinja --no-webui"}))


class PresetAndValidationTests(unittest.TestCase):

    def _save(self, body):
        storage = Mock()
        storage.get_preset.return_value = {}
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets.invalidate_alias_cache", create=True):
            resp = app.test_client().put("/api/presets/models/chat-Q4_K_M.gguf",
                                         json={"ctx_size": 4096, **body})
        self.assertEqual(resp.status_code, 200, resp.get_json())
        return storage.save_preset.call_args.args[1]

    def test_preset_stores_toggle_default_on(self):
        self.assertTrue(self._save({})["webui_enabled"])
        self.assertFalse(self._save({"webui_enabled": False})["webui_enabled"])

    def test_strata_accepts_either_value(self):
        with patch("config.STRATA_ENABLED", True):
            for v in (True, False):
                self.assertIsNone(validate_launch({"ctx_size": 32768, "webui_enabled": v},
                                                  "/strata/qwen-IQ2_XS", "cuda")[2], v)


if __name__ == "__main__":
    unittest.main()
