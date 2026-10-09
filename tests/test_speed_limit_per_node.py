# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""The global download speed limit is per node (core/node_settings): saved
into this node's namespace, read back from it by GET /api/settings, by the
subprocess settings mirror running downloads poll, and by new downloads'
launch environment. A value saved before it became per-node (top-level in
the shared settings) still applies until the node saves its own."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.downloads as downloads_api
import api.settings as settings_api
from core.cluster import get_node_id
from core.node_settings import NODE_SCOPED_KEYS


class FakeStorage:
    def __init__(self, settings):
        self.settings = settings

    def get_settings(self):
        return json.loads(json.dumps(self.settings))

    def merge_settings(self, patch):
        def merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    merge(dst[k], v)
                else:
                    dst[k] = v
        merge(self.settings, patch)
        return self.get_settings()


class PerNodeSpeedLimitTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.snapshot = os.path.join(self._tmp.name, "subprocess_settings.json")
        self.node = get_node_id()
        self.storage = FakeStorage({"nodes": {"peer-node": {"global_speed_limit_mbps": 5.0}}})
        self._p = [patch.object(settings_api, "_SUBPROCESS_SETTINGS_FILE", self.snapshot),
                   patch("api.settings.get_storage", return_value=self.storage),
                   patch("api.downloads.get_storage", return_value=self.storage),
                   patch("core.node_settings.get_storage", return_value=self.storage)]
        for p in self._p:
            p.start()
        app = Flask(__name__)
        app.register_blueprint(settings_api.bp)
        self.client = app.test_client()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()
        self._tmp.cleanup()

    def _snapshot_value(self):
        with open(self.snapshot) as f:
            return json.load(f)["global_speed_limit_mbps"]

    def test_is_node_scoped(self):
        self.assertIn("global_speed_limit_mbps", NODE_SCOPED_KEYS)

    def test_save_goes_to_this_node_only(self):
        resp = self.client.post("/api/settings", json={"global_speed_limit_mbps": "40"})
        self.assertEqual(resp.status_code, 200)
        nodes = self.storage.settings["nodes"]
        self.assertEqual(nodes[self.node]["global_speed_limit_mbps"], 40.0)  # coerced
        self.assertEqual(nodes["peer-node"]["global_speed_limit_mbps"], 5.0)  # untouched
        self.assertNotIn("global_speed_limit_mbps", self.storage.settings)    # not shared
        self.assertEqual(resp.get_json()["settings"]["global_speed_limit_mbps"], 40.0)
        self.assertEqual(self._snapshot_value(), 40.0)

    def test_get_settings_reports_this_nodes_value(self):
        self.storage.settings["nodes"][self.node] = {"global_speed_limit_mbps": 12.0}
        data = self.client.get("/api/settings").get_json()
        self.assertEqual(data["global_speed_limit_mbps"], 12.0)
        self.assertNotIn("nodes", data)

    def test_legacy_shared_value_still_applies(self):
        self.storage.settings["global_speed_limit_mbps"] = 30.0
        self.assertEqual(self.client.get("/api/settings").get_json()["global_speed_limit_mbps"], 30.0)
        settings_api.snapshot_subprocess_settings()
        self.assertEqual(self._snapshot_value(), 30.0)
        self.client.post("/api/settings", json={"global_speed_limit_mbps": 0})
        self.assertEqual(self._snapshot_value(), 0.0)  # node's own value wins

    def test_new_downloads_get_this_nodes_limit(self):
        self.storage.settings["nodes"][self.node] = {"global_speed_limit_mbps": 80.0}
        env = downloads_api._build_download_env("r", "/d", "f", "", 0)
        self.assertEqual(env["HF_SPEED_LIMIT"], str(int(80 * 1_000_000 / 8)))
        self.storage.settings["nodes"][self.node] = {"global_speed_limit_mbps": 0}
        env = downloads_api._build_download_env("r", "/d", "f", "", 16)
        self.assertEqual(env["HF_SPEED_LIMIT"], str(int(16 * 1_000_000 / 8)))  # per-model fallback

    def test_other_settings_stay_shared(self):
        self.client.post("/api/settings", json={"auto_retry_failed_downloads": True})
        self.assertTrue(self.storage.settings["auto_retry_failed_downloads"])


if __name__ == "__main__":
    unittest.main()
