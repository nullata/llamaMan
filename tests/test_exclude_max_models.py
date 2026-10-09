# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

# "Exclude from Max Models" is its own toggle: a decision model is kept out of
# LLAMAMAN_MAX_MODELS without --embeddings. Configs and presets saved before
# the toggle existed follow embedding_model, which used to be the only way.

import os
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.instances as instances_api
import api.llamaman as llamaman
from core.helpers import build_llama_cmd, excluded_from_max_models
from core.state import instances, instances_lock


class ExcludedFromMaxModelsTests(unittest.TestCase):
    def test_explicit_toggle_wins(self):
        self.assertTrue(excluded_from_max_models({"exclude_from_max_models": True}))
        self.assertFalse(excluded_from_max_models({"exclude_from_max_models": False,
                                                   "embedding_model": True}))
        self.assertTrue(excluded_from_max_models({"exclude_from_max_models": True,
                                                  "embedding_model": False}))

    def test_legacy_config_follows_embedding(self):
        self.assertTrue(excluded_from_max_models({"embedding_model": True}))
        self.assertFalse(excluded_from_max_models({"embedding_model": False}))
        self.assertFalse(excluded_from_max_models({}))
        self.assertFalse(excluded_from_max_models(None))

    def test_exclude_alone_adds_no_embeddings_flag(self):
        cmd = build_llama_cmd("/models/jev.gguf", 8000, {"exclude_from_max_models": True})
        self.assertNotIn("--embeddings", cmd)


def _inst(inst_id, config):
    return {"id": inst_id, "model_name": inst_id, "model_path": f"/models/{inst_id}.gguf",
            "port": 8000, "status": "healthy", "started_at": 1, "_last_request_at": 1,
            "_llamaman_managed": True, "config": config}


class CapCountingTests(unittest.TestCase):
    def setUp(self):
        with instances_lock:
            self._saved = {k: dict(v) for k, v in instances.items()}
            instances.clear()
            instances["chat"] = _inst("chat", {})
            instances["jev"] = _inst("jev", {"exclude_from_max_models": True})
            instances["emb_legacy"] = _inst("emb_legacy", {"embedding_model": True})
            instances["emb_counted"] = _inst("emb_counted", {"embedding_model": True,
                                                             "exclude_from_max_models": False})

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def test_counts(self):
        self.assertEqual(llamaman._count_running_instances(), 2)
        self.assertEqual(instances_api._count_running_chat_instances(), 2)

    def test_eviction_candidates(self):
        ids = {i["id"] for i in llamaman._get_all_evictable_instances()}
        self.assertEqual(ids, {"chat", "emb_counted"})
        ids = {i["id"] for i in instances_api._get_lru_chat_instances()}
        self.assertEqual(ids, {"chat", "emb_counted"})

    def test_incoming_excluded_never_blocked(self):
        with patch.object(instances_api, "LLAMAMAN_MAX_MODELS", 1):
            self.assertTrue(instances_api._would_ui_launch_exceed_limit())
            self.assertFalse(instances_api._would_ui_launch_exceed_limit(incoming_excluded=True))
        with patch.object(llamaman, "LLAMAMAN_MAX_MODELS", 1):
            self.assertTrue(llamaman._evict_llamaman_instances_if_needed(incoming_excluded=True,
                                                                          can_evict_admin=False))


class PresetSaveTests(unittest.TestCase):
    """The stored preset carries the resolved value, so a legacy body (only
    embedding_model) saves both on. Saving also applies it live."""

    def _save(self, body):
        import api.presets as presets_api
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        captured = {}
        with patch.object(presets_api, "get_storage") as gs, \
                patch.object(presets_api, "_apply_live_preset_changes"), \
                patch.object(presets_api, "invalidate_alias_cache"):
            gs.return_value.get_preset.return_value = {}
            gs.return_value.get_all_presets.return_value = {}
            gs.return_value.save_preset.side_effect = lambda path, data: captured.update(data)
            with app.test_client() as c:
                r = c.put("/api/presets/models/m.gguf", json={"ctx_size": 4096, **body})
        self.assertEqual(r.status_code, 200, r.get_json())
        return captured

    def test_legacy_body_saves_both_on(self):
        saved = self._save({"embedding_model": True})
        self.assertTrue(saved["embedding_model"])
        self.assertTrue(saved["exclude_from_max_models"])

    def test_explicit_values(self):
        self.assertFalse(self._save({"embedding_model": True,
                                     "exclude_from_max_models": False})["exclude_from_max_models"])
        self.assertTrue(self._save({"exclude_from_max_models": True})["exclude_from_max_models"])

    def test_live_apply(self):
        import api.presets as presets_api
        with instances_lock:
            saved = {k: dict(v) for k, v in instances.items()}
            instances.clear()
            instances["j"] = _inst("j", {})
        try:
            with patch("proxy.refresh_gate"), patch.object(presets_api, "save_state"):
                presets_api._apply_live_preset_changes("/models/j.gguf",
                                                       {"exclude_from_max_models": True})
            self.assertTrue(instances["j"]["config"]["exclude_from_max_models"])
        finally:
            with instances_lock:
                instances.clear()
                instances.update(saved)


class LaunchConfigTests(unittest.TestCase):
    def test_launch_kwargs_resolve_legacy(self):
        kw = instances_api.launch_kwargs_from_config({"embedding_model": True}, "/models/e.gguf")
        self.assertTrue(kw["exclude_from_max_models"])
        kw = instances_api.launch_kwargs_from_config({"exclude_from_max_models": True}, "/models/j.gguf")
        self.assertTrue(kw["exclude_from_max_models"])
        self.assertFalse(kw["embedding_model"])


if __name__ == "__main__":
    unittest.main()
