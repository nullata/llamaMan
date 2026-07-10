import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))

import api.presets as presets_api
from core.helpers import build_llama_cmd
from core.state import instances, instances_lock


class BuildSpecCmdTests(unittest.TestCase):
    def test_disabled_emits_no_spec_flags(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"spec_enabled": False})
        self.assertNotIn("--spec-type", cmd)
        self.assertNotIn("--model-draft", cmd)

    def test_defaults_to_mtp_without_a_draft_model(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"spec_enabled": True})
        self.assertIn("--spec-type", cmd)
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "draft-mtp")
        self.assertNotIn("--model-draft", cmd)

    def test_dflash_emits_draft_model_and_n_max(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "spec_enabled": True,
            "spec_type": "draft-dflash",
            "spec_draft_model": "/models/drafter/drafter.gguf",
            "spec_draft_n_max": 2,
        })
        self.assertEqual(cmd[cmd.index("--model-draft") + 1], "/models/drafter/drafter.gguf")
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "draft-dflash")
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "2")

    def test_mtp_ignores_a_leftover_draft_model(self):
        # The field is kept in the preset when the user switches type back to
        # MTP; MTP drafts from the main model's heads, so -md must not be sent.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "spec_enabled": True,
            "spec_type": "draft-mtp",
            "spec_draft_model": "/models/drafter/drafter.gguf",
        })
        self.assertNotIn("--model-draft", cmd)


class PresetSpecTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def test_preset_save_includes_spec_fields(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "spec_enabled": True,
                "spec_type": "draft-dflash",
                "spec_draft_model": "/models/drafter/drafter.gguf",
                "spec_draft_n_max": 2,
            })

        self.assertEqual(resp.status_code, 200)
        _, saved_preset = storage.save_preset.call_args.args
        self.assertTrue(saved_preset["spec_enabled"])
        self.assertEqual(saved_preset["spec_type"], "draft-dflash")
        self.assertEqual(saved_preset["spec_draft_model"], "/models/drafter/drafter.gguf")
        self.assertEqual(saved_preset["spec_draft_n_max"], 2)

    def test_preset_save_defaults_spec_type_to_mtp(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf",
                                   json={"ctx_size": 4096, "spec_enabled": True})

        self.assertEqual(resp.status_code, 200)
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["spec_type"], "draft-mtp")

    def test_preset_save_rejects_dflash_without_draft_model(self):
        resp = self.client.put("/api/presets/models/chat.gguf", json={
            "ctx_size": 4096,
            "spec_enabled": True,
            "spec_type": "draft-dflash",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(
            resp.get_json()["error"],
            "spec_draft_model is required when spec_type is draft-dflash",
        )

    def test_preset_save_rejects_unknown_spec_type(self):
        resp = self.client.put("/api/presets/models/chat.gguf",
                               json={"ctx_size": 4096, "spec_type": "eagle"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(
            resp.get_json()["error"],
            "spec_type must be one of: draft-mtp, draft-dflash",
        )

    def test_preset_save_allows_dflash_draft_model_while_spec_is_off(self):
        # Toggling spec off shouldn't make a saved drafter path an error.
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "spec_enabled": False,
                "spec_type": "draft-dflash",
            })
        self.assertEqual(resp.status_code, 200)


class LaunchSpecTests(unittest.TestCase):
    def setUp(self):
        import api.instances as instances_api
        app = Flask(__name__)
        app.register_blueprint(instances_api.bp)
        self.client = app.test_client()
        with instances_lock:
            self._saved_instances = {i: dict(v) for i, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    def test_launch_rejects_dflash_without_draft_model(self):
        resp = self.client.post("/api/instances", json={
            "model_path": "/models/chat.gguf",
            "ctx_size": 4096,
            "spec_enabled": True,
            "spec_type": "draft-dflash",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(
            resp.get_json()["error"],
            "spec_draft_model is required when spec_type is draft-dflash",
        )

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_stores_spec_config(self, _avail, run_container, _save):
        run_container.return_value = (Mock(id="cid"), None)
        with patch("api.instances._admin_ui_enforces_eviction", return_value=False), \
             patch("api.instances._would_ui_launch_exceed_limit", return_value=False):
            resp = self.client.post("/api/instances", json={
                "model_path": "/models/chat.gguf",
                "port": 8000,
                "ctx_size": 4096,
                "spec_enabled": True,
                "spec_type": "draft-dflash",
                "spec_draft_model": "/models/drafter/drafter.gguf",
                "spec_draft_n_max": 2,
            })

        self.assertEqual(resp.status_code, 201)
        config = run_container.call_args.args[4]
        self.assertEqual(config["spec_type"], "draft-dflash")
        self.assertEqual(config["spec_draft_model"], "/models/drafter/drafter.gguf")
        self.assertEqual(config["spec_draft_n_max"], 2)


if __name__ == "__main__":
    unittest.main()
