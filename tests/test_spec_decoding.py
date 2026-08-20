import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.presets as presets_api
from core.helpers import build_llama_cmd
from core.state import instances, instances_lock


class BuildSpecCmdTests(unittest.TestCase):
    def test_disabled_emits_no_spec_flags(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"spec_enabled": False})
        self.assertNotIn("--spec-type", cmd)
        self.assertNotIn("--model-draft", cmd)

    def test_defaults_to_mtp_without_a_draft_model(self):
        # No drafter set: llama-server falls back to the main model's MTP heads.
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

    def test_mtp_emits_a_separate_drafter_when_set(self):
        # A standalone MTP-head GGUF (e.g. Gemma 4's assistant-MTP drafter) is
        # passed as -md alongside --spec-type draft-mtp.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "spec_enabled": True,
            "spec_type": "draft-mtp",
            "spec_draft_model": "/models/gemma-4-12B-assistant-MTP.gguf",
            "spec_draft_n_max": 4,
        })
        self.assertEqual(cmd[cmd.index("--model-draft") + 1],
                         "/models/gemma-4-12B-assistant-MTP.gguf")
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "4")

    def test_spec_disabled_drops_a_saved_draft_model(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "spec_enabled": False,
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

    def test_preset_save_allows_mtp_with_a_draft_model(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "spec_enabled": True,
                "spec_type": "draft-mtp",
                "spec_draft_model": "/models/gemma-4-12B-assistant-MTP.gguf",
            })

        self.assertEqual(resp.status_code, 200)
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["spec_type"], "draft-mtp")
        self.assertEqual(saved_preset["spec_draft_model"],
                         "/models/gemma-4-12B-assistant-MTP.gguf")

    def test_preset_save_allows_mtp_without_a_draft_model(self):
        # Blank drafter stays valid for MTP: built-in heads are the fallback.
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "spec_enabled": True,
                "spec_type": "draft-mtp",
            })
        self.assertEqual(resp.status_code, 200)

    def test_preset_save_rejects_unknown_spec_type(self):
        resp = self.client.put("/api/presets/models/chat.gguf",
                               json={"ctx_size": 4096, "spec_type": "eagle"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(
            resp.get_json()["error"],
            "spec_type must be one of: "
            "draft-simple, draft-mtp, draft-dflash, draft-dspark, draft-eagle3",
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


class ExpandedDraftTypesTests(unittest.TestCase):
    """The Draft Type dropdown grew from 2 values to 5. Every new value must
    round-trip through parse_spec_config, emit --spec-type correctly, and
    (for the four non-MTP types) require a Draft Model - otherwise a valid
    UI selection silently launches without the flag or with an empty -md."""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def test_spec_types_contains_all_five(self):
        from core.spec_decoding import SPEC_TYPES
        self.assertEqual(
            set(SPEC_TYPES),
            {"draft-simple", "draft-mtp", "draft-dflash", "draft-dspark", "draft-eagle3"},
        )

    def test_only_mtp_is_optional_drafter(self):
        from core.spec_decoding import SPEC_TYPES_NEEDING_DRAFT_MODEL
        self.assertEqual(
            SPEC_TYPES_NEEDING_DRAFT_MODEL,
            {"draft-simple", "draft-dflash", "draft-dspark", "draft-eagle3"},
        )

    def test_build_llama_cmd_emits_each_new_draft_type(self):
        for spec_type in ("draft-simple", "draft-dspark", "draft-eagle3"):
            with self.subTest(spec_type=spec_type):
                cmd = build_llama_cmd("/models/m.gguf", 8080, {
                    "spec_enabled": True,
                    "spec_type": spec_type,
                    "spec_draft_model": "/models/drafter/d.gguf",
                })
                self.assertEqual(cmd[cmd.index("--spec-type") + 1], spec_type)
                self.assertEqual(cmd[cmd.index("--model-draft") + 1], "/models/drafter/d.gguf")

    def test_preset_save_rejects_each_new_type_without_draft_model(self):
        # Same guard that previously only fired on draft-dflash must fire on
        # all four "drafter mandatory" types now, otherwise the UI can save
        # a preset that will fail at launch with an unhelpful error.
        for spec_type in ("draft-simple", "draft-dspark", "draft-eagle3"):
            with self.subTest(spec_type=spec_type):
                resp = self.client.put("/api/presets/models/chat.gguf", json={
                    "ctx_size": 4096,
                    "spec_enabled": True,
                    "spec_type": spec_type,
                })
                self.assertEqual(resp.status_code, 400)
                self.assertEqual(
                    resp.get_json()["error"],
                    f"spec_draft_model is required when spec_type is {spec_type}",
                )

    def test_preset_save_rejects_unknown_draft_type(self):
        # A stale preset naming draft-dflash2 (not yet in llama.cpp) or a
        # typo must be rejected, not silently persisted for the next launch
        # to blow up on.
        resp = self.client.put("/api/presets/models/chat.gguf", json={
            "ctx_size": 4096,
            "spec_enabled": True,
            "spec_type": "draft-dflash2",
            "spec_draft_model": "/models/d.gguf",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("spec_type must be one of", resp.get_json()["error"])


class AdvancedSpecDecodingFieldsTests(unittest.TestCase):
    """The Advanced <details> subsection adds three optional numeric knobs:
    spec_draft_n_min / spec_draft_p_split / spec_draft_p_min. Contract: blank
    means "omit the flag entirely" so llama-server uses its own default (which
    drifts across versions - hard-coding a default here would silently
    override future improvements)."""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    # ---- build_llama_cmd ----

    def test_all_advanced_fields_absent_omits_all_flags(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "spec_enabled": True,
            "spec_type": "draft-mtp",
        })
        self.assertNotIn("--spec-draft-n-min", cmd)
        self.assertNotIn("--spec-draft-p-split", cmd)
        self.assertNotIn("--spec-draft-p-min", cmd)

    def test_advanced_fields_emit_when_set(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "spec_enabled": True,
            "spec_type": "draft-mtp",
            "spec_draft_n_min": 1,
            "spec_draft_p_split": 0.5,
            "spec_draft_p_min": 0.1,
        })
        self.assertEqual(cmd[cmd.index("--spec-draft-n-min") + 1], "1")
        # Floats stringify without loss for these values, but we check the
        # flag+value pair rather than exact string form to stay tolerant.
        self.assertIn("--spec-draft-p-split", cmd)
        self.assertEqual(float(cmd[cmd.index("--spec-draft-p-split") + 1]), 0.5)
        self.assertIn("--spec-draft-p-min", cmd)
        self.assertEqual(float(cmd[cmd.index("--spec-draft-p-min") + 1]), 0.1)

    def test_advanced_fields_only_emitted_when_spec_enabled(self):
        # The advanced flags belong to the spec_enabled block - they must not
        # leak out when spec decoding itself is off.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "spec_enabled": False,
            "spec_draft_n_min": 1,
            "spec_draft_p_split": 0.5,
            "spec_draft_p_min": 0.1,
        })
        self.assertNotIn("--spec-draft-n-min", cmd)
        self.assertNotIn("--spec-draft-p-split", cmd)
        self.assertNotIn("--spec-draft-p-min", cmd)

    def test_advanced_field_zero_is_a_real_value(self):
        # 0 is a valid value for these knobs and must reach llama-server,
        # not be dropped as falsy. Distinguishes "not set" (None) from
        # "explicitly zero".
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "spec_enabled": True,
            "spec_type": "draft-mtp",
            "spec_draft_p_min": 0,
        })
        self.assertIn("--spec-draft-p-min", cmd)
        self.assertEqual(float(cmd[cmd.index("--spec-draft-p-min") + 1]), 0.0)

    # ---- parse_spec_config validation ----

    def test_preset_save_persists_advanced_fields(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "spec_enabled": True,
                "spec_type": "draft-mtp",
                "spec_draft_n_min": 2,
                "spec_draft_p_split": 0.35,
                "spec_draft_p_min": 0.05,
            })
        self.assertEqual(resp.status_code, 200)
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["spec_draft_n_min"], 2)
        self.assertEqual(saved_preset["spec_draft_p_split"], 0.35)
        self.assertEqual(saved_preset["spec_draft_p_min"], 0.05)

    def test_preset_save_stores_none_when_advanced_fields_blank(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "spec_enabled": True,
                "spec_type": "draft-mtp",
            })
        _, saved_preset = storage.save_preset.call_args.args
        self.assertIsNone(saved_preset["spec_draft_n_min"])
        self.assertIsNone(saved_preset["spec_draft_p_split"])
        self.assertIsNone(saved_preset["spec_draft_p_min"])

    def test_preset_save_rejects_negative_n_min(self):
        resp = self.client.put("/api/presets/models/chat.gguf", json={
            "ctx_size": 4096,
            "spec_enabled": True,
            "spec_type": "draft-mtp",
            "spec_draft_n_min": -1,
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("spec_draft_n_min", resp.get_json()["error"])

    def test_preset_save_rejects_p_split_out_of_range(self):
        for bad in (-0.1, 1.5):
            with self.subTest(value=bad):
                resp = self.client.put("/api/presets/models/chat.gguf", json={
                    "ctx_size": 4096,
                    "spec_enabled": True,
                    "spec_type": "draft-mtp",
                    "spec_draft_p_split": bad,
                })
                self.assertEqual(resp.status_code, 400)
                self.assertIn("spec_draft_p_split", resp.get_json()["error"])

    def test_preset_save_rejects_non_numeric_p_min(self):
        resp = self.client.put("/api/presets/models/chat.gguf", json={
            "ctx_size": 4096,
            "spec_enabled": True,
            "spec_type": "draft-mtp",
            "spec_draft_p_min": "not-a-number",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("spec_draft_p_min", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
