# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the DRY (Don't Repeat Yourself) sampler wiring: parse_dry_config
boundary validation, build_llama_cmd emission, launch_instance / restart route
wiring, and preset persistence.

DRY is llama.cpp's sampling-time anti-repeat and the soft first line of
defense against output loops (the hard second line is the output loop
detector in core/loop_detect.py). Off by default because a non-zero
multiplier changes sampling for every request; users opt in per preset."""

import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.instances as instances_api
import api.presets as presets_api
from core.dry_sampling import (
    DRY_SAMPLER_KEYS, dry_enabled, parse_dry_config,
)
from core.helpers import build_llama_cmd
from core.state import instances, instances_lock


class ParseDryConfigTests(unittest.TestCase):
    """parse_dry_config is the single boundary validator - every API caller
    routes through it, so misbehavior here silently corrupts every downstream
    read. Pin the contract explicitly."""

    def test_empty_body_gives_zero_disabled_defaults(self):
        cfg, err = parse_dry_config({})
        self.assertIsNone(err)
        self.assertFalse(cfg["dry_enabled"])
        self.assertEqual(cfg["dry_multiplier"], 0.0)
        self.assertEqual(cfg["dry_base"], 1.75)
        self.assertEqual(cfg["dry_allowed_length"], 2)
        self.assertIsNone(cfg["dry_penalty_last_n"])

    def test_typical_enabled_values_pass_through(self):
        cfg, err = parse_dry_config({
            "dry_enabled": True,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
            "dry_penalty_last_n": 1024,
        })
        self.assertIsNone(err)
        self.assertTrue(cfg["dry_enabled"])
        self.assertEqual(cfg["dry_multiplier"], 0.8)
        self.assertEqual(cfg["dry_penalty_last_n"], 1024)

    def test_multiplier_negative_rejected(self):
        cfg, err = parse_dry_config({"dry_multiplier": -0.1})
        self.assertEqual(cfg, {})
        self.assertIn("dry_multiplier", err)

    def test_multiplier_over_cap_rejected(self):
        cfg, err = parse_dry_config({"dry_multiplier": 100.0})
        self.assertEqual(cfg, {})
        self.assertIn("dry_multiplier", err)

    def test_multiplier_non_numeric_rejected(self):
        cfg, err = parse_dry_config({"dry_multiplier": "high"})
        self.assertEqual(cfg, {})
        self.assertIn("dry_multiplier", err)

    def test_base_below_one_rejected(self):
        # llama.cpp silently ignores dry_base < 1.0 - reject at the boundary
        # so the user sees an error instead of a silent no-op at runtime.
        cfg, err = parse_dry_config({"dry_base": 0.5})
        self.assertEqual(cfg, {})
        self.assertIn("dry_base", err)

    def test_allowed_length_negative_rejected(self):
        cfg, err = parse_dry_config({"dry_allowed_length": -1})
        self.assertEqual(cfg, {})
        self.assertIn("dry_allowed_length", err)

    def test_penalty_last_n_negative_rejected(self):
        # llama.cpp's arg parser THROWS on negative user input - a silent
        # forward here would make the container die at startup.
        cfg, err = parse_dry_config({"dry_penalty_last_n": -1})
        self.assertEqual(cfg, {})
        self.assertIn("dry_penalty_last_n", err)

    def test_penalty_last_n_zero_is_valid(self):
        # 0 explicitly disables the penalty history per llama.cpp's help text.
        cfg, err = parse_dry_config({"dry_penalty_last_n": 0})
        self.assertIsNone(err)
        self.assertEqual(cfg["dry_penalty_last_n"], 0)

    def test_penalty_last_n_empty_string_folds_to_none(self):
        # Empty from a form field should mean "leave llama.cpp's default in
        # place" (the flag is omitted), same "empty = auto" contract used by
        # spec-decoding's advanced knobs.
        cfg, err = parse_dry_config({"dry_penalty_last_n": ""})
        self.assertIsNone(err)
        self.assertIsNone(cfg["dry_penalty_last_n"])


class DryEnabledHelperTests(unittest.TestCase):
    """dry_enabled() gates flag emission and must combine the toggle AND the
    multiplier > 0 check - either alone is a footgun.
    """

    def test_none_config_is_disabled(self):
        self.assertFalse(dry_enabled(None))

    def test_empty_config_is_disabled(self):
        self.assertFalse(dry_enabled({}))

    def test_toggle_off_is_disabled_even_with_multiplier(self):
        self.assertFalse(dry_enabled({"dry_enabled": False, "dry_multiplier": 0.8}))

    def test_toggle_on_but_multiplier_zero_is_disabled(self):
        # llama.cpp treats multiplier=0 as off, so emitting the flags with a
        # zero multiplier would be a noisy no-op.
        self.assertFalse(dry_enabled({"dry_enabled": True, "dry_multiplier": 0.0}))

    def test_toggle_on_and_multiplier_positive_is_enabled(self):
        self.assertTrue(dry_enabled({"dry_enabled": True, "dry_multiplier": 0.8}))

    def test_garbage_multiplier_is_disabled(self):
        self.assertFalse(dry_enabled({"dry_enabled": True, "dry_multiplier": "high"}))


class BuildLlamaCmdDryTests(unittest.TestCase):
    """build_llama_cmd emits DRY flags only when dry_enabled() is true. The
    companion flags (--dry-base / --dry-allowed-length / --dry-penalty-last-n)
    only ride along with --dry-multiplier - never on their own - so an
    off-toggle preset produces the exact same command line as before this
    feature."""

    def test_default_config_omits_all_dry_flags(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {})
        for f in ("--dry-multiplier", "--dry-base", "--dry-allowed-length", "--dry-penalty-last-n"):
            self.assertNotIn(f, cmd, f"unexpected {f} in default cmd")

    def test_toggle_off_omits_all_flags(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "dry_enabled": False,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
        })
        self.assertNotIn("--dry-multiplier", cmd)

    def test_toggle_on_but_zero_multiplier_omits(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "dry_enabled": True,
            "dry_multiplier": 0.0,
        })
        self.assertNotIn("--dry-multiplier", cmd)

    def test_typical_enabled_emits_multiplier_and_defaults(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "dry_enabled": True,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
        })
        self.assertEqual(cmd[cmd.index("--dry-multiplier") + 1], "0.8")
        self.assertEqual(cmd[cmd.index("--dry-base") + 1], "1.75")
        self.assertEqual(cmd[cmd.index("--dry-allowed-length") + 1], "2")
        # penalty_last_n omitted from the config -> flag omitted from cmd.
        self.assertNotIn("--dry-penalty-last-n", cmd)

    def test_penalty_last_n_explicit_zero_emits(self):
        # 0 disables the penalty window per llama.cpp's help text - it's a
        # real value distinct from "not set", so it must be emitted.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "dry_enabled": True,
            "dry_multiplier": 0.8,
            "dry_penalty_last_n": 0,
        })
        self.assertEqual(cmd[cmd.index("--dry-penalty-last-n") + 1], "0")

    def test_negative_penalty_last_n_from_corrupt_preset_is_dropped(self):
        # A hand-crafted API request or corrupt storage could carry a
        # negative value that llama.cpp would refuse at startup. The safe
        # fallback is to drop the flag rather than ship it - the emission
        # path is a defense in depth on top of parse_dry_config.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "dry_enabled": True,
            "dry_multiplier": 0.8,
            "dry_penalty_last_n": -5,
        })
        self.assertNotIn("--dry-penalty-last-n", cmd)


class LaunchInstanceDryFieldsTests(unittest.TestCase):
    """launch_instance() must copy the five DRY kwargs into inst["config"]
    with normalized numeric types so the running instance's config is
    authoritative for later reads - restart, cluster snapshot, live preset
    merge."""

    def setUp(self):
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_stores_dry_fields(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            dry_enabled=True,
            dry_multiplier=0.8,
            dry_base=1.75,
            dry_allowed_length=2,
            dry_penalty_last_n=1024,
        )
        self.assertIsNone(err)
        cfg = inst["config"]
        self.assertTrue(cfg["dry_enabled"])
        self.assertEqual(cfg["dry_multiplier"], 0.8)
        self.assertEqual(cfg["dry_base"], 1.75)
        self.assertEqual(cfg["dry_allowed_length"], 2)
        self.assertEqual(cfg["dry_penalty_last_n"], 1024)

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_defaults_are_disabled(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        # Callers that don't mention DRY still get all five keys in config at
        # their disabled defaults, so downstream reads never KeyError - same
        # invariant as flash_attn / cache types.
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
        )
        self.assertIsNone(err)
        cfg = inst["config"]
        self.assertFalse(cfg["dry_enabled"])
        self.assertEqual(cfg["dry_multiplier"], 0.0)
        self.assertEqual(cfg["dry_base"], 1.75)
        self.assertEqual(cfg["dry_allowed_length"], 2)
        self.assertIsNone(cfg["dry_penalty_last_n"])


class InstancesCreateRouteDryTests(unittest.TestCase):
    """POST /api/instances must lift the five DRY fields out of the JSON body
    and forward them as kwargs. If the route drops any, the user's UI choice
    vanishes silently at launch."""

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(instances_api.bp)
        self.client = self.app.test_client()
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    @patch("api.instances.launch_instance")
    def test_create_route_forwards_dry_fields(self, launch_mock):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "ctx_size": 4096,
                    "dry_enabled": True,
                    "dry_multiplier": 0.8,
                    "dry_base": 2.0,
                    "dry_allowed_length": 3,
                    "dry_penalty_last_n": 512,
                },
            )
        self.assertEqual(resp.status_code, 201)
        kw = launch_mock.call_args.kwargs
        self.assertTrue(kw["dry_enabled"])
        self.assertEqual(kw["dry_multiplier"], 0.8)
        self.assertEqual(kw["dry_base"], 2.0)
        self.assertEqual(kw["dry_allowed_length"], 3)
        self.assertEqual(kw["dry_penalty_last_n"], 512)

    @patch("api.instances.launch_instance")
    def test_create_route_rejects_bad_dry_at_boundary(self, launch_mock):
        # A boundary-invalid value must 400 before launch_instance is called,
        # so the user sees a clear error rather than a container crash.
        launch_mock.return_value = ({"id": "inst-1"}, None)
        resp = self.client.post(
            "/api/instances",
            json={
                "model_path": "/models/chat.gguf",
                "port": 8000,
                "ctx_size": 4096,
                "dry_multiplier": -0.5,
            },
        )
        self.assertEqual(resp.status_code, 400)
        launch_mock.assert_not_called()


class PresetDryTests(unittest.TestCase):
    """DRY fields are shared preset fields (behavior knobs, not topology),
    matching flash_attn / cache types / reasoning_format - not in
    PRESET_HARDWARE_KEYS. This lets the preset carry cluster-wide DRY
    settings without a per-node override."""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def test_dry_keys_are_NOT_in_hardware_keys(self):
        for key in DRY_SAMPLER_KEYS:
            self.assertNotIn(
                key, presets_api.PRESET_HARDWARE_KEYS,
                f"{key} accidentally landed in PRESET_HARDWARE_KEYS - it would "
                "then be silently overlaid per-node instead of shared "
                "cluster-wide"
            )

    def test_preset_save_persists_dry_fields(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "dry_enabled": True,
                "dry_multiplier": 0.8,
                "dry_base": 1.75,
                "dry_allowed_length": 2,
                "dry_penalty_last_n": 1024,
            })
        self.assertEqual(resp.status_code, 200)
        _, saved = storage.save_preset.call_args.args
        self.assertTrue(saved["dry_enabled"])
        self.assertEqual(saved["dry_multiplier"], 0.8)
        self.assertEqual(saved["dry_penalty_last_n"], 1024)

    def test_preset_save_default_is_disabled(self):
        # A save that doesn't mention DRY must still write it as disabled -
        # a subsequent save from an older form would otherwise leave a stale
        # value behind (same invariant as flash_attn).
        storage = Mock()
        storage.get_preset.return_value = {
            "dry_enabled": True,
            "dry_multiplier": 0.8,
        }
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={"ctx_size": 4096})
        _, saved = storage.save_preset.call_args.args
        self.assertFalse(saved["dry_enabled"])
        self.assertEqual(saved["dry_multiplier"], 0.0)

    def test_preset_save_rejects_bad_dry_at_boundary(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "dry_base": 0.1,
            })
        self.assertEqual(resp.status_code, 400)


class InstancesRestartRouteDryTests(unittest.TestCase):
    """POST /api/instances/<id>/restart forwards the stopped instance's DRY
    config back into launch_instance. Missing any of the five would mean a
    restart silently reverts to defaults - the feature "forgets" itself on
    any restart triggered for an unrelated reason."""

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(instances_api.bp)
        self.client = self.app.test_client()
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    @patch("api.instances.save_state")
    @patch("api.instances.release_instance_reservations")
    @patch("api.instances.is_port_available", return_value=True)
    @patch("api.instances._admin_ui_enforces_eviction", return_value=False)
    @patch("api.instances._would_ui_launch_exceed_limit", return_value=False)
    @patch("api.instances._merge_preset_into_config", side_effect=lambda _p, cfg: cfg)
    @patch("api.instances.launch_instance")
    def test_restart_route_preserves_dry_fields(
        self, launch_mock, _merge_mock, _would_exceed_mock, _admin_mock,
        _is_port_mock, _release_mock, _save_state_mock,
    ):
        launch_mock.return_value = ({"id": "inst-1"}, None)

        with instances_lock:
            instances["inst-1"] = {
                "id": "inst-1",
                "model_path": "/models/chat.gguf",
                "port": 8000,
                "status": "stopped",
                "stats": {},
                "config": {
                    "n_gpu_layers": -1,
                    "ctx_size": 4096,
                    "dry_enabled": True,
                    "dry_multiplier": 0.8,
                    "dry_base": 1.75,
                    "dry_allowed_length": 2,
                    "dry_penalty_last_n": 1024,
                },
            }

        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post("/api/instances/inst-1/restart", json={})

        self.assertIn(resp.status_code, (200, 201))
        kw = launch_mock.call_args.kwargs
        self.assertTrue(kw["dry_enabled"])
        self.assertEqual(kw["dry_multiplier"], 0.8)
        self.assertEqual(kw["dry_penalty_last_n"], 1024)


class MergePresetIntoConfigDryTests(unittest.TestCase):
    """_merge_preset_into_config is the "live preset apply" path. DRY is
    baked into the container at launch (a sampler flag, not something the
    proxy can retro-apply), so live-merging it only affects the NEXT relaunch
    - but it still has to reach the merged config so restart picks it up."""

    def test_merge_preset_overlays_dry_fields(self):
        base_config = {"n_gpu_layers": -1, "ctx_size": 4096}
        preset = {
            "dry_enabled": True,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
            "dry_penalty_last_n": 1024,
        }
        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)
        self.assertTrue(merged["dry_enabled"])
        self.assertEqual(merged["dry_multiplier"], 0.8)
        self.assertEqual(merged["dry_penalty_last_n"], 1024)


if __name__ == "__main__":
    unittest.main()
