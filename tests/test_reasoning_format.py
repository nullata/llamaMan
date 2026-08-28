# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the --reasoning-format launch control: normalize_reasoning_format
helper, build_llama_cmd flag emission, launch_instance / restart route wiring,
preset persistence, and the preset-merge whitelist that lets a live preset
edit reach a running instance without a relaunch.

--reasoning-format is llama.cpp's four-value knob (none|auto|deepseek|
deepseek-legacy, default auto). 'auto' matches llama.cpp's own default so we
omit the flag for the common case; unknown / missing values fold to 'auto' so
a corrupt preset or hand-crafted request can never make llama-server refuse to
start with an opaque error."""

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
from core.helpers import build_llama_cmd, normalize_reasoning_format
from core.state import instances, instances_lock


class NormalizeReasoningFormatTests(unittest.TestCase):
    """The helper is the single source of truth for reasoning_format coercion;
    every boundary (route, launch_instance, preset save, build_llama_cmd)
    routes through it. Pinning it here catches drift that would otherwise
    only surface as a wrong CLI arg buried in a launch."""

    def test_all_llamacpp_values_pass_through(self):
        for v in ("none", "auto", "deepseek", "deepseek-legacy"):
            self.assertEqual(normalize_reasoning_format(v), v)

    def test_case_and_whitespace_tolerant(self):
        self.assertEqual(normalize_reasoning_format("  AUTO "), "auto")
        self.assertEqual(normalize_reasoning_format("DeepSeek"), "deepseek")
        self.assertEqual(normalize_reasoning_format(" Deepseek-Legacy "), "deepseek-legacy")

    def test_empty_and_unknown_default_to_auto(self):
        # Anything llama-server wouldn't accept becomes 'auto' - the safe
        # default that matches its own behavior when the flag is absent.
        self.assertEqual(normalize_reasoning_format(None), "auto")
        self.assertEqual(normalize_reasoning_format(""), "auto")
        self.assertEqual(normalize_reasoning_format("garbage"), "auto")
        self.assertEqual(normalize_reasoning_format("deepseek-r2"), "auto")
        self.assertEqual(normalize_reasoning_format(42), "auto")
        self.assertEqual(normalize_reasoning_format(True), "auto")


class BuildLlamaCmdReasoningFormatTests(unittest.TestCase):
    """build_llama_cmd emits --reasoning-format only for the three non-default
    values. 'auto' (and missing / empty) omits the flag entirely to match
    llama.cpp's own default and keep the command line quiet."""

    def test_missing_omits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {})
        self.assertNotIn("--reasoning-format", cmd)

    def test_auto_omits_flag(self):
        # 'auto' IS llama.cpp's own default when the flag is absent; emitting
        # it would be noise for the common case.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"reasoning_format": "auto"})
        self.assertNotIn("--reasoning-format", cmd)

    def test_none_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"reasoning_format": "none"})
        self.assertEqual(cmd[cmd.index("--reasoning-format") + 1], "none")

    def test_deepseek_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"reasoning_format": "deepseek"})
        self.assertEqual(cmd[cmd.index("--reasoning-format") + 1], "deepseek")

    def test_deepseek_legacy_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"reasoning_format": "deepseek-legacy"})
        self.assertEqual(cmd[cmd.index("--reasoning-format") + 1], "deepseek-legacy")

    def test_case_and_whitespace_are_normalized(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"reasoning_format": "  DeepSeek "})
        self.assertEqual(cmd[cmd.index("--reasoning-format") + 1], "deepseek")

    def test_unknown_value_is_dropped(self):
        # A corrupt preset or hand-crafted request could send a value
        # llama-server would refuse - drop the flag instead of shipping it
        # and letting the container die with an opaque error.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"reasoning_format": "garbage"})
        self.assertNotIn("--reasoning-format", cmd)


class LaunchInstanceReasoningFormatTests(unittest.TestCase):
    """launch_instance() must copy reasoning_format into inst["config"] with
    normalization, so the running instance's config is authoritative for later
    reads - restart, cluster snapshot, live preset merge."""

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
    def test_launch_instance_stores_field(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            reasoning_format="deepseek",
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["reasoning_format"], "deepseek")

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_default_is_auto(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        # Callers that don't mention the new kwarg still get the key in
        # config at 'auto', so downstream reads never KeyError.
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["reasoning_format"], "auto")

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_normalizes_case(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            reasoning_format="  DeepSeek-Legacy  ",
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["reasoning_format"], "deepseek-legacy")

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_folds_unknown_to_auto(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        # A restart / adopt path handing back a garbage value must not
        # propagate it into the running config - fold to 'auto' at the
        # boundary so downstream stays clean.
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            reasoning_format="garbage",
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["reasoning_format"], "auto")


class InstancesCreateRouteReasoningFormatTests(unittest.TestCase):
    """POST /api/instances must lift reasoning_format out of the JSON body and
    forward it as a kwarg. If the route drops it, the user's UI choice
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
    def test_create_route_forwards_field(self, launch_mock):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "ctx_size": 4096,
                    "reasoning_format": "deepseek",
                },
            )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(launch_mock.call_args.kwargs["reasoning_format"], "deepseek")

    @patch("api.instances.launch_instance")
    def test_create_route_default_omitted(self, launch_mock):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "ctx_size": 4096,
                },
            )
        # No body value -> the default 'auto' (matching llama.cpp).
        self.assertEqual(launch_mock.call_args.kwargs["reasoning_format"], "auto")


class PresetReasoningFormatTests(unittest.TestCase):
    """reasoning_format is a shared preset field (behavior knob, not topology),
    so it belongs in the main preset body rather than in PRESET_HARDWARE_KEYS -
    matching flash_attn / cache types, which are also shared knobs."""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def test_reasoning_format_is_NOT_in_hardware_keys(self):
        # If it lands in PRESET_HARDWARE_KEYS by accident, the cluster path
        # silently overlays it per-node - so a preset saved on one node would
        # set an override for that node alone and other nodes would keep the
        # shared base indefinitely.
        self.assertNotIn("reasoning_format", presets_api.PRESET_HARDWARE_KEYS)

    def test_preset_save_persists_field(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "reasoning_format": "deepseek",
            })
        self.assertEqual(resp.status_code, 200)
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["reasoning_format"], "deepseek")

    def test_preset_save_normalizes_case(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "reasoning_format": " DeepSeek-Legacy ",
            })
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["reasoning_format"], "deepseek-legacy")

    def test_preset_save_default_is_auto(self):
        # A save that doesn't mention the key must still write it as 'auto' -
        # a subsequent save from an older form would otherwise leave a stale
        # value behind (same invariant as flash_attn / cache types).
        storage = Mock()
        storage.get_preset.return_value = {"reasoning_format": "deepseek"}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={"ctx_size": 4096})
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["reasoning_format"], "auto")


class InstancesRestartRouteReasoningFormatTests(unittest.TestCase):
    """POST /api/instances/<id>/restart forwards the stopped instance's
    reasoning_format back into launch_instance. Missing it would mean a
    restart silently reverts to 'auto' - the feature "forgets" itself the
    first time an operator stops and starts an instance to apply an
    unrelated change."""

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
    def test_restart_route_preserves_field(
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
                    "reasoning_format": "deepseek",
                },
            }

        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post("/api/instances/inst-1/restart", json={})

        self.assertIn(resp.status_code, (200, 201))
        self.assertEqual(launch_mock.call_args.kwargs["reasoning_format"], "deepseek")


class MergePresetIntoConfigReasoningFormatTests(unittest.TestCase):
    """_merge_preset_into_config is the "live preset apply" path - if
    reasoning_format isn't in the whitelist, a preset edit silently doesn't
    take effect for it, and the user thinks it did."""

    def test_merge_preset_overlays_field(self):
        base_config = {"n_gpu_layers": -1, "ctx_size": 4096}
        preset = {"reasoning_format": "deepseek"}

        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)

        self.assertEqual(merged["reasoning_format"], "deepseek")

    def test_merge_preset_leaves_base_when_preset_omits_field(self):
        # A pre-feature preset has no key. The merged config must keep the
        # base's value rather than clobbering with a default, which would
        # silently reset reasoning behavior on the next reap for a running
        # instance that was launched with it set.
        base_config = {"reasoning_format": "deepseek"}
        preset = {"ctx_size": 4096}  # pre-feature preset

        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)

        self.assertEqual(merged["reasoning_format"], "deepseek")


if __name__ == "__main__":
    unittest.main()
