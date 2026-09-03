# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the --load-mode launch control: normalize_load_mode helper,
build_llama_cmd flag emission, launch_instance / create-route / restart-route
wiring, preset persistence, and the preset-merge whitelist.

--load-mode is llama.cpp's six-value knob (auto|none|mmap|mlock|mmap+mlock|dio,
default auto), the successor to the deprecated --mlock / --mmap / --no-mmap /
--direct-io flags. 'auto' matches llama.cpp's own default so we omit the flag
for the common case; unknown / missing values fold to 'auto' so a corrupt
preset can never make llama-server refuse to start with an opaque error
(its arg parser throws on invalid values)."""

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
from core.helpers import build_llama_cmd, normalize_load_mode
from core.state import instances, instances_lock

ALL_VALUES = ("auto", "none", "mmap", "mlock", "mmap+mlock", "dio")


class NormalizeLoadModeTests(unittest.TestCase):
    """The helper is the single source of truth for load_mode coercion; every
    boundary (route, launch_instance, preset save, build_llama_cmd) routes
    through it."""

    def test_all_llamacpp_values_pass_through(self):
        for v in ALL_VALUES:
            self.assertEqual(normalize_load_mode(v), v)

    def test_case_and_whitespace_tolerant(self):
        self.assertEqual(normalize_load_mode("  AUTO "), "auto")
        self.assertEqual(normalize_load_mode("MMAP+MLOCK"), "mmap+mlock")
        self.assertEqual(normalize_load_mode(" Dio "), "dio")

    def test_empty_and_unknown_default_to_auto(self):
        # Anything llama-server wouldn't accept becomes 'auto' - the safe
        # default that matches its own behavior when the flag is absent.
        self.assertEqual(normalize_load_mode(None), "auto")
        self.assertEqual(normalize_load_mode(""), "auto")
        self.assertEqual(normalize_load_mode("garbage"), "auto")
        self.assertEqual(normalize_load_mode("read"), "auto")  # not a llama.cpp value
        self.assertEqual(normalize_load_mode("mlock+mmap"), "auto")  # wrong order
        self.assertEqual(normalize_load_mode(42), "auto")
        self.assertEqual(normalize_load_mode(True), "auto")


class BuildLlamaCmdLoadModeTests(unittest.TestCase):
    """build_llama_cmd emits --load-mode only for the five non-default values;
    'auto' (and missing / empty / corrupt) omits the flag entirely to match
    llama.cpp's own default and keep the command line quiet."""

    def test_missing_omits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {})
        self.assertNotIn("--load-mode", cmd)

    def test_auto_omits_flag(self):
        # 'auto' IS llama.cpp's own default when the flag is absent; emitting
        # it would be noise for the common case.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"load_mode": "auto"})
        self.assertNotIn("--load-mode", cmd)

    def test_each_non_default_value_emits_flag(self):
        for v in ALL_VALUES:
            if v == "auto":
                continue
            cmd = build_llama_cmd("/models/m.gguf", 8080, {"load_mode": v})
            self.assertEqual(cmd[cmd.index("--load-mode") + 1], v)

    def test_case_and_whitespace_are_normalized(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"load_mode": "  MMap+MLock "})
        self.assertEqual(cmd[cmd.index("--load-mode") + 1], "mmap+mlock")

    def test_unknown_value_is_dropped(self):
        # A corrupt preset or hand-crafted request could send a value
        # llama-server would refuse (its parser throws) - drop the flag
        # instead of shipping it and letting the container die at startup.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"load_mode": "garbage"})
        self.assertNotIn("--load-mode", cmd)


class LaunchInstanceLoadModeTests(unittest.TestCase):
    """launch_instance() must copy load_mode into inst["config"] with
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
            load_mode="mlock",
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["load_mode"], "mlock")

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
        self.assertEqual(inst["config"]["load_mode"], "auto")

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
            load_mode="  MMap+MLock  ",
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["load_mode"], "mmap+mlock")

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_folds_unknown_to_auto(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            load_mode="garbage",
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["load_mode"], "auto")


class InstancesCreateRouteLoadModeTests(unittest.TestCase):
    """POST /api/instances must lift load_mode out of the JSON body and
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
                    "load_mode": "dio",
                },
            )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(launch_mock.call_args.kwargs["load_mode"], "dio")

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
        self.assertEqual(launch_mock.call_args.kwargs["load_mode"], "auto")


class PresetLoadModeTests(unittest.TestCase):
    """load_mode is a shared preset field (a behavior knob, not topology), so
    it belongs in the main preset body rather than in PRESET_HARDWARE_KEYS -
    matching flash_attn / reasoning_format, which are also shared knobs."""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def test_load_mode_is_NOT_in_hardware_keys(self):
        # If it lands in PRESET_HARDWARE_KEYS by accident, the cluster path
        # silently overlays it per-node - a preset saved on one node would
        # set an override for that node alone.
        self.assertNotIn("load_mode", presets_api.PRESET_HARDWARE_KEYS)

    def test_preset_save_persists_field(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "load_mode": "mlock",
            })
        self.assertEqual(resp.status_code, 200)
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["load_mode"], "mlock")

    def test_preset_save_normalizes_case(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "load_mode": " MMap+MLock ",
            })
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["load_mode"], "mmap+mlock")

    def test_preset_save_default_is_auto(self):
        # A save that doesn't mention the key must still write it as 'auto' -
        # a subsequent save from an older form would otherwise leave a stale
        # value behind (same invariant as flash_attn / reasoning_format).
        storage = Mock()
        storage.get_preset.return_value = {"load_mode": "mlock"}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={"ctx_size": 4096})
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["load_mode"], "auto")


class MergePresetLoadModeTests(unittest.TestCase):
    """The preset-merge whitelist must carry load_mode, otherwise an
    auto-launch / restart merge silently drops the operator's saved choice."""

    def test_load_mode_in_merge_whitelist(self):
        import inspect
        src = inspect.getsource(instances_api._merge_preset_into_config)
        self.assertIn('"load_mode"', src)

    def test_merge_overlays_preset_value(self):
        storage = Mock()
        storage.get_preset.return_value = {"load_mode": "mmap"}
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config(
                "/models/chat.gguf", {"load_mode": "auto", "ctx_size": 4096},
            )
        self.assertEqual(merged["load_mode"], "mmap")


if __name__ == "__main__":
    unittest.main()
