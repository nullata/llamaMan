# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the --threads-batch launch control: build_llama_cmd flag
emission, launch_instance / restart route wiring, and preset persistence.

--threads-batch is llama.cpp's separate thread pool for batch and prompt
processing (the prefill / evaluation phase). When absent, llama-server falls
back to the --threads value on its own, so a blank UI field means we omit
the flag - never second-guess by mirroring --threads server-side."""

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
from core.helpers import build_llama_cmd
from core.state import instances, instances_lock


class BuildLlamaCmdThreadsBatchTests(unittest.TestCase):
    """--threads-batch is only emitted when set - a blank value means
    llama-server uses the --threads value (which itself may be omitted so
    llama-server auto-detects). We never re-emit --threads' value to match
    llama.cpp's own fallback."""

    def test_missing_omits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {})
        self.assertNotIn("--threads-batch", cmd)

    def test_none_omits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"threads_batch": None})
        self.assertNotIn("--threads-batch", cmd)

    def test_zero_omits_flag(self):
        # 0 is not a valid --threads-batch value and would come from a
        # cleared UI field being coerced to 0. Truthiness-check drops it.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"threads_batch": 0})
        self.assertNotIn("--threads-batch", cmd)

    def test_empty_string_omits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"threads_batch": ""})
        self.assertNotIn("--threads-batch", cmd)

    def test_integer_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"threads_batch": 16})
        self.assertEqual(cmd[cmd.index("--threads-batch") + 1], "16")

    def test_string_integer_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"threads_batch": "8"})
        self.assertEqual(cmd[cmd.index("--threads-batch") + 1], "8")

    def test_independent_from_threads(self):
        # The two flags are separately controlled - setting one must never
        # imply the other. This pins the "no cross-emission" behavior.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"threads": 4, "threads_batch": 12})
        self.assertEqual(cmd[cmd.index("--threads") + 1], "4")
        self.assertEqual(cmd[cmd.index("--threads-batch") + 1], "12")

        cmd = build_llama_cmd("/models/m.gguf", 8080, {"threads": 4})
        self.assertNotIn("--threads-batch", cmd)


class LaunchInstanceThreadsBatchTests(unittest.TestCase):
    """launch_instance() must copy threads_batch into inst["config"]
    unchanged (no normalization needed - it's an integer or None), so the
    running instance's config is authoritative for restart / cluster
    snapshot / live preset merge."""

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
            threads_batch=16,
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["threads_batch"], 16)

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_default_is_none(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        # Callers that don't mention the new kwarg still get the key in
        # config at None (== "omit flag, fall back to --threads value"),
        # so downstream reads never KeyError.
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
        )

        self.assertIsNone(err)
        self.assertIsNone(inst["config"]["threads_batch"])


class InstancesCreateRouteThreadsBatchTests(unittest.TestCase):
    """POST /api/instances must lift threads_batch out of the JSON body and
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
                    "threads_batch": 12,
                },
            )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(launch_mock.call_args.kwargs["threads_batch"], 12)

    @patch("api.instances.launch_instance")
    def test_create_route_default_omitted_forwards_none(self, launch_mock):
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
        self.assertIsNone(launch_mock.call_args.kwargs["threads_batch"])


class PresetThreadsBatchTests(unittest.TestCase):
    """threads_batch is per-node HARDWARE (CPU-core count varies by node) -
    it belongs in PRESET_HARDWARE_KEYS alongside threads. This lets a shared
    cluster preset carry a base value while each node can override with its
    own core count via node_overrides."""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def test_threads_batch_is_in_hardware_keys(self):
        # If it accidentally leaves this list, a node override would silently
        # be ignored and every node would use the base value - which is wrong
        # for machines with different CPU topologies.
        self.assertIn("threads_batch", presets_api.PRESET_HARDWARE_KEYS)

    def test_preset_save_persists_field(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "threads_batch": 16,
            })
        self.assertEqual(resp.status_code, 200)
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["threads_batch"], 16)

    def test_preset_save_default_is_none(self):
        # A save that doesn't mention the key must still write it as None -
        # a subsequent save from an older form would otherwise leave a stale
        # value behind (same invariant as threads).
        storage = Mock()
        storage.get_preset.return_value = {"threads_batch": 16}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={"ctx_size": 4096})
        _, saved_preset = storage.save_preset.call_args.args
        self.assertIsNone(saved_preset["threads_batch"])

    def test_node_override_overlays_base(self):
        # Two nodes with different CPU-core counts: the base preset carries a
        # cluster-wide default, and each node's override can supersede it.
        preset = {
            "threads_batch": 8,  # cluster-wide base
            "node_overrides": {"beefy": {"threads_batch": 32}},
        }
        merged = presets_api.resolve_preset_for_node(preset, "beefy")
        self.assertEqual(merged["threads_batch"], 32)

        # Other nodes see the base
        merged = presets_api.resolve_preset_for_node(preset, "small")
        self.assertEqual(merged["threads_batch"], 8)


class InstancesRestartRouteThreadsBatchTests(unittest.TestCase):
    """POST /api/instances/<id>/restart forwards the stopped instance's
    threads_batch back into launch_instance. Missing it would mean a restart
    silently reverts to None (== --threads fallback) - the feature "forgets"
    itself on any restart triggered for an unrelated reason."""

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
                    "threads": 4,
                    "threads_batch": 16,
                },
            }

        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post("/api/instances/inst-1/restart", json={})

        self.assertIn(resp.status_code, (200, 201))
        self.assertEqual(launch_mock.call_args.kwargs["threads_batch"], 16)


class MergePresetIntoConfigThreadsBatchTests(unittest.TestCase):
    """_merge_preset_into_config is the "live preset apply" path - if
    threads_batch isn't in the whitelist, a preset edit silently doesn't take
    effect. Note that this is a launch-time flag (baked into the container
    at spawn), so a live overlay only matters for the NEXT relaunch - but
    it still has to reach the merged config for restart to pick it up."""

    def test_merge_preset_overlays_field(self):
        base_config = {"n_gpu_layers": -1, "ctx_size": 4096}
        preset = {"threads_batch": 12}

        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)

        self.assertEqual(merged["threads_batch"], 12)


if __name__ == "__main__":
    unittest.main()
