# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the GPU Settings launch controls: --split-mode / --tensor-split
flag emission and preset persistence (which is per-node hardware, so both keys
must round-trip through the base preset AND through a node_overrides block)."""

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


class BuildLlamaCmdSplitModeTests(unittest.TestCase):
    """`build_llama_cmd` must emit --split-mode / --tensor-split only when the
    user asked for them - an empty config has to reproduce llama.cpp's own
    default (layer with even split), which means "no flag at all"."""

    def test_empty_config_backfills_to_layer(self):
        # An unset split_mode (e.g. a preset saved before this feature) is
        # backfilled to 'layer' so the emitted flag matches llama.cpp's own
        # default. Behavior is unchanged, just now explicit.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {})
        self.assertEqual(cmd[cmd.index("--split-mode") + 1], "layer")
        self.assertNotIn("--tensor-split", cmd)

    def test_split_mode_none_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"split_mode": "none"})
        self.assertEqual(cmd[cmd.index("--split-mode") + 1], "none")

    def test_split_mode_layer_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"split_mode": "layer"})
        self.assertEqual(cmd[cmd.index("--split-mode") + 1], "layer")

    def test_split_mode_row_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"split_mode": "row"})
        self.assertEqual(cmd[cmd.index("--split-mode") + 1], "row")

    def test_split_mode_is_lowercased_and_whitespace_trimmed(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"split_mode": "  ROW  "})
        self.assertEqual(cmd[cmd.index("--split-mode") + 1], "row")

    def test_unknown_split_mode_falls_back_to_no_flag(self):
        # A hand-crafted request or a corrupt preset can hand us a value
        # llama.cpp wouldn't accept. We shouldn't ship it - dropping the
        # flag lets llama.cpp use its own default rather than failing to
        # start. (The dropdown only allows the three real values, so a
        # UI-only path can't reach this.)
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"split_mode": "garbage"})
        self.assertNotIn("--split-mode", cmd)

    def test_tensor_split_flag_emitted_verbatim(self):
        # Values are relative weights, normalized inside llama.cpp - passing
        # them through unchanged is the whole contract.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"tensor_split": "24,16"})
        self.assertEqual(cmd[cmd.index("--tensor-split") + 1], "24,16")

    def test_tensor_split_whitespace_only_is_omitted(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"tensor_split": "   "})
        self.assertNotIn("--tensor-split", cmd)

    def test_both_flags_together(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "split_mode": "row",
            "tensor_split": "3,1",
        })
        self.assertEqual(cmd[cmd.index("--split-mode") + 1], "row")
        self.assertEqual(cmd[cmd.index("--tensor-split") + 1], "3,1")


class PresetGpuSplitTests(unittest.TestCase):
    """The two new keys are per-node hardware (split_mode + tensor_split describe
    a physical topology), so they must live in PRESET_HARDWARE_KEYS and honour
    the node_overrides plumbing that the launch form uses in cluster mode."""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def test_hardware_keys_include_split_fields(self):
        # If either key ever leaves the hardware-keys tuple, this test fails
        # loudly - because resolve_preset_for_node stops overlaying them, and
        # the launch form silently reverts to the shared base value on the
        # wrong-topology node.
        self.assertIn("split_mode", presets_api.PRESET_HARDWARE_KEYS)
        self.assertIn("tensor_split", presets_api.PRESET_HARDWARE_KEYS)

    def test_preset_save_persists_split_fields(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "split_mode": "layer",
                "tensor_split": "24,16",
            })
        self.assertEqual(resp.status_code, 200)
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["split_mode"], "layer")
        self.assertEqual(saved_preset["tensor_split"], "24,16")

    def test_preset_save_normalizes_split_mode_case_and_whitespace(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "split_mode": "  ROW ",
                "tensor_split": "  3,1  ",
            })
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["split_mode"], "row")
        self.assertEqual(saved_preset["tensor_split"], "3,1")

    def test_preset_save_defaults_are_empty_strings(self):
        # A save that doesn't mention the keys must still write them, otherwise
        # a subsequent save from an old form would leave a stale value behind.
        storage = Mock()
        storage.get_preset.return_value = {"split_mode": "row", "tensor_split": "3,1"}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={"ctx_size": 4096})
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["split_mode"], "")
        self.assertEqual(saved_preset["tensor_split"], "")

    def test_resolve_preset_for_node_overlays_split_fields(self):
        # Same base preset, different overrides on two nodes - each node sees
        # its own split_mode + tensor_split rather than the shared base.
        base = {
            "ctx_size": 4096,
            "split_mode": "layer",
            "tensor_split": "24,16",
            "node_overrides": {
                "srv-a": {"split_mode": "row", "tensor_split": "10,10,10"},
                "srv-b": {"split_mode": "layer", "tensor_split": "48,24"},
            },
        }
        a = presets_api.resolve_preset_for_node(base, "srv-a")
        b = presets_api.resolve_preset_for_node(base, "srv-b")
        self.assertEqual(a["split_mode"], "row")
        self.assertEqual(a["tensor_split"], "10,10,10")
        self.assertEqual(b["split_mode"], "layer")
        self.assertEqual(b["tensor_split"], "48,24")
        # Base stays untouched (resolve returns a new dict).
        self.assertEqual(base["split_mode"], "layer")
        self.assertEqual(base["tensor_split"], "24,16")


class LaunchInstanceSplitFieldsTests(unittest.TestCase):
    """`launch_instance()` must copy the split_mode / tensor_split kwargs into
    `inst["config"]` (and normalize them the same way `build_llama_cmd` does),
    so that a running instance's config dict is the authoritative source both
    for the launch command and for later reads (health checks, cluster
    snapshot, preset relaunch)."""

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
    def test_launch_instance_stores_split_fields_in_config(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        fake_container = Mock()
        fake_container.id = "abc123containerid"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            split_mode="row",
            tensor_split="24,16",
        )

        self.assertIsNone(err)
        self.assertIsNotNone(inst)
        self.assertEqual(inst["config"]["split_mode"], "row")
        self.assertEqual(inst["config"]["tensor_split"], "24,16")

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_normalizes_split_mode_case_and_whitespace(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        # Whatever the caller sends (a hand-crafted request, an old preset)
        # ends up in the container config; normalizing at the boundary means
        # downstream reads and build_llama_cmd don't each have to defend.
        fake_container = Mock()
        fake_container.id = "abc123containerid"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            split_mode="  ROW  ",
            tensor_split="  3,1  ",
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["split_mode"], "row")
        self.assertEqual(inst["config"]["tensor_split"], "3,1")

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_defaults_split_fields_to_empty(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        # Callers that don't mention the new kwargs (e.g. every path in the
        # codebase that predates this feature) still get the keys in the
        # config dict, at their zero values - so build_llama_cmd's backfill
        # kicks in uniformly rather than KeyError'ing on some paths.
        fake_container = Mock()
        fake_container.id = "abc123containerid"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
        )

        self.assertIsNone(err)
        self.assertIn("split_mode", inst["config"])
        self.assertIn("tensor_split", inst["config"])
        self.assertEqual(inst["config"]["split_mode"], "")
        self.assertEqual(inst["config"]["tensor_split"], "")


class InstancesCreateRouteSplitFieldsTests(unittest.TestCase):
    """POST /api/instances must lift split_mode + tensor_split out of the JSON
    body and forward them as kwargs to launch_instance. If it drops either,
    the launch silently uses defaults and the user's UI choice vanishes."""

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
    def test_create_route_forwards_split_fields_to_launch_instance(self, launch_mock):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "ctx_size": 4096,
                    "split_mode": "row",
                    "tensor_split": "24,16",
                },
            )

        self.assertEqual(resp.status_code, 201)
        launch_mock.assert_called_once()
        kwargs = launch_mock.call_args.kwargs
        self.assertEqual(kwargs["split_mode"], "row")
        self.assertEqual(kwargs["tensor_split"], "24,16")

    @patch("api.instances.launch_instance")
    def test_create_route_strips_surrounding_whitespace_before_forwarding(self, launch_mock):
        # The route trims per-field before forwarding; without this, a preset
        # that saved "  row  " would blow past launch_instance's normalization
        # and land in the config with the whitespace preserved.
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "ctx_size": 4096,
                    "split_mode": "  row  ",
                    "tensor_split": "  24,16  ",
                },
            )
        kwargs = launch_mock.call_args.kwargs
        self.assertEqual(kwargs["split_mode"], "row")
        self.assertEqual(kwargs["tensor_split"], "24,16")

    @patch("api.instances.launch_instance")
    def test_create_route_defaults_omitted_split_fields_to_empty(self, launch_mock):
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
        kwargs = launch_mock.call_args.kwargs
        self.assertEqual(kwargs["split_mode"], "")
        self.assertEqual(kwargs["tensor_split"], "")


class InstancesRestartRouteSplitFieldsTests(unittest.TestCase):
    """POST /api/instances/<id>/restart reads the stopped instance's config
    and calls launch_instance again with those values. If either split field
    isn't forwarded, a restart silently reverts them to defaults - which
    would look like the feature "forgot" itself the first time an operator
    stops+starts an instance to apply an unrelated change."""

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
    def test_restart_route_preserves_split_fields(
        self, launch_mock, _merge_mock, _would_exceed_mock, _admin_mock,
        _is_port_mock, _release_mock, _save_state_mock,
    ):
        # _merge_preset_into_config is patched to identity so this test isolates
        # the restart wiring from the merge behavior (which has its own tests
        # above) - here we care that config -> launch_instance is complete.
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
                    "split_mode": "row",
                    "tensor_split": "24,16",
                },
            }

        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post("/api/instances/inst-1/restart", json={})

        # A launched inst is returned as 200; the exact envelope isn't the
        # point - what matters is that launch_instance saw both fields.
        self.assertIn(resp.status_code, (200, 201))
        kwargs = launch_mock.call_args.kwargs
        self.assertEqual(kwargs["split_mode"], "row")
        self.assertEqual(kwargs["tensor_split"], "24,16")


class MergePresetIntoConfigSplitFieldsTests(unittest.TestCase):
    """_merge_preset_into_config is the "live preset apply" path - the reaper
    and the periodic re-read use it to fold a saved preset onto a running
    instance's config without a relaunch. If either new key isn't in the
    whitelist, a preset edit in the UI silently doesn't take effect for
    them, and the user thinks it did."""

    def test_merge_preset_overlays_split_mode_and_tensor_split(self):
        # Preset supplies both; base config has neither. Merged config must
        # carry the preset's values.
        base_config = {"n_gpu_layers": -1, "ctx_size": 4096}
        preset = {"split_mode": "row", "tensor_split": "24,16"}

        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)

        self.assertEqual(merged["split_mode"], "row")
        self.assertEqual(merged["tensor_split"], "24,16")

    def test_merge_preset_overrides_base_config_split_fields(self):
        # A preset edit AFTER launch has to win over the value baked in at
        # launch, otherwise the reaper's "re-read preset" behavior on live
        # instances doesn't reach these fields at all.
        base_config = {"split_mode": "layer", "tensor_split": ""}
        preset = {"split_mode": "row", "tensor_split": "24,16"}

        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)

        self.assertEqual(merged["split_mode"], "row")
        self.assertEqual(merged["tensor_split"], "24,16")

    def test_merge_preset_leaves_base_untouched_when_preset_omits_split_fields(self):
        # An older preset (saved before this feature) has neither key. The
        # merged config must keep the base's values rather than clobbering
        # them with empties - that would silently downgrade a running
        # instance to llama.cpp's default even split on the next reap.
        base_config = {"split_mode": "row", "tensor_split": "24,16"}
        preset = {"ctx_size": 4096}  # a pre-feature preset

        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)

        self.assertEqual(merged["split_mode"], "row")
        self.assertEqual(merged["tensor_split"], "24,16")


if __name__ == "__main__":
    unittest.main()
