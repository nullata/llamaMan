# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the `engine` field on instances and presets.

Contract: a missing / empty engine means llama.cpp, so every instance row,
preset and container label written before engines existed keeps loading as
llama.cpp, and llama.cpp configs are never stamped with the key (it would
land in the container's llamaman.config label). Only a non-default engine is
recorded. Presets and instances are JSON blobs in both storage backends
(presets.json / state.json; the MariaDB `data` TEXT columns), so the field
needs no schema migration - these tests pin the read-side default instead.

A throwaway "dummy" engine is registered for the non-default cases so the
data-model plumbing is tested independently of any real second engine.
"""

import os
import tempfile
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
from core.engines import ENGINES, engine_name, parse_engine
from core.engines.base import Engine
from core.state import instances, instances_lock
from storage.json_backend import JsonBackend


class DummyEngine(Engine):
    name = "dummy"
    label = "Dummy"

    def default_image(self) -> str:
        return "dummy:latest"

    def display_name(self, model_path: str) -> str:
        return "dummy:" + model_path


def _with_dummy_engine():
    return patch.dict(ENGINES, {"dummy": DummyEngine()})


class _InstancesIsolation(unittest.TestCase):
    def setUp(self):
        with instances_lock:
            self._saved_instances = {k: dict(v) for k, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)


class ParseEngineTests(unittest.TestCase):

    def test_missing_and_empty_mean_llamacpp(self):
        for body in ({}, None, {"engine": None}, {"engine": ""}, {"engine": "  "}):
            self.assertEqual(parse_engine(body), ("llamacpp", None))

    def test_known_engine_normalized(self):
        self.assertEqual(parse_engine({"engine": " LLAMACPP "}), ("llamacpp", None))

    def test_unknown_engine_is_an_error(self):
        name, err = parse_engine({"engine": "vllm-typo"})
        self.assertIsNotNone(err)
        self.assertIn("unknown engine", err)

    def test_non_string_is_an_error(self):
        _, err = parse_engine({"engine": 3})
        self.assertIsNotNone(err)


class LaunchInstanceEngineTests(_InstancesIsolation):

    def _launch(self, **kw):
        fake = Mock()
        fake.id = "cid"
        with patch("api.instances.save_state"), \
             patch("api.instances._run_container", return_value=(fake, None)) as run_mock, \
             patch("api.instances.is_port_available", return_value=True), \
             patch("api.instances._publish_cluster_heartbeat_safe"):
            inst, err = instances_api.launch_instance(
                model_path="/models/chat.gguf", port=8000, ctx_size=4096, **kw)
        self.assertIsNone(err)
        return inst, run_mock

    def test_default_launch_does_not_stamp_engine(self):
        inst, _ = self._launch()
        self.assertNotIn("engine", inst["config"])
        self.assertEqual(inst["model_name"], "chat.gguf")

    def test_explicit_llamacpp_does_not_stamp_engine(self):
        inst, _ = self._launch(engine="llamacpp")
        self.assertNotIn("engine", inst["config"])

    def test_public_instance_reports_llamacpp_for_legacy_config(self):
        inst, _ = self._launch()
        self.assertEqual(instances_api._public_instance(inst)["engine"], "llamacpp")

    def test_non_default_engine_recorded_in_config(self):
        with _with_dummy_engine():
            inst, run_mock = self._launch(engine="dummy")
            self.assertEqual(inst["config"]["engine"], "dummy")
            self.assertEqual(inst["config"]["image"], "dummy:latest")
            self.assertEqual(inst["model_name"], "dummy:/models/chat.gguf")
            self.assertEqual(instances_api._public_instance(inst)["engine"], "dummy")
            # The config handed to _run_container (and so to the container's
            # llamaman.config label, which orphan adoption reads) carries it.
            self.assertEqual(run_mock.call_args.args[4]["engine"], "dummy")


class InstancesRoutesEngineTests(_InstancesIsolation):

    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(instances_api.bp)
        self.client = app.test_client()

    @patch("api.instances.launch_instance")
    def test_create_route_defaults_to_llamacpp(self, launch_mock):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post("/api/instances", json={
                "model_path": "/models/chat.gguf", "port": 8000, "ctx_size": 4096,
            })
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(launch_mock.call_args.kwargs["engine"], "llamacpp")

    @patch("api.instances.launch_instance")
    def test_create_route_rejects_unknown_engine(self, launch_mock):
        resp = self.client.post("/api/instances", json={
            "model_path": "/models/chat.gguf", "port": 8000, "ctx_size": 4096,
            "engine": "nope",
        })
        self.assertEqual(resp.status_code, 400)
        launch_mock.assert_not_called()

    @patch("api.instances.launch_instance")
    def test_create_route_rejects_unavailable_engine(self, launch_mock):
        class Unavailable(DummyEngine):
            def availability(self, vendor):
                return False, "needs hardware this node lacks"
        with patch.dict(ENGINES, {"dummy": Unavailable()}):
            resp = self.client.post("/api/instances", json={
                "model_path": "/models/chat.gguf", "port": 8000, "ctx_size": 4096,
                "engine": "dummy",
            })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("hardware", resp.get_json()["error"])
        launch_mock.assert_not_called()

    @patch("api.instances.save_state")
    @patch("api.instances.release_instance_reservations")
    @patch("api.instances.is_port_available", return_value=True)
    @patch("api.instances._admin_ui_enforces_eviction", return_value=False)
    @patch("api.instances._would_ui_launch_exceed_limit", return_value=False)
    @patch("api.instances._merge_preset_into_config", side_effect=lambda _p, cfg: cfg)
    @patch("api.instances.launch_instance")
    def test_restart_route_preserves_engine(self, launch_mock, *_mocks):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        for cfg_engine, expected in ((None, None), ("dummy", "dummy")):
            cfg = {"n_gpu_layers": -1, "ctx_size": 4096}
            if cfg_engine:
                cfg["engine"] = cfg_engine
            with instances_lock:
                instances["inst-1"] = {
                    "id": "inst-1", "model_path": "/models/chat.gguf", "port": 8000,
                    "status": "stopped", "stats": {}, "config": cfg,
                }
            with patch("api.instances._public_instance", side_effect=lambda inst: inst):
                resp = self.client.post("/api/instances/inst-1/restart", json={})
            self.assertIn(resp.status_code, (200, 201))
            self.assertEqual(launch_mock.call_args.kwargs["engine"], expected)


class PresetEngineTests(unittest.TestCase):

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def _save(self, body, existing=None):
        storage = Mock()
        storage.get_preset.return_value = existing or {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json=body)
        saved = storage.save_preset.call_args.args[1] if storage.save_preset.called else None
        return resp, saved

    def test_preset_without_engine_saves_without_key(self):
        resp, saved = self._save({"ctx_size": 4096})
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("engine", saved)
        self.assertEqual(engine_name(saved), "llamacpp")

    def test_explicit_llamacpp_saves_without_key(self):
        _, saved = self._save({"ctx_size": 4096, "engine": "llamacpp"})
        self.assertNotIn("engine", saved)

    def test_non_default_engine_saved(self):
        with _with_dummy_engine():
            _, saved = self._save({"ctx_size": 4096, "engine": "dummy"})
        self.assertEqual(saved["engine"], "dummy")

    def test_unknown_engine_rejected(self):
        resp, saved = self._save({"ctx_size": 4096, "engine": "nope"})
        self.assertEqual(resp.status_code, 400)
        self.assertIsNone(saved)

    def test_engine_not_merged_into_running_config(self):
        # The engine is fixed for an instance's lifetime: a preset edit must
        # never flip a running llama.cpp instance's relaunch onto another
        # engine (or vice versa).
        storage = Mock()
        storage.get_preset.return_value = {"engine": "dummy", "ctx_size": 8192}
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config(
                "/models/chat.gguf", {"ctx_size": 4096})
        self.assertNotIn("engine", merged)


class LegacyStorageLoadsAsLlamaCppTests(unittest.TestCase):
    """Rows written before engines existed, through the real JSON backend."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.backend = JsonBackend(
            state_file=os.path.join(d, "state.json"),
            presets_file=os.path.join(d, "presets.json"),
            users_file=os.path.join(d, "users.json"),
            settings_file=os.path.join(d, "settings.json"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_legacy_preset_loads_as_llamacpp(self):
        legacy = {"n_gpu_layers": 99, "ctx_size": 8192, "flash_attn": True}
        self.backend.save_preset("/models/old.gguf", legacy)
        loaded = self.backend.get_preset("/models/old.gguf")
        self.assertEqual(loaded, legacy)  # untouched on read
        self.assertEqual(engine_name(loaded), "llamacpp")

    def test_engine_round_trips_through_preset_storage(self):
        self.backend.save_preset("/models/x", {"ctx_size": 4096, "engine": "dummy"})
        with _with_dummy_engine():
            self.assertEqual(engine_name(self.backend.get_preset("/models/x")), "dummy")

    def test_legacy_instance_row_loads_as_llamacpp(self):
        row = {"id": "i1", "model_name": "old.gguf", "model_path": "/models/old.gguf",
               "port": 8000, "status": "stopped", "config": {"ctx_size": 4096}}
        self.backend.save_state([row], [], node_id="test-node")
        (loaded,) = self.backend.load_instances("test-node")
        self.assertNotIn("engine", loaded["config"])
        self.assertEqual(engine_name(loaded["config"]), "llamacpp")


if __name__ == "__main__":
    unittest.main()
