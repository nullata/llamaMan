# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the mmproj GPU-offload control (--mmproj-offload /
--no-mmproj-offload boolean pair; llama.cpp default is ENABLED).

The whole feature is default-True end to end: parse_mmproj_config,
launch_instance, the restart route, the preset-merge whitelist, and
build_llama_cmd must all treat a missing key as "offload on", so every
existing preset, instance and hand-written config keeps a byte-identical CLI.
--no-mmproj-offload is emitted ONLY for an explicit false, and only when an
mmproj is actually loaded (gated on mmproj_enabled: harmless upstream but
confusing noise in the CLI the operator reads).
"""

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
from core.multimodal import MMPROJ_CONFIG_KEYS, parse_mmproj_config
from core.state import instances, instances_lock

MM = {"mmproj_enabled": True, "mmproj_path": "/models/mm.gguf"}


class ParseMmprojConfigOffloadTests(unittest.TestCase):
    """The parse boundary is the single normalizer; both callers
    (launch route, preset save) spread its output, so one True default
    covers both."""

    def test_key_is_in_mmproj_config_keys(self):
        # The whitelist the create route and preset save validate against.
        self.assertIn("mmproj_offload", MMPROJ_CONFIG_KEYS)

    def test_missing_key_defaults_true(self):
        cfg, err = parse_mmproj_config(dict(MM))
        self.assertIsNone(err)
        self.assertIs(cfg["mmproj_offload"], True)

    def test_explicit_false_survives(self):
        cfg, err = parse_mmproj_config({**MM, "mmproj_offload": False})
        self.assertIsNone(err)
        self.assertIs(cfg["mmproj_offload"], False)

    def test_explicit_true_survives(self):
        cfg, err = parse_mmproj_config({**MM, "mmproj_offload": True})
        self.assertIsNone(err)
        self.assertIs(cfg["mmproj_offload"], True)

    def test_null_defaults_true_not_false(self):
        # JSON null must not become falsy-off: an old client sending null
        # means "no opinion", which is llama.cpp's on-by-default.
        cfg, err = parse_mmproj_config({**MM, "mmproj_offload": None})
        self.assertIsNone(err)
        self.assertIs(cfg["mmproj_offload"], True)

    def test_default_true_even_when_mmproj_disabled(self):
        # Shape consistency with pdf_dpi etc.: the key is always present so a
        # saved preset never has holes.
        cfg, err = parse_mmproj_config({})
        self.assertIsNone(err)
        self.assertIs(cfg["mmproj_offload"], True)


class BuildLlamaCmdOffloadTests(unittest.TestCase):
    """Emission rule: nothing when on (llama.cpp's own default), the bare
    boolean flag when explicitly off, never when no projector is loaded."""

    def test_default_config_emits_nothing(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, MM)
        self.assertNotIn("--no-mmproj-offload", cmd)

    def test_offload_true_emits_nothing(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {**MM, "mmproj_offload": True})
        self.assertNotIn("--no-mmproj-offload", cmd)

    def test_explicit_off_emits_bare_flag(self):
        # Bare flag - upstream takes no value. Pin that the flag is the LAST
        # argv element: if we ever emitted a value after it, llama.cpp would
        # swallow it as an unrelated argument and corrupt the command line.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {**MM, "mmproj_offload": False})
        self.assertIn("--no-mmproj-offload", cmd)
        self.assertEqual(cmd[-1], "--no-mmproj-offload")

    def test_off_but_no_mmproj_emits_nothing(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080,
                              {"mmproj_enabled": False, "mmproj_offload": False})
        self.assertNotIn("--no-mmproj-offload", cmd)

    def test_legacy_config_without_key_is_byte_identical(self):
        # The pre-feature config shape must produce the exact same argv as
        # the same config with mmproj_offload: True - that is the no-
        # migration guarantee, pinned.
        legacy = dict(MM)
        with_offload = {**MM, "mmproj_offload": True}
        self.assertEqual(build_llama_cmd("/models/m.gguf", 8080, legacy),
                         build_llama_cmd("/models/m.gguf", 8080, with_offload))


class LaunchInstanceOffloadTests(unittest.TestCase):
    def setUp(self):
        with instances_lock:
            self._saved = {i: dict(v) for i, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def _launch(self, **kw):
        with patch("api.instances.save_state"), \
             patch("api.instances._run_container") as run, \
             patch("api.instances.is_port_available", return_value=True):
            c = Mock()
            c.id = "abc123"
            run.return_value = (c, None)
            return instances_api.launch_instance(
                model_path="/models/chat.gguf", port=8000, ctx_size=4096, **kw)

    def test_default_stores_true(self):
        inst, err = self._launch()
        self.assertIsNone(err)
        self.assertIs(inst["config"]["mmproj_offload"], True)

    def test_explicit_false_stored(self):
        inst, err = self._launch(mmproj_enabled=True, mmproj_path="/models/mm.gguf",
                                 mmproj_offload=False)
        self.assertIsNone(err)
        self.assertIs(inst["config"]["mmproj_offload"], False)


class RestartRouteOffloadTests(unittest.TestCase):
    """The restart route re-types launch_instance's signature by hand and is
    the boundary where default-True gets lost most easily (.get(k, False)
    beside .get(k, True)). Legacy config without the key must restart with
    offload still ON."""

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(instances_api.bp)
        self.client = self.app.test_client()
        with instances_lock:
            self._saved = {i: dict(v) for i, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def _restart_with_config(self, config):
        inst = {"id": "inst-1", "status": "stopped", "model_path": "/models/m.gguf",
                "port": 8000, "config": config}
        with instances_lock:
            instances["inst-1"] = inst
        with patch("api.instances._merge_preset_into_config",
                   side_effect=lambda p, c: c), \
             patch("api.instances.kill_instance_process"), \
             patch("api.instances.stop_container"), \
             patch("api.instances.launch_instance") as launch:
            launch.return_value = ({"id": "inst-2"}, None)
            self.client.post("/api/instances/inst-1/restart")
            return launch

    def test_legacy_config_restarts_with_true(self):
        cfg = {"mmproj_enabled": True, "mmproj_path": "/models/mm.gguf"}
        del cfg  # spelled out: no mmproj_offload key at all
        launch = self._restart_with_config({"ctx_size": 4096,
                                            "mmproj_enabled": True,
                                            "mmproj_path": "/models/mm.gguf"})
        self.assertIs(launch.call_args.kwargs["mmproj_offload"], True)

    def test_stored_false_survives_restart(self):
        launch = self._restart_with_config({"ctx_size": 4096,
                                            "mmproj_enabled": True,
                                            "mmproj_path": "/models/mm.gguf",
                                            "mmproj_offload": False})
        self.assertIs(launch.call_args.kwargs["mmproj_offload"], False)


class MergeWhitelistTests(unittest.TestCase):
    def test_mmproj_offload_in_merge_whitelist(self):
        import inspect
        src = inspect.getsource(instances_api._merge_preset_into_config)
        self.assertIn('"mmproj_offload"', src)

    def test_merge_overlays_preset_value(self):
        storage = Mock()
        storage.get_preset.return_value = {"mmproj_offload": False}
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config(
                "/models/m.gguf",
                {"mmproj_offload": True, "ctx_size": 4096})
        self.assertIs(merged["mmproj_offload"], False)


class PresetSaveOffloadTests(unittest.TestCase):
    """The preset body must persist the value through the same
    parse_mmproj_config spread, and a save that omits the key must write
    True - leaving a stale false behind would be a silent behavior change."""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def _save(self, preset_on_disk, body):
        storage = Mock()
        storage.get_preset.return_value = preset_on_disk
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/m.gguf",
                                      json={**body, "ctx_size": 4096})
        self.assertEqual(resp.status_code, 200)
        _, saved = storage.save_preset.call_args.args
        return saved

    def test_preset_save_persists_false(self):
        saved = self._save({}, {**MM, "mmproj_offload": False})
        self.assertIs(saved["mmproj_offload"], False)

    def test_preset_save_omitting_key_writes_true(self):
        # Stale-false protection: an older form re-saving a preset that had
        # offload off must not preserve the false by accident...
        saved = self._save({"mmproj_offload": False}, {**MM})
        self.assertIs(saved["mmproj_offload"], True)


if __name__ == "__main__":
    unittest.main()
