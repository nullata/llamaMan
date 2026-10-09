# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the inference engine abstraction (core/engines).

The load-bearing one is LlamaCppContainerSnapshotTests: the docker
containers.run(**kwargs) that _run_container produces for llama.cpp must be
identical to what it produced before the engine abstraction existed. The
expected values in tests/fixtures/llamacpp_container_spec.json were captured
from the pre-refactor code (see tests/fixtures_llamacpp_cases.py).
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import config
import api.instances as instances_api
from core.engines import DEFAULT_ENGINE, ENGINES, engine_name, get_engine
from core.helpers import build_llama_cmd
from tests.fixtures_llamacpp_cases import CASES, VENDORS, normalize_spec

SNAPSHOT_FILE = os.path.join(REPO_ROOT, "tests", "fixtures", "llamacpp_container_spec.json")


def _run_container_kwargs(model_path: str, config_dict: dict, vendor):
    fake_container = Mock()
    fake_container.id = "cid"
    client = Mock()
    client.containers.run.return_value = fake_container
    with patch("api.instances.get_docker_client", return_value=client), \
         patch("api.instances.ensure_docker_network"), \
         patch("api.instances._start_log_relay"), \
         patch("api.instances.get_vendor", return_value=vendor), \
         patch("core.gpu.resolve_render_gids", return_value=[44, 107]):
        container, err = instances_api._run_container(
            "inst-1", "llamaman-inst-1", model_path, 9001, dict(config_dict), "/tmp/x.log",
        )
    assert err is None, err
    return client.containers.run.call_args.kwargs


class LlamaCppContainerSnapshotTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(SNAPSHOT_FILE) as f:
            cls.snapshot = json.load(f)

    def test_snapshot_covers_every_case(self):
        expected = {f"{c}|{v}" for c in CASES for v in VENDORS}
        self.assertEqual(set(self.snapshot), expected)

    def test_container_spec_identical_to_pre_refactor(self):
        for case_name, (model_path, cfg) in CASES.items():
            for vendor in VENDORS:
                key = f"{case_name}|{vendor}"
                with self.subTest(key):
                    got = normalize_spec(
                        _run_container_kwargs(model_path, cfg, vendor),
                        REPO_ROOT, config.LLAMA_IMAGE,
                    )
                    self.assertEqual(got, self.snapshot[key])

    def test_engine_command_is_build_llama_cmd(self):
        # The engine must not grow its own flag logic: build_llama_cmd stays
        # the single source of truth (the preview endpoint calls it directly).
        engine = get_engine({})
        for model_path, cfg in CASES.values():
            self.assertEqual(engine.command(model_path, cfg),
                             build_llama_cmd(model_path, 8080, cfg))


class EngineRegistryTests(unittest.TestCase):

    def test_missing_engine_means_llamacpp(self):
        self.assertEqual(DEFAULT_ENGINE, "llamacpp")
        for cfg in (None, {}, {"engine": ""}, {"engine": None}, {"ctx_size": 4096}):
            self.assertEqual(engine_name(cfg), "llamacpp")
            self.assertIs(get_engine(cfg), ENGINES["llamacpp"])

    def test_unknown_engine_falls_back_to_llamacpp(self):
        # A config written by a newer llamaman (or hand-edited) must not make
        # restore / orphan adoption crash; it runs as llama.cpp.
        self.assertEqual(engine_name({"engine": "nonexistent"}), "llamacpp")

    def test_engine_name_case_insensitive(self):
        self.assertEqual(engine_name({"engine": " LlamaCpp "}), "llamacpp")

    def test_llamacpp_describe(self):
        d = get_engine("llamacpp").describe("cuda")
        self.assertTrue(d["available"])
        self.assertIsNone(d["launch_fields"])
        self.assertEqual(d["capabilities"]["max_concurrency"], 0)

    def test_llamacpp_load_timeout_is_model_load_timeout(self):
        self.assertEqual(get_engine("llamacpp").load_timeout(), config.MODEL_LOAD_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
