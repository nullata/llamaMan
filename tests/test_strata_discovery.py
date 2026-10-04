# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Strata models in discovery (/api/models, /api/tags, /v1/models, /api/ps),
Ollama/OpenAI name resolution and auto-launch, per-instance proxy model-name
matching, /api/engines and the cluster snapshot.

The library lists only files on disk. Strata's catalogue is offered for
download in the Strata settings, not listed; a downloaded shard is an
ordinary file tagged with the engine that runs it (engine_models), and
answers to Strata's names for it. GGUF matching must be unchanged.
"""

import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.engines as engines_api
import api.llamaman as llamaman
import api.models as models_api
from core.helpers import model_name_from_path
from core.state import instances, instances_lock
from proxy import _model_matches

QWEN = "/strata/qwen-IQ2_XS"
GGUF = {"name": "alpha", "path": "/models/alpha-Q4_K_M.gguf", "type": "gguf",
        "quant": "Q4_K_M", "size_bytes": 99, "size_display": "99 B"}
# A downloaded shard of strata/qwen-IQ2_XS, as discover_models lists it.
SHARD_NAME = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002"
SHARD = {"name": SHARD_NAME, "path": f"/models/strata/iq2_xs/IQ2_XS/{SHARD_NAME}.gguf", "type": "gguf",
         "quant": "IQ2_XS", "size_bytes": 50, "size_display": "50 B",
         "engine_models": {"strata": "strata/qwen-IQ2_XS"}}


def strata_on(vendor="cuda"):
    """Context: Strata enabled on a node of the given vendor."""
    class _Ctx:
        def __enter__(self):
            self.ps = [patch("config.STRATA_ENABLED", True),
                       patch("core.gpu.get_vendor", return_value=vendor)]
            for p in self.ps:
                p.start()

        def __exit__(self, *a):
            for p in reversed(self.ps):
                p.stop()
    return _Ctx()


class ModelNameTests(unittest.TestCase):

    def test_virtual_model_named_by_id(self):
        self.assertEqual(model_name_from_path(QWEN), "strata/qwen-iq2_xs")
        self.assertEqual(model_name_from_path("/strata/unsloth-UD-Q4_K_XL"), "strata/unsloth-ud-q4_k_xl")

    def test_gguf_names_unchanged(self):
        self.assertEqual(model_name_from_path("/models/Foo-Q4_K_M.gguf"), "foo-q4_k_m")
        self.assertEqual(model_name_from_path("/models/sub/x-00001-of-00002.gguf"), "x-00001-of-00002")
        # Not in Strata's catalogue -> just a path.
        self.assertEqual(model_name_from_path("/strata/notamodel.gguf"), "notamodel")


class ListModelsTests(unittest.TestCase):

    def test_library_is_files_only(self):
        for vendor in ("cuda", "rocm"):
            with strata_on(vendor), patch("api.models.discover_models", return_value=[dict(GGUF)]):
                self.assertEqual([m["path"] for m in models_api.list_models("/models")], [GGUF["path"]])

    def test_downloaded_shard_tagged_with_its_engine(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "strata", "iq2_xs", "IQ2_XS")
            os.makedirs(sub)
            for i in (1, 2):
                open(os.path.join(sub, f"Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{i}-of-00002.gguf"), "wb").close()
            open(os.path.join(d, "plain-Q4_K_M.gguf"), "wb").close()
            models = {m["name"]: m for m in models_api.discover_models(d)}
        self.assertEqual(models[SHARD_NAME]["engine_models"], {"strata": "strata/qwen-IQ2_XS"})
        self.assertNotIn("engine_models", models["plain-Q4_K_M"])

    def test_api_models_route(self):
        app = Flask(__name__)
        app.register_blueprint(models_api.bp)
        with strata_on(), patch("api.models.discover_models", return_value=[dict(SHARD)]), \
             patch("api.models.get_storage") as st:
            st.return_value.get_settings.return_value = {}
            data = app.test_client().get("/api/models").get_json()
        self.assertEqual([m["name"] for m in data], [SHARD_NAME])
        self.assertNotIn("strata/qwen-IQ2_XS", [m["name"] for m in data])


class CompatListingTests(unittest.TestCase):

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(llamaman.bp)
        self.client = app.test_client()
        self._p = [patch("api.llamaman.discover_models", return_value=[dict(GGUF)]),
                   patch("api.llamaman._cluster_group_entries", return_value=[])]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()

    def test_tags_and_v1_models_list_files_only(self):
        storage = Mock()
        storage.get_preset.return_value = None
        with strata_on(), patch("api.llamaman.get_storage", return_value=storage):
            tags = [m["name"] for m in self.client.get("/api/tags").get_json()["models"]]
            ids = [m["id"] for m in self.client.get("/v1/models").get_json()["data"]]
        self.assertEqual(tags, ["alpha-q4_k_m"])
        self.assertEqual(ids, ["alpha-q4_k_m"])

    @patch("api.llamaman._probe_server_ready", return_value=True)
    @patch("api.llamaman._instance_container_alive", return_value=True)
    def test_ps_lists_running_strata_instance(self, *_):
        with instances_lock:
            saved = dict(instances)
            instances.clear()
            instances["s1"] = {"id": "s1", "model_name": "strata/qwen-IQ2_XS", "model_path": QWEN,
                               "port": 8001, "status": "healthy", "container_id": "c",
                               "started_at": 1000, "config": {"engine": "strata", "ctx_size": 65536}}
        try:
            with strata_on():
                models = self.client.get("/api/ps").get_json()["models"]
        finally:
            with instances_lock:
                instances.clear()
                instances.update(saved)
        (m,) = models
        self.assertEqual(m["name"], "strata/qwen-iq2_xs")
        self.assertEqual(m["context_length"], 65536)
        self.assertEqual(m["size"], 68_000_000_000)


class NameResolutionTests(unittest.TestCase):

    def _find(self, name, files=(GGUF, SHARD)):
        with strata_on(), patch("api.llamaman.discover_models", return_value=[dict(f) for f in files]):
            return llamaman._find_model_by_name(name)

    def test_strata_names_resolve_to_the_downloaded_file(self):
        for name in ("strata/qwen-IQ2_XS", "STRATA/QWEN-iq2_xs", "strata/qwen-IQ2_XS:latest",
                     "qwen3.8-flash-next-iq2_xs"):
            m = self._find(name)
            self.assertEqual(m["path"], SHARD["path"], name)
            self.assertEqual(m["_engine"], "strata", name)

    def test_file_name_resolves_without_engine_hint(self):
        m = self._find(SHARD_NAME.lower())
        self.assertEqual(m["path"], SHARD["path"])
        self.assertNotIn("_engine", m)

    def test_not_found_when_not_downloaded(self):
        self.assertIsNone(self._find("strata/qwen-IQ2_XS", files=(GGUF,)))
        self.assertIsNone(self._find("qwen3.8-flash-next-iq2_xs", files=(GGUF,)))

    def test_gguf_still_wins_its_names(self):
        self.assertEqual(self._find("alpha-q4_k_m")["path"], GGUF["path"])
        self.assertEqual(self._find("alpha")["path"], GGUF["path"])

    def _auto_launch(self, name, preset):
        storage = Mock()
        storage.get_preset.return_value = preset
        storage.get_settings.return_value = {}
        launched = {"id": "new", "status": "starting", "port": 8003}
        with instances_lock:
            saved = dict(instances)
            instances.clear()
        try:
            with strata_on(), \
                 patch("api.llamaman.discover_models", return_value=[dict(SHARD)]), \
                 patch("api.llamaman.get_storage", return_value=storage), \
                 patch("api.llamaman.find_available_port", return_value=8003), \
                 patch("api.instances.launch_instance", return_value=(launched, None)) as launch:
                inst, err = llamaman._ensure_model_running(name)
        finally:
            with instances_lock:
                instances.clear()
                instances.update(saved)
        self.assertIsNone(err)
        return launch.call_args.kwargs

    def test_auto_launch_by_strata_name_runs_the_file_on_strata(self):
        kw = self._auto_launch("strata/qwen-IQ2_XS", {"strata_vision": "cpu", "strata_kv": "int8"})
        self.assertEqual(kw["model_path"], SHARD["path"])     # launch_instance maps it to the model id
        self.assertEqual(kw["engine"], "strata")
        self.assertEqual(kw["ctx_size"], 32768)               # Strata's default, not llama.cpp's 4096
        self.assertEqual(kw["engine_options"], {"strata_vision": "cpu", "strata_kv": "int8"})

    def test_auto_launch_by_file_name_follows_the_preset(self):
        kw = self._auto_launch(SHARD_NAME.lower(), {"engine": "strata", "ctx_size": 65536})
        self.assertEqual((kw["engine"], kw["ctx_size"]), ("strata", 65536))
        kw = self._auto_launch(SHARD_NAME.lower(), {"ctx_size": 4096})
        self.assertEqual(kw["engine"], "llamacpp")           # no engine in the preset: llama.cpp

    def test_running_strata_instance_found_by_its_source_file(self):
        with instances_lock:
            saved = dict(instances)
            instances.clear()
            instances["s1"] = {"id": "s1", "model_path": QWEN, "status": "healthy", "port": 8001,
                               "config": {"engine": "strata", "engine_source_path": SHARD["path"]}}
        try:
            self.assertEqual(llamaman._find_running_instance_for_model(SHARD["path"])["id"], "s1")
            self.assertEqual(llamaman._find_any_instance_for_model(QWEN)["id"], "s1")
            self.assertIsNone(llamaman._find_running_instance_for_model(GGUF["path"]))
        finally:
            with instances_lock:
                instances.clear()
                instances.update(saved)


class ProxyModelMatchTests(unittest.TestCase):

    def test_strata_id_and_served_name(self):
        for req in ("strata/qwen-IQ2_XS", "STRATA/QWEN-IQ2_XS:latest", "qwen3.8-flash-next-iq2_xs"):
            self.assertTrue(_model_matches(QWEN, req), req)

    def test_strata_mismatch(self):
        for req in ("strata/qwen-Q2_0", "swift-1.5-iq2_xs", "llama3"):
            self.assertFalse(_model_matches(QWEN, req), req)

    def test_gguf_stem_prefix_matching_unchanged(self):
        path = "/models/Qwen2.5-14B-Q4_K_M.gguf"
        self.assertTrue(_model_matches(path, "qwen2.5-14b-q4_k_m"))
        self.assertTrue(_model_matches(path, "Qwen2.5-14B"))
        self.assertFalse(_model_matches(path, "llama3"))
        self.assertFalse(_model_matches(path, "qwen3.8-flash-next-iq2_xs"))


class EnginesEndpointTests(unittest.TestCase):

    def _get(self, vendor, enabled=True):
        app = Flask(__name__)
        app.register_blueprint(engines_api.bp)
        with patch("config.STRATA_ENABLED", enabled), \
             patch("api.engines.get_vendor", return_value=vendor):
            return app.test_client().get("/api/engines").get_json()

    def test_nvidia_node(self):
        data = self._get("cuda")
        by = {e["name"]: e for e in data["engines"]}
        self.assertTrue(by["llamacpp"]["available"])
        self.assertTrue(by["strata"]["available"])
        self.assertEqual(by["strata"]["capabilities"]["max_concurrency"], 1)
        self.assertEqual(len(by["strata"]["models"]), 8)
        self.assertIn("docker build", by["strata"]["build_command"])
        self.assertNotIn("models", by["llamacpp"])

    def test_rocm_node_reports_reason(self):
        by = {e["name"]: e for e in self._get("rocm")["engines"]}
        self.assertFalse(by["strata"]["available"])
        self.assertIn("NVIDIA", by["strata"]["reason"])

    def test_cluster_snapshot_carries_engines(self):
        import api.cluster as cluster_api
        with patch("config.STRATA_ENABLED", True), patch("api.engines.get_vendor", return_value="cuda"):
            snap = cluster_api.build_local_snapshot()
        names = [e["name"] for e in snap["system"]["engines"]["engines"]]
        self.assertEqual(names, ["llamacpp", "strata"])


if __name__ == "__main__":
    unittest.main()
