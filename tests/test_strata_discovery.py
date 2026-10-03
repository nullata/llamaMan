# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Strata models in discovery (/api/models, /api/tags, /v1/models, /api/show,
/api/ps), Ollama/OpenAI name resolution and auto-launch, per-instance proxy
model-name matching, /api/engines and the cluster snapshot.

Strata models are virtual (no file): they're listed only when STRATA_ENABLED
is set and the node is NVIDIA, after the files on disk so filename lookups
keep precedence. GGUF matching must be unchanged.
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

    def _list(self):
        with patch("api.models.discover_models", return_value=[dict(GGUF)]):
            return models_api.list_models("/models")

    def test_disabled_lists_files_only(self):
        with patch("config.STRATA_ENABLED", False), patch("core.gpu.get_vendor", return_value="cuda"):
            self.assertEqual([m["path"] for m in self._list()], [GGUF["path"]])

    def test_enabled_nvidia_appends_catalogue_after_files(self):
        with strata_on():
            models = self._list()
        self.assertEqual(models[0]["path"], GGUF["path"])
        virtual = models[1:]
        self.assertEqual(len(virtual), 8)
        q = next(m for m in virtual if m["path"] == QWEN)
        self.assertEqual((q["name"], q["type"], q["engine"], q["quant"]),
                         ("strata/qwen-IQ2_XS", "strata", "strata", "IQ2_XS"))
        self.assertFalse(q["local_shards"])

    def test_non_nvidia_lists_no_virtual_models(self):
        for vendor in ("rocm", "intel", "vulkan", None):
            with strata_on(vendor):
                self.assertEqual(len(self._list()), 1, vendor)

    def test_api_models_route(self):
        app = Flask(__name__)
        app.register_blueprint(models_api.bp)
        with strata_on(), patch("api.models.discover_models", return_value=[]), \
             patch("api.models.get_storage") as st:
            st.return_value.get_settings.return_value = {}
            data = app.test_client().get("/api/models").get_json()
        self.assertIn("strata/qwen-IQ2_XS", [m["name"] for m in data])


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

    def test_tags_lists_strata_only_when_enabled(self):
        with patch("config.STRATA_ENABLED", False):
            names = [m["name"] for m in self.client.get("/api/tags").get_json()["models"]]
        self.assertEqual(names, ["alpha-q4_k_m"])
        with strata_on():
            models = self.client.get("/api/tags").get_json()["models"]
        entry = next(m for m in models if m["name"] == "strata/qwen-iq2_xs")
        self.assertEqual(entry["details"]["family"], "qwen3.8-flash-next")
        self.assertEqual(entry["details"]["quantization_level"], "IQ2_XS")

    def test_v1_models_context_default_and_preset(self):
        storage = Mock()
        storage.get_preset.return_value = None
        with strata_on(), patch("api.llamaman.get_storage", return_value=storage):
            data = self.client.get("/v1/models").get_json()["data"]
        entry = next(m for m in data if m["id"] == "strata/qwen-iq2_xs")
        self.assertEqual(entry["context_length"], 32768)
        storage.get_preset.side_effect = lambda p: {"ctx_size": 131072} if p == QWEN else None
        with strata_on(), patch("api.llamaman.get_storage", return_value=storage):
            data = self.client.get("/v1/models").get_json()["data"]
        entry = next(m for m in data if m["id"] == "strata/qwen-iq2_xs")
        self.assertEqual(entry["context_length"], 131072)

    def test_show_strata_model(self):
        storage = Mock()
        storage.get_preset.return_value = None
        with strata_on(), patch("api.llamaman.get_storage", return_value=storage):
            resp = self.client.post("/api/show", json={"model": "strata/qwen-IQ2_XS"})
        self.assertEqual(resp.status_code, 200)
        info = resp.get_json()["model_info"]
        self.assertEqual(info["general.architecture"], "qwen3.8-flash-next")
        self.assertEqual(info["qwen3.8-flash-next.context_length"], 32768)

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

    def _find(self, name):
        with strata_on(), patch("api.llamaman.discover_models", return_value=[dict(GGUF)]):
            m = llamaman._find_model_by_name(name)
        return m["path"] if m else None

    def test_resolves_id_case_and_tag_insensitively(self):
        for name in ("strata/qwen-IQ2_XS", "STRATA/QWEN-iq2_xs", "strata/qwen-IQ2_XS:latest"):
            self.assertEqual(self._find(name), QWEN, name)

    def test_resolves_strata_served_name(self):
        self.assertEqual(self._find("qwen3.8-flash-next-iq2_xs"), QWEN)
        self.assertEqual(self._find("swift-1.5-iq3_xxs"), "/strata/swift-IQ3_XXS")

    def test_gguf_still_wins_its_names(self):
        self.assertEqual(self._find("alpha-q4_k_m"), GGUF["path"])
        self.assertEqual(self._find("alpha"), GGUF["path"])

    def test_not_found_when_disabled(self):
        with patch("config.STRATA_ENABLED", False), \
             patch("api.llamaman.discover_models", return_value=[]):
            self.assertIsNone(llamaman._find_model_by_name("strata/qwen-IQ2_XS"))

    def test_auto_launch_uses_strata_engine_and_preset_options(self):
        storage = Mock()
        storage.get_preset.return_value = {"strata_vision": "cpu", "strata_kv": "int8"}
        storage.get_settings.return_value = {}
        launched = {"id": "new", "status": "starting", "port": 8003}
        with instances_lock:
            saved = dict(instances)
            instances.clear()
        try:
            with strata_on(), \
                 patch("api.llamaman.discover_models", return_value=[]), \
                 patch("api.llamaman.get_storage", return_value=storage), \
                 patch("api.llamaman.find_available_port", return_value=8003), \
                 patch("api.instances.launch_instance", return_value=(launched, None)) as launch:
                inst, err = llamaman._ensure_model_running("strata/qwen-IQ2_XS")
        finally:
            with instances_lock:
                instances.clear()
                instances.update(saved)
        self.assertIsNone(err)
        kw = launch.call_args.kwargs
        self.assertEqual(kw["model_path"], QWEN)
        self.assertEqual(kw["engine"], "strata")
        self.assertEqual(kw["ctx_size"], 32768)  # Strata's default, not llama.cpp's 4096
        self.assertEqual(kw["engine_options"], {"strata_vision": "cpu", "strata_kv": "int8"})


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
