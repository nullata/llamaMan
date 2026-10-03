# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for the Strata engine (core/engines/strata.py).

Strata is configured only through env vars read by its image's entrypoint,
serves one request at a time, needs memlock unlimited, and runs on NVIDIA
only. These tests pin: the catalogue and model-path parsing, the container
spec (env / mounts / ulimits / GPU), REINSTALL fingerprinting, capability
enforcement (gate forced to 1, llama.cpp-only fields rejected by the API),
vendor gating, and load-stage parsing of its first-start log.
"""

import json
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
from core.engines import ENGINES, engine_for_path, get_engine, load_timeout_for, validate_launch
from core.engines import strata as S
from core.state import instances, instances_lock

STRATA = ENGINES["strata"]
QWEN = "/strata/qwen-IQ2_XS"


class _TmpDirs(unittest.TestCase):
    """Isolate DATA_DIR (setup fingerprints) and MODELS_DIR (shard reuse)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self._tmp.name, "data")
        self.models_dir = os.path.join(self._tmp.name, "models")
        os.makedirs(self.data_dir)
        os.makedirs(self.models_dir)
        self._patches = [
            patch("config.DATA_DIR", self.data_dir),
            patch("config.MODELS_DIR", self.models_dir),
            patch("config.HOST_MODELS_DIR", "/host/models"),
            patch("config.HOST_STRATA_DATA_DIR", ""),
            patch("config.STRATA_DATA_VOLUME", "llamaman-strata-data"),
            patch("config.STRATA_IMAGE", "strata:latest"),
            patch("config.LLAMA_GPU_DEVICES", ""),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()


class CatalogueTests(unittest.TestCase):

    def test_catalogue_matches_upstream_families_and_sizes(self):
        ids = {m["id"] for m in S.catalogue()}
        self.assertEqual(ids, {
            "strata/qwen-Q2_0", "strata/qwen-IQ2_XS", "strata/qwen-IQ3_XXS", "strata/qwen-IQ3_S",
            "strata/swift-IQ2_XS", "strata/swift-IQ3_XXS",
            "strata/coder-IQ1_M",
            "strata/unsloth-UD-Q4_K_XL",
        })

    def test_served_name_matches_setup_py(self):
        # setup.py: model_name = f"{fam['name']}-{model.lower()}"
        by_id = {m["id"]: m for m in S.catalogue()}
        self.assertEqual(by_id["strata/qwen-IQ2_XS"]["served_name"], "qwen3.8-flash-next-iq2_xs")
        self.assertEqual(by_id["strata/swift-IQ3_XXS"]["served_name"], "swift-1.5-iq3_xxs")

    def test_parse_model_path(self):
        self.assertEqual(S.parse_model_path(QWEN), ("qwen", "IQ2_XS"))
        self.assertEqual(S.parse_model_path("strata/QWEN-iq2_xs"), ("qwen", "IQ2_XS"))
        self.assertEqual(S.parse_model_path("/strata/unsloth-UD-Q4_K_XL"), ("unsloth", "UD-Q4_K_XL"))
        for bad in ("/strata/swift-Q2_0", "/strata/coder-IQ2_XS", "/strata/qwen",
                    "/strata/nope-IQ2_XS", "/models/qwen-IQ2_XS.gguf", None, ""):
            self.assertIsNone(S.parse_model_path(bad), bad)

    def test_setup_tag_matches_entrypoint(self):
        # docker-entrypoint.sh: qwen has an empty family prefix.
        self.assertEqual(S.setup_tag("qwen", "IQ2_XS"), "iq2_xs")
        self.assertEqual(S.setup_tag("coder", "IQ1_M"), "coder-iq1_m")
        self.assertEqual(S.setup_tag("unsloth", "UD-Q4_K_XL"), "unsloth-ud-q4_k_xl")

    def test_repo_files(self):
        self.assertEqual(S.repo_files("qwen", "IQ2_XS"), [
            "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
            "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
        ])
        self.assertEqual(S.repo_files("swift", "IQ2_XS")[0],
                         "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf")
        self.assertEqual(len(S.repo_files("unsloth", "UD-Q4_K_XL")), 4)

    def test_engine_owns_only_virtual_paths(self):
        self.assertEqual(engine_for_path(QWEN), "strata")
        self.assertIsNone(engine_for_path("strata/qwen-IQ2_XS"))  # ids aren't paths
        self.assertIsNone(engine_for_path("/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"))

    def test_served_model_names(self):
        self.assertEqual(STRATA.served_model_names(QWEN),
                         ["strata/qwen-iq2_xs", "qwen3.8-flash-next-iq2_xs"])
        self.assertEqual(STRATA.display_name(QWEN), "strata/qwen-IQ2_XS")


class StrataContainerSpecTests(_TmpDirs):

    def _spec(self, config=None, model_path=QWEN, vendor="cuda"):
        cfg = {"engine": "strata", "ctx_size": 32768, "strata_vision": "no",
               "strata_kv": "", "strata_low_ram": "auto", **(config or {})}
        fake = Mock()
        fake.id = "cid"
        client = Mock()
        client.containers.run.return_value = fake
        with patch("api.instances.get_docker_client", return_value=client), \
             patch("api.instances.ensure_docker_network"), \
             patch("api.instances._start_log_relay"), \
             patch("api.instances.get_vendor", return_value=vendor):
            c, err = instances_api._run_container("inst-1", "llamaman-inst-1", model_path,
                                                  9001, cfg, "/tmp/x.log")
        self.assertIsNone(err)
        return client.containers.run.call_args.kwargs

    def test_basic_spec(self):
        kw = self._spec()
        self.assertEqual(kw["image"], "strata:latest")
        self.assertIsNone(kw["command"])  # entrypoint reads env only
        self.assertEqual(kw["ports"], {8080: 9001})
        self.assertEqual(kw["network"], "llamaman-net")
        env = kw["environment"]
        self.assertEqual(env["FAMILY"], "qwen")
        self.assertEqual(env["MODEL"], "IQ2_XS")
        self.assertEqual(env["CONTEXT"], "32768")
        self.assertEqual(env["VISION"], "no")
        self.assertEqual(env["PORT"], "8080")
        self.assertEqual(env["HOST"], "0.0.0.0")
        self.assertEqual(env["LOW_RAM"], "auto")
        self.assertNotIn("KV", env)
        self.assertNotIn("API_KEY", env)
        self.assertEqual(json.loads(kw["labels"]["llamaman.config"])["engine"], "strata")
        self.assertEqual(kw["labels"]["llamaman.model_path"], QWEN)

    def test_memlock_unlimited_always(self):
        (ul,) = self._spec()["ulimits"]
        self.assertEqual((ul["Name"], ul["Soft"], ul["Hard"]), ("memlock", -1, -1))

    def test_nvidia_device_requests_even_off_vendor(self):
        # Availability refuses non-NVIDIA nodes at the API; the spec itself
        # never takes the ROCm / Intel / Vulkan device branches.
        for vendor in ("cuda", "rocm", None):
            kw = self._spec(vendor=vendor)
            self.assertIn("device_requests", kw)
            self.assertNotIn("devices", kw)
            self.assertNotIn("group_add", kw)

    def test_gpu_pinning_renumbers_inside_container(self):
        kw = self._spec({"gpu_devices": "2"})
        self.assertEqual(kw["device_requests"][0]["DeviceIDs"], ["2"])
        self.assertEqual(kw["environment"]["GPU"], "0")
        self.assertNotIn("GPUS", kw["environment"])
        kw = self._spec({"gpu_devices": "1,3"})
        self.assertEqual(kw["environment"]["GPUS"], "0,1")
        for d in ("", "all"):
            env = self._spec({"gpu_devices": d})["environment"]
            self.assertNotIn("GPU", env)
            self.assertNotIn("GPUS", env)

    def test_memory_limit_forces_low_ram(self):
        kw = self._spec({"memory_limit": "48g", "strata_low_ram": "off"})
        self.assertEqual(kw["mem_limit"], "48g")
        self.assertEqual(kw["environment"]["LOW_RAM"], "on")

    def test_options_reach_env(self):
        env = self._spec({"ctx_size": 131072, "strata_vision": "cpu", "strata_kv": "k8v4",
                          "strata_low_ram": "on"})["environment"]
        self.assertEqual((env["CONTEXT"], env["VISION"], env["KV"], env["LOW_RAM"]),
                         ("131072", "cpu", "k8v4", "on"))

    def test_no_llamacpp_flags_or_nano_cpus(self):
        kw = self._spec({"threads": 8, "n_gpu_layers": 20})
        self.assertNotIn("nano_cpus", kw)
        self.assertNotIn("--n-gpu-layers", json.dumps(kw.get("command")))

    def test_data_volume_and_logs_mounted(self):
        vols = self._spec()["volumes"]
        self.assertEqual(vols["llamaman-strata-data"], {"bind": "/data", "mode": "rw"})
        self.assertEqual(len(vols), 2)  # data + logs; no shard mount without shards

    def test_host_data_dir_overrides_volume(self):
        with patch("config.HOST_STRATA_DATA_DIR", "/srv/strata"):
            vols = self._spec()["volumes"]
        self.assertIn("/srv/strata", vols)
        self.assertNotIn("llamaman-strata-data", vols)

    def test_existing_shards_are_bind_mounted(self):
        # llamaman's downloader keeps the repo's size folder: <tag>/IQ2_XS/<shards>
        d = os.path.join(self.models_dir, "strata", "iq2_xs", "IQ2_XS")
        os.makedirs(d)
        for name in S.shard_files("qwen", "IQ2_XS"):
            open(os.path.join(d, name), "w").close()
        vols = self._spec()["volumes"]
        self.assertEqual(vols["/host/models/strata/iq2_xs/IQ2_XS"],
                         {"bind": "/data/models/iq2_xs", "mode": "rw"})

    def test_partial_shard_set_not_mounted(self):
        d = os.path.join(self.models_dir, "strata", "iq2_xs")
        os.makedirs(d)
        open(os.path.join(d, S.shard_files("qwen", "IQ2_XS")[0]), "w").close()
        self.assertEqual(len(self._spec()["volumes"]), 2)

    def test_missing_image_message_says_build(self):
        msg = STRATA.image_missing_message("strata:latest")
        self.assertIn("docker build -t strata:latest .", msg)


class ReinstallFingerprintTests(_TmpDirs):

    def _cfg(self, **kw):
        return {"engine": "strata", "ctx_size": 32768, "strata_vision": "no", **kw}

    def test_first_launch_reinstalls(self):
        self.assertEqual(STRATA.environment(QWEN, self._cfg()).get("REINSTALL"), "1")

    def test_unchanged_settings_after_healthy_skip_reinstall(self):
        STRATA.on_ready({"model_path": QWEN, "config": self._cfg()})
        self.assertNotIn("REINSTALL", STRATA.environment(QWEN, self._cfg()))

    def test_changed_context_reinstalls(self):
        STRATA.on_ready({"model_path": QWEN, "config": self._cfg()})
        self.assertEqual(STRATA.environment(QWEN, self._cfg(ctx_size=65536)).get("REINSTALL"), "1")

    def test_fingerprint_is_per_model_and_per_volume(self):
        STRATA.on_ready({"model_path": QWEN, "config": self._cfg()})
        self.assertEqual(STRATA.environment("/strata/qwen-Q2_0", self._cfg()).get("REINSTALL"), "1")
        with patch("config.HOST_STRATA_DATA_DIR", "/other"):
            self.assertEqual(STRATA.environment(QWEN, self._cfg()).get("REINSTALL"), "1")


class AvailabilityTests(unittest.TestCase):

    def test_disabled_by_default(self):
        with patch("config.STRATA_ENABLED", False):
            ok, reason = STRATA.availability("cuda")
        self.assertFalse(ok)
        self.assertIn("STRATA_ENABLED", reason)

    def test_nvidia_only(self):
        with patch("config.STRATA_ENABLED", True):
            self.assertEqual(STRATA.availability("cuda"), (True, ""))
            for vendor in ("rocm", "intel", "vulkan", None):
                ok, reason = STRATA.availability(vendor)
                self.assertFalse(ok, vendor)
                self.assertIn("NVIDIA", reason)

    def test_load_timeout(self):
        with patch("config.STRATA_LOAD_TIMEOUT", 4321):
            self.assertEqual(STRATA.load_timeout(), 4321)
            self.assertEqual(load_timeout_for({"model_path": QWEN, "config": {"engine": "strata"}}), 4321)


class ValidateLaunchTests(unittest.TestCase):

    def setUp(self):
        self._p = patch("config.STRATA_ENABLED", True)
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_engine_inferred_from_model_path(self):
        name, opts, err = validate_launch({"ctx_size": 32768}, QWEN, "cuda")
        self.assertIsNone(err)
        self.assertEqual(name, "strata")
        self.assertEqual(opts, {"strata_vision": "no", "strata_kv": "", "strata_low_ram": "auto"})

    def test_llamacpp_form_defaults_accepted(self):
        # What the launch form sends for fields it hides: their defaults.
        body = {"ctx_size": 32768, "n_gpu_layers": -1, "n_cpu_moe_layers": 0, "threads": "",
                "parallel": None, "extra_args": "", "spec_enabled": False, "split_mode": "layer",
                "flash_attn": "auto", "reasoning_format": "auto", "load_mode": "auto",
                "cache_type_k": "f16", "cache_type_v": "", "dry_enabled": False,
                "embedding_model": False, "mmproj_enabled": False, "tensor_split": ""}
        self.assertIsNone(validate_launch(body, QWEN, "cuda")[2])

    def test_llamacpp_only_fields_rejected(self):
        for field, value in (("n_gpu_layers", 20), ("spec_enabled", True), ("threads", 8),
                             ("flash_attn", "on"), ("extra_args", "--jinja"),
                             ("embedding_model", True), ("cache_type_k", "q8_0")):
            err = validate_launch({"ctx_size": 32768, field: value}, QWEN, "cuda")[2]
            self.assertIsNotNone(err, field)
            self.assertIn(field, err)

    def test_vendor_gating(self):
        err = validate_launch({"ctx_size": 32768}, QWEN, "rocm")[2]
        self.assertIn("NVIDIA", err)

    def test_bad_options(self):
        for body in ({"strata_vision": "maybe"}, {"strata_kv": "q8"}, {"strata_low_ram": "x"}):
            self.assertIsNotNone(validate_launch({"ctx_size": 1, **body}, QWEN, "cuda")[2])
        # The Unsloth file has no image encoder.
        err = validate_launch({"ctx_size": 1, "strata_vision": "yes"},
                              "/strata/unsloth-UD-Q4_K_XL", "cuda")[2]
        self.assertIn("image encoder", err)

    def test_unknown_strata_model(self):
        _, _, err = validate_launch({"ctx_size": 1, "engine": "strata"}, "/strata/qwen-IQ9", "cuda")
        self.assertIsNotNone(err)

    def test_llamacpp_unaffected(self):
        self.assertEqual(validate_launch({"ctx_size": 4096, "n_gpu_layers": 20},
                                         "/models/chat.gguf", "rocm"), ("llamacpp", {}, None))


class _Instances(unittest.TestCase):
    def setUp(self):
        with instances_lock:
            self._saved = {k: dict(v) for k, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)


class CapabilityEnforcementTests(_Instances):

    def _launch(self, **kw):
        fake = Mock()
        fake.id = "cid"
        with patch("api.instances.save_state"), \
             patch("api.instances._run_container", return_value=(fake, None)) as run_mock, \
             patch("api.instances.is_port_available", return_value=True), \
             patch("api.instances.find_available_port", return_value=9005), \
             patch("api.instances.start_idle_proxy") as proxy_mock, \
             patch("api.instances.create_gate") as gate_mock, \
             patch("api.instances._publish_cluster_heartbeat_safe"):
            inst, err = instances_api.launch_instance(model_path=QWEN, port=8001,
                                                      ctx_size=32768, **kw)
        self.assertIsNone(err)
        return inst, run_mock, proxy_mock, gate_mock

    def test_gate_forced_to_one(self):
        for requested in (0, 1, 4):
            with instances_lock:
                instances.clear()
            inst, _, proxy_mock, gate_mock = self._launch(max_concurrent=requested)
            self.assertEqual(inst["config"]["max_concurrent"], 1)
            self.assertEqual(gate_mock.call_args.args[1], 1)
            # The gate needs the sidecar proxy: public port -> internal port.
            proxy_mock.assert_called_once_with(inst["id"], 8001, 9005)

    def test_one_live_instance_per_model(self):
        self._launch()
        fake = Mock()
        fake.id = "cid2"
        with patch("api.instances._run_container", return_value=(fake, None)) as run_mock, \
             patch("api.instances.is_port_available", return_value=True):
            inst, err = instances_api.launch_instance(model_path=QWEN, port=8002, ctx_size=32768)
        self.assertIsNone(inst)
        self.assertIn("one instance per model", err)
        run_mock.assert_not_called()

    def test_strata_launch_records_engine_and_options(self):
        inst, run_mock, _, _ = self._launch(engine_options={"strata_vision": "cpu", "strata_kv": "",
                                                            "strata_low_ram": "auto"})
        cfg = inst["config"]
        self.assertEqual(cfg["engine"], "strata")
        self.assertEqual(cfg["strata_vision"], "cpu")
        self.assertEqual(cfg["image"], STRATA.default_image())
        self.assertEqual(inst["model_name"], "strata/qwen-IQ2_XS")

    def test_embedding_and_spec_cleared(self):
        inst, _, _, _ = self._launch(embedding_model=True, spec_enabled=True)
        self.assertFalse(inst["config"]["embedding_model"])
        self.assertFalse(inst["config"]["spec_enabled"])

    def test_preset_merge_cannot_lift_the_gate(self):
        storage = Mock()
        storage.get_preset.return_value = {"max_concurrent": 0, "strata_vision": "yes", "ctx_size": 65536}
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config(
                QWEN, {"engine": "strata", "max_concurrent": 1, "strata_vision": "no"})
        self.assertEqual(merged["max_concurrent"], 1)
        self.assertEqual(merged["strata_vision"], "yes")  # engine options merge
        self.assertEqual(merged["ctx_size"], 65536)


class StrataRoutesTests(_Instances):

    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(instances_api.bp)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()
        self._p = patch("config.STRATA_ENABLED", True)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        super().tearDown()

    @patch("api.instances.get_vendor", return_value="cuda")
    @patch("api.instances.launch_instance")
    def test_create_route_forwards_engine_and_options(self, launch_mock, _v):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post("/api/instances", json={
                "model_path": QWEN, "port": 8001, "ctx_size": 65536, "strata_kv": "int8",
            })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        kw = launch_mock.call_args.kwargs
        self.assertEqual(kw["engine"], "strata")
        self.assertEqual(kw["engine_options"]["strata_kv"], "int8")

    @patch("api.instances.get_vendor", return_value="cuda")
    @patch("api.instances.launch_instance")
    def test_create_route_rejects_hidden_fields(self, launch_mock, _v):
        resp = self.client.post("/api/instances", json={
            "model_path": QWEN, "port": 8001, "ctx_size": 32768, "n_gpu_layers": 30,
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("n_gpu_layers", resp.get_json()["error"])
        launch_mock.assert_not_called()

    @patch("api.instances.get_vendor", return_value="rocm")
    @patch("api.instances.launch_instance")
    def test_create_route_refuses_non_nvidia_node(self, launch_mock, _v):
        resp = self.client.post("/api/instances", json={
            "model_path": QWEN, "port": 8001, "ctx_size": 32768,
        })
        self.assertEqual(resp.status_code, 400)
        launch_mock.assert_not_called()

    @patch("api.instances.save_state")
    @patch("api.instances.release_instance_reservations")
    @patch("api.instances.is_port_available", return_value=True)
    @patch("api.instances._admin_ui_enforces_eviction", return_value=False)
    @patch("api.instances._would_ui_launch_exceed_limit", return_value=False)
    @patch("api.instances._merge_preset_into_config", side_effect=lambda _p, cfg: cfg)
    @patch("api.instances.launch_instance")
    def test_restart_preserves_options(self, launch_mock, *_):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with instances_lock:
            instances["inst-1"] = {
                "id": "inst-1", "model_path": QWEN, "port": 8001, "status": "stopped", "stats": {},
                "config": {"engine": "strata", "ctx_size": 65536, "strata_vision": "cpu",
                           "strata_kv": "q4_0", "strata_low_ram": "on", "max_concurrent": 1},
            }
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post("/api/instances/inst-1/restart", json={})
        self.assertIn(resp.status_code, (200, 201))
        kw = launch_mock.call_args.kwargs
        self.assertEqual(kw["engine"], "strata")
        self.assertEqual(kw["engine_options"], {"strata_vision": "cpu", "strata_kv": "q4_0",
                                                "strata_low_ram": "on"})

    def test_preset_save_strata(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/strata/qwen-IQ2_XS", json={
                "ctx_size": 65536, "max_concurrent": 3, "strata_vision": "yes",
            })
            self.assertEqual(resp.status_code, 200, resp.get_json())
            key, saved = storage.save_preset.call_args.args
            self.assertEqual(key, QWEN)
            self.assertEqual(saved["engine"], "strata")
            self.assertEqual(saved["strata_vision"], "yes")
            self.assertEqual(saved["max_concurrent"], 1)

            resp = self.client.put("/api/presets/strata/qwen-IQ2_XS", json={
                "ctx_size": 65536, "spec_enabled": True,
            })
            self.assertEqual(resp.status_code, 400)


class OrphanAdoptionTests(_Instances):

    def test_adopts_strata_container_by_label(self):
        cfg = {"engine": "strata", "ctx_size": 32768, "max_concurrent": 1}
        container = Mock()
        container.id = "strata-cid"
        container.name = "llamaman-abcd1234"
        container.labels = {
            "llamaman.instance_id": "abcd1234-0000", "llamaman.model_path": QWEN,
            "llamaman.port": "9003", "llamaman.config": json.dumps(cfg),
        }
        storage = Mock()
        storage.get_preset.return_value = {}
        from core import state
        with patch("core.helpers.list_llama_containers", return_value=[container]), \
             patch("storage.get_storage", return_value=storage), \
             patch("core.state.start_container_log_relay"), \
             patch("core.state.save_state"):
            self.assertEqual(state.adopt_orphans(), 1)
        inst = instances["abcd1234-0000"]
        self.assertEqual(inst["model_name"], "strata/qwen-IQ2_XS")
        self.assertEqual(inst["config"]["engine"], "strata")
        self.assertEqual(get_engine(inst["config"], inst["model_path"]).name, "strata")


class LoadStageTests(unittest.TestCase):

    def test_nothing_yet(self):
        self.assertIsNone(S.load_stage_from_log(""))
        self.assertIsNone(S.load_stage_from_log("random noise\n"))

    def test_setting_up(self):
        st = S.load_stage_from_log("Setting up iq2_xs: downloading the model\n\n=== Step 1: checking your PC ===\n")
        self.assertEqual(st["stage"], "setting up")

    def test_downloading_with_progress(self):
        log = ("=== Step 5: downloading Qwen3.8-Flash-Next IQ2_XS ===\n"
               "\r  Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf:   1.00 / 49.10 GB (2%)   "
               "\r  Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf:  12.34 / 49.10 GB (25%)   ")
        st = S.load_stage_from_log(log)
        self.assertEqual(st["stage"], "downloading")
        self.assertEqual(st["percent"], 25)
        self.assertIn("12.34 / 49.10 GB", st["detail"])

    def test_preparing_pack(self):
        st = S.load_stage_from_log("=== Step 5: downloading x ===\n=== Step 6: preparing the model for Strata ===\n")
        self.assertEqual(st["stage"], "preparing pack")

    def test_loading(self):
        log = "=== Step 7: writing the start script ===\nloading the model (the first start takes a minute or two) ...\n"
        self.assertEqual(S.load_stage_from_log(log)["stage"], "loading")
        # A later start skips setup entirely and goes straight to loading.
        self.assertEqual(S.load_stage_from_log("loading the model ...\n")["stage"], "loading")

    def test_failed(self):
        log = "=== Step 5: downloading x ===\n  [X]  cannot reach huggingface.co\nSetup stopped. Fix the item above"
        self.assertEqual(S.load_stage_from_log(log)["stage"], "setup failed")

    def test_public_instance_exposes_stage_while_starting(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write("=== Step 6: preparing the model for Strata ===\n")
            path = f.name
        try:
            inst = {"id": "i", "model_path": QWEN, "status": "starting", "log_file": path,
                    "config": {"engine": "strata"}, "port": 8001}
            self.assertEqual(instances_api._public_instance(inst)["load_stage"]["stage"], "preparing pack")
            inst["status"] = "healthy"
            self.assertNotIn("load_stage", instances_api._public_instance(inst))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
