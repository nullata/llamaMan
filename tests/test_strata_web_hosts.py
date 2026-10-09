# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Strata (no API key) refuses Host / Origin names it doesn't know, and the
instance-card link opens its web app on the host the browser used for
llamaMan. UI launches and restarts record that host (config["web_hosts"],
Strata only) and it goes into STRATA_ALLOWED_HOSTS. A restart also relaunches
from the file the instance was started from (engine_source_path)."""

import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.instances as instances_api
from core.engines.strata import StrataEngine
from core.state import instances, instances_lock

QWEN = "/strata/qwen-IQ2_XS"
SHARD = "/models/uploads/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"


class BrowserHostTests(unittest.TestCase):

    def _host(self, **headers):
        with Flask(__name__).test_request_context("/", headers=headers):
            return instances_api._browser_host()

    def test_parsing(self):
        self.assertEqual(self._host(Host="nullsrv:5005"), "nullsrv")
        self.assertEqual(self._host(Host="192.168.0.116:5005"), "192.168.0.116")
        self.assertEqual(self._host(Host="[fd00::5]:5005"), "fd00::5")
        self.assertEqual(self._host(Host="nullsrv:5005", **{"X-Forwarded-Host": "llm.home.lan"}), "llm.home.lan")


class AllowedHostsTests(unittest.TestCase):

    def test_recorded_hosts_allowed(self):
        with patch("config.CLUSTER_ADVERTISE_URL", ""), patch("config.STRATA_WEB_HOSTS", []), \
             patch("config.LLAMA_HOST_ADDR", ""):
            hosts = StrataEngine.allowed_hosts("llamaman-x", {"web_hosts": ["nullsrv", "NullSrv", "bad host"]})
        self.assertEqual(hosts, ["llamaman-x", "nullsrv"])


class _Instances(unittest.TestCase):
    def setUp(self):
        with instances_lock:
            self._saved = dict(instances)
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def _launch(self, **kw):
        fake = Mock()
        fake.id = "cid"
        with patch("api.instances.save_state"), \
             patch("api.instances._run_container", return_value=(fake, None)), \
             patch("api.instances.is_port_available", return_value=True), \
             patch("api.instances.find_available_port", return_value=9005), \
             patch("api.instances.start_idle_proxy"), patch("api.instances.create_gate"), \
             patch("api.instances._publish_cluster_heartbeat_safe"), \
             patch("core.engines.strata.local_shard_dir", return_value="/models/strata/iq2_xs/IQ2_XS"):
            inst, err = instances_api.launch_instance(port=8001, ctx_size=32768, **kw)
        self.assertIsNone(err)
        return inst


class RecordingTests(_Instances):

    def test_strata_launch_records_hosts(self):
        inst = self._launch(model_path=QWEN, engine="strata", web_hosts=["nullsrv", "nullsrv", ""])
        self.assertEqual(inst["config"]["web_hosts"], ["nullsrv"])

    def test_llamacpp_launch_does_not(self):
        inst = self._launch(model_path="/models/chat.gguf", web_hosts=["nullsrv"])
        self.assertNotIn("web_hosts", inst["config"])


class RestartTests(_Instances):

    def test_restart_relaunches_from_source_file_and_adds_host(self):
        with instances_lock:
            instances["s1"] = {"id": "s1", "model_path": QWEN, "model_name": "strata/qwen-IQ2_XS",
                               "port": 8001, "status": "stopped",
                               "config": {"engine": "strata", "ctx_size": 32768,
                                          "engine_source_path": SHARD, "web_hosts": ["192.168.0.116"]}}
        app = Flask(__name__)
        app.register_blueprint(instances_api.bp)
        launched = {"id": "s2", "model_path": QWEN, "port": 8001, "status": "starting", "config": {}}
        with patch("api.instances.launch_instance", return_value=(launched, None)) as launch, \
             patch("api.instances.is_port_available", return_value=True), \
             patch("api.instances.save_state"), \
             patch("api.instances._merge_preset_into_config", side_effect=lambda p, c: dict(c)), \
             patch("api.instances._admin_ui_enforces_eviction", return_value=False), \
             patch("api.instances._would_ui_launch_exceed_limit", return_value=False), \
             patch("api.instances.release_instance_reservations", create=True):
            resp = app.test_client().post("/api/instances/s1/restart", json={},
                                          headers={"Host": "nullsrv:5005"})
        self.assertEqual(resp.status_code, 201, resp.get_json())
        kw = launch.call_args.kwargs
        self.assertEqual(kw["model_path"], SHARD)
        self.assertEqual(kw["engine"], "strata")
        self.assertEqual(kw["web_hosts"], ["192.168.0.116", "nullsrv"])


if __name__ == "__main__":
    unittest.main()
