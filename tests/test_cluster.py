import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from flask import Flask

import config
import core.cluster as cluster
from storage.json_backend import JsonBackend


class NodeIdentityTests(unittest.TestCase):
    def test_node_id_is_env_value(self):
        with patch.object(config, "LLAMAMAN_NODE_NAME", "srv-xyz"):
            self.assertEqual(cluster.get_node_id(), "srv-xyz")

    def test_local_descriptor_uses_id_as_name(self):
        # id and name are the same operator-chosen string.
        with patch.object(config, "LLAMAMAN_NODE_NAME", "srv-xyz"):
            desc = cluster.local_node_descriptor()
        self.assertEqual(desc["node_id"], "srv-xyz")
        self.assertEqual(desc["node_name"], "srv-xyz")

    def test_config_refuses_to_load_without_env(self):
        # config.py validates LLAMAMAN_NODE_NAME at import time and exits 78
        # (EX_CONFIG) with a banner on stderr. Test in a subprocess because the
        # current process already imported config successfully.
        env = {k: v for k, v in os.environ.items() if k != "LLAMAMAN_NODE_NAME"}
        result = subprocess.run(
            [sys.executable, "-c", "import config"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 78)
        self.assertIn("LLAMAMAN_NODE_NAME", result.stderr)

    def test_cluster_disabled_without_secret(self):
        with patch.object(config, "CLUSTER_ENABLED", True), \
             patch.object(config, "CLUSTER_SECRET", ""):
            self.assertFalse(cluster.is_cluster_enabled())
        with patch.object(config, "CLUSTER_ENABLED", True), \
             patch.object(config, "CLUSTER_SECRET", "s3cr3t"):
            self.assertTrue(cluster.is_cluster_enabled())


class ClusterSecretTests(unittest.TestCase):
    def test_verify_cluster_secret(self):
        with patch.object(config, "CLUSTER_SECRET", "topsecret"):
            self.assertTrue(cluster.verify_cluster_secret("topsecret"))
            self.assertFalse(cluster.verify_cluster_secret("wrong"))
            self.assertFalse(cluster.verify_cluster_secret(""))
        with patch.object(config, "CLUSTER_SECRET", ""):
            self.assertFalse(cluster.verify_cluster_secret("anything"))

    def test_cluster_request_attaches_secret_header_and_builds_url(self):
        captured = {}

        def fake_request(method, url, headers=None, timeout=None, **kwargs):
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = headers
            return "resp"

        node = {"node_id": "n1", "advertise_url": "http://srv2:5000/"}
        with patch.object(config, "CLUSTER_SECRET", "abc"), \
             patch.object(cluster._session, "request", side_effect=fake_request):
            out = cluster.cluster_request(node, "GET", "/api/instances")
        self.assertEqual(out, "resp")
        self.assertEqual(captured["url"], "http://srv2:5000/api/instances")
        # Secret rides its own header, never as a client Bearer.
        self.assertEqual(captured["headers"]["X-Cluster-Secret"], "abc")
        self.assertNotIn("Authorization", captured["headers"])

    def test_cluster_secret_is_not_a_client_bearer(self):
        import api.auth as auth
        with patch.object(config, "CLUSTER_SECRET", "the-secret"):
            # The cluster secret presented as a Bearer is rejected (not an API key).
            self.assertIsNotNone(auth.verify_bearer_token("Bearer the-secret", strict=True))

    def test_cluster_request_rejects_missing_advertise_url(self):
        with self.assertRaises(ValueError):
            cluster.cluster_request({"node_id": "x"}, "GET", "/api/instances")

    def test_cluster_request_prefers_caller_auth(self):
        # A forwarded client API key must override the default cluster secret.
        captured = {}

        def fake_request(method, url, headers=None, timeout=None, **kwargs):
            captured["headers"] = headers
            return "resp"

        node = {"node_id": "n", "advertise_url": "http://x:5000"}
        with patch.object(config, "CLUSTER_SECRET", "the-secret"), \
             patch.object(cluster._session, "request", side_effect=fake_request):
            cluster.cluster_request(node, "POST", "/p",
                                    headers={"Authorization": "Bearer llm-clientkey"})
        self.assertEqual(captured["headers"]["Authorization"], "Bearer llm-clientkey")


class JsonRegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"),
            os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"),
            os.path.join(d, "settings.json"),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _node(self, nid, name="srv", url="http://srv:5000"):
        return {"node_id": nid, "node_name": name, "advertise_url": url,
                "vendor": "cuda", "llama_image": "img"}

    def test_register_list_get_remove(self):
        self.assertEqual(self.storage.list_nodes(), [])

        self.storage.register_node(self._node("a"), snapshot={"cpu": 1})
        self.storage.register_node(self._node("b", name="srv2"))

        nodes = {n["node_id"]: n for n in self.storage.list_nodes()}
        self.assertEqual(set(nodes), {"a", "b"})
        self.assertEqual(nodes["a"]["snapshot"], {"cpu": 1})
        self.assertEqual(nodes["b"]["snapshot"], {})
        self.assertTrue(nodes["a"]["last_heartbeat_at"])

        self.assertIsNone(self.storage.get_node("missing"))
        self.assertEqual(self.storage.get_node("b")["node_name"], "srv2")

        self.storage.remove_node("a")
        self.assertIsNone(self.storage.get_node("a"))
        self.assertEqual([n["node_id"] for n in self.storage.list_nodes()], ["b"])

    def test_register_updates_identity_and_preserves_snapshot(self):
        self.storage.register_node(self._node("a"), snapshot={"v": 1})
        first_hb = self.storage.get_node("a")["last_heartbeat_at"]
        # Identity-only re-register keeps the prior snapshot
        self.storage.register_node(self._node("a", name="renamed", url="http://new:5000"))
        node = self.storage.get_node("a")
        self.assertEqual(node["node_name"], "renamed")
        self.assertEqual(node["advertise_url"], "http://new:5000")
        self.assertEqual(node["snapshot"], {"v": 1})
        self.assertGreaterEqual(node["last_heartbeat_at"], first_hb)


class SnapshotAndOnlineTests(unittest.TestCase):
    def test_is_online_window(self):
        import api.cluster as cluster_api
        from core.timeutil import now_iso
        self.assertTrue(cluster_api._is_online(now_iso()))
        self.assertFalse(cluster_api._is_online("2000-01-01T00:00:00.000Z"))
        self.assertFalse(cluster_api._is_online(None))
        self.assertFalse(cluster_api._is_online("not-a-date"))

    def test_build_local_snapshot_shape(self):
        import api.cluster as cluster_api
        from core.state import instances, instances_lock, downloads, downloads_lock
        with instances_lock:
            saved_i = dict(instances); instances.clear()
        with downloads_lock:
            saved_d = dict(downloads); downloads.clear()
        try:
            snap = cluster_api.build_local_snapshot()
        finally:
            with instances_lock:
                instances.update(saved_i)
            with downloads_lock:
                downloads.update(saved_d)
        self.assertEqual(set(snap), {"system", "gpus", "instances", "downloads", "models", "updated_at"})
        self.assertEqual(snap["instances"], [])
        self.assertIsInstance(snap["gpus"], list)
        self.assertIsInstance(snap["models"], list)


class NodeScopedStateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"),
            os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"),
            os.path.join(d, "settings.json"),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _ids(self, rows):
        return sorted(r["id"] for r in rows)

    def test_legacy_rows_adopted_by_local_node(self):
        # Simulate a pre-cluster state file (no node_id on rows).
        self.storage.save_state([{"id": "i1"}, {"id": "i2"}], [{"id": "d1"}], node_id=None)
        # A scoped load adopts the untagged rows.
        self.assertEqual(self._ids(self.storage.load_instances("nodeA")), ["i1", "i2"])
        self.assertEqual(self._ids(self.storage.load_downloads("nodeA")), ["d1"])
        # Re-saving under nodeA stamps them; nodeB no longer sees them.
        self.storage.save_state([{"id": "i1"}, {"id": "i2"}], [{"id": "d1"}], node_id="nodeA")
        self.assertEqual(self._ids(self.storage.load_instances("nodeB")), [])
        self.assertEqual(self._ids(self.storage.load_instances("nodeA")), ["i1", "i2"])

    def test_nodes_do_not_clobber_each_other(self):
        self.storage.save_state([{"id": "a1"}], [], node_id="A")
        self.storage.save_state([{"id": "b1"}], [], node_id="B")
        self.assertEqual(self._ids(self.storage.load_instances("A")), ["a1"])
        self.assertEqual(self._ids(self.storage.load_instances("B")), ["b1"])
        # A stops everything; B's rows survive.
        self.storage.save_state([], [], node_id="A")
        self.assertEqual(self._ids(self.storage.load_instances("A")), [])
        self.assertEqual(self._ids(self.storage.load_instances("B")), ["b1"])
        self.assertEqual(self._ids(self.storage.load_instances(None)), ["b1"])

    def test_migration_registered(self):
        from core.migrations import CURRENT_SCHEMA_VERSION, MIGRATIONS
        self.assertGreaterEqual(CURRENT_SCHEMA_VERSION, 3)
        self.assertIn(3, MIGRATIONS)


class ClusterApiTests(unittest.TestCase):
    def setUp(self):
        import api.cluster as cluster_api
        self.cluster_api = cluster_api
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"),
            os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"),
            os.path.join(d, "settings.json"),
        )
        self._orig_data_dir = config.DATA_DIR
        config.DATA_DIR = d

        self.app = Flask(__name__)
        self.app.register_blueprint(cluster_api.bp)
        self.client = self.app.test_client()

    def tearDown(self):
        config.DATA_DIR = self._orig_data_dir
        self._tmp.cleanup()

    def test_nodes_disabled(self):
        with patch.object(config, "CLUSTER_ENABLED", False):
            resp = self.client.get("/api/cluster/nodes")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertFalse(body["enabled"])
        self.assertEqual(body["nodes"], [])

    def test_proxy_unknown_node(self):
        with patch.object(config, "CLUSTER_ENABLED", True), \
             patch.object(config, "CLUSTER_SECRET", "shh"), \
             patch("api.cluster.get_storage", return_value=self.storage):
            resp = self.client.get("/api/cluster/nodes/ghost/proxy/api/instances")
        self.assertEqual(resp.status_code, 404)

    def test_proxy_disabled(self):
        with patch.object(config, "CLUSTER_ENABLED", False):
            resp = self.client.post("/api/cluster/nodes/x/proxy/api/instances")
        self.assertEqual(resp.status_code, 400)

    def test_proxy_forwards_with_auth(self):
        self.storage.register_node(
            {"node_id": "peer", "node_name": "srv2", "advertise_url": "http://srv2:5000",
             "vendor": "rocm", "llama_image": "img"})

        captured = {}

        class FakeResp:
            status_code = 201
            content = b'{"ok": true}'
            headers = {"Content-Type": "application/json"}

        def fake_cluster_request(node, method, path, **kwargs):
            captured["node"] = node
            captured["method"] = method
            captured["path"] = path
            captured["headers"] = kwargs.get("headers")
            return FakeResp()

        with patch.object(config, "CLUSTER_ENABLED", True), \
             patch.object(config, "CLUSTER_SECRET", "shh"), \
             patch("api.cluster.get_storage", return_value=self.storage), \
             patch("core.cluster.cluster_request", side_effect=fake_cluster_request):
            resp = self.client.post(
                "/api/cluster/nodes/peer/proxy/api/instances?x=1",
                json={"model_path": "/m"},
            )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/api/instances?x=1")
        self.assertEqual(captured["node"]["advertise_url"], "http://srv2:5000")

    def test_nodes_enabled_registers_self(self):
        with patch.object(config, "CLUSTER_ENABLED", True), \
             patch.object(config, "CLUSTER_SECRET", "shh"), \
             patch.object(config, "LLAMAMAN_NODE_NAME", "srv-test"), \
             patch("api.cluster.get_storage", return_value=self.storage):
            resp = self.client.get("/api/cluster/nodes")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body["enabled"])
        self.assertEqual(len(body["nodes"]), 1)
        node = body["nodes"][0]
        self.assertTrue(node["is_self"])
        self.assertTrue(node["online"])
        self.assertEqual(node["node_name"], "srv-test")
        self.assertEqual(node["node_id"], body["self_id"])
        self.assertEqual(node["node_id"], "srv-test")


class _FakeGate:
    """Minimal RequestGate stand-in for the cluster wrapper."""
    def __init__(self, acquire_result=False, draining=False):
        self._acquire_result = acquire_result
        self._draining = draining  # the wrapper's find_target reads this

    def acquire(self, timeout=None):
        return self._acquire_result

    def acquire_or_overflow(self, timeout, poll, find_target, do_forward):
        if self._acquire_result:
            return ("acquired", None)
        target = find_target()
        if target is None:
            return ("rejected", None)
        return ("overflow", do_forward(target))


class DispatchTests(unittest.TestCase):
    def setUp(self):
        import api.cluster as cluster_api
        from core.state import instances, instances_lock
        self.cluster_api = cluster_api
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"),
        )
        self._orig_data_dir = config.DATA_DIR
        config.DATA_DIR = d
        with instances_lock:
            self._saved = dict(instances); instances.clear()
        self.app = Flask(__name__)

    def tearDown(self):
        from core.state import instances, instances_lock
        config.DATA_DIR = self._orig_data_dir
        with instances_lock:
            instances.clear(); instances.update(self._saved)
        self._tmp.cleanup()

    def _peer_with_model(self, name="srv2", active=0):
        return {
            "node_id": "peer", "node_name": name, "advertise_url": "http://srv2:5000",
            "vendor": "cuda", "llama_image": "img",
        }, {
            "instances": [{
                "id": "p1", "model_name": "gpt-oss-20b.gguf",
                "model_path": "/models/gpt-oss-20b.gguf", "status": "healthy",
                "config": {"share_queue": True}, "queue": {"active": active},
            }],
        }

    def test_no_dispatch_when_disabled(self):
        with self.app.test_request_context("/v1/chat/completions", method="POST",
                                           json={"model": "gpt-oss-20b"}):
            with patch.object(config, "CLUSTER_ENABLED", False):
                self.assertIsNone(self.cluster_api.dispatch_inference("gpt-oss-20b"))

    def test_no_dispatch_on_forwarded_hop(self):
        with self.app.test_request_context("/v1/chat/completions", method="POST",
                                           json={"model": "gpt-oss-20b"},
                                           headers={"X-Cluster-Dispatch": "1"}):
            with patch.object(config, "CLUSTER_ENABLED", True), \
                 patch.object(config, "CLUSTER_SECRET", "s"):
                self.assertIsNone(self.cluster_api.dispatch_inference("gpt-oss-20b"))

    def test_no_dispatch_without_peer(self):
        # No peers host the model -> serve locally (return None).
        with self.app.test_request_context("/v1/chat/completions", method="POST",
                                           json={"model": "gpt-oss-20b"}):
            with patch.object(config, "CLUSTER_ENABLED", True), \
                 patch.object(config, "CLUSTER_SECRET", "s"), \
                 patch("api.cluster.get_storage", return_value=self.storage):
                self.assertIsNone(self.cluster_api.dispatch_inference("gpt-oss-20b"))

    def test_forwards_to_peer_group_member(self):
        node, snap = self._peer_with_model(active=0)
        self.storage.register_node(node, snapshot=snap)
        captured = {}

        def fake_cluster_request(n, method, path, **kwargs):
            captured["path"] = path
            captured["dispatch_header"] = kwargs.get("headers", {}).get("X-Cluster-Dispatch")

            class R:
                status_code = 200
                headers = {"Content-Type": "application/json"}
                def iter_content(self, chunk_size=None): yield b'{"ok":1}'
                def close(self): pass
            return R()

        with self.app.test_request_context("/v1/chat/completions", method="POST",
                                           json={"model": "gpt-oss-20b"}):
            with patch.object(config, "CLUSTER_ENABLED", True), \
                 patch.object(config, "CLUSTER_SECRET", "s"), \
                 patch("api.cluster.get_storage", return_value=self.storage), \
                 patch("api.cluster._peer_live_load",
                       return_value={"active": 0, "queued": 0, "free": 1}), \
                 patch("core.cluster.cluster_request", side_effect=fake_cluster_request):
                resp = self.cluster_api.dispatch_inference("gpt-oss-20b")
        self.assertIsNotNone(resp)
        self.assertEqual(captured["path"], "/v1/chat/completions")
        self.assertEqual(captured["dispatch_header"], "1")

    def test_round_robin_spreads_ties(self):
        # Equal backlog (the burst case) must alternate, not pin to self.
        self.cluster_api._rr_counter = 0
        cands = [{"load": 0, "is_self": True, "node": {"node_name": "A"}},
                 {"load": 0, "is_self": False, "node": {"node_name": "B"}}]
        firsts = [self.cluster_api._order_candidates(cands)[0]["is_self"] for _ in range(4)]
        self.assertEqual(firsts, [True, False, True, False])

    def test_least_loaded_wins_over_tiebreak(self):
        # A genuinely less-loaded peer beats self regardless of round-robin.
        cands = [{"load": 5, "is_self": True, "node": {"node_name": "A"}},
                 {"load": 0, "is_self": False, "node": {"node_name": "B"}}]
        self.assertFalse(self.cluster_api._order_candidates(cands)[0]["is_self"])

    def test_dispatch_forwards_client_api_key(self):
        node, snap = self._peer_with_model()
        self.storage.register_node(node, snapshot=snap)
        captured = {}

        class FakeResp:
            status_code = 200
            headers = {"Content-Type": "application/json"}
            def iter_content(self, chunk_size=None): yield b"{}"
            def close(self): pass

        def fake_cluster_request(n, method, path, **kwargs):
            captured["auth"] = (kwargs.get("headers") or {}).get("Authorization")
            return FakeResp()

        with self.app.test_request_context(
                "/v1/chat/completions", method="POST",
                json={"model": "gpt-oss-20b"},
                headers={"Authorization": "Bearer llm-clientkey"}):
            with patch.object(config, "CLUSTER_ENABLED", True), \
                 patch.object(config, "CLUSTER_SECRET", "s"), \
                 patch("api.cluster.get_storage", return_value=self.storage), \
                 patch("api.cluster._peer_live_load",
                       return_value={"active": 0, "queued": 0, "free": 1}), \
                 patch("core.cluster.cluster_request", side_effect=fake_cluster_request):
                resp = self.cluster_api.dispatch_inference("gpt-oss-20b")
        self.assertIsNotNone(resp)
        self.assertEqual(captured["auth"], "Bearer llm-clientkey")

    def test_no_redispatch_when_already_hopped(self):
        # A request already forwarded by a peer must not re-run entry dispatch.
        with self.app.test_request_context(
                "/v1/chat/completions", method="POST", json={"model": "gpt-oss-20b"},
                headers={"X-Cluster-Dispatch": "2"}):
            with patch.object(config, "CLUSTER_ENABLED", True), \
                 patch.object(config, "CLUSTER_SECRET", "s"):
                self.assertIsNone(self.cluster_api.dispatch_inference("gpt-oss-20b"))

    def test_acquire_or_overflow_serves_locally(self):
        with self.app.test_request_context("/v1/chat/completions", method="POST"):
            with patch.object(config, "CLUSTER_ENABLED", False):
                acquired, overflow, reason = self.cluster_api.acquire_or_overflow(_FakeGate(True), "m")
        self.assertTrue(acquired)
        self.assertIsNone(overflow)
        self.assertIsNone(reason)

    def test_acquire_or_overflow_migrates_to_free_peer(self):
        sentinel = object()
        with self.app.test_request_context(
                "/v1/chat/completions", method="POST", json={"model": "m"}):
            with patch.object(config, "CLUSTER_ENABLED", True), \
                 patch.object(config, "CLUSTER_SECRET", "s"), \
                 patch("api.cluster._find_free_peer",
                       return_value={"node_id": "B", "advertise_url": "http://b:5000"}), \
                 patch("api.cluster._forward_inference", return_value=sentinel) as fwd:
                acquired, overflow, reason = self.cluster_api.acquire_or_overflow(_FakeGate(False), "m")
        self.assertFalse(acquired)
        self.assertIs(overflow, sentinel)
        self.assertIsNone(reason)
        self.assertEqual(fwd.call_args.kwargs.get("hops"), 1)  # entry(0) + 1

    def test_evacuation_targets_busy_alive_peer(self):
        # A dead local worker must offload its queue to an alive peer EVEN IF the
        # peer is full - work-stealing's free-slot rule would otherwise strand the
        # queue on the dead node (the bug: peer at Max Concurrent => no transfer).
        from core.timeutil import now_iso
        busy = {"node_id": "B", "advertise_url": "http://b:5000",
                "last_heartbeat_at": now_iso()}
        with patch.object(self.cluster_api, "get_storage") as gs, \
             patch.object(self.cluster_api.cl, "get_node_id", return_value="A"), \
             patch.object(self.cluster_api, "_peer_live_load",
                          return_value={"active": 2, "queued": 3, "free": 0, "max_concurrent": 2}):
            gs.return_value.list_nodes.return_value = [busy]
            self.assertIsNone(self.cluster_api._find_free_peer("m"))        # no free slot
            self.assertIs(self.cluster_api._find_evacuation_peer("m"), busy)  # evacuate anyway

    def test_evacuation_skips_peer_without_model(self):
        from core.timeutil import now_iso
        peer = {"node_id": "B", "advertise_url": "http://b:5000",
                "last_heartbeat_at": now_iso()}
        with patch.object(self.cluster_api, "get_storage") as gs, \
             patch.object(self.cluster_api.cl, "get_node_id", return_value="A"), \
             patch.object(self.cluster_api, "_peer_live_load", return_value=None):
            gs.return_value.list_nodes.return_value = [peer]
            self.assertIsNone(self.cluster_api._find_evacuation_peer("m"))

    def test_local_load_by_model(self):
        from proxy import create_gate, remove_gate
        from core.state import instances, instances_lock
        create_gate("i1", 2, 10, model_path="/models/gpt-oss-20b.gguf", share_queue=True)
        with instances_lock:
            instances["i1"] = {"id": "i1", "model_path": "/models/gpt-oss-20b.gguf",
                               "status": "healthy", "config": {"share_queue": True}}
        try:
            load = self.cluster_api.local_load_by_model()
        finally:
            remove_gate("i1")
            with instances_lock:
                instances.pop("i1", None)
        self.assertIn("gpt-oss-20b", load)
        self.assertEqual(load["gpt-oss-20b"]["free"], 2)
        self.assertEqual(load["gpt-oss-20b"]["max_concurrent"], 2)

    def test_forward_inflight_increments_before_response_and_cleans_up(self):
        # The in-flight count must rise the moment we commit to a peer (not after
        # its first byte) so a burst doesn't funnel onto a slow peer; and it must
        # be rolled back on every non-success path.
        node = {"node_id": "B", "node_name": "B", "advertise_url": "http://b:5000"}
        self.cluster_api._inflight.clear()
        ctx = dict(json={"model": "m"}, method="POST")

        # transport failure -> rolled back to 0
        with self.app.test_request_context("/v1/chat/completions", **ctx):
            with patch("core.cluster.cluster_request", side_effect=RuntimeError("down")):
                self.assertIsNone(self.cluster_api._forward_inference(
                    node, "/v1/chat/completions", b"{}", "application/json"))
        self.assertEqual(self.cluster_api._inflight_get("B"), 0)

        # peer gate full (429) -> rolled back to 0
        class R429:
            status_code = 429
            headers = {}
            def close(self): pass
        with self.app.test_request_context("/v1/chat/completions", **ctx):
            with patch("core.cluster.cluster_request", return_value=R429()):
                self.assertIsNone(self.cluster_api._forward_inference(
                    node, "/v1/chat/completions", b"{}", "application/json"))
        self.assertEqual(self.cluster_api._inflight_get("B"), 0)

        # success -> counted immediately, and stays counted while the request
        # runs; draining the relay must NOT decrement it (it ages out instead).
        class ROK:
            status_code = 200
            headers = {"Content-Type": "application/json"}
            def iter_content(self, chunk_size=None): yield b"{}"
            def close(self): pass
        with self.app.test_request_context("/v1/chat/completions", **ctx):
            with patch("core.cluster.cluster_request", return_value=ROK()):
                resp = self.cluster_api._forward_inference(
                    node, "/v1/chat/completions", b"{}", "application/json")
            self.assertEqual(self.cluster_api._inflight_get("B"), 1)
            list(resp.response)  # draining no longer decrements
            self.assertEqual(self.cluster_api._inflight_get("B"), 1)
        # ...but it ages out of the time window (the fix for the stuck queue:
        # a held-forever count made a free peer look full and blocked migration)
        with self.cluster_api._inflight_lock:
            self.cluster_api._inflight["B"] = [
                t - self.cluster_api._INFLIGHT_WINDOW_S - 1
                for t in self.cluster_api._inflight["B"]]
        self.assertEqual(self.cluster_api._inflight_get("B"), 0)
        self.cluster_api._inflight.clear()

    def test_probe_peer_reachable(self):
        node = {"node_id": "B", "advertise_url": "http://b:5000"}

        class OK:
            status_code = 200
        self.cluster_api._reach_cache.clear()
        with patch("core.cluster.cluster_request", return_value=OK()):
            ok, err = self.cluster_api.probe_peer_reachable(node)
        self.assertTrue(ok)
        self.assertEqual(err, "")

        self.cluster_api._reach_cache.clear()
        with patch("core.cluster.cluster_request",
                   side_effect=ConnectionError("connection refused")):
            ok, err = self.cluster_api.probe_peer_reachable(node)
        self.assertFalse(ok)
        self.assertIn("refused", err)

        # No advertise_url -> immediate fail, never touches the network.
        ok, err = self.cluster_api.probe_peer_reachable({"node_id": "C", "advertise_url": ""})
        self.assertFalse(ok)
        self.assertEqual(err, "no advertise_url")
        self.cluster_api._reach_cache.clear()

    def test_match_model_load_is_case_insensitive_and_fuzzy(self):
        # local-load maps are keyed by the LOWERCASE file stem; a client asking
        # with a mixed-case quant suffix (or just the base name) must still hit.
        load = {"gpt-oss-20b-q4_k_m": {"active": 0, "queued": 0, "free": 2}}
        m = self.cluster_api._match_model_load
        self.assertEqual(m(load, "gpt-oss-20b-Q4_K_M")["free"], 2)  # case only
        self.assertEqual(m(load, "gpt-oss-20b")["free"], 2)          # base name
        self.assertEqual(m(load, "gpt-oss-20b-q4_k_m:latest")["free"], 2)
        self.assertIsNone(m(load, "llama-3"))
        self.assertIsNone(m(None, "gpt-oss-20b"))

    def test_peer_live_load_matches_quant_suffixed_request(self):
        # Regression: a peer hosting gpt-oss-20b-Q4_K_M must not vanish from
        # routing/work-stealing just because the request name has uppercase.
        node = {"node_id": "peer", "advertise_url": "http://srv2:5000"}

        class R:
            status_code = 200
            def json(self): return {"gpt-oss-20b-q4_k_m": {"active": 0, "queued": 0, "free": 3}}

        self.cluster_api._load_cache.clear()
        with patch.object(config, "CLUSTER_SECRET", "s"), \
             patch("core.cluster.cluster_request", return_value=R()):
            live = self.cluster_api._peer_live_load(node, "gpt-oss-20b-Q4_K_M")
        self.cluster_api._load_cache.clear()
        self.assertIsNotNone(live)
        self.assertEqual(live["free"], 3)

    def test_group_override_roundtrip(self):
        with patch.object(config, "CLUSTER_ENABLED", True), \
             patch.object(config, "CLUSTER_SECRET", "s"), \
             patch("api.cluster.get_storage", return_value=self.storage):
            self.cluster_api.record_group_overrides("/models/gpt-oss-20b.gguf", {
                "share_queue": True,
                "proxy_sampling_override_enabled": True,
                "proxy_sampling_temperature": 0.3,
            })
            inst = {"model_path": "/models/gpt-oss-20b.gguf",
                    "config": {"share_queue": True, "proxy_sampling_temperature": 0.9}}
            eff = self.cluster_api.effective_inference_config(inst)
        self.assertTrue(eff["proxy_sampling_override_enabled"])
        self.assertEqual(eff["proxy_sampling_temperature"], 0.3)


class NodeSettingsTests(unittest.TestCase):
    def setUp(self):
        import core.node_settings as ns
        self.ns = ns
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_data_dir = config.DATA_DIR
        config.DATA_DIR = self._tmp.name
        self.nid = cluster.get_node_id()

    def tearDown(self):
        config.DATA_DIR = self._orig_data_dir
        self._tmp.cleanup()

    def test_legacy_fallback_then_node_override(self):
        # Legacy top-level value is used until a node-scoped value exists.
        legacy = {"admin_ui_enforce_max_models": True}
        self.assertTrue(self.ns.effective_from_settings(legacy, "admin_ui_enforce_max_models", False))
        # Node-scoped value wins over legacy.
        scoped = {"admin_ui_enforce_max_models": True,
                  "nodes": {self.nid: {"admin_ui_enforce_max_models": False}}}
        self.assertFalse(self.ns.effective_from_settings(scoped, "admin_ui_enforce_max_models", True))
        # Default when absent everywhere.
        self.assertEqual(self.ns.effective_from_settings({}, "missing", "d"), "d")

    def test_two_nodes_keep_separate_docker_images(self):
        d = self._tmp.name
        storage = JsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"),
        )
        with patch("core.node_settings.get_storage", return_value=storage):
            self.ns.merge_node_settings({"docker_images": {"vendor": "cuda"}}, node_id="A")
            self.ns.merge_node_settings({"docker_images": {"vendor": "rocm"}}, node_id="B")
            self.assertEqual(self.ns.get_node_settings("A")["docker_images"]["vendor"], "cuda")
            self.assertEqual(self.ns.get_node_settings("B")["docker_images"]["vendor"], "rocm")


class PresetOverrideTests(unittest.TestCase):
    def setUp(self):
        import api.presets as presets_api
        self.presets_api = presets_api
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"),
        )
        self.app = Flask(__name__)
        self.app.register_blueprint(presets_api.bp)
        self.client = self.app.test_client()

    def tearDown(self):
        self._tmp.cleanup()

    def test_resolve_preset_for_node(self):
        preset = {"n_gpu_layers": -1, "threads": 4,
                  "node_overrides": {"B": {"n_gpu_layers": 32}}}
        self.assertEqual(self.presets_api.resolve_preset_for_node(preset, "B")["n_gpu_layers"], 32)
        self.assertEqual(self.presets_api.resolve_preset_for_node(preset, "B")["threads"], 4)
        self.assertEqual(self.presets_api.resolve_preset_for_node(preset, "A")["n_gpu_layers"], -1)
        self.assertIsNone(self.presets_api.resolve_preset_for_node(None, "B"))

    def test_put_with_override_node_preserves_base(self):
        with patch("api.presets.get_storage", return_value=self.storage):
            # Base save (no target): hardware is the base.
            self.client.put("/api/presets/models/m.gguf",
                            json={"ctx_size": 8192, "n_gpu_layers": -1, "threads": 4})
            # Override save for node B: hardware goes to B, base preserved.
            self.client.put("/api/presets/models/m.gguf",
                            json={"ctx_size": 8192, "n_gpu_layers": 32, "threads": 8,
                                  "override_node_id": "B"})
        preset = self.storage.get_preset("/models/m.gguf")
        self.assertEqual(preset["n_gpu_layers"], -1)          # base untouched
        self.assertEqual(preset["threads"], 4)
        self.assertEqual(preset["node_overrides"]["B"]["n_gpu_layers"], 32)
        self.assertEqual(preset["node_overrides"]["B"]["threads"], 8)
        # And resolution applies B's override.
        resolved = self.presets_api.resolve_preset_for_node(preset, "B")
        self.assertEqual(resolved["n_gpu_layers"], 32)


class SettingsRoutingTests(unittest.TestCase):
    def setUp(self):
        import api.settings as settings_api
        self.settings_api = settings_api
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"),
        )
        self._orig_data_dir = config.DATA_DIR
        config.DATA_DIR = d
        self.nid = cluster.get_node_id()
        self.app = Flask(__name__)
        self.app.register_blueprint(settings_api.bp)
        self.client = self.app.test_client()

    def tearDown(self):
        config.DATA_DIR = self._orig_data_dir
        self._tmp.cleanup()

    def test_eviction_toggle_routed_per_node_recording_shared(self):
        with patch("api.settings.get_storage", return_value=self.storage), \
             patch("core.node_settings.get_storage", return_value=self.storage):
            resp = self.client.post("/api/settings", json={
                "admin_ui_enforce_max_models": True,
                "recording_mode": "per_request",
            })
            self.assertEqual(resp.status_code, 200)
            raw = self.storage.get_settings()
            # Per-node key landed in this node's namespace, not top-level.
            self.assertTrue(raw["nodes"][self.nid]["admin_ui_enforce_max_models"])
            self.assertNotIn("admin_ui_enforce_max_models", {k: v for k, v in raw.items() if k != "nodes"})
            # Shared key stays top-level.
            self.assertEqual(raw["recording_mode"], "per_request")
            # GET overlays the node value and never leaks the namespace.
            got = self.client.get("/api/settings").get_json()
            self.assertTrue(got["admin_ui_enforce_max_models"])
            self.assertNotIn("nodes", got)


class GateOverflowTests(unittest.TestCase):
    def test_acquires_when_slot_free(self):
        from proxy import RequestGate
        g = RequestGate(max_concurrent=1, max_queue_depth=10)
        outcome, _ = g.acquire_or_overflow(
            timeout=1, poll=0.05, find_target=lambda: None, do_forward=lambda t: None)
        self.assertEqual(outcome, "acquired")

    def test_overflow_fires_when_saturated(self):
        from proxy import RequestGate
        g = RequestGate(max_concurrent=1, max_queue_depth=10)
        self.assertTrue(g.acquire(timeout=1))  # take the only slot
        outcome, value = g.acquire_or_overflow(
            timeout=1, poll=0.02, find_target=lambda: "PEER", do_forward=lambda t: t)
        self.assertEqual(outcome, "overflow")
        self.assertEqual(value, "PEER")

    def test_queued_stays_counted_while_waiting(self):
        # A genuinely-waiting request (no free peer) must show up in `queued`
        # while it waits on the gate - not flap to 0 every poll as the old
        # acquire()-in-a-loop did.
        import threading
        import time as _t
        from proxy import RequestGate
        g = RequestGate(max_concurrent=1, max_queue_depth=10)
        self.assertTrue(g.acquire(timeout=1))  # hold the slot

        stop = threading.Event()

        def waiter():
            g.acquire_or_overflow(
                timeout=2, poll=0.02,
                find_target=lambda: None,      # never a peer -> keep waiting
                do_forward=lambda t: None)

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        # It sits in the condition wait the vast majority of the time; confirm
        # the counted state is observable (not stuck at 0).
        deadline = _t.time() + 1
        seen = False
        while _t.time() < deadline:
            if g.queued == 1:
                seen = True
                break
            _t.sleep(0.005)
        self.assertTrue(seen)
        g.release()                     # let the waiter acquire and finish
        t.join(2)

    def test_queued_stays_counted_through_search_cycles(self):
        # The flicker bug: a waiting request must stay counted as queued across
        # the periodic peer searches (find_target), not drop to 0 between polls.
        import threading
        import time as _t
        from proxy import RequestGate
        g = RequestGate(max_concurrent=1, max_queue_depth=10)
        self.assertTrue(g.acquire(timeout=1))  # saturate the only slot

        searches = []

        def find_target():
            searches.append(1)
            return None                         # no peer, ever -> keep waiting

        def waiter():
            g.acquire_or_overflow(timeout=1.0, poll=0.02,
                                  find_target=find_target, do_forward=lambda t: None)

        t = threading.Thread(target=waiter, daemon=True)
        t.start()

        saw_one = False
        min_after = 99
        deadline = _t.time() + 0.6           # spans many poll/search cycles
        while _t.time() < deadline:
            q = g.queued
            if q == 1:
                saw_one = True
            if saw_one:
                min_after = min(min_after, q)
            _t.sleep(0.003)

        self.assertTrue(saw_one)
        self.assertEqual(min_after, 1)        # never flickered to 0 mid-search
        self.assertGreater(len(searches), 3)  # multiple search cycles actually ran
        g.release()
        t.join(2)

    def test_migrating_request_not_counted_as_queued(self):
        # The phantom-queue bug: a request handed off via on_overflow must not
        # also show as queued while the (possibly long, non-streaming) forward
        # is in flight. Here on_overflow blocks like a real forward does.
        import threading
        import time as _t
        from proxy import RequestGate
        g = RequestGate(max_concurrent=1, max_queue_depth=10)
        self.assertTrue(g.acquire(timeout=1))  # saturate the only slot

        in_handoff = threading.Event()
        release = threading.Event()
        out = {}

        def blocking_forward(_target):
            in_handoff.set()
            release.wait(2)     # simulate the forward blocking for the request's life
            return "PEER"

        def waiter():
            out["res"] = g.acquire_or_overflow(
                timeout=3, poll=0.02,
                find_target=lambda: "PEER",     # peer found immediately
                do_forward=blocking_forward)    # ...then the forward blocks

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        self.assertTrue(in_handoff.wait(1))  # we're inside the blocking handoff
        _t.sleep(0.05)
        self.assertEqual(g.queued, 0)        # NOT counted while migrating
        release.set()
        t.join(2)
        self.assertEqual(out["res"], ("overflow", "PEER"))
        self.assertEqual(g.queued, 0)


class GateDrainTests(unittest.TestCase):
    """A worker dying drains its gate so QUEUED requests migrate to peers
    (work-stealing) instead of being rejected - the whole point of the feature."""

    def test_drain_migrates_even_with_free_slot(self):
        # The worker is dead, so a locally-"free" slot is meaningless: the request
        # must NOT acquire locally - it must overflow to a peer.
        from proxy import RequestGate
        g = RequestGate(max_concurrent=1, max_queue_depth=10)
        g.drain()
        outcome, value = g.acquire_or_overflow(
            timeout=1, poll=0.02, find_target=lambda: "PEER", do_forward=lambda t: t)
        self.assertEqual(outcome, "overflow")
        self.assertEqual(value, "PEER")

    def test_drain_rejects_when_no_peer(self):
        # Single-node / no free peer: draining degrades to a clean 429 (the
        # fail-fast-no-peer behavior), it does not hang forever.
        from proxy import RequestGate
        g = RequestGate(max_concurrent=1, max_queue_depth=10)
        g.drain()
        outcome, _ = g.acquire_or_overflow(
            timeout=0.3, poll=0.02, find_target=lambda: None, do_forward=lambda t: None)
        self.assertEqual(outcome, "rejected")

    def test_acquire_false_when_draining(self):
        # The non-cluster / max-hops path has no migration, so a drained gate
        # simply refuses -> 429.
        from proxy import RequestGate
        g = RequestGate(max_concurrent=1, max_queue_depth=10)
        g.drain()
        self.assertFalse(g.acquire(timeout=0.2))

    def test_drain_wakes_saturated_waiter_to_migrate(self):
        # A request already parked in the queue (saturated, sitting in the
        # condition wait) must be woken by drain() and migrate immediately,
        # not sit until its poll slice elapses.
        import threading
        import time as _t
        from proxy import RequestGate
        g = RequestGate(max_concurrent=1, max_queue_depth=10)
        self.assertTrue(g.acquire(timeout=1))  # saturate the only slot
        out = {}

        def waiter():
            # Long poll: without a wake it would block in the condition wait.
            out["res"] = g.acquire_or_overflow(
                timeout=5, poll=5, find_target=lambda: "PEER", do_forward=lambda t: t)

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        deadline = _t.time() + 1
        while _t.time() < deadline and g.queued != 1:
            _t.sleep(0.005)
        self.assertEqual(g.queued, 1)        # it's parked, waiting
        g.drain()                            # wake it -> should migrate at once
        t.join(2)
        self.assertEqual(out.get("res"), ("overflow", "PEER"))


class DrainGateRegistryTests(unittest.TestCase):
    """drain_gate(): mark the dying instance's gate draining and detach it so a
    relaunch builds a fresh one, unless a live sibling still shares it."""

    def setUp(self):
        import proxy
        import core.state as cstate
        self._proxy = proxy
        self._cstate = cstate
        proxy._instance_gates.clear()
        proxy._shared_queue_gates.clear()
        proxy._instance_gate_configs.clear()
        self._saved_instances = dict(cstate.instances)
        cstate.instances.clear()

    def tearDown(self):
        self._proxy._instance_gates.clear()
        self._proxy._shared_queue_gates.clear()
        self._proxy._instance_gate_configs.clear()
        self._cstate.instances.clear()
        self._cstate.instances.update(self._saved_instances)

    def test_drains_and_detaches_solo_gate(self):
        from proxy import create_gate, get_gate, drain_gate
        create_gate("i1", 1, 10, model_path="/m.gguf", share_queue=True)
        gate = get_gate("i1")
        self.assertIsNotNone(gate)
        drain_gate("i1")
        self.assertTrue(gate._draining)
        self.assertIsNone(get_gate("i1"))                       # detached for relaunch
        self.assertNotIn("/m.gguf", self._proxy._shared_queue_gates)

    def test_skips_when_sibling_alive(self):
        from proxy import create_gate, get_gate, drain_gate
        create_gate("i1", 1, 10, model_path="/m.gguf", share_queue=True)
        create_gate("i2", 1, 10, model_path="/m.gguf", share_queue=True)
        gate = get_gate("i1")
        self._cstate.instances.update({"i2": {"id": "i2", "status": "healthy"}})
        drain_gate("i1")
        self.assertFalse(gate._draining)                        # sibling still serves
        self.assertIs(get_gate("i1"), gate)                     # left attached
        self.assertIn("/m.gguf", self._proxy._shared_queue_gates)


class ShareQueueAliasTests(unittest.TestCase):
    """Alias-based grouping: share_queue_group as the cluster group key, plus
    --alias on the llama-server cmd so direct hits also see the group name."""

    def test_build_llama_cmd_adds_alias_when_group_set(self):
        from core.helpers import build_llama_cmd
        cmd = build_llama_cmd("/models/qwen2.5-14b-q4.gguf", 8080,
                              {"share_queue_group": "qwen2.5-14b"})
        self.assertIn("--alias", cmd)
        self.assertEqual(cmd[cmd.index("--alias") + 1], "qwen2.5-14b")

    def test_build_llama_cmd_omits_alias_when_group_empty(self):
        from core.helpers import build_llama_cmd
        cmd = build_llama_cmd("/models/m.gguf", 8080, {})
        self.assertNotIn("--alias", cmd)
        cmd2 = build_llama_cmd("/models/m.gguf", 8080, {"share_queue_group": "  "})
        self.assertNotIn("--alias", cmd2)

    def test_effective_group_key_uses_alias_when_set(self):
        import api.cluster as ca
        self.assertEqual(
            ca.effective_group_key("/models/qwen2.5-14b-Q4_K_M.gguf",
                                   {"share_queue_group": "qwen2.5-14b"}),
            "qwen2.5-14b")
        # Empty alias falls back to filename stem (existing behavior).
        self.assertEqual(
            ca.effective_group_key("/models/qwen2.5-14b-Q4_K_M.gguf", {}),
            "qwen2.5-14b-q4_k_m")
        # Whitespace-only alias is treated as unset.
        self.assertEqual(
            ca.effective_group_key("/models/m.gguf", {"share_queue_group": "  "}),
            "m")

    def test_local_load_keyed_by_alias_and_aggregates(self):
        # Two locally-running aliased instances (different files, same group)
        # collapse into one entry under the alias - that's the whole point.
        import api.cluster as ca
        from core.state import instances, instances_lock
        from proxy import create_gate, remove_gate
        with instances_lock:
            saved = dict(instances); instances.clear()
        try:
            create_gate("a", 2, 10, model_path="/models/qwen-q4.gguf", share_queue=True)
            create_gate("b", 2, 10, model_path="/models/qwen-q8.gguf", share_queue=True)
            with instances_lock:
                instances["a"] = {"id": "a", "model_path": "/models/qwen-q4.gguf",
                                  "status": "healthy",
                                  "config": {"share_queue": True,
                                             "share_queue_group": "qwen2.5-14b"}}
                instances["b"] = {"id": "b", "model_path": "/models/qwen-q8.gguf",
                                  "status": "healthy",
                                  "config": {"share_queue": True,
                                             "share_queue_group": "qwen2.5-14b"}}
            load = ca.local_load_by_model()
        finally:
            remove_gate("a"); remove_gate("b")
            with instances_lock:
                instances.clear(); instances.update(saved)
        self.assertIn("qwen2.5-14b", load)
        # Two q=2 instances under the same alias key -> max_concurrent = 4.
        self.assertEqual(load["qwen2.5-14b"]["max_concurrent"], 4)
        # Filename keys must NOT appear when an alias is set.
        self.assertNotIn("qwen-q4", load)
        self.assertNotIn("qwen-q8", load)

    def test_find_running_instance_by_alias(self):
        from api.llamaman import _find_running_instance_by_alias
        from core.state import instances, instances_lock
        with instances_lock:
            saved = dict(instances); instances.clear()
            instances["x"] = {"id": "x", "model_path": "/models/qwen-q4.gguf",
                              "status": "healthy",
                              "config": {"share_queue_group": "qwen2.5-14b"}}
        try:
            self.assertIsNotNone(_find_running_instance_by_alias("qwen2.5-14b"))
            # Case insensitive + substring match (same rule as cluster matching).
            self.assertIsNotNone(_find_running_instance_by_alias("QWEN2.5-14B"))
            self.assertIsNotNone(_find_running_instance_by_alias("qwen2.5-14b:latest"))
            self.assertIsNone(_find_running_instance_by_alias("llama-3"))
            # Stopped instances don't match.
            with instances_lock:
                instances["x"]["status"] = "stopped"
            self.assertIsNone(_find_running_instance_by_alias("qwen2.5-14b"))
        finally:
            with instances_lock:
                instances.clear(); instances.update(saved)


class FallbackRoutingTests(unittest.TestCase):
    """Fallback-only nodes: serve a group ONLY when every non-fallback is at
    capacity or unreachable. Tier comes from the load dict's `fallback` flag,
    not the load magnitude."""

    def test_order_candidates_puts_fallback_last_even_when_idle(self):
        # Even with load=0, a fallback must sort behind a saturated primary -
        # otherwise the whole point (serve only when needed) is lost.
        import api.cluster as ca
        cands = [
            {"load": 99, "is_self": False, "fallback": False, "node": {"node_name": "P"}},
            {"load": 0,  "is_self": False, "fallback": True,  "node": {"node_name": "F"}},
        ]
        ordered = ca._order_candidates(cands)
        self.assertEqual(ordered[0]["node"]["node_name"], "P")
        self.assertEqual(ordered[1]["node"]["node_name"], "F")

    def test_order_candidates_no_primaries_uses_fallback(self):
        # When every primary is gone/unreachable, fallbacks become the only
        # candidates and should serve normally (load-ordered).
        import api.cluster as ca
        cands = [
            {"load": 5, "is_self": False, "fallback": True, "node": {"node_name": "F1"}},
            {"load": 2, "is_self": False, "fallback": True, "node": {"node_name": "F2"}},
        ]
        ordered = ca._order_candidates(cands)
        self.assertEqual(ordered[0]["node"]["node_name"], "F2")
        self.assertEqual(ordered[1]["node"]["node_name"], "F1")

    def test_find_free_peer_skips_fallback_if_primary_has_room(self):
        # Work-stealing: an idle fallback is invisible while ANY primary has
        # a free slot, so the queue stays in the primary tier until it can't.
        import api.cluster as ca
        from core.timeutil import now_iso
        nodes = [
            {"node_id": "P", "advertise_url": "http://p:5000", "last_heartbeat_at": now_iso()},
            {"node_id": "F", "advertise_url": "http://f:5000", "last_heartbeat_at": now_iso()},
        ]
        loads = {
            "P": {"active": 0, "queued": 0, "free": 1, "fallback": False},
            "F": {"active": 0, "queued": 0, "free": 10, "fallback": True},
        }
        with patch.object(ca, "get_storage") as gs, \
             patch.object(ca.cl, "get_node_id", return_value="SELF"), \
             patch.object(ca, "_peer_live_load",
                          side_effect=lambda node, _m: loads[node["node_id"]]):
            gs.return_value.list_nodes.return_value = nodes
            peer = ca._find_free_peer("m")
        self.assertEqual(peer["node_id"], "P")

    def test_find_free_peer_uses_fallback_when_primaries_full(self):
        import api.cluster as ca
        from core.timeutil import now_iso
        nodes = [
            {"node_id": "P", "advertise_url": "http://p:5000", "last_heartbeat_at": now_iso()},
            {"node_id": "F", "advertise_url": "http://f:5000", "last_heartbeat_at": now_iso()},
        ]
        loads = {
            "P": {"active": 2, "queued": 5, "free": 0, "fallback": False},
            "F": {"active": 0, "queued": 0, "free": 3, "fallback": True},
        }
        with patch.object(ca, "get_storage") as gs, \
             patch.object(ca.cl, "get_node_id", return_value="SELF"), \
             patch.object(ca, "_peer_live_load",
                          side_effect=lambda node, _m: loads[node["node_id"]]):
            gs.return_value.list_nodes.return_value = nodes
            peer = ca._find_free_peer("m")
        self.assertEqual(peer["node_id"], "F")

    def test_local_group_load_flags_pure_fallback_node(self):
        # All matching instances are fallback -> the whole node is fallback.
        import api.cluster as ca
        from core.state import instances, instances_lock
        from proxy import create_gate, remove_gate
        with instances_lock:
            saved = dict(instances); instances.clear()
        try:
            create_gate("a", 1, 10, model_path="/models/m.gguf", share_queue=True)
            with instances_lock:
                instances["a"] = {"id": "a", "model_path": "/models/m.gguf",
                                  "status": "healthy",
                                  "config": {"share_queue": True,
                                             "share_queue_group": "g",
                                             "share_queue_fallback": True}}
            result = ca._local_group_load("g")
        finally:
            remove_gate("a")
            with instances_lock:
                instances.clear(); instances.update(saved)
        self.assertIsNotNone(result)
        load, is_fb = result
        self.assertEqual(load, 0)
        self.assertTrue(is_fb)

    def test_local_group_load_one_primary_flips_node_to_primary(self):
        # Mixed instances on one node: any primary makes the node primary.
        import api.cluster as ca
        from core.state import instances, instances_lock
        from proxy import create_gate, remove_gate
        with instances_lock:
            saved = dict(instances); instances.clear()
        try:
            create_gate("p", 1, 10, model_path="/models/m1.gguf", share_queue=True)
            create_gate("f", 1, 10, model_path="/models/m2.gguf", share_queue=True)
            with instances_lock:
                instances["p"] = {"id": "p", "model_path": "/models/m1.gguf",
                                  "status": "healthy",
                                  "config": {"share_queue": True,
                                             "share_queue_group": "g",
                                             "share_queue_fallback": False}}
                instances["f"] = {"id": "f", "model_path": "/models/m2.gguf",
                                  "status": "healthy",
                                  "config": {"share_queue": True,
                                             "share_queue_group": "g",
                                             "share_queue_fallback": True}}
            result = ca._local_group_load("g")
        finally:
            remove_gate("p"); remove_gate("f")
            with instances_lock:
                instances.clear(); instances.update(saved)
        self.assertIsNotNone(result)
        _load, is_fb = result
        self.assertFalse(is_fb)


if __name__ == "__main__":
    unittest.main()
