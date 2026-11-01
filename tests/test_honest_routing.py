# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""#111 Phase 2: listings and routing agree on which peers count, a failed probe
isn't mistaken for a peer hosting nothing, and when nobody can take a request the
client hears why rather than "model not found"."""

import json
import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.cluster as ca
import api.llamaman as llamaman
from core.state import instances, instances_lock

SELF = ca.cl.get_node_id()


def _inst(status="healthy", group="chat-pool", fallback=False, share_queue=True, path="/peer/m.gguf"):
    return {"status": status, "model_path": path,
            "config": {"share_queue": share_queue, "share_queue_group": group,
                       "share_queue_fallback": fallback, "ctx_size": 4096}}


def _peer(name, *insts, url=True, online=True):
    return {"node_id": name, "node_name": name, "online": online,
            "advertise_url": f"http://{name}:5000" if url else "",
            "snapshot": {"system": {"features": ["alias_wake"]}, "instances": list(insts)}}


def _live(healthy=1, active=0, queued=0, free=1, sleeping=0):
    return {"active": active, "queued": queued, "free": free, "max_concurrent": 2, "fallback": False,
            "healthy": healthy, "starting": 0, "sleeping": sleeping}


class _Case(unittest.TestCase):

    def setUp(self):
        with instances_lock:
            saved = dict(instances)
            instances.clear()

        def restore():
            with instances_lock:
                instances.clear()
                instances.update(saved)
        self.addCleanup(restore)
        ca._load_cache.clear()
        ca._unreachable_peers.clear()
        self.nodes = []
        self.storage = Mock()
        self.storage.list_nodes.side_effect = lambda: self.nodes
        self.addCleanup(ca._load_cache.clear)
        self.addCleanup(ca._unreachable_peers.clear)
        for target, value in (
            ("api.cluster.get_storage", lambda: self.storage),
            ("api.llamaman.get_storage", lambda: self.storage),
            ("api.cluster._is_online", lambda n: n.get("online", True)),
            ("api.cluster._inflight_get", lambda _n: 0),
            ("api.cluster.cl.is_cluster_enabled", lambda: True),
            ("core.cluster.is_cluster_enabled", lambda: True),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def app(self):
        app = Flask(__name__)
        app.register_blueprint(llamaman.bp)
        return app


class AdvertisingTests(_Case):

    def _names(self):
        return [g["name"] for g in ca.cluster_group_models()]

    def test_a_routable_peers_group_is_listed(self):
        self.nodes = [_peer("a", _inst())]
        self.assertEqual(self._names(), ["chat-pool"])

    def test_a_view_only_peers_group_is_not_listed(self):
        self.nodes = [_peer("a", _inst(), url=False)]
        self.assertEqual(self._names(), [])

    def test_an_offline_peers_group_is_not_listed(self):
        self.nodes = [_peer("a", _inst(), online=False)]
        self.assertEqual(self._names(), [])

    def test_a_group_without_share_queue_is_not_listed(self):
        self.nodes = [_peer("a", _inst(share_queue=False))]
        self.assertEqual(self._names(), [])
        instances["x"] = {"id": "x", **_inst(share_queue=False)}
        self.assertEqual(self._names(), [])

    def test_a_sleeping_peer_member_is_still_listed(self):
        self.nodes = [_peer("a", _inst("sleeping"))]
        self.assertEqual(self._names(), ["chat-pool"])

    def test_a_stopped_peer_member_is_not_listed(self):
        self.nodes = [_peer("a", _inst("stopped"))]
        self.assertEqual(self._names(), [])

    def test_a_local_member_is_listed_with_no_peers(self):
        instances["x"] = {"id": "x", **_inst(path="/models/x.gguf")}
        self.assertEqual(self._names(), ["chat-pool"])

    def test_min_ctx_ignores_a_peer_that_cannot_serve(self):
        small = _inst()
        small["config"]["ctx_size"] = 1024
        self.nodes = [_peer("a", _inst()), _peer("view", small, url=False),
                      _peer("gone", small, online=False)]
        members = list(llamaman._iter_group_members_across_cluster("chat-pool"))
        self.assertEqual([m["node_id"] for m in members], ["a"])


class LoadProbeTests(_Case):

    NODE = {"node_id": "a", "node_name": "a", "advertise_url": "http://a:5000"}

    def _probe(self, **kwargs):
        with patch("core.cluster.cluster_request", **kwargs) as req:
            return ca._peer_live_load(self.NODE, "chat-pool"), req

    def test_a_timeout_is_unreachable(self):
        live, _ = self._probe(side_effect=TimeoutError("slow"))
        self.assertIs(live, ca.PEER_UNREACHABLE)
        self.assertFalse(live)

    def test_a_non_200_is_unreachable(self):
        live, _ = self._probe(return_value=Mock(status_code=403))
        self.assertIs(live, ca.PEER_UNREACHABLE)

    def test_a_bad_body_is_unreachable(self):
        live, _ = self._probe(return_value=Mock(status_code=200, json=Mock(return_value=[])))
        self.assertIs(live, ca.PEER_UNREACHABLE)

    def test_no_entry_is_not_unreachable(self):
        live, _ = self._probe(return_value=Mock(status_code=200, json=Mock(return_value={})))
        self.assertIsNone(live)

    def test_an_entry_is_returned(self):
        entry = _live()
        live, _ = self._probe(return_value=Mock(status_code=200, json=Mock(return_value={"chat-pool": entry})))
        self.assertEqual(live, entry)

    def test_a_failure_is_cached_like_a_success(self):
        self._probe(side_effect=TimeoutError("slow"))
        _, req = self._probe(side_effect=TimeoutError("slow"))
        req.assert_not_called()

    def test_it_logs_once_per_change_of_state(self):
        with self.assertLogs("llamaman", level="INFO") as logs:
            for _ in range(3):
                ca._load_cache.clear()
                self._probe(side_effect=TimeoutError("slow"))
            ca._load_cache.clear()
            self._probe(return_value=Mock(status_code=200, json=Mock(return_value={})))
        self.assertEqual(len([m for m in logs.output if "cannot read live load" in m]), 1)
        self.assertEqual(len([m for m in logs.output if "readable again" in m]), 1)


class UnreachablePeerRoutingTests(_Case):

    def setUp(self):
        super().setUp()
        self.loads = {}
        patcher = patch("api.cluster._peer_live_load", lambda node, _m: self.loads[node["node_id"]])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_snapshot_member_stays_routable(self):
        # The blip that made a served group briefly "not found".
        self.nodes = [_peer("a", _inst())]
        self.loads = {"a": ca.PEER_UNREACHABLE}
        candidates = ca._group_candidates("chat-pool")
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["unknown"])

    def test_it_ranks_behind_every_peer_that_answered(self):
        self.nodes = [_peer("dead", _inst()), _peer("busy", _inst()), _peer("asleep", _inst("sleeping"))]
        self.loads = {"dead": ca.PEER_UNREACHABLE, "busy": _live(active=50, free=0),
                      "asleep": _live(healthy=0, sleeping=1, free=0)}
        ordered = ca._order_candidates(ca._group_candidates("chat-pool"))
        self.assertEqual([c["node"]["node_id"] for c in ordered], ["busy", "asleep", "dead"])

    def test_a_peer_not_hosting_the_group_is_not_a_candidate(self):
        self.nodes = [_peer("a", _inst(group="other"))]
        self.loads = {"a": ca.PEER_UNREACHABLE}
        self.assertEqual(ca._group_candidates("chat-pool"), [])

    def test_a_stopped_snapshot_member_is_not_a_candidate(self):
        self.nodes = [_peer("a", _inst("stopped"))]
        self.loads = {"a": ca.PEER_UNREACHABLE}
        self.assertEqual(ca._group_candidates("chat-pool"), [])

    def test_an_older_peer_needs_a_healthy_snapshot_member(self):
        node = _peer("a", _inst("sleeping"))
        node["snapshot"]["system"]["features"] = []
        self.nodes = [node]
        self.loads = {"a": ca.PEER_UNREACHABLE}
        self.assertEqual(ca._group_candidates("chat-pool"), [])

    def test_it_stays_out_of_work_stealing_and_evacuation(self):
        self.nodes = [_peer("a", _inst())]
        self.loads = {"a": ca.PEER_UNREACHABLE}
        self.assertIsNone(ca._find_free_peer("chat-pool"))
        self.assertIsNone(ca._find_evacuation_peer("chat-pool"))

    def test_it_is_no_evidence_for_eager_wake(self):
        self.nodes = [_peer("a", _inst())]
        with patch("api.cluster._peer_load_map", return_value=ca.PEER_UNREACHABLE):
            self.assertFalse(ca.group_wants_wake("chat-pool", {"sleeping": 1}))


class UnroutableErrorTests(_Case):

    def _error(self, name="chat-pool", refusals=None, headers=None):
        from flask import g
        with self.app().test_request_context("/v1/chat/completions", method="POST", headers=headers or {}):
            if refusals:
                g.forward_refusals = refusals
            return ca.group_unroutable_error(name)

    def test_each_node_is_named_with_its_reason(self):
        self.nodes = [_peer("gone", _inst(), online=False), _peer("view", _inst(), url=False)]
        err = self._error()
        self.assertEqual(err["http_status"], 503)
        self.assertEqual(err["code"], "group_unavailable")
        self.assertIn("gone (offline)", err["message"])
        self.assertIn("view (view-only, no CLUSTER_ADVERTISE_URL)", err["message"])

    def test_a_full_peer_is_429(self):
        self.nodes = [_peer("a", _inst()), _peer("gone", _inst(), online=False)]
        err = self._error(refusals={"a": ca._REFUSED_FULL})
        self.assertEqual(err["http_status"], 429)
        self.assertEqual(err["code"], "group_full")
        self.assertIn("a (queue full)", err["message"])

    def test_a_refused_forward_is_reported_as_such(self):
        self.nodes = [_peer("a", _inst())]
        err = self._error(refusals={"a": ca._REFUSED_UNREACHABLE})
        self.assertEqual(err["http_status"], 503)
        self.assertIn("a (unreachable)", err["message"])

    def test_a_partial_name_that_was_forwarded_counts(self):
        # Routing is fuzzy, so a refusal means it did match something.
        self.nodes = [_peer("a", _inst())]
        self.assertIn("a (queue full)", self._error("chat", {"a": ca._REFUSED_FULL})["message"])

    def test_an_unrelated_name_is_none(self):
        self.nodes = [_peer("a", _inst())]
        self.assertIsNone(self._error("llama-3"))

    def test_a_partial_name_nobody_tried_is_none(self):
        self.nodes = [_peer("a", _inst(), online=False)]
        self.assertIsNone(self._error("chat"))

    def test_only_stopped_members_is_none(self):
        self.nodes = [_peer("a", _inst("stopped"))]
        self.assertIsNone(self._error())

    def test_this_node_is_not_reported(self):
        self.nodes = [_peer(SELF, _inst())]
        self.assertIsNone(self._error())

    def test_a_forwarded_request_is_left_to_its_entry_node(self):
        self.nodes = [_peer("gone", _inst(), online=False)]
        self.assertIsNone(self._error(headers={"X-Cluster-Dispatch": "1"}))

    def test_clustering_off_is_none(self):
        self.nodes = [_peer("gone", _inst(), online=False)]
        with patch("api.cluster.cl.is_cluster_enabled", return_value=False):
            self.assertIsNone(self._error())

    def test_outside_a_request_is_none(self):
        self.nodes = [_peer("gone", _inst(), online=False)]
        self.assertIsNone(ca.group_unroutable_error("chat-pool"))

    def test_a_stem_only_pool_counts(self):
        self.nodes = [_peer("gone", _inst(group=""), online=False)]
        self.assertIsNotNone(self._error("m"))


class SurfaceTests(_Case):
    """Entry node C, hosting nothing, on every surface."""

    OPENAI = {"model": "chat-pool", "messages": [{"role": "user", "content": "hi"}]}
    ROUTES = (
        ("/api/chat", OPENAI),
        ("/v1/chat/completions", OPENAI),
        ("/v1/completions", {"model": "chat-pool", "prompt": "hi"}),
        ("/v1/embeddings", {"model": "chat-pool", "input": "hi"}),
    )

    def setUp(self):
        super().setUp()
        self.library = []
        for target, value in (
            ("api.llamaman._library", lambda: self.library),
            ("core.model_alias.resolve_to_path", Mock(return_value=None)),
            ("core.model_alias.canonical_name", lambda n: n),
            ("core.engines.virtual_model_key", lambda _n: None),
            ("api.llamaman._openai_can_evict_admin_instances", lambda: False),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _post(self, path, body, probe):
        with patch("core.cluster.cluster_request", **probe):
            return self.app().test_client().post(path, json=body)

    def _message(self, response):
        error = response.get_json()["error"]
        return error if isinstance(error, str) else error["message"]

    def test_every_peer_full_is_429_everywhere(self):
        self.nodes = [_peer("a", _inst())]

        def request(_node, method, _path, **_kw):
            if method == "GET":
                return Mock(status_code=200, json=Mock(return_value={"chat-pool": _live(free=0)}))
            return Mock(status_code=429, headers={})

        for path, body in self.ROUTES:
            with self.subTest(path=path):
                ca._load_cache.clear()
                r = self._post(path, body, {"side_effect": request})
                self.assertEqual(r.status_code, 429, r.get_data(as_text=True))
                self.assertIn("a (queue full)", self._message(r))

    def test_only_a_view_only_member_is_503_everywhere(self):
        self.nodes = [_peer("view", _inst(), url=False)]
        for path, body in self.ROUTES:
            with self.subTest(path=path):
                r = self._post(path, body, {"side_effect": AssertionError("not contacted")})
                self.assertEqual(r.status_code, 503, r.get_data(as_text=True))
                self.assertIn("view-only", self._message(r))

    def test_each_surface_keeps_its_shape(self):
        self.nodes = [_peer("view", _inst(), url=False)]
        r = self._post("/api/chat", self.OPENAI, {})
        self.assertIsInstance(r.get_json()["error"], str)
        r = self._post("/v1/chat/completions", self.OPENAI, {})
        self.assertEqual(r.get_json()["error"]["code"], "group_unavailable")

    def test_a_dead_peer_is_tried_and_reported(self):
        self.nodes = [_peer("dead", _inst())]
        r = self._post("/v1/chat/completions", self.OPENAI, {"side_effect": ConnectionError("down")})
        self.assertEqual(r.status_code, 503)
        self.assertIn("dead (unreachable)", self._message(r))

    def test_it_does_not_launch_a_file_the_name_substring_matches(self):
        self.nodes = [_peer("view", _inst(), url=False)]
        self.library = [{"path": "/models/chat-pool-unrelated.gguf", "name": "x"}]
        with patch("api.instances.launch_instance") as launch:
            r = self._post("/v1/chat/completions", self.OPENAI, {})
        self.assertEqual(r.status_code, 503)
        launch.assert_not_called()

    def test_a_file_with_exactly_that_name_still_runs_here(self):
        self.nodes = [_peer("view", _inst(), url=False)]
        self.library = [{"path": "/models/chat-pool.gguf", "name": "x"}]
        running = {"id": "local", "status": "healthy"}
        with patch.object(llamaman, "_find_running_instance_for_model", return_value=running):
            inst, err = llamaman._ensure_model_running("chat-pool")
        self.assertIsNone(err)
        self.assertEqual(inst["id"], "local")

    def test_an_unknown_model_is_still_not_found(self):
        r = self._post("/v1/chat/completions", {**self.OPENAI, "model": "llama-3"}, {})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.get_json(), {"error": {"message": "model 'llama-3' not found"}})


class TimeoutShapeTests(_Case):

    def _timeout(self, path):
        import requests
        node = {"node_id": "a", "node_name": "a"}
        with self.app().test_request_context(path, method="POST"), \
                patch("core.cluster.cluster_request", side_effect=requests.ReadTimeout("slow")):
            out = ca._forward_inference(node, path, b"{}", None)
        self.assertEqual(out.status_code, 504)
        return json.loads(out.get_data())["error"]

    def test_ollama_gets_a_plain_string(self):
        self.assertIn("did not respond", self._timeout("/api/chat"))

    def test_openai_gets_the_nested_shape(self):
        self.assertIn("did not respond", self._timeout("/v1/chat/completions")["message"])


class EmbeddingRoutingTests(_Case):

    def test_an_embedding_group_is_forwarded_to_its_peer(self):
        self.nodes = [_peer("a", _inst())]
        peer = Mock(status_code=200, headers={"Content-Type": "application/json"},
                    content=b'{"data": []}')

        def request(_node, method, _path, **_kw):
            if method == "GET":
                return Mock(status_code=200, json=Mock(return_value={"chat-pool": _live()}))
            return peer

        with patch("core.cluster.cluster_request", side_effect=request):
            r = self.app().test_client().post("/v1/embeddings", json={"model": "chat-pool", "input": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"data": []})


if __name__ == "__main__":
    unittest.main()
