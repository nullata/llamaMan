# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Cluster routing and queue sharing with Strata instances.

A two-node cluster, simulated in-process: this node's instances and gates are
real (core.state / proxy), the peer is a registered node whose live load
(_peer_live_load) and HTTP answers (core.cluster.cluster_request) are faked.

Pinned here:
  * a Strata instance's group key is its model id ("strata/qwen-iq3_s"), on
    both sides of the wire (local load map, peer load matching);
  * every name a client may use reaches the group - the model id, its short
    form, and Strata's own served name ("qwen3.8-flash-next-iq3_s"), which
    did not route across nodes before;
  * least-load dispatch, fallback tiering and work-stealing treat a Strata
    member like any other (max_concurrent 1: one busy request = saturated);
  * a share_queue_group pools a Strata instance with a llama.cpp one;
  * a Strata answer relayed through a peer streams (HTTP/1.0 SSE, no
    chunked framing) instead of arriving all at once.
"""

import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from flask import Flask

import api.cluster as cluster_api
import config
import proxy
from core.engines import virtual_model_key
from core.state import instances, instances_lock

STRATA_PATH = "/strata/qwen-IQ3_S"
GROUP_KEY = "strata/qwen-iq3_s"
SERVED_NAME = "qwen3.8-flash-next-iq3_s"
PEER = {"node_id": "peer", "node_name": "srv2", "advertise_url": "http://srv2:5000"}


class _Storage:
    def __init__(self, nodes):
        self._nodes = nodes

    def list_nodes(self):
        return [dict(n) for n in self._nodes]


class _ClusterCase(unittest.TestCase):
    """This node + one peer; cluster on; no real network."""

    def setUp(self):
        with instances_lock:
            self._saved = dict(instances)
            instances.clear()
        cluster_api._load_cache.clear()
        with cluster_api._inflight_lock:
            cluster_api._inflight.clear()
        self.peer_load = None          # what the peer's /api/cluster/local-load reports
        self._gates = []
        self._p = [
            patch.object(config, "CLUSTER_ENABLED", True),
            patch.object(config, "CLUSTER_SECRET", "s"),
            patch("api.cluster.get_storage", return_value=_Storage([PEER])),
            patch("api.cluster._is_online", return_value=True),
            patch("api.cluster._peer_live_load",
                  side_effect=lambda node, name: cluster_api._match_model_load(self.peer_load or {}, name)),
        ]
        for p in self._p:
            p.start()
        self.app = Flask(__name__)

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()
        for inst_id in self._gates:
            proxy.remove_gate(inst_id)
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def add_local(self, inst_id, model_path, busy=0, **cfg):
        config_ = {"share_queue": True, "max_concurrent": 1, "max_queue_depth": 10, **cfg}
        if model_path.startswith("/strata/"):
            config_.setdefault("engine", "strata")
        with instances_lock:
            instances[inst_id] = {"id": inst_id, "model_path": model_path, "status": "healthy",
                                  "port": 8001, "config": config_}
        proxy.create_gate(inst_id, config_["max_concurrent"], config_["max_queue_depth"],
                          model_path=model_path, share_queue=True)
        self._gates.append(inst_id)
        gate = proxy.get_gate(inst_id)
        for _ in range(busy):
            self.assertTrue(gate.acquire(timeout=1))
        self.addCleanup(lambda: [gate.release() for _ in range(busy)])
        return gate

    def dispatch(self, model, forward_resp=None):
        """Run dispatch_inference for a chat request; returns (response or
        None, the peer paths forwarded to)."""
        forwarded = []

        def fake_cluster_request(node, method, path, **kw):
            forwarded.append((node["node_id"], path, kw.get("headers", {}).get("X-Cluster-Dispatch")))
            return forward_resp or _JsonResp()

        with self.app.test_request_context("/v1/chat/completions", method="POST",
                                           json={"model": model, "messages": []}):
            with patch("core.cluster.cluster_request", side_effect=fake_cluster_request):
                resp = cluster_api.dispatch_inference(model)
        return resp, forwarded


class _JsonResp:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    content = b'{"ok": 1}'

    def iter_content(self, chunk_size=None):
        yield self.content

    def close(self):
        pass


def _peer_load(active=0, queued=0, free=1, fallback=False, key=GROUP_KEY):
    return {key: {"active": active, "queued": queued, "free": free,
                  "max_concurrent": 1, "fallback": fallback}}


class StrataGroupKeyTests(_ClusterCase):

    def test_group_key_is_the_model_id(self):
        self.assertEqual(cluster_api.effective_group_key(STRATA_PATH, {"engine": "strata"}), GROUP_KEY)
        self.assertEqual(cluster_api.effective_group_key(STRATA_PATH, {"share_queue_group": "Qwen-Next"}),
                         "qwen-next")

    def test_local_load_map_is_keyed_for_peers(self):
        self.add_local("s1", STRATA_PATH, busy=1)
        load = cluster_api.local_load_by_model()
        self.assertEqual(load[GROUP_KEY], {"active": 1, "queued": 0, "free": 0,
                                           "max_concurrent": 1, "fallback": False})

    def test_every_client_name_reaches_the_group(self):
        inst = {"model_path": STRATA_PATH, "config": {"engine": "strata", "share_queue": True}}
        for name in ("strata/qwen-IQ3_S", "strata/qwen-iq3_s", "qwen-IQ3_S"):
            self.assertTrue(cluster_api._inst_matches_request(inst, name), name)
            self.assertIsNotNone(cluster_api._match_model_load(_peer_load(), name), name)

    def test_served_name_translates_to_group_key(self):
        self.assertEqual(virtual_model_key(SERVED_NAME), GROUP_KEY)
        self.assertEqual(virtual_model_key(SERVED_NAME.upper() + ":latest"), GROUP_KEY)
        self.assertEqual(virtual_model_key("qwen3.8-flash-next-unsloth-ud-q4_k_xl"),
                         "strata/unsloth-ud-q4_k_xl")
        self.assertIsNone(virtual_model_key("gpt-oss-20b"))
        self.assertIsNone(virtual_model_key(""))


class StrataDispatchTests(_ClusterCase):

    def test_busy_local_forwards_to_idle_peer(self):
        self.add_local("s1", STRATA_PATH, busy=1)          # max_concurrent 1: saturated
        self.peer_load = _peer_load(active=0, free=1)
        resp, fwd = self.dispatch("strata/qwen-IQ3_S")
        self.assertIsNotNone(resp)
        self.assertEqual(fwd, [("peer", "/v1/chat/completions", "1")])

    def test_served_name_forwards_too(self):
        # Regression: the served name matched no group, so no cross-node routing.
        self.add_local("s1", STRATA_PATH, busy=1)
        self.peer_load = _peer_load(active=0, free=1)
        resp, fwd = self.dispatch(SERVED_NAME)
        self.assertIsNotNone(resp)
        self.assertEqual([f[0] for f in fwd], ["peer"])

    def test_peer_only_member_gets_the_request(self):
        # This node doesn't run it at all; the peer does.
        self.peer_load = _peer_load()
        resp, fwd = self.dispatch(SERVED_NAME)
        self.assertIsNotNone(resp)
        self.assertEqual([f[0] for f in fwd], ["peer"])

    def test_idle_local_beats_busy_peer(self):
        self.add_local("s1", STRATA_PATH, busy=0)
        self.peer_load = _peer_load(active=1, queued=3, free=0)
        resp, fwd = self.dispatch("strata/qwen-IQ3_S")
        self.assertIsNone(resp)                              # serve here
        self.assertEqual(fwd, [])

    def test_no_peer_member_serves_locally(self):
        self.add_local("s1", STRATA_PATH, busy=1)
        self.peer_load = {}
        resp, fwd = self.dispatch("strata/qwen-IQ3_S")
        self.assertIsNone(resp)
        self.assertEqual(fwd, [])

    def test_other_strata_size_is_a_different_group(self):
        self.add_local("s1", STRATA_PATH, busy=1)
        self.peer_load = _peer_load(key="strata/qwen-iq2_xs")
        resp, fwd = self.dispatch("strata/qwen-IQ3_S")
        self.assertIsNone(resp)
        self.assertEqual(fwd, [])

    def test_idle_fallback_peer_waits_behind_primary(self):
        self.add_local("s1", STRATA_PATH, busy=0)
        self.peer_load = _peer_load(active=0, free=1, fallback=True)
        cands = cluster_api._order_candidates(cluster_api._group_candidates(GROUP_KEY))
        self.assertTrue(cands[0]["is_self"])
        self.assertTrue(cands[-1]["fallback"])

    def test_peer_429_falls_back_to_local(self):
        self.add_local("s1", STRATA_PATH, busy=1)
        self.peer_load = _peer_load(active=0, free=1)

        class Full(_JsonResp):
            status_code = 429
        resp, fwd = self.dispatch("strata/qwen-IQ3_S", forward_resp=Full())
        self.assertIsNone(resp)                              # queue locally instead
        self.assertEqual([f[0] for f in fwd], ["peer"])


class StrataWorkStealingTests(_ClusterCase):

    def test_queued_request_finds_free_peer(self):
        self.peer_load = _peer_load(active=0, free=1)
        self.assertEqual(cluster_api._find_free_peer(GROUP_KEY)["node_id"], "peer")

    def test_saturated_peer_is_not_a_target(self):
        self.peer_load = _peer_load(active=1, free=0)
        self.assertIsNone(cluster_api._find_free_peer(GROUP_KEY))

    def test_recent_forward_counts_against_peer(self):
        self.peer_load = _peer_load(active=0, free=1)
        cluster_api._inflight_inc("peer")                     # one already heading there
        self.assertIsNone(cluster_api._find_free_peer(GROUP_KEY))


class MixedEngineGroupTests(_ClusterCase):

    def test_strata_and_llamacpp_pool_under_one_alias(self):
        self.add_local("s1", STRATA_PATH, busy=1, share_queue_group="qwen-next")
        self.peer_load = _peer_load(active=0, free=1, key="qwen-next")   # a llama.cpp copy on the peer
        cands = cluster_api._group_candidates("qwen-next")
        self.assertEqual(sorted(c["is_self"] for c in cands), [False, True])
        resp, fwd = self.dispatch("qwen-next")
        self.assertIsNotNone(resp)
        self.assertEqual([f[0] for f in fwd], ["peer"])

    def test_alias_load_reported_under_alias(self):
        self.add_local("s1", STRATA_PATH, busy=0, share_queue_group="qwen-next")
        self.add_local("l1", "/models/Qwen-Next-Q4_K_M.gguf", busy=1, share_queue_group="qwen-next")
        load = cluster_api.local_load_by_model()
        self.assertEqual(list(load), ["qwen-next"])
        self.assertEqual((load["qwen-next"]["active"], load["qwen-next"]["free"]), (1, 1))


_GATE = threading.Event()


class _StrataLikePeer(BaseHTTPRequestHandler):
    """A peer answering like Strata does: HTTP/1.0 SSE, no chunked framing."""
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n')
        self.wfile.flush()
        _GATE.wait(5)
        self.wfile.write(b"data: [DONE]\n\n")


class StreamedRelayThroughPeerTests(_ClusterCase):

    def test_strata_stream_relayed_through_peer_is_not_buffered(self):
        _GATE.clear()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _StrataLikePeer)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_port}"

        def real_request(node, method, path, data=None, headers=None, stream=False, timeout=None):
            return requests.request(method, base + path, data=data, headers=headers,
                                    stream=stream, timeout=timeout)

        self.add_local("s1", STRATA_PATH, busy=1)
        self.peer_load = _peer_load(active=0, free=1)
        with self.app.test_request_context("/v1/chat/completions", method="POST",
                                           json={"model": SERVED_NAME, "stream": True}):
            with patch("core.cluster.cluster_request", side_effect=real_request):
                t = time.monotonic()
                resp = cluster_api.dispatch_inference(SERVED_NAME)
                self.assertIsNotNone(resp)
                body = iter(resp.response)
                first = next(body)
                delay = time.monotonic() - t
                _GATE.set()
                rest = b"".join(body)
        self.assertLess(delay, 2.0)                           # not held until the peer closes
        self.assertIn(b'"Hi"', first)
        self.assertEqual(rest, b"data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
